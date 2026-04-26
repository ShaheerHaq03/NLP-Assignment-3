
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import gc
import json
import math
import time
import argparse
import random
import warnings
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset as TorchDataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup

warnings.filterwarnings("ignore")


# ============================================================
# 1. PROFILES
# ============================================================

PROFILES = {
    # Use this first only to verify everything works.
    "smoke": {
        "output_dir": "outputs_smoke",
        "model_name": "t5-small",
        "max_train_per_public_dataset": 1000,
        "max_train_combo": 1500,
        "max_eval_per_dataset": 300,
        "gnli_examples_per_domain_length": 12,
        "epochs": 1,
        "batch_size": 4,
        "eval_batch_size": 8,
        "grad_accum_steps": 2,
        "learning_rate": 5e-4,
        "max_input_length": 256,
        "max_target_length": 8,
        "run_combinations": True,
        "run_maw": True,
        "run_enhanced_all": False,
        "run_true_if_available": False,
        "save_models": False,
    },

    # Best balanced config for RTX 4060 + 16GB RAM.
    "final": {
        "output_dir": "outputs_final",
        "model_name": "t5-small",
        "max_train_per_public_dataset": 12000,
        "max_train_combo": 16000,
        "max_eval_per_dataset": 1500,
        "gnli_examples_per_domain_length": 90,
        "epochs": 1,
        "batch_size": 4,
        "eval_batch_size": 12,
        "grad_accum_steps": 4,
        "learning_rate": 5e-4,
        "max_input_length": 384,
        "max_target_length": 8,
        "run_combinations": True,
        "run_maw": True,
        "run_enhanced_all": True,
        "run_true_if_available": True,
        "save_models": False,
    },

    # Use this only after final succeeds. It will take longer.
    "enhanced": {
        "output_dir": "outputs_enhanced",
        "model_name": "t5-small",
        "max_train_per_public_dataset": 20000,
        "max_train_combo": 26000,
        "max_eval_per_dataset": 2500,
        "gnli_examples_per_domain_length": 150,
        "epochs": 2,
        "batch_size": 4,
        "eval_batch_size": 12,
        "grad_accum_steps": 4,
        "learning_rate": 5e-4,
        "max_input_length": 512,
        "max_target_length": 8,
        "run_combinations": True,
        "run_maw": True,
        "run_enhanced_all": True,
        "run_true_if_available": True,
        "save_models": False,
    }
}

LABELS_3WAY = ["entailment", "neutral", "contradiction"]
LABELS_BINARY = ["entailment", "non-entailment"]


# ============================================================
# 2. UTILITIES
# ============================================================

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sanitize_name(x: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", x)


def clean_text(x) -> str:
    if x is None:
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def unwrap_model(model):
    """Return the original model if DataParallel is used."""
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def maybe_wrap_multi_gpu(model, cfg):
    """Use both Kaggle GPUs through DataParallel when available."""
    use_multi_gpu = cfg.get("use_multi_gpu", True)

    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and use_multi_gpu:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    return model


def setup_torch():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)



def get_secret_or_env(name: str):
    """
    Reads secret from Kaggle Secrets first, then environment variables.
    Never prints secret values.
    """
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        value = user_secrets.get_secret(name)
        if value:
            return value
    except Exception:
        pass

    return os.environ.get(name)


def run_shell(cmd, cwd=None, allow_fail=True):
    """
    Runs shell command safely without printing secrets.
    """
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd,
        text=True,
        capture_output=True
    )

    if result.returncode != 0:
        print(f"[GitHub backup] Command failed with code {result.returncode}")
        if result.stderr:
            print(result.stderr[-1000:])
        if not allow_fail:
            raise RuntimeError(result.stderr)

    return result


def prepare_github_backup_repo(cfg: Dict):
    """
    Clones or updates the GitHub backup repo.
    Requires Kaggle Secrets:
        GITHUB_USERNAME
        GITHUB_TOKEN
        GITHUB_REPO
    """
    if not cfg.get("push_to_github", False):
        return None

    username = get_secret_or_env("GITHUB_USERNAME")
    token = get_secret_or_env("GITHUB_TOKEN")
    repo = get_secret_or_env("GITHUB_REPO") or cfg.get("github_repo")

    if not username or not token or not repo:
        print("[GitHub backup] Missing GitHub secrets. Skipping backup.")
        print("[GitHub backup] Required secrets: GITHUB_USERNAME, GITHUB_TOKEN, GITHUB_REPO")
        return None

    backup_root = Path("/kaggle/working/github_auto_backup")
    backup_root.parent.mkdir(parents=True, exist_ok=True)

    clone_url = f"https://{username}:{token}@github.com/{username}/{repo}.git"

    run_shell('git config --global user.email "kaggle-run@example.com"', allow_fail=True)
    run_shell('git config --global user.name "Kaggle Runner"', allow_fail=True)

    if not backup_root.exists():
        print("[GitHub backup] Cloning backup repository...")
        result = subprocess.run(
            ["git", "clone", clone_url, str(backup_root)],
            text=True,
            capture_output=True
        )
        if result.returncode != 0:
            print("[GitHub backup] Clone failed.")
            print(result.stderr[-1000:])
            return None
    else:
        print("[GitHub backup] Repository already exists. Pulling latest changes...")
        run_shell("git pull", cwd=backup_root, allow_fail=True)

    return backup_root


def remove_large_files(folder: Path, max_mb: int = 90):
    """
    GitHub blocks files above 100MB.
    This removes oversized files from the backup copy.
    """
    if not folder.exists():
        return

    max_bytes = max_mb * 1024 * 1024

    for file_path in folder.rglob("*"):
        if file_path.is_file():
            try:
                if file_path.stat().st_size > max_bytes:
                    print(f"[GitHub backup] Removing large file from backup copy: {file_path}")
                    file_path.unlink()
            except Exception:
                pass


def sync_outputs_to_github(output_dir: Path, cfg: Dict, message: str = "Backup experiment outputs"):
    """
    Copies current outputs to GitHub repo and pushes them.
    This is called after each completed model and again at the end.
    """
    if not cfg.get("push_to_github", False):
        return

    backup_root = prepare_github_backup_repo(cfg)
    if backup_root is None:
        return

    output_dir = Path(output_dir)
    if not output_dir.exists():
        print(f"[GitHub backup] Output folder does not exist yet: {output_dir}")
        return

    subdir = cfg.get("github_subdir") or f"assignment3_{output_dir.name}"
    destination = backup_root / subdir
    destination.mkdir(parents=True, exist_ok=True)

    # Copy output folder
    copied_output = destination / output_dir.name

    ignore_patterns = shutil.ignore_patterns(
        "__pycache__",
        ".ipynb_checkpoints",
        ".cache",
        "model",
        "tokenizer",
        "*.bin",
        "*.pt",
        "*.pth",
        "*.safetensors"
    )

    print(f"[GitHub backup] Copying {output_dir} to GitHub backup folder...")
    shutil.copytree(output_dir, copied_output, dirs_exist_ok=True, ignore=ignore_patterns)

    # Copy script and logs if present
    if Path("nli_enhancement.py").exists():
        shutil.copy2("nli_enhancement.py", destination / "nli_enhancement.py")

    for log_file in Path("/kaggle/working").glob("*.txt"):
        try:
            shutil.copy2(log_file, destination / log_file.name)
        except Exception:
            pass

    # Save sync metadata
    sync_info = {
        "message": message,
        "output_dir": str(output_dir),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu_info": get_gpu_info(),
        "config": cfg
    }
    save_json(sync_info, destination / "last_sync_info.json")

    # Avoid GitHub file size failure
    remove_large_files(destination, max_mb=90)

    # Commit and push
    print("[GitHub backup] Committing and pushing outputs...")
    run_shell("git add .", cwd=backup_root, allow_fail=True)

    commit_msg = message.replace('"', "'")
    commit_result = run_shell(f'git commit -m "{commit_msg}"', cwd=backup_root, allow_fail=True)

    # Push even if commit says nothing changed
    push_result = run_shell("git push", cwd=backup_root, allow_fail=True)

    if push_result.returncode == 0:
        print("[GitHub backup] Push completed.")
    else:
        print("[GitHub backup] Push may have failed. Check token/repo permissions.")


def normalize_label(raw, label_names=None) -> str:
    if isinstance(raw, (int, np.integer)):
        if label_names is not None:
            try:
                return normalize_label(label_names[int(raw)])
            except Exception:
                pass
        mapping = {0: "entailment", 1: "neutral", 2: "contradiction"}
        return mapping.get(int(raw), "unknown")

    s = str(raw).strip().lower()
    s = s.replace("entails", "entailment")
    s = s.replace("entail", "entailment")
    s = s.replace("contradicts", "contradiction")
    s = s.replace("contradictory", "contradiction")

    if s in ["e", "entailment"]:
        return "entailment"
    if s in ["n", "neutral"]:
        return "neutral"
    if s in ["c", "contradiction"]:
        return "contradiction"
    if s in ["not_entailment", "not entailment", "non-entailment", "non_entailment"]:
        return "non-entailment"

    if "contradiction" in s:
        return "contradiction"
    if "neutral" in s:
        return "neutral"
    if "entailment" in s and "non" not in s:
        return "entailment"

    return "unknown"


def to_binary_label(label: str) -> str:
    label = normalize_label(label)
    return "entailment" if label == "entailment" else "non-entailment"


def stratified_sample(records: List[Dict], max_n: Optional[int], seed: int = 42) -> List[Dict]:
    if max_n is None or len(records) <= max_n:
        return list(records)

    df = pd.DataFrame(records)
    if "label" not in df.columns:
        return df.sample(n=max_n, random_state=seed).to_dict("records")

    output = []
    per_label = max_n // 3

    for label in LABELS_3WAY:
        sub = df[df["label"] == label]
        if len(sub) > 0:
            take = min(per_label, len(sub))
            output.extend(sub.sample(n=take, random_state=seed).to_dict("records"))

    remaining = max_n - len(output)
    if remaining > 0:
        selected_ids = set(id(x) for x in output)
        extra_df = df.sample(frac=1, random_state=seed + 1)
        extra = []
        for row in extra_df.to_dict("records"):
            if len(extra) >= remaining:
                break
            extra.append(row)
        output.extend(extra[:remaining])

    random.Random(seed).shuffle(output)
    return output[:max_n]


def make_nli_input(premise: str, hypothesis: str) -> str:
    return f"nli premise: {premise} hypothesis: {hypothesis} label:"


def make_examples(records: List[Dict], binary: bool = False) -> List[Dict]:
    examples = []
    for r in records:
        label = normalize_label(r["label"])
        if label not in LABELS_3WAY:
            continue

        target = to_binary_label(label) if binary else label
        examples.append({
            "input_text": make_nli_input(r["premise"], r["hypothesis"]),
            "target_text": target,
            "label": target,
            "source": r.get("source", "unknown"),
        })
    return examples


def get_gpu_info() -> Dict:
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device": str(get_device()),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
    return info


# ============================================================
# 3. LOAD PUBLIC DATASETS
# ============================================================

def get_label_names(ds, label_col="label"):
    try:
        feature = ds.features[label_col]
        if hasattr(feature, "names"):
            return feature.names
    except Exception:
        return None
    return None


def extract_records(ds, dataset_name: str) -> List[Dict]:
    cols = ds.column_names

    premise_col = None
    hyp_col = None
    label_col = None

    for c in ["premise", "sentence1", "context", "grounding"]:
        if c in cols:
            premise_col = c
            break

    for c in ["hypothesis", "sentence2", "generated_text"]:
        if c in cols:
            hyp_col = c
            break

    for c in ["label", "gold", "gold_label"]:
        if c in cols:
            label_col = c
            break

    if premise_col is None or hyp_col is None or label_col is None:
        raise ValueError(f"Could not identify columns for {dataset_name}. Columns={cols}")

    label_names = get_label_names(ds, label_col)
    records = []

    for row in ds:
        label = normalize_label(row[label_col], label_names=label_names)
        if label not in LABELS_3WAY:
            continue

        premise = clean_text(row[premise_col])
        hypothesis = clean_text(row[hyp_col])

        if not premise or not hypothesis:
            continue

        records.append({
            "premise": premise,
            "hypothesis": hypothesis,
            "label": label,
            "source": dataset_name,
        })

    return records


def load_mnli() -> Tuple[List[Dict], List[Dict]]:
    print("\nLoading MNLI...")
    ds = load_dataset("nyu-mll/multi_nli")
    train = extract_records(ds["train"], "MNLI")
    eval_split = "validation_matched" if "validation_matched" in ds else list(ds.keys())[1]
    eval_records = extract_records(ds[eval_split], "MNLI")
    return train, eval_records


def load_anli() -> Tuple[List[Dict], List[Dict]]:
    print("\nLoading ANLI...")
    ds = load_dataset("facebook/anli")

    train_splits = [s for s in ["train_r1", "train_r2", "train_r3"] if s in ds]
    dev_splits = [s for s in ["dev_r1", "dev_r2", "dev_r3"] if s in ds]

    if not train_splits:
        raise RuntimeError("ANLI train splits not found.")
    if not dev_splits:
        raise RuntimeError("ANLI dev splits not found.")

    train_ds = concatenate_datasets([ds[s] for s in train_splits])
    dev_ds = concatenate_datasets([ds[s] for s in dev_splits])

    train = extract_records(train_ds, "ANLI")
    dev = extract_records(dev_ds, "ANLI")
    return train, dev


def load_wanli() -> Tuple[List[Dict], List[Dict]]:
    print("\nLoading WANLI...")
    ds = load_dataset("alisawuffles/WANLI")
    split_names = list(ds.keys())

    train_split = "train" if "train" in split_names else split_names[0]
    eval_split = "test" if "test" in split_names else ("validation" if "validation" in split_names else split_names[-1])

    train = extract_records(ds[train_split], "WANLI")
    eval_records = extract_records(ds[eval_split], "WANLI")
    return train, eval_records


# ============================================================
# 4. GNLI-STYLE SYNTHETIC DATA
# ============================================================

GNLI_DOMAINS = [
    "ads", "blog post", "book reviews", "casual dialog",
    "chat message", "email", "essay", "fans forum",
    "forum post", "google play reviews", "government documents",
    "legal", "legal document", "medical", "movie plot",
    "movie reviews", "news", "news comments", "news headlines",
    "phone conversation", "place reviews", "quora", "recipe",
    "reddit comment", "reddit title", "research paper abstract",
    "scientific article", "shopping reviews", "song lyrics",
    "sports news", "story for kids", "student forum",
    "student papers", "support forum", "travel guides",
    "twitter", "wikipedia", "youtube comments"
]

PEOPLE = ["Amina", "Bilal", "Sara", "Usman", "Hina", "Omar", "Maya", "Daniel", "Noah", "Fatima"]
ITEMS = ["phone", "laptop", "book", "camera", "medical report", "rental listing", "software update", "restaurant order"]
PLACES = ["Islamabad", "Lahore", "Karachi", "London", "a small clinic", "a university office", "a travel agency"]
DAYS = ["Monday", "Tuesday", "Friday", "the next morning", "last week", "two days later"]
ISSUES = ["a delayed response", "a broken screen", "missing information", "a payment error", "slow service", "unclear instructions"]
OUTCOMES = ["was resolved", "remained unresolved", "was delayed", "was approved", "was rejected", "needed another review"]


def make_domain_premise(domain: str, length: str, idx: int) -> Dict:
    rng = random.Random(1000 + idx + len(domain))

    person = rng.choice(PEOPLE)
    item = rng.choice(ITEMS)
    place = rng.choice(PLACES)
    day = rng.choice(DAYS)
    issue = rng.choice(ISSUES)
    outcome = rng.choice(OUTCOMES)

    if length == "short":
        premise = (
            f"In a {domain} text, {person} says the {item} in {place} had {issue} on {day}, "
            f"and the case {outcome}."
        )
    else:
        premise = (
            f"In a {domain} text, {person} describes a situation involving a {item} in {place}. "
            f"The issue started on {day} and included {issue}. "
            f"Several details were checked before the final update. "
            f"According to the text, the case {outcome}."
        )

    label_type = idx % 3

    if label_type == 0:
        hypothesis = f"{person} describes an issue involving a {item}."
        label = "entailment"
    elif label_type == 1:
        hypothesis = f"{person} says there was no issue involving the {item}."
        label = "contradiction"
    else:
        hypothesis = f"{person} later shared the story with a close friend."
        label = "neutral"

    return {
        "premise": premise,
        "hypothesis": hypothesis,
        "label": label,
        "source": "GNLI_PROXY",
        "domain": domain,
        "length": length,
    }


def build_gnli_proxy(cfg: Dict) -> Tuple[List[Dict], List[Dict]]:
    print("\nBuilding GNLI-style synthetic proxy dataset...")

    records = []
    n = cfg["gnli_examples_per_domain_length"]

    for domain in GNLI_DOMAINS:
        for length in ["short", "paragraph"]:
            for i in range(n):
                records.append(make_domain_premise(domain, length, i))

    random.Random(42).shuffle(records)

    split = int(0.8 * len(records))
    train = records[:split]
    test = records[split:]

    return train, test


# ============================================================
# 5. DATASET CLASS AND COLLATE
# ============================================================

class Text2TextTorchDataset(TorchDataset):
    def __init__(self, examples: List[Dict]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch, tokenizer, cfg):
    inputs = [x["input_text"] for x in batch]
    targets = [x["target_text"] for x in batch]

    enc = tokenizer(
        inputs,
        padding=True,
        truncation=True,
        max_length=cfg["max_input_length"],
        return_tensors="pt",
    )

    target_enc = tokenizer(
        targets,
        padding=True,
        truncation=True,
        max_length=cfg["max_target_length"],
        return_tensors="pt",
    )

    labels = target_enc["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100

    enc["labels"] = labels
    return enc


# ============================================================
# 6. CANDIDATE SCORING EVALUATION
# ============================================================

@torch.no_grad()
def score_candidates(
    model,
    tokenizer,
    input_texts: List[str],
    candidates: List[str],
    cfg: Dict,
    device
) -> np.ndarray:
    model.eval()

    enc = tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        max_length=cfg["max_input_length"],
        return_tensors="pt",
    ).to(device)

    all_scores = []

    for cand in candidates:
        target = tokenizer(
            [cand] * len(input_texts),
            padding=True,
            truncation=True,
            max_length=cfg["max_target_length"],
            return_tensors="pt",
        ).to(device)

        labels = target["input_ids"]
        mask = labels != tokenizer.pad_token_id

        decoder_input_ids = unwrap_model(model)._shift_right(labels)

        out = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            decoder_input_ids=decoder_input_ids,
        )

        logits = out.logits
        log_probs = F.log_softmax(logits, dim=-1)

        token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        token_log_probs = token_log_probs * mask

        seq_scores = token_log_probs.sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        all_scores.append(seq_scores.detach().cpu().numpy())

    return np.stack(all_scores, axis=1)


def evaluate_examples(
    model,
    tokenizer,
    examples: List[Dict],
    cfg: Dict,
    candidates: List[str],
    device,
    desc: str = "eval"
) -> Dict:
    y_true = []
    y_pred = []
    y_scores = []

    for start in tqdm(range(0, len(examples), cfg["eval_batch_size"]), desc=desc, leave=False):
        batch = examples[start:start + cfg["eval_batch_size"]]
        inputs = [x["input_text"] for x in batch]
        labels = [x["label"] for x in batch]

        scores = score_candidates(model, tokenizer, inputs, candidates, cfg, device)
        pred_idx = scores.argmax(axis=1)
        preds = [candidates[i] for i in pred_idx]

        y_true.extend(labels)
        y_pred.extend(preds)
        y_scores.extend(scores.tolist())

    acc = accuracy_score(y_true, y_pred)

    return {
        "accuracy": acc,
        "y_true": y_true,
        "y_pred": y_pred,
        "scores": y_scores,
        "candidates": candidates,
    }


# ============================================================
# 7. TRAINING
# ============================================================

def train_model(
    train_name: str,
    train_records: List[Dict],
    dev_records: List[Dict],
    cfg: Dict,
    output_dir: Path,
    binary: bool = False
):
    device = get_device()
    model_name = cfg["model_name"]

    run_name = sanitize_name(f"{train_name}_{'binary' if binary else '3way'}")
    run_dir = output_dir / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_examples = make_examples(train_records, binary=binary)
    dev_examples = make_examples(dev_records, binary=binary)
    dev_examples = stratified_sample(dev_examples, min(600, len(dev_examples)), seed=42)

    print(f"\n{'=' * 80}")
    print(f"Training: {run_name}")
    print(f"Model: {model_name}")
    print(f"Train examples: {len(train_examples)}")
    print(f"Dev examples during training: {len(dev_examples)}")
    print(f"Device: {device}")
    print(f"{'=' * 80}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.config.use_cache = False
    model.to(device)
    model = maybe_wrap_multi_gpu(model, cfg)

    train_ds = Text2TextTorchDataset(train_examples)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, cfg),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=0.01,
    )

    total_update_steps = math.ceil(len(train_loader) / cfg["grad_accum_steps"]) * cfg["epochs"]
    warmup_steps = int(0.06 * total_update_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    logs = []
    global_step = 0

    start_time = time.time()

    for epoch in range(cfg["epochs"]):
        model.train()
        optimizer.zero_grad()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"{run_name} epoch {epoch + 1}/{cfg['epochs']}")

        for step, batch in enumerate(pbar):
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(**batch)
                raw_loss = out.loss
                if raw_loss.dim() > 0:
                    raw_loss = raw_loss.mean()
                loss = raw_loss / cfg["grad_accum_steps"]

            scaler.scale(loss).backward()
            running_loss += loss.item() * cfg["grad_accum_steps"]

            if (step + 1) % cfg["grad_accum_steps"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                avg_loss = running_loss / max(1, step + 1)
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "step": global_step})

                if global_step % 25 == 0:
                    logs.append({
                        "run_name": run_name,
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "train_loss": avg_loss,
                        "learning_rate": scheduler.get_last_lr()[0],
                    })

        candidates = LABELS_BINARY if binary else LABELS_3WAY
        dev_result = evaluate_examples(
            model=model,
            tokenizer=tokenizer,
            examples=dev_examples,
            cfg=cfg,
            candidates=candidates,
            device=device,
            desc=f"dev {run_name}",
        )

        logs.append({
            "run_name": run_name,
            "epoch": epoch + 1,
            "global_step": global_step,
            "train_loss": running_loss / max(1, len(train_loader)),
            "dev_accuracy": dev_result["accuracy"],
            "learning_rate": scheduler.get_last_lr()[0],
        })

        print(f"Dev accuracy after epoch {epoch + 1}: {dev_result['accuracy']:.4f}")

    elapsed = round(time.time() - start_time, 2)

    log_df = pd.DataFrame(logs)
    log_path = run_dir / "training_log.csv"
    log_df.to_csv(log_path, index=False)

    plot_training_curve(log_df, run_dir / "training_curve.png", run_name)

    save_json({
        "run_name": run_name,
        "train_dataset": train_name,
        "binary": binary,
        "model_name": model_name,
        "train_examples": len(train_examples),
        "dev_examples": len(dev_examples),
        "elapsed_seconds": elapsed,
        "config": cfg,
        "gpu_info": get_gpu_info(),
    }, run_dir / "run_metadata.json")

    if cfg["save_models"]:
        unwrap_model(model).save_pretrained(run_dir / "model")
        tokenizer.save_pretrained(run_dir / "tokenizer")

    return model, tokenizer, run_name, elapsed


def plot_training_curve(log_df: pd.DataFrame, path: Path, title: str):
    if log_df.empty or "train_loss" not in log_df.columns:
        return

    df = log_df.dropna(subset=["train_loss"])
    if df.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(df["global_step"], df["train_loss"], marker="o")
    plt.xlabel("Optimizer step")
    plt.ylabel("Training loss")
    plt.title(f"Training Curve: {title}")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


# ============================================================
# 8. RESULT SAVING
# ============================================================

def save_eval_outputs(
    eval_result: Dict,
    output_dir: Path,
    run_name: str,
    eval_name: str
):
    safe_eval = sanitize_name(eval_name)
    out_dir = output_dir / "eval_details" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = eval_result["candidates"]
    y_true = eval_result["y_true"]
    y_pred = eval_result["y_pred"]

    cm = confusion_matrix(y_true, y_pred, labels=candidates)
    cm_df = pd.DataFrame(cm, index=candidates, columns=candidates)
    cm_df.to_csv(out_dir / f"confusion_{safe_eval}.csv")

    report = classification_report(
        y_true,
        y_pred,
        labels=candidates,
        output_dict=True,
        zero_division=0,
    )
    save_json(report, out_dir / f"classification_report_{safe_eval}.json")

    pred_df = pd.DataFrame({
        "gold": y_true,
        "pred": y_pred,
    })
    pred_df.to_csv(out_dir / f"predictions_{safe_eval}.csv", index=False)


def plot_cross_dataset_chart(results_df: pd.DataFrame, output_dir: Path):
    if results_df.empty:
        return

    df = results_df.copy()
    df["pair"] = df["train_dataset"] + " → " + df["eval_dataset"]

    plt.figure(figsize=(14, 6))
    plt.bar(range(len(df)), df["accuracy"])
    plt.xticks(range(len(df)), df["pair"], rotation=90)
    plt.ylabel("Accuracy")
    plt.title("Cross-Dataset NLI Accuracy")
    plt.tight_layout()
    plt.savefig(output_dir / "cross_dataset_accuracy_chart.png", dpi=240)
    plt.close()


def save_paper_reference_tables(output_dir: Path):
    table6_t5_small = [
        {"paper_table": "Table 6", "model_size": "T5-small", "train_dataset": "MNLI", "MNLI": 83.37, "ANLI": 31.34, "WANLI": 56.52, "GNLI_Human": 75.31},
        {"paper_table": "Table 6", "model_size": "T5-small", "train_dataset": "ANLI", "MNLI": 70.35, "ANLI": 48.31, "WANLI": 52.70, "GNLI_Human": 67.14},
        {"paper_table": "Table 6", "model_size": "T5-small", "train_dataset": "WANLI", "MNLI": 60.40, "ANLI": 36.41, "WANLI": 72.60, "GNLI_Human": 57.76},
        {"paper_table": "Table 6", "model_size": "T5-small", "train_dataset": "GNLI", "MNLI": 82.18, "ANLI": 33.00, "WANLI": 56.56, "GNLI_Human": 77.14},
        {"paper_table": "Table 6", "model_size": "T5-small", "train_dataset": "MNLI+GNLI", "MNLI": 82.66, "ANLI": 30.94, "WANLI": 55.82, "GNLI_Human": 77.76},
        {"paper_table": "Table 6", "model_size": "T5-small", "train_dataset": "ANLI+GNLI", "MNLI": 72.89, "ANLI": 37.94, "WANLI": 48.65, "GNLI_Human": 69.80},
        {"paper_table": "Table 6", "model_size": "T5-small", "train_dataset": "WANLI+GNLI", "MNLI": 78.02, "ANLI": 34.87, "WANLI": 86.69, "GNLI_Human": 76.53},
    ]

    table5_avg = [
        {"paper_table": "Table 5 TRUE avg", "model_size": "T5-small", "train_dataset": "MNLI", "TRUE_AUC_avg": 63.48},
        {"paper_table": "Table 5 TRUE avg", "model_size": "T5-small", "train_dataset": "ANLI", "TRUE_AUC_avg": 51.77},
        {"paper_table": "Table 5 TRUE avg", "model_size": "T5-small", "train_dataset": "WANLI", "TRUE_AUC_avg": 65.21},
        {"paper_table": "Table 5 TRUE avg", "model_size": "T5-small", "train_dataset": "M+A+W", "TRUE_AUC_avg": 65.33},
        {"paper_table": "Table 5 TRUE avg", "model_size": "T5-small", "train_dataset": "GNLI", "TRUE_AUC_avg": 72.06},
    ]

    pd.DataFrame(table6_t5_small).to_csv(output_dir / "paper_reference_table6_t5_small.csv", index=False)
    pd.DataFrame(table5_avg).to_csv(output_dir / "paper_reference_table5_true_avg.csv", index=False)


# ============================================================
# 9. OPTIONAL TRUE EVALUATION
# ============================================================

def load_true_csvs(true_dir: str = "true/data") -> Dict[str, List[Dict]]:
    true_path = Path(true_dir)
    if not true_path.exists():
        return {}

    csvs = list(true_path.rglob("*.csv"))
    true_sets = {}

    for csv_path in csvs:
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        needed = {"grounding", "generated_text", "label"}
        if not needed.issubset(set(df.columns)):
            continue

        records = []
        for _, row in df.iterrows():
            grounding = clean_text(row["grounding"])
            generated = clean_text(row["generated_text"])

            try:
                label = int(row["label"])
            except Exception:
                continue

            if not grounding or not generated:
                continue

            records.append({
                "input_text": make_nli_input(grounding, generated),
                "label": label,
            })

        if records:
            true_sets[csv_path.stem] = records

    return true_sets


def evaluate_true_if_available(model, tokenizer, cfg: Dict, output_dir: Path, run_name: str, train_name: str):
    true_sets = load_true_csvs("true/data")
    if not true_sets:
        return []

    print("\nTRUE files found. Running optional TRUE evaluation...")

    device = get_device()
    rows = []

    for true_name, records in true_sets.items():
        records = records[:cfg["max_eval_per_dataset"]]
        y_true = []
        y_score = []

        for start in tqdm(range(0, len(records), cfg["eval_batch_size"]), desc=f"TRUE {true_name}", leave=False):
            batch = records[start:start + cfg["eval_batch_size"]]
            inputs = [x["input_text"] for x in batch]
            labels = [x["label"] for x in batch]

            scores = score_candidates(model, tokenizer, inputs, LABELS_3WAY, cfg, device)
            probs = torch.softmax(torch.tensor(scores), dim=1).numpy()

            entail_idx = LABELS_3WAY.index("entailment")
            entail_prob = probs[:, entail_idx]

            y_true.extend(labels)
            y_score.extend(entail_prob.tolist())

        try:
            auc = roc_auc_score(y_true, y_score)
        except Exception:
            auc = np.nan

        rows.append({
            "train_dataset": train_name,
            "run_name": run_name,
            "true_dataset": true_name,
            "roc_auc": auc,
            "num_examples": len(records),
        })

    if rows:
        df = pd.DataFrame(rows)
        true_dir = output_dir / "true_results"
        true_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(true_dir / f"true_results_{run_name}.csv", index=False)

    return rows


# ============================================================
# 10. MAIN PIPELINE
# ============================================================

def build_train_eval_sets(cfg: Dict, output_dir: Path):
    train_sets = {}
    eval_sets = {}

    mnli_train, mnli_eval = load_mnli()
    anli_train, anli_eval = load_anli()
    wanli_train, wanli_eval = load_wanli()
    gnli_train, gnli_eval = build_gnli_proxy(cfg)

    train_sets["MNLI"] = stratified_sample(mnli_train, cfg["max_train_per_public_dataset"], seed=42)
    train_sets["ANLI"] = stratified_sample(anli_train, cfg["max_train_per_public_dataset"], seed=43)
    train_sets["WANLI"] = stratified_sample(wanli_train, cfg["max_train_per_public_dataset"], seed=44)
    train_sets["GNLI_PROXY"] = stratified_sample(gnli_train, cfg["max_train_combo"], seed=45)

    eval_sets["MNLI"] = stratified_sample(mnli_eval, cfg["max_eval_per_dataset"], seed=52)
    eval_sets["ANLI"] = stratified_sample(anli_eval, cfg["max_eval_per_dataset"], seed=53)
    eval_sets["WANLI"] = stratified_sample(wanli_eval, cfg["max_eval_per_dataset"], seed=54)
    eval_sets["GNLI_PROXY"] = stratified_sample(gnli_eval, cfg["max_eval_per_dataset"], seed=55)

    if cfg["run_combinations"]:
        for base in ["MNLI", "ANLI", "WANLI"]:
            base_part = stratified_sample(train_sets[base], cfg["max_train_combo"] // 2, seed=100)
            gnli_part = stratified_sample(train_sets["GNLI_PROXY"], cfg["max_train_combo"] - len(base_part), seed=101)
            train_sets[f"{base}+GNLI_PROXY"] = base_part + gnli_part
            random.Random(200).shuffle(train_sets[f"{base}+GNLI_PROXY"])

    if cfg["run_maw"]:
        maw_public = (
            stratified_sample(train_sets["MNLI"], cfg["max_train_combo"] // 3, seed=61)
            + stratified_sample(train_sets["ANLI"], cfg["max_train_combo"] // 3, seed=62)
            + stratified_sample(train_sets["WANLI"], cfg["max_train_combo"] // 3, seed=63)
        )
        random.Random(64).shuffle(maw_public)
        train_sets["MNLI+ANLI+WANLI"] = maw_public[:cfg["max_train_combo"]]

    if cfg["run_enhanced_all"]:
        per = max(1, cfg["max_train_combo"] // 4)
        enhanced = (
            stratified_sample(train_sets["MNLI"], per, seed=71)
            + stratified_sample(train_sets["ANLI"], per, seed=72)
            + stratified_sample(train_sets["WANLI"], per, seed=73)
            + stratified_sample(train_sets["GNLI_PROXY"], cfg["max_train_combo"] - 3 * per, seed=74)
        )
        random.Random(75).shuffle(enhanced)
        train_sets["ENHANCED_ALL_PUBLIC+GNLI_PROXY"] = enhanced[:cfg["max_train_combo"]]

    # Save dataset summary
    summary_rows = []
    for name, records in train_sets.items():
        counts = pd.Series([r["label"] for r in records]).value_counts().to_dict()
        summary_rows.append({
            "dataset": name,
            "split": "train",
            "num_examples": len(records),
            "label_counts": json.dumps(counts),
        })

    for name, records in eval_sets.items():
        counts = pd.Series([r["label"] for r in records]).value_counts().to_dict()
        summary_rows.append({
            "dataset": name,
            "split": "eval",
            "num_examples": len(records),
            "label_counts": json.dumps(counts),
        })

    pd.DataFrame(summary_rows).to_csv(output_dir / "dataset_summary.csv", index=False)

    # Save GNLI proxy for report/debugging
    pd.DataFrame(gnli_train + gnli_eval).to_csv(output_dir / "gnli_proxy_dataset.csv", index=False)

    return train_sets, eval_sets


def select_dev_set(train_name: str, eval_sets: Dict[str, List[Dict]]) -> List[Dict]:
    # The paper tuned combined models on GNLI dev.
    if "+" in train_name or train_name.startswith("ENHANCED"):
        return eval_sets["GNLI_PROXY"]

    if train_name in eval_sets:
        return eval_sets[train_name]

    return eval_sets["MNLI"]


def run_pipeline(profile: str):
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile: {profile}. Choose from {list(PROFILES.keys())}")

    cfg = PROFILES[profile].copy()

    # Auto-backup to GitHub only for enhanced profile by default.
    # Requires Kaggle Secrets: GITHUB_USERNAME, GITHUB_TOKEN, GITHUB_REPO.
    cfg.setdefault("push_to_github", profile == "enhanced")
    cfg.setdefault("github_sync_each_model", True)
    cfg.setdefault("github_subdir", f"assignment3_{profile}")

    seed_everything(42)
    setup_torch()

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eval_details").mkdir(exist_ok=True)

    save_json(cfg, output_dir / "run_config.json")
    save_json(get_gpu_info(), output_dir / "hardware_info.json")
    save_paper_reference_tables(output_dir)

    print("\nCONFIG")
    print(json.dumps(cfg, indent=2))
    print("\nHARDWARE")
    print(json.dumps(get_gpu_info(), indent=2))

    train_sets, eval_sets = build_train_eval_sets(cfg, output_dir)

    results_rows = []
    true_rows_all = []

    train_order = [
        "MNLI",
        "ANLI",
        "WANLI",
        "GNLI_PROXY",
    ]

    if cfg["run_maw"]:
        train_order.append("MNLI+ANLI+WANLI")

    if cfg["run_combinations"]:
        train_order.extend([
            "MNLI+GNLI_PROXY",
            "ANLI+GNLI_PROXY",
            "WANLI+GNLI_PROXY",
        ])

    if cfg["run_enhanced_all"]:
        train_order.append("ENHANCED_ALL_PUBLIC+GNLI_PROXY")

    train_order = [x for x in train_order if x in train_sets]

    for train_name in train_order:
        train_records = train_sets[train_name]
        dev_records = select_dev_set(train_name, eval_sets)

        model, tokenizer, run_name, elapsed = train_model(
            train_name=train_name,
            train_records=train_records,
            dev_records=dev_records,
            cfg=cfg,
            output_dir=output_dir,
            binary=False,
        )

        device = get_device()

        for eval_name, eval_records in eval_sets.items():
            eval_examples = make_examples(eval_records, binary=False)

            eval_result = evaluate_examples(
                model=model,
                tokenizer=tokenizer,
                examples=eval_examples,
                cfg=cfg,
                candidates=LABELS_3WAY,
                device=device,
                desc=f"{run_name} on {eval_name}",
            )

            save_eval_outputs(eval_result, output_dir, run_name, eval_name)

            row = {
                "profile": profile,
                "model_name": cfg["model_name"],
                "train_dataset": train_name,
                "eval_dataset": eval_name,
                "accuracy": eval_result["accuracy"],
                "train_examples": len(train_records),
                "eval_examples": len(eval_examples),
                "elapsed_train_seconds": elapsed,
                "run_name": run_name,
            }
            results_rows.append(row)

            print(f"{train_name} -> {eval_name}: accuracy={eval_result['accuracy']:.4f}")

        if cfg["run_true_if_available"]:
            true_rows = evaluate_true_if_available(model, tokenizer, cfg, output_dir, run_name, train_name)
            true_rows_all.extend(true_rows)

        # Save after every model so you do not lose progress.
        results_df = pd.DataFrame(results_rows)
        results_df.to_csv(output_dir / "cross_dataset_accuracy_results.csv", index=False)

        pivot = results_df.pivot_table(
            index=["train_dataset"],
            columns="eval_dataset",
            values="accuracy",
            aggfunc="mean",
        ).reset_index()
        pivot.to_csv(output_dir / "cross_dataset_accuracy_pivot.csv", index=False)

        plot_cross_dataset_chart(results_df, output_dir)

        if cfg.get("push_to_github", False) and cfg.get("github_sync_each_model", True):
            sync_outputs_to_github(
                output_dir=output_dir,
                cfg=cfg,
                message=f"Backup after completing {train_name}"
            )

        if true_rows_all:
            true_df = pd.DataFrame(true_rows_all)
            true_df.to_csv(output_dir / "true_all_results.csv", index=False)
            true_avg = true_df.groupby("train_dataset")["roc_auc"].mean().reset_index()
            true_avg.to_csv(output_dir / "true_average_auc.csv", index=False)

        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    make_experiment_summary(output_dir, profile, cfg)

    if cfg.get("push_to_github", False):
        sync_outputs_to_github(
            output_dir=output_dir,
            cfg=cfg,
            message=f"Final backup for {profile} run"
        )

    print("\nDONE.")
    print(f"Results saved in: {output_dir.resolve()}")
    print("\nImportant files:")
    print(f"- {output_dir / 'dataset_summary.csv'}")
    print(f"- {output_dir / 'cross_dataset_accuracy_results.csv'}")
    print(f"- {output_dir / 'cross_dataset_accuracy_pivot.csv'}")
    print(f"- {output_dir / 'cross_dataset_accuracy_chart.png'}")
    print(f"- {output_dir / 'gnli_proxy_dataset.csv'}")
    print(f"- {output_dir / 'runs/*/training_log.csv'}")
    print(f"- {output_dir / 'runs/*/training_curve.png'}")
    print(f"- {output_dir / 'experiment_summary.md'}")


def make_experiment_summary(output_dir: Path, profile: str, cfg: Dict):
    results_path = output_dir / "cross_dataset_accuracy_results.csv"
    pivot_path = output_dir / "cross_dataset_accuracy_pivot.csv"

    lines = []
    lines.append(f"# NLI Reproduction Experiment Summary\n")
    lines.append(f"Profile: `{profile}`\n")
    lines.append(f"Model: `{cfg['model_name']}`\n")
    lines.append(f"Hardware: `{json.dumps(get_gpu_info())}`\n")
    lines.append("\n## Important Reproducibility Note\n")
    lines.append(
        "The original GNLI dataset was not released by the paper authors. "
        "This experiment therefore uses a GNLI-style proxy generated with the same domain/length/label design. "
        "The experiment reproduces the structure of the paper at laptop scale using T5-small.\n"
    )

    if pivot_path.exists():
        pivot = pd.read_csv(pivot_path)
        lines.append("\n## Cross-Dataset Accuracy Pivot\n")
        lines.append(pivot.to_markdown(index=False))
        lines.append("\n")

    lines.append("\n## Output Files\n")
    lines.append("- `dataset_summary.csv`\n")
    lines.append("- `cross_dataset_accuracy_results.csv`\n")
    lines.append("- `cross_dataset_accuracy_pivot.csv`\n")
    lines.append("- `cross_dataset_accuracy_chart.png`\n")
    lines.append("- `gnli_proxy_dataset.csv`\n")
    lines.append("- `runs/*/training_log.csv`\n")
    lines.append("- `runs/*/training_curve.png`\n")
    lines.append("- `eval_details/*/confusion_*.csv`\n")
    lines.append("- `eval_details/*/classification_report_*.json`\n")

    with open(output_dir / "experiment_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# 11. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        type=str,
        default="final",
        choices=list(PROFILES.keys()),
        help="Run profile: smoke, final, enhanced",
    )
    args = parser.parse_args()

    run_pipeline(args.profile)
