# Core experiment pipeline copied from the original uploaded script.
# Do not edit unless you intentionally change experiment behavior.

# Assignment 2: Reproduction of Results
# Paper: "A Synthetic Data Approach for Domain Generalization of NLI Models"
#
# What this script reproduces:
# 1. Loads public NLI datasets: MNLI, ANLI, WANLI
# 2. Builds a GNLI-style proxy dataset because original GNLI was not released
# 3. Fine-tunes T5-small for 3-way NLI classification
# 4. Evaluates cross-dataset accuracy similar to paper Table 6
# 5. Includes paper-style mixtures:
#       MNLI
#       ANLI
#       WANLI
#       GNLI_PROXY
#       MNLI + ANLI + WANLI
#       MNLI + GNLI_PROXY
#       ANLI + GNLI_PROXY
#       WANLI + GNLI_PROXY
# 6. Saves:
#       dataset_summary.csv
#       cross_dataset_accuracy_results.csv
#       cross_dataset_accuracy_pivot.csv
#       cross_dataset_accuracy_chart.png
#       training logs
#       training curves
#       confusion matrices
#       classification reports
#
# Run:
#   python assignment2_nli_reproduction.py --profile smoke
#   python assignment2_nli_reproduction.py --profile final

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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup

warnings.filterwarnings("ignore")


# ============================================================
# 1. PROFILES
# ============================================================

PROFILES = {
    # Very short run to verify CUDA, dataset loading, training, evaluation and outputs.
    # This is NOT for the final report results.
    "smoke": {
        "output_dir": "outputs_assignment2_smoke",
        "model_name": "t5-small",
        "max_train_per_public_dataset": 1000,
        "max_train_combo": 1500,
        "max_eval_per_dataset": 300,
        "gnli_examples_per_domain_length": 12,

        # Paper-style training mechanics, scaled down for smoke test.
        "batch_size": 4,
        "grad_accum_steps": 8,      # 4 x 8 = effective batch size 32, matching paper.
        "eval_batch_size": 8,
        "learning_rate": 5e-4,
        "max_input_length": 512,
        "max_target_length": 8,
        "checkpoint_every_steps": 100,
        "select_best_checkpoint": True,

        "use_paper_steps": False,
        "max_steps_by_dataset": {
            "MNLI": 100,
            "ANLI": 100,
            "WANLI": 100,
            "GNLI_PROXY": 100,
            "MNLI+ANLI+WANLI": 100,
            "MNLI+GNLI_PROXY": 100,
            "ANLI+GNLI_PROXY": 100,
            "WANLI+GNLI_PROXY": 100,
        },

        # Original paper T5-small 3-way selected steps from Table 9.
        # This is here for reproducibility documentation; smoke does not use it.
        "paper_steps_by_dataset": {
            "MNLI": 40000,
            "ANLI": 45000,
            "WANLI": 10000,
            "GNLI_PROXY": 50000,
            "MNLI+ANLI+WANLI": 40000,
            "MNLI+GNLI_PROXY": 25000,
            "ANLI+GNLI_PROXY": 5000,
            "WANLI+GNLI_PROXY": 50000,
        },

        "run_combinations": True,
        "run_maw": True,
        "save_models": False,
    },

    # Recommended final Assignment 2 run for RTX 4060 8GB VRAM + 16GB RAM.
    # It follows the paper's step-based schedule structure, effective batch size,
    # input length, LR, and checkpoint-evaluation style, but with fewer steps.
    "final": {
        "output_dir": "outputs_assignment2_final",
        "model_name": "t5-small",

        # More paper-faithful than the earlier 12K version, but still laptop-safe.
        "max_train_per_public_dataset": 50000,
        "max_train_combo": 50000,
        "max_eval_per_dataset": 2000,

        # Original GNLI has ~670K train examples and is unreleased.
        # This creates 38 domains x 2 lengths x 250 = 19,000 proxy examples.
        "gnli_examples_per_domain_length": 250,

        # Paper-style T5 fine-tuning settings.
        "batch_size": 4,
        "grad_accum_steps": 8,      # effective batch size 32, matching the paper.
        "eval_batch_size": 8,
        "learning_rate": 5e-4,      # T5-small 3-way Table 9 uses 5e-4 for all listed datasets.
        "max_input_length": 512,    # matches paper NLI inference/training input length.
        "max_target_length": 8,
        "checkpoint_every_steps": 1000,
        "select_best_checkpoint": True,

        # Keep False for laptop-safe reproduction. Set True only for a very long run.
        "use_paper_steps": False,

        # Laptop-safe scaled-down step schedule. Same structure as paper, smaller scale.
        "max_steps_by_dataset": {
            "MNLI": 4000,
            "ANLI": 4500,
            "WANLI": 2000,
            "GNLI_PROXY": 4000,
            "MNLI+ANLI+WANLI": 4000,
            "MNLI+GNLI_PROXY": 3000,
            "ANLI+GNLI_PROXY": 1500,
            "WANLI+GNLI_PROXY": 4000,
        },

        # Original paper T5-small 3-way selected steps from Table 9.
        # WARNING: use_paper_steps=True may take many hours to more than a day.
        "paper_steps_by_dataset": {
            "MNLI": 40000,
            "ANLI": 45000,
            "WANLI": 10000,
            "GNLI_PROXY": 50000,
            # Paper does not report a Table-9 3-way value for M+A+W, so 40K is used as a neutral proxy.
            "MNLI+ANLI+WANLI": 40000,
            "MNLI+GNLI_PROXY": 25000,
            "ANLI+GNLI_PROXY": 5000,
            "WANLI+GNLI_PROXY": 50000,
        },

        "run_combinations": True,
        "run_maw": True,
        "save_models": False,
    },

    # Optional true paper-steps run. It is included only for completeness.
    # Do not run this unless you are ready for a very long experiment.
    "paper_steps": {
        "output_dir": "outputs_assignment2_paper_steps",
        "model_name": "t5-small",
        "max_train_per_public_dataset": 50000,
        "max_train_combo": 50000,
        "max_eval_per_dataset": 2000,
        "gnli_examples_per_domain_length": 250,
        "batch_size": 4,
        "grad_accum_steps": 8,
        "eval_batch_size": 8,
        "learning_rate": 5e-4,
        "max_input_length": 512,
        "max_target_length": 8,
        "checkpoint_every_steps": 1000,
        "select_best_checkpoint": True,
        "use_paper_steps": True,
        "max_steps_by_dataset": {},
        "paper_steps_by_dataset": {
            "MNLI": 40000,
            "ANLI": 45000,
            "WANLI": 10000,
            "GNLI_PROXY": 50000,
            "MNLI+ANLI+WANLI": 40000,
            "MNLI+GNLI_PROXY": 25000,
            "ANLI+GNLI_PROXY": 5000,
            "WANLI+GNLI_PROXY": 50000,
        },
        "run_combinations": True,
        "run_maw": True,
        "save_models": False,
    },
}

LABELS_3WAY = ["entailment", "neutral", "contradiction"]


# ============================================================
# 2. BASIC UTILITIES
# ============================================================

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_torch():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_gpu_info() -> Dict:
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device": str(get_device()),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2
        )
    return info


def sanitize_name(x: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", x)


def clean_text(x) -> str:
    if x is None:
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def normalize_label(raw, label_names=None) -> str:
    """
    Normalizes labels into:
        entailment
        neutral
        contradiction
    """
    if isinstance(raw, (int, np.integer)):
        if label_names is not None:
            try:
                return normalize_label(label_names[int(raw)])
            except Exception:
                pass

        # Common NLI order in MNLI/ANLI:
        mapping = {
            0: "entailment",
            1: "neutral",
            2: "contradiction",
        }
        return mapping.get(int(raw), "unknown")

    s = str(raw).strip().lower()
    s = s.replace("entails", "entailment")
    s = s.replace("entail", "entailment")
    s = s.replace("contradicts", "contradiction")
    s = s.replace("contradictory", "contradiction")
    s = s.replace("not_entailment", "neutral")
    s = s.replace("non-entailment", "neutral")
    s = s.replace("non_entailment", "neutral")

    if s in ["e", "entailment"]:
        return "entailment"
    if s in ["n", "neutral"]:
        return "neutral"
    if s in ["c", "contradiction"]:
        return "contradiction"

    if "contradiction" in s:
        return "contradiction"
    if "neutral" in s:
        return "neutral"
    if "entailment" in s:
        return "entailment"

    return "unknown"


def stratified_sample(records: List[Dict], max_n: Optional[int], seed: int = 42) -> List[Dict]:
    """
    Samples records while trying to keep labels balanced.
    """
    if max_n is None or len(records) <= max_n:
        return list(records)

    df = pd.DataFrame(records)
    if "label" not in df.columns:
        return df.sample(n=max_n, random_state=seed).to_dict("records")

    output = []
    per_label = max_n // len(LABELS_3WAY)

    for label in LABELS_3WAY:
        sub = df[df["label"] == label]
        if len(sub) > 0:
            take = min(per_label, len(sub))
            output.extend(sub.sample(n=take, random_state=seed).to_dict("records"))

    remaining = max_n - len(output)
    if remaining > 0:
        selected_df = pd.DataFrame(output)
        full_df = df.sample(frac=1, random_state=seed + 1)

        if not selected_df.empty:
            selected_keys = set(
                zip(
                    selected_df["premise"].astype(str),
                    selected_df["hypothesis"].astype(str),
                    selected_df["label"].astype(str),
                )
            )
        else:
            selected_keys = set()

        extra = []
        for _, row in full_df.iterrows():
            key = (str(row["premise"]), str(row["hypothesis"]), str(row["label"]))
            if key not in selected_keys:
                extra.append(row.to_dict())
            if len(extra) >= remaining:
                break

        output.extend(extra)

    random.Random(seed).shuffle(output)
    return output[:max_n]


def make_nli_input(premise: str, hypothesis: str) -> str:
    return f"nli premise: {premise} hypothesis: {hypothesis} label:"


def make_examples(records: List[Dict]) -> List[Dict]:
    examples = []

    for r in records:
        label = normalize_label(r["label"])
        if label not in LABELS_3WAY:
            continue

        examples.append({
            "input_text": make_nli_input(r["premise"], r["hypothesis"]),
            "target_text": label,
            "label": label,
            "source": r.get("source", "unknown"),
        })

    return examples


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
    """
    Extracts premise/hypothesis/label from different NLI dataset schemas.
    """
    cols = ds.column_names

    premise_col = None
    hypothesis_col = None
    label_col = None

    for c in ["premise", "sentence1", "context", "grounding"]:
        if c in cols:
            premise_col = c
            break

    for c in ["hypothesis", "sentence2", "generated_text"]:
        if c in cols:
            hypothesis_col = c
            break

    for c in ["label", "gold", "gold_label"]:
        if c in cols:
            label_col = c
            break

    if premise_col is None or hypothesis_col is None or label_col is None:
        raise ValueError(
            f"Could not identify columns for {dataset_name}. Columns={cols}"
        )

    label_names = get_label_names(ds, label_col)
    records = []

    for row in ds:
        label = normalize_label(row[label_col], label_names=label_names)
        if label not in LABELS_3WAY:
            continue

        premise = clean_text(row[premise_col])
        hypothesis = clean_text(row[hypothesis_col])

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

    print(f"MNLI train={len(train)}, eval={len(eval_records)}")
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

    print(f"ANLI train={len(train)}, eval={len(dev)}")
    return train, dev


def load_wanli() -> Tuple[List[Dict], List[Dict]]:
    print("\nLoading WANLI...")
    ds = load_dataset("alisawuffles/WANLI")
    split_names = list(ds.keys())

    train_split = "train" if "train" in split_names else split_names[0]
    eval_split = "test" if "test" in split_names else (
        "validation" if "validation" in split_names else split_names[-1]
    )

    train = extract_records(ds[train_split], "WANLI")
    eval_records = extract_records(ds[eval_split], "WANLI")

    print(f"WANLI train={len(train)}, eval={len(eval_records)}")
    return train, eval_records


# ============================================================
# 4. GNLI-STYLE PROXY DATASET
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

PEOPLE = [
    "Amina", "Bilal", "Sara", "Usman", "Hina",
    "Omar", "Maya", "Daniel", "Noah", "Fatima"
]

ITEMS = [
    "phone", "laptop", "book", "camera", "medical report",
    "rental listing", "software update", "restaurant order",
    "legal notice", "travel booking"
]

PLACES = [
    "Islamabad", "Lahore", "Karachi", "London",
    "a small clinic", "a university office", "a travel agency",
    "an online forum", "a shopping website"
]

DAYS = [
    "Monday", "Tuesday", "Friday", "the next morning",
    "last week", "two days later", "after the meeting"
]

ISSUES = [
    "a delayed response", "a broken screen", "missing information",
    "a payment error", "slow service", "unclear instructions",
    "a confusing update", "a scheduling conflict"
]

OUTCOMES = [
    "was resolved", "remained unresolved", "was delayed",
    "was approved", "was rejected", "needed another review"
]


def make_domain_premise(domain: str, length: str, idx: int) -> Dict:
    """
    Creates one GNLI-style synthetic example.

    The original paper used FLAN-PaLM2 for premise generation and a prompt-tuned
    FLAN-PaLM model for hypothesis+label generation. Since those are unavailable
    and the GNLI dataset is unreleased, this function creates a reproducible
    proxy dataset using the same design idea:
        - many domains
        - short/paragraph premises
        - balanced entailment/neutral/contradiction labels
    """
    rng = random.Random(1000 + idx + len(domain) + len(length))

    person = rng.choice(PEOPLE)
    item = rng.choice(ITEMS)
    place = rng.choice(PLACES)
    day = rng.choice(DAYS)
    issue = rng.choice(ISSUES)
    outcome = rng.choice(OUTCOMES)

    if length == "short":
        premise = (
            f"In a {domain} text, {person} says the {item} in {place} had "
            f"{issue} on {day}, and the case {outcome}."
        )
    else:
        premise = (
            f"In a {domain} text, {person} describes a situation involving a "
            f"{item} in {place}. The issue started on {day} and included {issue}. "
            f"Several details were checked before the final update. According to "
            f"the text, the case {outcome}."
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
    print("\nBuilding GNLI-style proxy dataset...")

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

    print(f"GNLI_PROXY train={len(train)}, eval={len(test)}")
    return train, test


# ============================================================
# 5. TORCH DATASET
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
# 6. CANDIDATE-SCORING EVALUATION
# ============================================================

@torch.no_grad()
def score_candidates(
    model,
    tokenizer,
    input_texts: List[str],
    candidates: List[str],
    cfg: Dict,
    device,
) -> np.ndarray:
    """
    Scores each candidate label using sequence likelihood.
    This is more stable than free generation because labels are fixed.
    """
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

        decoder_input_ids = model._shift_right(labels)

        out = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            decoder_input_ids=decoder_input_ids,
        )

        logits = out.logits
        log_probs = F.log_softmax(logits, dim=-1)

        token_log_probs = log_probs.gather(
            dim=-1,
            index=labels.unsqueeze(-1)
        ).squeeze(-1)

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
    desc: str = "eval",
) -> Dict:
    y_true = []
    y_pred = []
    y_scores = []

    for start in tqdm(
        range(0, len(examples), cfg["eval_batch_size"]),
        desc=desc,
        leave=False,
    ):
        batch = examples[start:start + cfg["eval_batch_size"]]
        inputs = [x["input_text"] for x in batch]
        labels = [x["label"] for x in batch]

        scores = score_candidates(
            model=model,
            tokenizer=tokenizer,
            input_texts=inputs,
            candidates=candidates,
            cfg=cfg,
            device=device,
        )

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

def resolve_max_train_steps(train_name: str, cfg: Dict) -> int:
    """
    Paper uses step-based fine-tuning. This helper chooses either the original
    paper's T5-small 3-way step count or the laptop-scaled step count.
    """
    if cfg.get("use_paper_steps", False):
        return int(cfg["paper_steps_by_dataset"].get(train_name, 50000))
    return int(cfg["max_steps_by_dataset"].get(train_name, 4000))


def train_model(
    train_name: str,
    train_records: List[Dict],
    dev_records: List[Dict],
    cfg: Dict,
    output_dir: Path,
):
    device = get_device()
    model_name = cfg["model_name"]

    run_name = sanitize_name(f"{train_name}_3way")
    run_dir = output_dir / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_examples = make_examples(train_records)
    dev_examples = make_examples(dev_records)
    dev_examples = stratified_sample(dev_examples, min(600, len(dev_examples)), seed=42)

    max_train_steps = resolve_max_train_steps(train_name, cfg)
    effective_batch_size = cfg["batch_size"] * cfg["grad_accum_steps"]

    print(f"\n{'=' * 80}")
    print(f"Training: {run_name}")
    print(f"Model: {model_name}")
    print(f"Train examples: {len(train_examples)}")
    print(f"Dev examples during training: {len(dev_examples)}")
    print(f"Training schedule: step-based | max_train_steps={max_train_steps}")
    print(f"Batch size: {cfg['batch_size']} | grad_accum_steps={cfg['grad_accum_steps']} | effective_batch_size={effective_batch_size}")
    print(f"Input length: {cfg['max_input_length']} | LR: {cfg['learning_rate']} | checkpoint_every_steps={cfg['checkpoint_every_steps']}")
    print(f"Device: {device}")
    print(f"{'=' * 80}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # T5 default dropout is already 0.1, but set it explicitly for method reporting.
    if hasattr(model.config, "dropout_rate"):
        model.config.dropout_rate = 0.1

    model.to(device)
    model.config.use_cache = False

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

    total_update_steps = max_train_steps
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
    micro_step = 0
    epoch = 0
    examples_seen = 0
    running_loss = 0.0
    best_dev_accuracy = -1.0
    best_global_step = 0
    best_state_dict = None
    start_time = time.time()

    model.train()
    optimizer.zero_grad()

    while global_step < max_train_steps:
        epoch += 1

        pbar = tqdm(
            train_loader,
            desc=f"{run_name} epoch {epoch} | steps {global_step}/{max_train_steps}",
        )

        for batch in pbar:
            if global_step >= max_train_steps:
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            examples_seen += int(batch["input_ids"].size(0))
            micro_step += 1

            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(**batch)
                loss = out.loss / cfg["grad_accum_steps"]

            scaler.scale(loss).backward()
            running_loss += float(loss.item() * cfg["grad_accum_steps"])

            if micro_step % cfg["grad_accum_steps"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1
                avg_loss = running_loss / max(1, global_step * cfg["grad_accum_steps"])

                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "step": f"{global_step}/{max_train_steps}",
                })

                if global_step % 25 == 0 or global_step == 1:
                    logs.append({
                        "run_name": run_name,
                        "epoch": epoch,
                        "global_step": global_step,
                        "train_loss": avg_loss,
                        "dev_accuracy": np.nan,
                        "learning_rate": scheduler.get_last_lr()[0],
                        "examples_seen": examples_seen,
                        "effective_batch_size": effective_batch_size,
                    })

                # Paper-style checkpoint evaluation every 1K steps.
                if global_step % cfg["checkpoint_every_steps"] == 0 or global_step == max_train_steps:
                    dev_result = evaluate_examples(
                        model=model,
                        tokenizer=tokenizer,
                        examples=dev_examples,
                        cfg=cfg,
                        candidates=LABELS_3WAY,
                        device=device,
                        desc=f"checkpoint dev {run_name} step {global_step}",
                    )

                    dev_acc = float(dev_result["accuracy"])
                    logs.append({
                        "run_name": run_name,
                        "epoch": epoch,
                        "global_step": global_step,
                        "train_loss": avg_loss,
                        "dev_accuracy": dev_acc,
                        "learning_rate": scheduler.get_last_lr()[0],
                        "examples_seen": examples_seen,
                        "effective_batch_size": effective_batch_size,
                    })

                    print(f"Checkpoint step {global_step}: dev_accuracy={dev_acc:.4f}")

                    if cfg.get("select_best_checkpoint", True) and dev_acc > best_dev_accuracy:
                        best_dev_accuracy = dev_acc
                        best_global_step = global_step
                        # Keep best checkpoint in memory so final evaluation uses selected checkpoint.
                        best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

                if global_step >= max_train_steps:
                    break

    # If no checkpoint was evaluated due to tiny smoke settings, evaluate once at the end.
    if best_state_dict is None:
        final_dev_result = evaluate_examples(
            model=model,
            tokenizer=tokenizer,
            examples=dev_examples,
            cfg=cfg,
            candidates=LABELS_3WAY,
            device=device,
            desc=f"final dev {run_name}",
        )
        best_dev_accuracy = float(final_dev_result["accuracy"])
        best_global_step = global_step
        best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Restore selected checkpoint, mirroring paper's checkpoint-based model selection.
    if cfg.get("select_best_checkpoint", True) and best_state_dict is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})
        print(f"Loaded best checkpoint for {run_name}: step={best_global_step}, dev_accuracy={best_dev_accuracy:.4f}")

    elapsed = round(time.time() - start_time, 2)

    log_df = pd.DataFrame(logs)
    log_df.to_csv(run_dir / "training_log.csv", index=False)

    plot_training_curve(log_df, run_dir / "training_curve.png", run_name)

    save_json({
        "run_name": run_name,
        "train_dataset": train_name,
        "model_name": model_name,
        "train_examples": len(train_examples),
        "dev_examples": len(dev_examples),
        "elapsed_seconds": elapsed,
        "training_schedule": "step_based",
        "max_train_steps": max_train_steps,
        "best_global_step": best_global_step,
        "best_dev_accuracy": best_dev_accuracy,
        "effective_batch_size": effective_batch_size,
        "checkpoint_every_steps": cfg["checkpoint_every_steps"],
        "dropout_rate": getattr(model.config, "dropout_rate", None),
        "config": cfg,
        "gpu_info": get_gpu_info(),
    }, run_dir / "run_metadata.json")

    if cfg["save_models"]:
        model.save_pretrained(run_dir / "model")
        tokenizer.save_pretrained(run_dir / "tokenizer")

    # Free CPU copy of best state after loading.
    del best_state_dict
    gc.collect()

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
# 8. SAVE EVALUATION OUTPUTS
# ============================================================

def save_eval_outputs(
    eval_result: Dict,
    output_dir: Path,
    run_name: str,
    eval_name: str,
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
    """
    Reference values from the paper for comparison in the report.
    These are not reproduced values; they are original paper values.
    """
    table6_t5_small = [
        {
            "paper_table": "Table 6",
            "model_size": "T5-small",
            "train_dataset": "MNLI",
            "MNLI": 83.37,
            "ANLI": 31.34,
            "WANLI": 56.52,
            "GNLI_Human": 75.31,
        },
        {
            "paper_table": "Table 6",
            "model_size": "T5-small",
            "train_dataset": "ANLI",
            "MNLI": 70.35,
            "ANLI": 48.31,
            "WANLI": 52.70,
            "GNLI_Human": 67.14,
        },
        {
            "paper_table": "Table 6",
            "model_size": "T5-small",
            "train_dataset": "WANLI",
            "MNLI": 60.40,
            "ANLI": 36.41,
            "WANLI": 72.60,
            "GNLI_Human": 57.76,
        },
        {
            "paper_table": "Table 6",
            "model_size": "T5-small",
            "train_dataset": "GNLI",
            "MNLI": 82.18,
            "ANLI": 33.00,
            "WANLI": 56.56,
            "GNLI_Human": 77.14,
        },
        {
            "paper_table": "Table 6",
            "model_size": "T5-small",
            "train_dataset": "MNLI+GNLI",
            "MNLI": 82.66,
            "ANLI": 30.94,
            "WANLI": 55.82,
            "GNLI_Human": 77.76,
        },
        {
            "paper_table": "Table 6",
            "model_size": "T5-small",
            "train_dataset": "ANLI+GNLI",
            "MNLI": 72.89,
            "ANLI": 37.94,
            "WANLI": 48.65,
            "GNLI_Human": 69.80,
        },
        {
            "paper_table": "Table 6",
            "model_size": "T5-small",
            "train_dataset": "WANLI+GNLI",
            "MNLI": 78.02,
            "ANLI": 34.87,
            "WANLI": 86.69,
            "GNLI_Human": 76.53,
        },
    ]

    table5_true_avg = [
        {
            "paper_table": "Table 5 TRUE avg",
            "model_size": "T5-small",
            "train_dataset": "MNLI",
            "TRUE_AUC_avg": 63.48,
        },
        {
            "paper_table": "Table 5 TRUE avg",
            "model_size": "T5-small",
            "train_dataset": "ANLI",
            "TRUE_AUC_avg": 51.77,
        },
        {
            "paper_table": "Table 5 TRUE avg",
            "model_size": "T5-small",
            "train_dataset": "WANLI",
            "TRUE_AUC_avg": 65.21,
        },
        {
            "paper_table": "Table 5 TRUE avg",
            "model_size": "T5-small",
            "train_dataset": "MNLI+ANLI+WANLI",
            "TRUE_AUC_avg": 65.33,
        },
        {
            "paper_table": "Table 5 TRUE avg",
            "model_size": "T5-small",
            "train_dataset": "GNLI",
            "TRUE_AUC_avg": 72.06,
        },
    ]

    pd.DataFrame(table6_t5_small).to_csv(
        output_dir / "paper_reference_table6_t5_small.csv",
        index=False,
    )

    pd.DataFrame(table5_true_avg).to_csv(
        output_dir / "paper_reference_table5_true_avg.csv",
        index=False,
    )


# ============================================================
# 9. BUILD TRAIN/EVAL SETS
# ============================================================

def build_train_eval_sets(cfg: Dict, output_dir: Path):
    train_sets = {}
    eval_sets = {}

    mnli_train, mnli_eval = load_mnli()
    anli_train, anli_eval = load_anli()
    wanli_train, wanli_eval = load_wanli()
    gnli_train, gnli_eval = build_gnli_proxy(cfg)

    train_sets["MNLI"] = stratified_sample(
        mnli_train,
        cfg["max_train_per_public_dataset"],
        seed=42,
    )
    train_sets["ANLI"] = stratified_sample(
        anli_train,
        cfg["max_train_per_public_dataset"],
        seed=43,
    )
    train_sets["WANLI"] = stratified_sample(
        wanli_train,
        cfg["max_train_per_public_dataset"],
        seed=44,
    )
    train_sets["GNLI_PROXY"] = stratified_sample(
        gnli_train,
        cfg["max_train_combo"],
        seed=45,
    )

    eval_sets["MNLI"] = stratified_sample(
        mnli_eval,
        cfg["max_eval_per_dataset"],
        seed=52,
    )
    eval_sets["ANLI"] = stratified_sample(
        anli_eval,
        cfg["max_eval_per_dataset"],
        seed=53,
    )
    eval_sets["WANLI"] = stratified_sample(
        wanli_eval,
        cfg["max_eval_per_dataset"],
        seed=54,
    )
    eval_sets["GNLI_PROXY"] = stratified_sample(
        gnli_eval,
        cfg["max_eval_per_dataset"],
        seed=55,
    )

    # Paper-style public mixture: MNLI + ANLI + WANLI.
    if cfg["run_maw"]:
        per = max(1, cfg["max_train_combo"] // 3)
        maw_public = (
            stratified_sample(train_sets["MNLI"], per, seed=61)
            + stratified_sample(train_sets["ANLI"], per, seed=62)
            + stratified_sample(train_sets["WANLI"], cfg["max_train_combo"] - 2 * per, seed=63)
        )
        random.Random(64).shuffle(maw_public)
        train_sets["MNLI+ANLI+WANLI"] = maw_public[:cfg["max_train_combo"]]

    # Paper-style GNLI augmentation.
    if cfg["run_combinations"]:
        for base in ["MNLI", "ANLI", "WANLI"]:
            base_part = stratified_sample(
                train_sets[base],
                cfg["max_train_combo"] // 2,
                seed=100,
            )
            gnli_part = stratified_sample(
                train_sets["GNLI_PROXY"],
                cfg["max_train_combo"] - len(base_part),
                seed=101,
            )
            combined_name = f"{base}+GNLI_PROXY"
            train_sets[combined_name] = base_part + gnli_part
            random.Random(200).shuffle(train_sets[combined_name])

    # Save dataset summary.
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

    pd.DataFrame(summary_rows).to_csv(
        output_dir / "dataset_summary.csv",
        index=False,
    )

    # Save the generated GNLI proxy data.
    pd.DataFrame(gnli_train + gnli_eval).to_csv(
        output_dir / "gnli_proxy_dataset.csv",
        index=False,
    )

    return train_sets, eval_sets


def select_dev_set(train_name: str, eval_sets: Dict[str, List[Dict]]) -> List[Dict]:
    """
    The paper tunes combined GNLI models on GNLI dev.
    This mirrors that idea at reduced scale.
    """
    if "+GNLI_PROXY" in train_name:
        return eval_sets["GNLI_PROXY"]

    if train_name in eval_sets:
        return eval_sets[train_name]

    if train_name == "MNLI+ANLI+WANLI":
        return eval_sets["MNLI"]

    return eval_sets["MNLI"]


# ============================================================
# 10. MAIN PIPELINE
# ============================================================

def make_experiment_summary(output_dir: Path, profile: str, cfg: Dict):
    pivot_path = output_dir / "cross_dataset_accuracy_pivot.csv"

    lines = []
    lines.append("# Assignment 2 NLI Reproduction Experiment Summary\n")
    lines.append(f"Profile: `{profile}`\n")
    lines.append(f"Model: `{cfg['model_name']}`\n")
    lines.append(f"Hardware: `{json.dumps(get_gpu_info())}`\n")

    lines.append("\n## Reproducibility Note\n")
    lines.append(
        "The original paper's GNLI dataset was not released, and the original "
        "synthetic data generation used FLAN-PaLM/FLAN-PaLM2 infrastructure. "
        "Therefore, this assignment reproduces the experimental structure at "
        "laptop scale using public NLI datasets and a GNLI-style proxy dataset "
        "generated with the same domain, length, and label-balancing design. "
        "The training loop is step-based, uses effective batch size 32 through "
        "gradient accumulation, input length 512, LR 5e-4, and checkpoint-style "
        "development evaluation.\n"
    )

    if pivot_path.exists():
        pivot = pd.read_csv(pivot_path)
        lines.append("\n## Cross-Dataset Accuracy Pivot\n")
        try:
            lines.append(pivot.to_markdown(index=False))
        except Exception:
            lines.append(pivot.to_string(index=False))
        lines.append("\n")

    lines.append("\n## Output Files\n")
    lines.append("- `dataset_summary.csv`\n")
    lines.append("- `gnli_proxy_dataset.csv`\n")
    lines.append("- `cross_dataset_accuracy_results.csv`\n")
    lines.append("- `cross_dataset_accuracy_pivot.csv`\n")
    lines.append("- `cross_dataset_accuracy_chart.png`\n")
    lines.append("- `paper_reference_table6_t5_small.csv`\n")
    lines.append("- `paper_reference_table5_true_avg.csv`\n")
    lines.append("- `runs/*/training_log.csv`\n")
    lines.append("- `runs/*/training_curve.png`\n")
    lines.append("- `eval_details/*/confusion_*.csv`\n")
    lines.append("- `eval_details/*/classification_report_*.json`\n")
    lines.append("- `eval_details/*/predictions_*.csv`\n")

    with open(output_dir / "experiment_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_pipeline(profile: str):
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile: {profile}. Choose from {list(PROFILES.keys())}")

    cfg = PROFILES[profile].copy()

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

    train_order = [x for x in train_order if x in train_sets]

    results_rows = []

    for train_name in train_order:
        train_records = train_sets[train_name]
        dev_records = select_dev_set(train_name, eval_sets)

        model, tokenizer, run_name, elapsed = train_model(
            train_name=train_name,
            train_records=train_records,
            dev_records=dev_records,
            cfg=cfg,
            output_dir=output_dir,
        )

        device = get_device()

        for eval_name, eval_records in eval_sets.items():
            eval_examples = make_examples(eval_records)

            eval_result = evaluate_examples(
                model=model,
                tokenizer=tokenizer,
                examples=eval_examples,
                cfg=cfg,
                candidates=LABELS_3WAY,
                device=device,
                desc=f"{run_name} on {eval_name}",
            )

            save_eval_outputs(
                eval_result=eval_result,
                output_dir=output_dir,
                run_name=run_name,
                eval_name=eval_name,
            )

            row = {
                "profile": profile,
                "model_name": cfg["model_name"],
                "train_dataset": train_name,
                "eval_dataset": eval_name,
                "accuracy": eval_result["accuracy"],
                "accuracy_percent": eval_result["accuracy"] * 100,
                "train_examples": len(train_records),
                "eval_examples": len(eval_examples),
                "elapsed_train_seconds": elapsed,
                "run_name": run_name,
            }

            results_rows.append(row)

            print(
                f"{train_name} -> {eval_name}: "
                f"accuracy={eval_result['accuracy']:.4f}"
            )

        # Save progress after each trained model.
        results_df = pd.DataFrame(results_rows)
        results_df.to_csv(
            output_dir / "cross_dataset_accuracy_results.csv",
            index=False,
        )

        pivot = results_df.pivot_table(
            index=["train_dataset"],
            columns="eval_dataset",
            values="accuracy_percent",
            aggfunc="mean",
        ).reset_index()

        pivot.to_csv(
            output_dir / "cross_dataset_accuracy_pivot.csv",
            index=False,
        )

        plot_cross_dataset_chart(results_df, output_dir)

        # Free VRAM before next model.
        del model
        del tokenizer
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    make_experiment_summary(output_dir, profile, cfg)

    print("\nDONE.")
    print(f"Results saved in: {output_dir.resolve()}")

    print("\nImportant files:")
    print(f"- {output_dir / 'dataset_summary.csv'}")
    print(f"- {output_dir / 'gnli_proxy_dataset.csv'}")
    print(f"- {output_dir / 'cross_dataset_accuracy_results.csv'}")
    print(f"- {output_dir / 'cross_dataset_accuracy_pivot.csv'}")
    print(f"- {output_dir / 'cross_dataset_accuracy_chart.png'}")
    print(f"- {output_dir / 'experiment_summary.md'}")
    print(f"- {output_dir / 'runs'}")
    print(f"- {output_dir / 'eval_details'}")


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
        help="Run profile: smoke, final, or paper_steps",
    )

    args = parser.parse_args()
    run_pipeline(args.profile)