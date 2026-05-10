import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.pipeline import make_nli_input, score_candidates, get_device, PROFILES

LABELS = ["entailment", "neutral", "contradiction"]

def load_model(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    device = get_device()
    model.to(device)
    model.eval()
    return model, tokenizer, device

def predict(premise: str, hypothesis: str, model_dir: str, profile: str):
    cfg = PROFILES[profile].copy()
    model, tokenizer, device = load_model(model_dir)
    input_text = make_nli_input(premise, hypothesis)
    scores = score_candidates(model, tokenizer, [input_text], LABELS, cfg, device)[0]
    pred_idx = int(scores.argmax())
    return {"prediction": LABELS[pred_idx], "scores": dict(zip(LABELS, [float(x) for x in scores]))}

def main():
    parser = argparse.ArgumentParser(description="Run NLI inference.")
    parser.add_argument("--premise", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--model_dir", default="t5-small", help="Path to saved checkpoint model folder, or HF model name.")
    parser.add_argument("--profile", default="smoke", choices=list(PROFILES.keys()))
    args = parser.parse_args()
    print(predict(args.premise, args.hypothesis, args.model_dir, args.profile))

if __name__ == "__main__":
    main()
