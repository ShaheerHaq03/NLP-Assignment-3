import argparse
from src.pipeline import run_pipeline, PROFILES

def main():
    parser = argparse.ArgumentParser(description="Train/evaluate Assignment 2 NLI Reproduction.")
    parser.add_argument("--profile", default="smoke", choices=list(PROFILES.keys()))
    args = parser.parse_args()
    run_pipeline(args.profile)

if __name__ == "__main__":
    main()
