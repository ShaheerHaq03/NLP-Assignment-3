        # NLI Enhancement Experiment

        Enhanced NLI workflow with public+GNLI proxy mixture, optional TRUE evaluation, and optional GitHub backup for Kaggle runs.

        ## Repository structure

        ```text
        project-root/
        ├── README.md
        ├── requirements.txt
        ├── train.py
        ├── inference.py
        ├── config.yaml
        ├── data/
        │   └── sample_data.csv
        ├── notebooks/
        │   └── 01_inference_demo.ipynb
        ├── src/
        │   ├── __init__.py
        │   ├── pipeline.py
        │   ├── model.py
        │   ├── dataset.py
        │   └── utils.py
        ├── results/
        │   ├── baseline_metrics.json
        │   ├── improved_metrics.json
        │   └── training_log.csv
        └── checkpoints/
            └── README.md
        ```

        `src/pipeline.py` contains the original experiment logic. The other files are lightweight wrappers so the workflow stays intact.

        ## Setup

        ```bash
        python -m venv .venv
        source .venv/bin/activate  # Windows: .venv\Scripts\activate
        pip install --upgrade pip
        pip install -r requirements.txt
        ```

        ## Run a quick smoke test

        ```bash
        python train.py --profile smoke
        ```

        This verifies dataset loading, tokenization, training, evaluation, and output writing.

        ## Run the main training profile

        ```bash
        python train.py --profile final
        ```

        Available profiles: `smoke, final, enhanced`.

        ## Long/full run

        ```bash
        python train.py --profile enhanced
        ```

        Use this only after the smoke and final profile work correctly.

        ## Inference

        By default, inference uses `t5-small`. For meaningful predictions, train with model saving enabled first.

        1. Open `src/pipeline.py`.
        2. In the selected profile, change:

        ```python
        "save_models": False
        ```

        to:

        ```python
        "save_models": True
        ```

        3. Train again.
        4. Run inference using the saved model folder:

        ```bash
        python inference.py \
          --premise "Amina submitted the report before the deadline." \
          --hypothesis "Amina submitted something before the deadline." \
          --model_dir outputs_*/runs/<run_name>/model \
          --profile smoke
        ```

        ## Expected generated outputs

        Training writes files into the profile output directory, for example:

        - `dataset_summary.csv`
        - `gnli_proxy_dataset.csv`
        - `cross_dataset_accuracy_results.csv`
        - `cross_dataset_accuracy_pivot.csv`
        - `cross_dataset_accuracy_chart.png`
        - `runs/*/training_log.csv`
        - `runs/*/training_curve.png`
        - `eval_details/*/classification_report_*.json`
        - `eval_details/*/confusion_*.csv`
        - `eval_details/*/predictions_*.csv`

        Before final submission, copy the final key metrics into `results/baseline_metrics.json`, `results/improved_metrics.json`, and `results/training_log.csv` if your instructor requires those exact paths.


## Enhancement-specific notes
- `ENHANCED_ALL_PUBLIC+GNLI_PROXY` mixes MNLI, ANLI, WANLI, and GNLI_PROXY.
- Optional TRUE evaluation runs only when `true/data/*.csv` exists with columns `grounding`, `generated_text`, and `label`.
- Kaggle GitHub backup is enabled by default for the `enhanced` profile if Kaggle Secrets `GITHUB_USERNAME`, `GITHUB_TOKEN`, and `GITHUB_REPO` are available.

        ## Notes

        - The GNLI dataset from the paper is not publicly released, so this code uses a GNLI-style proxy dataset.
        - Do not upload large model weights directly to GitHub unless Git LFS is enabled.
        - Keep only 5–10 rows in `data/sample_data.csv` for submission.
