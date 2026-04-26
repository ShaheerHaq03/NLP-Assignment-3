# NLI Reproduction Experiment Summary

Profile: `smoke`

Model: `t5-small`

Hardware: `{"cuda_available": true, "device": "cuda", "gpu_name": "Tesla T4", "gpu_memory_gb": 14.56}`


## Important Reproducibility Note

The original GNLI dataset was not released by the paper authors. This experiment therefore uses a GNLI-style proxy generated with the same domain/length/label design. The experiment reproduces the structure of the paper at laptop scale using T5-small.


## Cross-Dataset Accuracy Pivot

| train_dataset    |     ANLI |   GNLI_PROXY |     MNLI |    WANLI |
|:-----------------|---------:|-------------:|---------:|---------:|
| ANLI             | 0.39     |     0.912568 | 0.693333 | 0.563333 |
| ANLI+GNLI_PROXY  | 0.403333 |     1        | 0.696667 | 0.55     |
| GNLI_PROXY       | 0.33     |     1        | 0.596667 | 0.493333 |
| MNLI             | 0.29     |     0.989071 | 0.776667 | 0.546667 |
| MNLI+ANLI+WANLI  | 0.356667 |     0.939891 | 0.703333 | 0.59     |
| MNLI+GNLI_PROXY  | 0.286667 |     1        | 0.733333 | 0.536667 |
| WANLI            | 0.38     |     0.874317 | 0.643333 | 0.536667 |
| WANLI+GNLI_PROXY | 0.323333 |     1        | 0.653333 | 0.57     |



## Output Files

- `dataset_summary.csv`

- `cross_dataset_accuracy_results.csv`

- `cross_dataset_accuracy_pivot.csv`

- `cross_dataset_accuracy_chart.png`

- `gnli_proxy_dataset.csv`

- `runs/*/training_log.csv`

- `runs/*/training_curve.png`

- `eval_details/*/confusion_*.csv`

- `eval_details/*/classification_report_*.json`
