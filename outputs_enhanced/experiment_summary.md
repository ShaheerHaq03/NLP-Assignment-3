# NLI Reproduction Experiment Summary

Profile: `enhanced`

Model: `t5-small`

Hardware: `{"cuda_available": true, "device": "cuda", "gpu_name": "Tesla T4", "gpu_memory_gb": 14.56}`


## Important Reproducibility Note

The original GNLI dataset was not released by the paper authors. This experiment therefore uses a GNLI-style proxy generated with the same domain/length/label design. The experiment reproduces the structure of the paper at laptop scale using T5-small.


## Cross-Dataset Accuracy Pivot

| train_dataset                  |   ANLI |   GNLI_PROXY |   MNLI |   WANLI |
|:-------------------------------|-------:|-------------:|-------:|--------:|
| ANLI                           | 0.4152 |     0.999561 | 0.7204 |  0.5428 |
| ANLI+GNLI_PROXY                | 0.4116 |     1        | 0.6972 |  0.5452 |
| ENHANCED_ALL_PUBLIC+GNLI_PROXY | 0.3844 |     1        | 0.764  |  0.632  |
| GNLI_PROXY                     | 0.3292 |     1        | 0.622  |  0.4852 |
| MNLI                           | 0.3004 |     0.973246 | 0.7792 |  0.5592 |
| MNLI+ANLI+WANLI                | 0.3896 |     0.997807 | 0.778  |  0.6116 |
| MNLI+GNLI_PROXY                | 0.3008 |     1        | 0.78   |  0.5464 |
| WANLI                          | 0.362  |     0.990351 | 0.6656 |  0.6492 |
| WANLI+GNLI_PROXY               | 0.3668 |     1        | 0.6736 |  0.64   |



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
