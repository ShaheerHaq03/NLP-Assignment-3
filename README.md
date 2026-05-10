<div align="center">

# Synthetic Data for Domain Generalization of NLI Models

### Reproduction & Enhancement Study

Based on the research paper:

**A Synthetic Data Approach for Domain Generalization of NLI Models**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)
![Research](https://img.shields.io/badge/Research-NLP-purple.svg)

</div>

---

# Project Overview

This repository contains two complete implementations related to **Domain Generalization in Natural Language Inference (NLI)**:

* **Assignment 2 — Reproduction**
* **Assignment 3 — Enhancement & Extension**

The goal of this project was to reproduce and extend the methodology proposed in the original research paper using publicly available datasets and a GNLI-style synthetic proxy dataset.

The implementation focuses on:

* Cross-domain Natural Language Inference
* Synthetic data generation
* Domain generalization
* T5-small fine-tuning
* Cross-dataset evaluation
* Enhanced dataset mixtures
* Research reproducibility

---

# Repository Structure

```bash
.
├── Assignmentment 2 - Reproduction
│   ├── README.md
│   ├── train.py
│   ├── inference.py
│   ├── config.yaml
│   ├── requirements.txt
│   ├── data/
│   ├── notebooks/
│   ├── results/
│   ├── checkpoints/
│   └── src/
│
├── Assignment 3 - Enhancement
│   ├── README.md
│   ├── train.py
│   ├── inference.py
│   ├── config.yaml
│   ├── requirements.txt
│   ├── data/
│   ├── notebooks/
│   ├── results/
│   ├── checkpoints/
│   └── src/
```

---

# Assignment 2 — Reproduction

## Objective

Reproduce the experimental structure of the original paper using:

* Public NLI datasets
* GNLI-style synthetic data
* T5-small fine-tuning
* Cross-domain evaluation

## Features

* T5-small training pipeline
* Step-based fine-tuning
* Cross-dataset accuracy evaluation
* GNLI-style synthetic proxy generation
* Confusion matrices
* Classification reports
* Training curves
* Experimental logging

## Datasets Used

| Dataset    | Purpose                                  |
| ---------- | ---------------------------------------- |
| MNLI       | Multi-domain NLI benchmark               |
| ANLI       | Adversarial NLI benchmark                |
| WANLI      | Weakly-supervised adversarial NLI        |
| GNLI_PROXY | Synthetic domain-generalized NLI dataset |

---

# Assignment 3 — Enhancement

## Objective

Extend the reproduced pipeline with enhanced experimentation and additional training strategies.

## Enhancements Added

* Enhanced ALL_PUBLIC + GNLI dataset mixture
* TRUE benchmark evaluation support
* Multi-GPU support
* Automated GitHub backup support
* Larger synthetic dataset generation
* Improved evaluation pipeline
* Extended experiment tracking

## Additional Features

| Feature            | Description                              |
| ------------------ | ---------------------------------------- |
| TRUE Evaluation    | Optional AUC-based factuality evaluation |
| Multi-GPU Training | DataParallel support                     |
| Auto Backup        | Automatic GitHub output backup           |
| Enhanced GNLI      | Larger synthetic dataset generation      |
| Expanded Mixtures  | Additional dataset combinations          |

---

# Model Architecture

```text
Model: T5-small
Framework: HuggingFace Transformers
Backend: PyTorch
Task: Text-to-Text NLI Classification
```

Input Format:

```text
nli premise: <premise>
hypothesis: <hypothesis>
label:
```

Output Labels:

```text
entailment
neutral
contradiction
```

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

---

## 2. Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Experiments

# Assignment 2

## Smoke Test

```bash
python train.py --profile smoke
```

## Final Reproduction Run

```bash
python train.py --profile final
```

## Paper-Scale Step Run

```bash
python train.py --profile paper_steps
```

---

# Assignment 3

## Smoke Test

```bash
python train.py --profile smoke
```

## Final Run

```bash
python train.py --profile final
```

## Enhanced Run

```bash
python train.py --profile enhanced
```

---

# Inference

```bash
python inference.py
```

Example:

```text
Premise:
The meeting was delayed because of rain.

Hypothesis:
Bad weather caused the delay.

Prediction:
entailment
```

---

# Output Files

The pipeline automatically generates:

```bash
results/
│
├── cross_dataset_accuracy_results.csv
├── cross_dataset_accuracy_pivot.csv
├── dataset_summary.csv
├── training_log.csv
├── confusion matrices
├── classification reports
├── training curves
└── experiment summaries
```

---

# Synthetic GNLI Proxy Dataset

The original GNLI dataset used in the paper was not publicly released.

To reproduce the experimental structure, this project generates a GNLI-style synthetic dataset using:

* Multi-domain templates
* Balanced NLI labels
* Short and paragraph-style text generation
* Controlled synthetic sampling

## Domains Included

```text
news
legal
medical
reddit
twitter
forums
reviews
emails
scientific articles
student papers
youtube comments
and more...
```

---

# Technologies Used

| Category       | Tools                    |
| -------------- | ------------------------ |
| Deep Learning  | PyTorch                  |
| NLP            | HuggingFace Transformers |
| Datasets       | HuggingFace Datasets     |
| Evaluation     | Scikit-learn             |
| Visualization  | Matplotlib               |
| Language Model | T5-small                 |

---

# Hardware Used

The experiments were designed for consumer-grade GPUs.

```text
RTX 4060 Laptop GPU
8GB VRAM
16GB RAM
```

Gradient accumulation and memory-aware configurations were used to maintain stable training.

---

# Key Research Goals

* Domain Generalization
* Synthetic Data Augmentation
* Cross-Dataset NLI Evaluation
* Robustness in Natural Language Inference
* Reproducibility of NLP Research

---

# Important Reproducibility Note

The original GNLI dataset and FLAN-PaLM synthetic generation pipeline from the paper were not publicly released.

Therefore, this implementation reproduces the experimental methodology using:

* Public NLI datasets
* GNLI-style synthetic proxy generation
* T5-small fine-tuning
* Cross-domain evaluation

The focus of this work is reproducibility, methodology alignment, and experimental extension under consumer hardware constraints.

---

# Future Improvements

* T5-base / T5-large experiments
* LoRA fine-tuning
* Better synthetic generation using modern LLMs
* PGVector / FAISS integration
* Distributed multi-node training
* Improved TRUE benchmark evaluation

---
