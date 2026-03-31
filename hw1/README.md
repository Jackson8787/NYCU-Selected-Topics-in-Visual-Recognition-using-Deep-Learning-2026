# NYCU Selected Topics in Visual Recognition using Deep Learning 2026 HW1

- Student ID: `110654013`
- Name: `簡惟捷`

## Introduction

This project is a ResNet-based image classification system for the NYCU Selected Topics in Visual Recognition using Deep Learning 2026 Homework 1 competition.  
The final method uses two ResNet-50 models with different random seeds and one ResNet-101 model, followed by test-time augmentation and logit averaging ensemble.

Project structure:

```text
Deep Learning/
├── data/
│   ├── train/
│   │   ├── 0/
│   │   ├── 1/
│   │   ├── ...
│   │   └── 99/
│   ├── val/
│   │   ├── 0/
│   │   ├── 1/
│   │   ├── ...
│   │   └── 99/
│   └── test/
│       ├── *.jpg
│       └── ...
└── hw1/
    ├── README.md
    ├── requirements.txt
    ├── train.py
    ├── infer.py
    ├── .flake8
    └── Performance_Snapshot.png
```

The training and inference commands in this repository assume the following relative layout:

```text
../data
```

Repository files:

```text
hw1/
├── README.md
├── requirements.txt
├── train.py
├── infer.py
├── .flake8
└── Performance_Snapshot.png
```

Dataset source:

[Google Drive Download Link](https://drive.google.com/file/d/1vxiXJHUo6ZPGxBGXwrsSutOpqfJ6HN9D/view?usp=sharing)

## Environment Setup

Create and activate your environment, then install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Training

Train a single model:

```bash
python train.py \
  --data-root ../data \
  --output-dir ./outputs/resnet50_mix_tta \
  --model-name resnet50 \
  --epochs 20 \
  --batch-size 128 \
  --lr 5e-4 \
  --weight-decay 5e-5 \
  --label-smoothing 0.1 \
  --amp \
  --num-workers 8 \
  --prefetch-factor 4 \
  --mixup-alpha 0.2 \
  --cutmix-alpha 1.0 \
  --mix-prob 0.8 \
  --make-submission \
  --submission-tta 2
```

Train the ResNet-101 version:

```bash
python train.py \
  --data-root ../data \
  --output-dir ./outputs/resnet101_mix_tta_094 \
  --model-name resnet101 \
  --epochs 20 \
  --batch-size 64 \
  --lr 3e-4 \
  --weight-decay 5e-5 \
  --label-smoothing 0.1 \
  --amp \
  --num-workers 8 \
  --prefetch-factor 4 \
  --mixup-alpha 0.2 \
  --cutmix-alpha 1.0 \
  --mix-prob 0.8 \
  --make-submission \
  --submission-tta 2
```

### Inference

Generate a single-model submission:

```bash
python infer.py \
  --data-root ../data \
  --checkpoint ./outputs/resnet101_mix_tta_094/best.pt \
  --output-csv ./outputs/resnet101_mix_tta_094/prediction.csv \
  --num-workers 8 \
  --tta 2
```

Generate the final three-model ensemble submission:

```bash
python infer.py \
  --data-root ../data \
  --checkpoint \
    ./outputs/resnet50_mix_tta/best.pt \
    ./outputs/resnet50_mix_tta_seed7/best.pt \
    ./outputs/resnet101_mix_tta_094/best.pt \
  --output-csv ./outputs/final_ensemble/prediction.csv \
  --num-workers 8 \
  --tta 2
```

Then compress the output:

```bash
zip submission.zip prediction.csv
```

## Performance Snapshot

Final selected method:

- ResNet-50
- ResNet-50 (seed 7)
- ResNet-101
- TTA with horizontal flip
- Logit averaging ensemble

Leaderboard snapshot:

![Performance Snapshot](Performance_Snapshot.png)
