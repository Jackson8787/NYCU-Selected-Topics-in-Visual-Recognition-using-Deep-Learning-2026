# NYCU Visual Recognition Using Deep Learning 2026 HW4

- Student ID: `110654013`
- Name: `簡惟捷`

## Introduction

This repository contains the code used for NYCU Visual Recognition Using Deep
Learning 2026 Homework 4: all-in-one image restoration for rain and snow
degradation.

The implementation is based on PromptIR and follows the same overall direction
as the official PromptIR repository and the public HW4 reference implementation
shared in class. The final selected method combines:

- a strong PromptIR MSE continuation model trained on all released pairs
- a reference-style PromptIR model trained with L1, PSNR, SSIM, and VGG
  perceptual loss
- test-time augmentation
- weighted prediction ensemble

The best confirmed public leaderboard score of this final legal pipeline is
**30.40 PSNR**.

References:

- PromptIR official repository:
  [va1shn9v/PromptIR](https://github.com/va1shn9v/PromptIR)


## Performance Snapshot

Public leaderboard result:

- Submission score: **30.40 PSNR**
- Final method: `PromptIR + reference-style branch + TTA weighted ensemble`

![Public leaderboard score](assets/public_score_30_40.png)

## Project Structure

```text
hw4/
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- assets/
|   `-- public_score_30_40.png
|-- scripts/
|   |-- _bootstrap.py
|   |-- prepare_data.py
|   |-- smoke_test.py
|   |-- train.py
|   |-- evaluate.py
|   |-- infer.py
|   |-- evaluate_weighted_ensemble.py
|   |-- infer_weighted_ensemble.py
|   `-- ensemble_npz.py
|-- src/
|   `-- hw4/
|       |-- __init__.py
|       |-- data.py
|       |-- engine.py
|       `-- model.py
`-- tests/
    `-- test_perceptual_loss.py
```

The released dataset and training outputs are intentionally excluded from this
GitHub package.

## Environment Setup


Install the required packages:

```powershell
python pip install -r requirements.txt
```

## Data Setup

Place the released archive in the project root, then extract and validate it:


This command extracts the release ZIP and verifies:

- 3200 paired training samples
- 100 test images
- RGB format
- image size `256 x 256`

## Method Summary

### 1. PromptIR Main Branch

The first branch is a PromptIR model fine-tuned toward PSNR with MSE-based
continuation on all released training pairs. The strongest single-model branch
used:

- `patch_size=192`
- `AdamW`
- `weight_decay=0`
- ultra-low constant learning rate continuation
- degradation-aware weighting with stronger snow emphasis
- TTA during inference

### 2. Reference-Style Branch

The second branch follows the public HW4 reference direction more closely:

- PromptIR baseline profile
- `AdamW(lr=2e-4, weight_decay=1e-4)`
- OneCycle learning-rate schedule
- composite loss:
  `L1 + 0.1 * PSNR + 0.1 * SSIM + 0.01 * perceptual`
- perceptual backbone: ImageNet-pretrained VGG16

This branch was not the strongest single model by itself, but it produced
useful complementarity for the final ensemble.

### 3. Final Ensemble

The final submission averages the two model predictions after TTA:

- Model A: strongest MSE continuation checkpoint
- Model B: strongest reference-style checkpoint
- weighted ensemble with `weight_b = 0.35`

This was the method that reached the final public score of **30.40**.

## Usage

### Smoke Test

Run a quick GPU forward/backward check:

```powershell
python smoke_test.py --batch-size 4 --patch-size 128
```

### Train the Baseline PromptIR Branch

```powershell
python train.py `
  --train-all `
  --patch-size 192 `
  --batch-size 4 `
  --accumulation-steps 2 `
  --learning-rate 2.5e-7 `
  --weight-decay 0 `
  --scheduler constant `
  --loss mse `
  --rain-weight 1.0 `
  --snow-weight 2.0 `
  --init-checkpoint outputs/trainall192_mse_wd0_const2p5e7_1ep_from_constbest_v4/checkpoints/best.pt `
  --output-dir outputs/trainall192_mse_wd0_const2p5e7_snoww2p0_from_v4
```

### Train the Reference-Style Branch

```powershell
python train.py `
  --patch-size 128 `
  --batch-size 4 `
  --epochs 150 `
  --learning-rate 2e-4 `
  --weight-decay 1e-4 `
  --scheduler onecycle `
  --onecycle-pct-start 0.1 `
  --onecycle-div-factor 25 `
  --onecycle-final-div-factor 10000 `
  --loss l1_psnr_ssim_perceptual `
  --psnr-weight 0.1 `
  --ssim-weight 0.1 `
  --perceptual-weight 0.01 `
  --perceptual-backbone vgg16 `
  --perceptual-layer-preset reference `
  --no-amp `
  --output-dir outputs/ref_full_l1psnrssimp_vgg16_150ep
```

### Generate the Final Ensemble Submission

```powershell
python infer_weighted_ensemble.py `
  --checkpoint-a outputs/trainall192_mse_wd0_const2p5e7_snoww2p0_from_v4/checkpoints/best.pt `
  --checkpoint-b outputs/ref_full_l1psnrssimp_vgg16_150ep/checkpoints/best.pt `
  --weight-b 0.35 `
  --tta `
  --output-dir outputs/neural_ensemble_mse_snoww2p0_refmirror_wb035_tta
```

The final ZIP will be created as:

```text
outputs/neural_ensemble_mse_snoww2p0_refmirror_wb035_tta/hw4_promptir_weighted_ensemble_submission.zip
```
