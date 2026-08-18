# Steel Plate Defect Segmentation

Using UNet-based models for defect image segmentation on steel plate surfaces.

## Overview

This project applies deep learning-based semantic segmentation to detect and localize surface defects on steel plates. The model takes a grayscale steel plate image as input and outputs pixel-level segmentation masks for 4 defect classes.

## Models

| Model | Backbone |
|---|---|
| Attention UNet + ResNet34 | ResNet34 |
| Attention UNet + EfficientNetB4 | EfficientNetB4 |
| Attention UNet + InceptionV3 | InceptionV3 |

All models use a shared encoder with attention gates (AG) for improved feature selection.

## Dataset

[Severstal Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection) (Kaggle)

- Image size: 1600 × 256 (resized to 800 × 128 for training)
- 4 defect classes
- Pixel-level RLE annotations

## Demo

Upload a steel plate image to get:
- Defect segmentation overlay
- Per-class segmentation masks
- Dice Score and IoU metrics (when Ground Truth is available)

## Setup

```bash
conda create -n steel python=3.11 -y
conda activate steel
pip install tensorflow streamlit opencv-python keras pandas numpy
streamlit run app.py
```

## Evaluation Metrics

- **Dice Score** — measures overlap between predicted and ground truth masks
- **IoU (Intersection over Union)** — measures the ratio of intersection to union of predicted and ground truth masks
- Both metrics are computed only over classes with ground truth labels (macro average)
