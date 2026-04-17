---
title: CRC Tissue Classifier
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
pinned: false
license: mit
tags:
  - medical
  - histopathology
  - image-classification
  - pytorch
  - efficientnet
  - colorectal-cancer
  - computer-vision
---

# Colorectal Cancer Tissue Classifier

**EfficientNet-B0** fine-tuned on the [NCT-CRC-HE-100K](https://zenodo.org/record/1214456) dataset to classify H&E-stained colorectal tissue patches into **9 classes**.

---

## Model Performance

| Metric | Value |
|--------|-------|
| Validation Accuracy | **99.64%** |
| ROC-AUC (all 9 classes) | **1.000** |
| Architecture | EfficientNet-B0 |
| Parameters | ~5.3M |
| Training Epochs | 25 |

---

## Tissue Classes

| Class | Description |
|-------|-------------|
| ADI | Adipose (fat) tissue |
| BACK | Background / non-tissue |
| DEB | Debris / cellular artifacts |
| LYM | Lymphocytes (immune cells) |
| MUC | Mucus |
| MUS | Smooth muscle |
| NORM | Normal colon mucosa |
| STR | Tumor-associated stroma |
| TUM | Adenocarcinoma epithelium (tumor) |

---

## Dataset

- **NCT-CRC-HE-100K-NONORM** — 100,000 non-normalized H&E patches (224×224 px, 0.5 µm/px)
- 9 tissue classes, ~11,000 images per class
- Source: [Kather et al., 2019 — Zenodo](https://zenodo.org/record/1214456)

---

## Training Details

| Component | Choice |
|-----------|--------|
| Backbone | EfficientNet-B0 (ImageNet pretrained) |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR |
| Loss | CrossEntropyLoss + label smoothing (0.1) |
| Augmentation | RandomResizedCrop, HFlip, VFlip, Rotate90, ColorJitter, HueSaturation, GaussNoise |
| Batch Size | 64 |
| Val Split | 15% (random, stratified by class) |

---

## Features

- Single-image inference with **confidence scores** for all 9 classes
- **Grad-CAM heatmap** showing which regions drive the prediction
- **Probability bar chart** for all classes
- Sample patches included for quick demo

---

## Project Structure

```
├── app.py                  # Gradio demo (HF Spaces entry point)
├── requirements.txt
├── outputs/
│   └── best_model.pth      # Trained model checkpoint
├── assets/
│   └── samples/            # Example H&E patches for demo
└── src/
    ├── dataset.py          # Data loading & augmentation
    ├── model.py            # EfficientNet-B0 architecture
    ├── train.py            # Training loop
    ├── evaluate.py         # Metrics & visualisation
    └── gradcam.py          # Grad-CAM utilities
```

---

## How to Run Locally

```bash
git clone https://github.com/your-username/crc-tissue-classifier
cd crc-tissue-classifier
pip install -r requirements.txt
python app.py
```

---

## Citation

```bibtex
@article{kather2019predicting,
  title     = {Predicting survival from colorectal cancer histology slides using deep learning},
  author    = {Kather, Jakob Nikolas and others},
  journal   = {PLOS Medicine},
  year      = {2019},
  publisher = {Public Library of Science}
}
```

---

## License

MIT License. Model weights released for research and educational use only.  
**Not intended for clinical diagnosis.**
