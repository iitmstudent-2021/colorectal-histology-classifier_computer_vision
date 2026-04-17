---
title: CRC Tissue Classifier
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.29.0"
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

## Live Demo

**Try it instantly — no setup required:**

> **https://huggingface.co/spaces/PANKAJ-MOHAN/colorectal-histology-classifier**

---

## How to Use the App

### Step 1 — Open the Demo
Visit the live demo link above. The app loads in your browser — no installation needed.

### Step 2 — Upload an H&E Patch
Click the **"Upload H&E Patch"** image box and upload any colorectal histopathology image (`.tif`, `.png`, `.jpg`).

> **Recommended:** Use the sample images provided in the `Sample_Data/` folder of this repository — one patch per tissue class, ready to test.

### Step 3 — Click "Classify"
Hit the blue **Classify** button (or the image will auto-classify on upload).

### Step 4 — Read the Results
The app returns three outputs:

| Output | What It Shows |
|--------|--------------|
| **Grad-CAM Heatmap** | Highlighted regions the model focused on — red = high attention |
| **Prediction Card** | Top predicted class with confidence %, description, and Top-3 predictions |
| **Class Probabilities Chart** | Horizontal bar chart showing confidence across all 9 classes |

---

## Sample Test Images

The `Sample_Data/` folder contains **9 ready-to-use test patches** — one per tissue class:

| File | Class | Expected Prediction |
|------|-------|-------------------|
| `ADI-AAQEFMRI.tif` | Adipose tissue | ADI |
| `BACK-AAQSHYMA.tif` | Background | BACK |
| `DEB-ACHGAMCT.tif` | Debris / artifacts | DEB |
| `LYM-AATTSRNN.tif` | Lymphocytes | LYM |
| `MUC-AAQTMGMA.tif` | Mucus | MUC |
| `MUS-AACVWIEK.tif` | Smooth muscle | MUS |
| `NORM-AAAWMSFI.tif` | Normal colon mucosa | NORM |
| `STR-AACRYYNQ.tif` | Tumor-associated stroma | STR |
| `TUM-AACHEHDV.tif` | Adenocarcinoma (tumor) | TUM |

Download any of these and drag-drop into the app to test.

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

| Class | Description | Clinical Significance |
|-------|-------------|----------------------|
| ADI | Adipose (fat) tissue | Tumor boundary marker |
| BACK | Background / non-tissue | Quality control |
| DEB | Debris / cellular artifacts | Necrosis indicator |
| LYM | Lymphocytes (immune cells) | Immunotherapy response predictor |
| MUC | Mucus | Mucinous adenocarcinoma subtype |
| MUS | Smooth muscle | Muscularis invasion (T-staging) |
| NORM | Normal colon mucosa | Healthy baseline tissue |
| STR | Tumor-associated stroma | Linked to poor prognosis |
| TUM | Adenocarcinoma epithelium | The cancer itself |

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

## Project Structure

```
├── app.py                  # Gradio demo (HF Spaces entry point)
├── requirements.txt
├── Sample_Data/            # 9 test patches — one per tissue class
│   ├── ADI-AAQEFMRI.tif
│   ├── BACK-AAQSHYMA.tif
│   ├── DEB-ACHGAMCT.tif
│   ├── LYM-AATTSRNN.tif
│   ├── MUC-AAQTMGMA.tif
│   ├── MUS-AACVWIEK.tif
│   ├── NORM-AAAWMSFI.tif
│   ├── STR-AACRYYNQ.tif
│   └── TUM-AACHEHDV.tif
├── outputs/
│   └── best_model.pth      # Trained model checkpoint
├── docs/
│   └── problem_statement.md  # Problem statement & stakeholder analysis
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
git clone https://github.com/iitmstudent-2021/colorectal-histology-classifier_computer_vision
cd colorectal-histology-classifier_computer_vision
pip install -r requirements.txt
python app.py
# Open http://127.0.0.1:7860 in your browser
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
