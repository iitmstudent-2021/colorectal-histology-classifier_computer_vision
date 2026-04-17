"""
evaluate.py  —  Full evaluation: confusion matrix, per-class metrics,
                Cohen's kappa, ROC curves, training curves.

Usage:
    python src/evaluate.py \
        --data_root  data/NCT-CRC-HE-100K-NONORM \
        --ckpt_path  outputs/checkpoints/best_model.pth \
        --output_dir outputs/figures
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report, cohen_kappa_score,
    confusion_matrix, roc_auc_score, roc_curve,
)
from tqdm import tqdm

from dataset import CRCDataset, get_val_transforms, CLASS_NAMES, NUM_CLASSES
from model import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",   default="data/NCT-CRC-HE-100K-NONORM")
    p.add_argument("--ckpt_path",   default="outputs/checkpoints/best_model.pth")
    p.add_argument("--output_dir",  default="outputs/figures")
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────
@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        logits = model(images)
        probs  = F.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_probs.extend(probs)

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ──────────────────────────────────────────────
# Plot: Confusion Matrix
# ──────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        linewidths=0.5, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True",      fontsize=12)
    ax.set_title("Normalised Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


# ──────────────────────────────────────────────
# Plot: ROC Curves (one-vs-rest)
# ──────────────────────────────────────────────
def plot_roc_curves(y_true, y_probs, save_path):
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))

    for i, (cls, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        auc = roc_auc_score(y_bin[:, i], y_probs[:, i])
        ax.plot(fpr, tpr, color=color, label=f"{cls}  (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — One-vs-Rest")
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


# ──────────────────────────────────────────────
# Plot: Training curves from history.json
# ──────────────────────────────────────────────
def plot_training_curves(history_path: str, save_path: str):
    import json
    with open(history_path) as f:
        h = json.load(f)

    epochs = range(1, len(h["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, h["train_loss"], label="Train")
    ax1.plot(epochs, h["val_loss"],   label="Val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(epochs, h["train_acc"], label="Train")
    ax2.plot(epochs, h["val_acc"],   label="Val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.suptitle("Training History", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    ckpt = torch.load(args.ckpt_path, map_location=device)
    model = build_model(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    # Build dataset (full set as test)
    from torch.utils.data import DataLoader
    ds = CRCDataset(args.data_root, transform=get_val_transforms())
    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)

    # Predictions
    y_true, y_pred, y_probs = get_predictions(model, loader, device)

    # Metrics
    acc   = (y_true == y_pred).mean()
    kappa = cohen_kappa_score(y_true, y_pred)
    print(f"\nOverall accuracy : {acc:.4f}")
    print(f"Cohen's kappa    : {kappa:.4f}")
    print("\nPer-class report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

    # Plots
    plot_confusion_matrix(y_true, y_pred,  out_dir / "confusion_matrix.png")
    plot_roc_curves(y_true, y_probs,        out_dir / "roc_curves.png")

    history_path = Path(args.ckpt_path).parent.parent / "history.json"
    if history_path.exists():
        plot_training_curves(str(history_path), str(out_dir / "training_curves.png"))


if __name__ == "__main__":
    main()
