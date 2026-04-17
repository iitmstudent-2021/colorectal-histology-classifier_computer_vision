"""
train.py  —  Training script for CRC tissue classifier.
Run locally or paste into a Kaggle notebook cell-by-cell.

Usage:
    python src/train.py --data_root data/NCT-CRC-HE-100K-NONORM \
                        --output_dir outputs \
                        --epochs 30 \
                        --batch_size 64
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import build_dataloaders, CLASS_NAMES
from model import build_model, count_parameters


# ──────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",   default="data/NCT-CRC-HE-100K-NONORM")
    p.add_argument("--output_dir",  default="outputs")
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--patience",    type=int,   default=7)
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


# ──────────────────────────────────────────────
# One epoch helpers
# ──────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  val  ", leave=False):
        images, labels = images.to(device), labels.to(device)
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += images.size(0)

    return total_loss / total, correct / total


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(args.output_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    # ── Data ──
    print("Loading data...")
    train_loader, val_loader, class_weights = build_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    print(f"  Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    # ── Model ──
    model = build_model(num_classes=9).to(device)
    print(count_parameters(model))

    # ── Loss, Optimiser, Scheduler ──
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler    = torch.cuda.amp.GradScaler()

    # ── Training loop ──
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss,   val_acc   = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"({elapsed:.0f}s)"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            ckpt_path = out_dir / "checkpoints" / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "class_names": CLASS_NAMES,
            }, ckpt_path)
            print(f"  ✓ Saved best model  val_acc={val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Save training history
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print("Training complete.")


if __name__ == "__main__":
    main()
