"""
gradcam.py  —  Grad-CAM heatmap generation and overlay utilities.

Requires: pip install grad-cam
"""

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from dataset import get_val_transforms, IMAGENET_MEAN, IMAGENET_STD
from model import build_model, get_gradcam_target_layer


# ──────────────────────────────────────────────
# Core
# ──────────────────────────────────────────────
def load_model(ckpt_path: str, device: torch.device):
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = build_model(num_classes=9).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def preprocess_image(pil_img: Image.Image) -> tuple[torch.Tensor, np.ndarray]:
    """
    Returns:
        tensor  — (1, 3, 224, 224) normalised tensor  (for model)
        rgb_arr — (224, 224, 3) float32 in [0, 1]     (for overlay)
    """
    img = pil_img.convert("RGB").resize((224, 224))
    rgb_arr = np.array(img).astype(np.float32) / 255.0

    transform = get_val_transforms()
    tensor = transform(image=np.array(img))["image"].unsqueeze(0)
    return tensor, rgb_arr


def generate_gradcam(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    rgb_arr: np.ndarray,
    target_class: int = None,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """
    Returns a (224, 224, 3) uint8 image with Grad-CAM overlay.
    If target_class is None, uses the predicted class.
    """
    target_layer = get_gradcam_target_layer(model)

    targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None

    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=tensor.to(device), targets=targets)

    grayscale_cam = grayscale_cam[0]  # (224, 224)
    overlay = show_cam_on_image(rgb_arr, grayscale_cam, use_rgb=True)
    return overlay   # (224, 224, 3) uint8


# ──────────────────────────────────────────────
# Batch visualisation  (grid of patches)
# ──────────────────────────────────────────────
def visualise_batch(
    image_paths: list[str],
    model: torch.nn.Module,
    class_names: list[str],
    device: torch.device,
    save_path: str = "outputs/figures/gradcam_grid.png",
):
    """
    Given a list of image paths, generate a grid showing:
        original | Grad-CAM overlay | predicted label
    """
    import matplotlib.pyplot as plt

    n = len(image_paths)
    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
    if n == 1:
        axes = [axes]

    for ax_row, path in zip(axes, image_paths):
        pil_img = Image.open(path).convert("RGB")
        tensor, rgb_arr = preprocess_image(pil_img)

        with torch.no_grad():
            logits = model(tensor.to(device))
            pred   = logits.argmax(dim=1).item()
            conf   = torch.softmax(logits, dim=1)[0, pred].item()

        overlay = generate_gradcam(model, tensor, rgb_arr, target_class=pred, device=device)

        ax_row[0].imshow(pil_img.resize((224, 224)))
        ax_row[0].set_title("Original", fontsize=8)
        ax_row[0].axis("off")

        ax_row[1].imshow(overlay)
        ax_row[1].set_title(f"Pred: {class_names[pred]}  ({conf:.1%})", fontsize=8)
        ax_row[1].axis("off")

    plt.suptitle("Grad-CAM Visualisations", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")
