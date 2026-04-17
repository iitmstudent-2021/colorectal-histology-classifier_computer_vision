"""
app.py  —  Gradio demo for NCT-CRC-HE-100K colorectal tissue classifier.
Deployed on Hugging Face Spaces.
"""

import os
import sys
import asyncio
import numpy as np

# Suppress WinError 10054 (Windows asyncio connection-reset noise)
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import torch
import gradio as gr
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Path setup (so src/ imports work on HF Spaces) ──────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from model import build_model, get_gradcam_target_layer
from dataset import get_val_transforms, CLASS_NAMES, CLASS_DESCRIPTIONS

# ── Constants ────────────────────────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH  = os.path.join(os.path.dirname(__file__), "outputs", "best_model.pth")

CLASS_COLORS = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
    "#59a14f", "#edc948", "#b07aa1", "#ff9da7", "#9c755f",
]

# ── Load model once at startup ───────────────────────────────────────────────
def load_model():
    # weights=None avoids downloading ImageNet weights we'll overwrite anyway
    from torchvision.models import efficientnet_b0
    import torch.nn as nn
    model = efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(1280, 9),
    )
    ckpt  = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model

MODEL = load_model()


# ── Grad-CAM (pure torch, no extra library needed) ───────────────────────────
def compute_gradcam(model, tensor, target_class):
    """Returns (224,224) numpy heatmap in [0,1]."""
    target_layer = get_gradcam_target_layer(model)
    activations, gradients = {}, {}

    def fwd_hook(_, __, out):
        activations["value"] = out.detach()

    def bwd_hook(_, __, grad_out):
        gradients["value"] = grad_out[0].detach()

    fh = target_layer.register_forward_hook(fwd_hook)
    bh = target_layer.register_full_backward_hook(bwd_hook)

    tensor = tensor.to(DEVICE).requires_grad_(True)
    logits = model(tensor)
    model.zero_grad()
    logits[0, target_class].backward()

    fh.remove()
    bh.remove()

    acts  = activations["value"][0]          # (C, H, W)
    grads = gradients["value"][0]            # (C, H, W)
    weights = grads.mean(dim=(1, 2))         # (C,)
    cam = (weights[:, None, None] * acts).sum(0)
    cam = torch.relu(cam).cpu().numpy()

    # Normalise to [0, 1]
    if cam.max() > 0:
        cam = cam / cam.max()

    # Upsample to 224×224
    import cv2
    cam = cv2.resize(cam, (224, 224))
    return cam


def overlay_heatmap(rgb_arr, cam):
    """Blend heatmap onto image. rgb_arr: float32 (224,224,3) in [0,1]."""
    import cv2
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap  = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay  = 0.5 * rgb_arr + 0.5 * heatmap
    return np.clip(overlay, 0, 1)


# ── Main inference function ──────────────────────────────────────────────────
def predict(pil_image: Image.Image):
    if pil_image is None:
        return None, None, "Please upload an image."

    # Pre-process
    img_rgb  = pil_image.convert("RGB").resize((224, 224))
    rgb_arr  = np.array(img_rgb).astype(np.float32) / 255.0
    transform = get_val_transforms(img_size=224)
    tensor   = transform(image=np.array(img_rgb))["image"].unsqueeze(0)

    # Forward pass
    with torch.no_grad():
        logits = MODEL(tensor.to(DEVICE))
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx  = int(np.argmax(probs))
    pred_name = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    # Grad-CAM
    cam     = compute_gradcam(MODEL, tensor.clone(), pred_idx)
    overlay = overlay_heatmap(rgb_arr, cam)
    overlay_pil = Image.fromarray(np.uint8(overlay * 255))

    # Probability bar chart
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.barh(CLASS_NAMES, probs * 100, color=CLASS_COLORS, edgecolor="white", height=0.6)
    ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9)
    ax.set_xlim(0, 115)
    ax.set_xlabel("Confidence (%)", fontsize=10)
    ax.set_title("Class Probabilities", fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.close(fig)

    # Result markdown
    description = CLASS_DESCRIPTIONS.get(pred_name, "")
    result_md = (
        f"### Prediction: **{pred_name}** ({confidence:.1%})\n\n"
        f"> {description}\n\n"
        f"**Top-3 predictions:**\n"
    )
    top3 = np.argsort(probs)[::-1][:3]
    for i, idx in enumerate(top3, 1):
        result_md += f"{i}. `{CLASS_NAMES[idx]}` — {probs[idx]:.1%}\n"

    return overlay_pil, fig, result_md


# ── Sample images (bundled in repo under assets/) ───────────────────────────
SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "assets", "samples")
SAMPLES = []
if os.path.isdir(SAMPLE_DIR):
    for f in sorted(os.listdir(SAMPLE_DIR))[:9]:
        if f.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")):
            SAMPLES.append(os.path.join(SAMPLE_DIR, f))


# ── Gradio UI ────────────────────────────────────────────────────────────────
DESCRIPTION = """
## Colorectal Cancer Tissue Classifier
**EfficientNet-B0** fine-tuned on [NCT-CRC-HE-100K](https://zenodo.org/record/1214456) — 100,000 H&E stained patches across **9 tissue classes**.

| Class | Description |
|-------|-------------|
| ADI | Adipose (fat) tissue |
| BACK | Background / non-tissue |
| DEB | Debris / artifacts |
| LYM | Lymphocytes |
| MUC | Mucus |
| MUS | Smooth muscle |
| NORM | Normal colon mucosa |
| STR | Tumor-associated stroma |
| TUM | Adenocarcinoma (tumor) |

**Validation accuracy: 99.64% · AUC: 1.000 (all classes)**
"""

with gr.Blocks(title="CRC Tissue Classifier") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload H&E Patch (224×224 recommended)", height=280)
            run_btn     = gr.Button("Classify", variant="primary", size="lg")

            if SAMPLES:
                gr.Examples(
                    examples=SAMPLES,
                    inputs=image_input,
                    label="Sample patches",
                    examples_per_page=9,
                )

        with gr.Column(scale=1):
            gradcam_out = gr.Image(type="pil", label="Grad-CAM Heatmap", height=280)
            result_md   = gr.Markdown(label="Result")

    prob_plot = gr.Plot(label="Class Probabilities")

    run_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[gradcam_out, prob_plot, result_md],
    )
    image_input.change(
        fn=predict,
        inputs=image_input,
        outputs=[gradcam_out, prob_plot, result_md],
    )

    gr.Markdown(
        "---\n"
        "**Model:** EfficientNet-B0 · **Framework:** PyTorch + Albumentations · "
        "**Dataset:** NCT-CRC-HE-100K-NONORM (Kather et al., 2019) · "
        "**Code:** [GitHub](https://github.com/your-username/crc-tissue-classifier)"
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
