import os
from io import BytesIO
import requests
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50

import streamlit as st
import cv2


# ---------------- CONFIG ----------------
MODEL_URL = "https://github.com/shreyashreepani123/visionai-app/releases/download/v1.1/checkpoint.pth"
CHECKPOINT_PATH = "checkpoint.pth"

NUM_CLASSES = 91           # <- exactly what you trained with
IMAGE_SIZE = 256           # <- your training resize
DEVICE = torch.device("cpu")


# ---------------- UTIL ----------------
def ensure_checkpoint():
    """Download the checkpoint once."""
    if os.path.exists(CHECKPOINT_PATH):
        return
    st.write("⬇️ Downloading checkpoint…")
    r = requests.get(MODEL_URL, timeout=300, allow_redirects=True)
    r.raise_for_status()
    with open(CHECKPOINT_PATH, "wb") as f:
        f.write(r.content)
    st.write("✅ Download complete!")


def build_model():
    """
    Match the training setup you shared:
      model = deeplabv3_resnet50(pretrained=True)
      model.classifier[4] = nn.Conv2d(256, 91, 1)
    At inference we do not need the aux head; removing it avoids shape clashes.
    """
    model = deeplabv3_resnet50(
        weights=None,           # we’ll load YOUR weights
        num_classes=NUM_CLASSES,
        aux_loss=False          # no aux head in the runtime model
    )
    # (If torchvision ignores num_classes for some versions, enforce head:)
    if getattr(model.classifier[4], "out_channels", None) != NUM_CLASSES:
        model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    return model


def strip_module_prefix(sd):
    """Handle DataParallel checkpoints."""
    out = {}
    for k, v in sd.items():
        if k.startswith("module."):
            out[k[len("module."):]] = v
        else:
            out[k] = v
    return out


def load_weights_safely(model, ckpt_path):
    """
    Load only keys that exist AND have the same shape.
    This avoids the shape-mismatch RuntimeError you saw.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    else:
        state_dict = ckpt

    state_dict = strip_module_prefix(state_dict)
    model_sd = model.state_dict()

    compatible = {k: v for k, v in state_dict.items()
                  if (k in model_sd and model_sd[k].shape == v.shape)}

    # Load the compatible subset
    res = model.load_state_dict(compatible, strict=False)

    # Report what was skipped (for transparency)
    missing = list(res.missing_keys)
    unexpected = list(res.unexpected_keys)
    if missing:
        st.info(f"ℹ️ Missing keys not loaded (expected by model, absent in ckpt): {missing[:6]}{' …' if len(missing)>6 else ''}")
    if unexpected:
        st.info(f"ℹ️ Unexpected keys ignored (present in ckpt, not in model): {unexpected[:6]}{' …' if len(unexpected)>6 else ''}")


# ------------- PRE/POST ---------------
transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # same as training script
    T.ToTensor(),                        # no normalization (to match your code)
])


def predict_logits(model, pil_img):
    """Forward pass -> logits resized back to original size."""
    orig_w, orig_h = pil_img.size
    inp = transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(inp)["out"]                         # [1, C, h, w]
        up = F.interpolate(out, size=(orig_h, orig_w),  # back to original
                            mode="bilinear", align_corners=False)
    return up.squeeze(0).cpu()  # [C, H, W]


def refine_binary(binary, open_radius=3, close_radius=5):
    """Light morphological denoise to remove pepper noise & fill small holes."""
    if open_radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_radius, open_radius))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    if close_radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_radius, close_radius))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary


def colorize_mask(mask, num_classes=NUM_CLASSES):
    rng = np.random.RandomState(12345)
    palette = rng.randint(0, 256, size=(num_classes, 3), dtype=np.uint8)
    palette[0] = np.array([0, 0, 0], dtype=np.uint8)  # background black
    return palette[mask]


# ------------- STREAMLIT --------------
st.set_page_config(page_title="VisionAI Segmentation (Your Weights)", layout="centered")
st.title("🔍 VisionAI Segmentation — using your checkpoint")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

with st.expander("Advanced (post-processing)"):
    conf_thresh = st.slider("Confidence threshold for NON-background",
                            0.05, 0.95, 0.50, 0.01)
    open_r = st.slider("Morphology OPEN radius (noise removal)", 0, 9, 2, 1)
    close_r = st.slider("Morphology CLOSE radius (hole fill)", 0, 13, 6, 1)

if uploaded is not None:
    pil = Image.open(uploaded).convert("RGB")
    st.subheader("Uploaded Image")
    st.image(pil, use_column_width=True)

    # Build + load
    ensure_checkpoint()
    model = build_model().to(DEVICE).eval()
    load_weights_safely(model, CHECKPOINT_PATH)

    # Inference
    logits = predict_logits(model, pil)         # [C, H, W]
    probs = torch.softmax(logits, dim=0)        # per-class probabilities

    # Background assumed to be class 0 (matches your training target)
    p_bg = probs[0].numpy()
    p_obj = 1.0 - p_bg                            # any class except background
    binary = (p_obj >= conf_thresh).astype(np.uint8) * 255

    # Refine (optional)
    binary = refine_binary(binary, open_radius=open_r, close_radius=close_r)

    # Color mask (keep original colors where binary = 255)
    img_np = np.array(pil)
    color_mask = np.zeros_like(img_np)
    color_mask[binary == 255] = img_np[binary == 255]

    # Show results
    st.subheader("Binary Mask (objects vs background)")
    st.image(binary, use_column_width=True)

    st.download_button(
        "⬇️ Download Binary Mask (PNG)",
        data=BytesIO(cv2.imencode(".png", binary)[1].tobytes()),
        file_name="binary_mask.png",
        mime="image/png",
    )

    st.subheader("Color Mask (Original colors on black)")
    st.image(color_mask, use_column_width=True)

    st.download_button(
        "⬇️ Download Color Mask (PNG)",
        data=BytesIO(cv2.imencode(".png", cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))[1].tobytes()),
        file_name="color_mask.png",
        mime="image/png",
    )

    # (Optional) pseudo-color visualization by predicted class
    with st.expander("Show pseudo-colored full segmentation"):
        pred_classes = torch.argmax(logits, dim=0).numpy().astype(np.int32)
        st.image(colorize_mask(pred_classes), use_column_width=True)















