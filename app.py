# app.py
# VisionAI – COCO-style semantic segmentation demo (91 classes)
# Loads your checkpoint.pth (state_dict under "model_state") and runs DeepLabV3-ResNet50.
# Outputs: Binary mask, color cut-out, and multi-class colored mask, each with a download button.

import os
import io
import time
import requests
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models.segmentation as segmodels

import streamlit as st

# ---------------------------- CONFIG ----------------------------
CHECKPOINT_PATH = "checkpoint.pth"   # Must exist or will be downloaded from CHECKPOINT_URL
CHECKPOINT_URL  = "https://github.com/shreyashreepani123/visionai-app/releases/download/v1.1/checkpoint.pth"
DEVICE = torch.device("cpu")         # force CPU for Streamlit Cloud stability
IMAGE_SIZE = 256                     # your training resize
MIN_COMPONENT = 800                  # remove tiny blobs in binary mask
TITLE = "VisionAI Segmentation (COCO 91)"

# A stable 91-color palette: background=black; a few COCO-like overrides; rest deterministic.
def build_palette_91():
    rng = np.random.RandomState(0)
    pal = rng.randint(0, 256, size=(91, 3), dtype=np.uint8)
    pal[0] = np.array([0,0,0], np.uint8)           # background
    # Helpful overrides for common classes to be recognizable
    # (indexes follow training: 1=person, 2=bicycle, 3=car, etc.)
    if pal.shape[0] > 4:
        pal[1] = np.array([220, 20, 60], np.uint8)  # person -> red
        pal[2] = np.array([119, 11, 32], np.uint8)  # bicycle -> wine
        pal[3] = np.array([0, 0, 142], np.uint8)    # car -> blue
        pal[4] = np.array([0, 0, 230], np.uint8)    # motorbike -> bright blue
    return pal

PALETTE_91 = build_palette_91()

# ----------------------- CHECKPOINT HANDLING --------------------
def ensure_checkpoint():
    """Make sure checkpoint.pth exists; if not, try to download it."""
    if os.path.exists(CHECKPOINT_PATH) and os.path.getsize(CHECKPOINT_PATH) > 0:
        return
    st.info("Downloading checkpoint…")
    try:
        r = requests.get(CHECKPOINT_URL, timeout=300, allow_redirects=True)
        r.raise_for_status()
        with open(CHECKPOINT_PATH, "wb") as f:
            f.write(r.content)
    except Exception as e:
        st.error(
            f"Could not find or download the checkpoint at '{CHECKPOINT_PATH}'. "
            f"Please place your trained file there or update CHECKPOINT_URL.\n\nError: {e}"
        )
        st.stop()

def detect_num_classes_from_state(state_dict) -> int:
    """Infer number of classes from the classifier weight if available; default to 91."""
    for k in ("classifier.4.weight", "module.classifier.4.weight"):
        if k in state_dict:
            return int(state_dict[k].shape[0])
    return 91

def build_model(num_classes: int, aux: bool = False):
    model = segmodels.deeplabv3_resnet50(weights=None, aux_loss=aux)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    if aux and getattr(model, "aux_classifier", None) is not None:
        model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model

@st.cache_resource(show_spinner=False)
def load_model():
    ensure_checkpoint()
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))

    num_classes = detect_num_classes_from_state(state)
    model = build_model(num_classes=num_classes, aux=False)

    # Some trainers save aux keys; ignore those safely.
    filtered = {k: v for k, v in state.items() if not k.startswith("aux_classifier")}
    load_report = model.load_state_dict(filtered, strict=False)
    # Optional: sanity checks without spamming UI
    # st.write("Missing keys:", load_report.missing_keys)
    # st.write("Unexpected keys:", load_report.unexpected_keys)

    model.to(DEVICE).eval()
    return model, num_classes

# --------------------------- TRANSFORMS -------------------------
transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# --------------------------- INFERENCE --------------------------
def forward_image(model, pil_img: Image.Image):
    """Run model and return logits at original size."""
    W, H = pil_img.size
    x = transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x)["out"]  # (1, C, h, w)
    up = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
    return up  # (1, C, H, W)

def to_masks(logits, orig_rgb: np.ndarray, palette: np.ndarray, min_component: int = 800):
    """Convert logits to (binary mask uint8, cutout RGB, colorized class RGB)."""
    H, W = logits.shape[-2], logits.shape[-1]
    probs = torch.softmax(logits, dim=1)
    pred = probs.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)  # (H,W)

    # ---- Binary (objects vs background) ----
    binary = (pred != 0).astype(np.uint8) * 255

    # Clean tiny blobs
    if binary.any():
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        keep = np.zeros_like(binary)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_component:
                keep[labels == i] = 255
        if keep.sum() > 0:
            binary = keep

    # Morphology (close then open)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # ---- Cut-out (original colors on black) ----
    cutout = np.zeros_like(orig_rgb)
    cutout[binary == 255] = orig_rgb[binary == 255]

    # ---- Multi-class colorized ----
    colorized = np.zeros((H, W, 3), dtype=np.uint8)
    uniq = np.unique(pred)
    for cid in uniq:
        if cid < len(palette):
            colorized[pred == cid] = palette[cid]
        else:
            # safety – should not happen for 91 classes
            colorized[pred == cid] = np.array([255, 255, 255], np.uint8)

    return binary, cutout, colorized

def rgba_cutout(orig_rgb: np.ndarray, binary_mask: np.ndarray):
    """Create transparent PNG (RGBA) using the binary mask as alpha."""
    alpha = (binary_mask > 0).astype(np.uint8) * 255
    rgba = np.dstack([orig_rgb, alpha]).astype(np.uint8)
    return rgba

def to_png_bytes(arr: np.ndarray, is_bgr=False) -> bytes:
    if is_bgr:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    if arr.ndim == 2:
        # grayscale
        pil = Image.fromarray(arr)
    else:
        pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()

# --------------------------- UI -------------------------------
st.set_page_config(page_title=TITLE, layout="centered")
st.title(TITLE)

st.caption(
    "Loads your **checkpoint.pth** (COCO-style 91 classes). "
    "Outputs: binary mask, color cut-out, and multi-class colored mask."
)
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
min_area = st.slider("Min object area (px) to keep in binary mask", 0, 5000, MIN_COMPONENT, 50)

if uploaded is not None:
    pil = Image.open(uploaded).convert("RGB")
    orig_rgb = np.array(pil)

    st.subheader("Input")
    st.image(orig_rgb, use_column_width=True)

    model, nc = load_model()

    t0 = time.time()
    logits = forward_image(model, pil)
    binary, cutout, colorized = to_masks(logits, orig_rgb, PALETTE_91, min_component=min_area)
    dt = time.time() - t0

    st.write(f"Inference time: {dt:.2f}s (CPU)")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Binary Mask (objects vs background)")
        st.image(binary, clamp=True, channels="GRAY", use_column_width=True)
        st.download_button("⬇ Download Binary Mask (PNG)",
            data=to_png_bytes(binary), file_name="binary_mask.png", mime="image/png")
    with col2:
        st.subheader("Color Cut-out (original colors)")
        st.image(cutout, use_column_width=True)
        st.download_button("⬇ Download Cut-out (PNG)",
            data=to_png_bytes(cutout), file_name="color_cutout.png", mime="image/png")

    st.subheader("Multi-class Colored Mask")
    st.image(colorized, use_column_width=True)
    st.download_button("⬇ Download Multi-class Mask (PNG)",
        data=to_png_bytes(colorized), file_name="class_mask.png", mime="image/png")

    # Transparent PNG export
    st.subheader("Transparent Cut-out (RGBA)")
    rgba = rgba_cutout(orig_rgb, binary)
    st.image(rgba, use_column_width=True)
    st.download_button("⬇ Download Transparent PNG",
        data=to_png_bytes(rgba), file_name="cutout_transparent.png", mime="image/png")
else:
    st.info("Upload a JPG/PNG image to begin.")









