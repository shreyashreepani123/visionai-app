# app.py
# VisionAI – Auto Binary Mask Extraction
# Automatically picks the main object (largest non-background region) and produces a binary mask.

import os
import io
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

# ---------------- CONFIG ----------------
CHECKPOINT_PATH = "checkpoint.pth"
CHECKPOINT_URL  = "https://github.com/shreyashreepani123/visionai-app/releases/download/v1.1/checkpoint.pth"
DEVICE = torch.device("cpu")
IMAGE_SIZE = 256

# ---------------- CHECKPOINT ----------------
def ensure_checkpoint():
    if os.path.exists(CHECKPOINT_PATH) and os.path.getsize(CHECKPOINT_PATH) > 0:
        return
    st.info("Downloading checkpoint…")
    r = requests.get(CHECKPOINT_URL, timeout=300)
    r.raise_for_status()
    with open(CHECKPOINT_PATH, "wb") as f:
        f.write(r.content)

def build_model(num_classes=91):
    model = segmodels.deeplabv3_resnet50(weights=None, aux_loss=False)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model

@st.cache_resource
def load_model():
    ensure_checkpoint()
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    state = ckpt.get("model_state", ckpt)
    model = build_model(num_classes=91)
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    return model

# ---------------- TRANSFORM ----------------
transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
])

# ---------------- INFERENCE ----------------
def forward_image(model, pil_img):
    W, H = pil_img.size
    x = transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x)["out"]
    up = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
    return up

def auto_binary_mask(logits):
    """Pick largest non-background class → foreground=white, rest=black"""
    pred = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

    # Ignore background (id=0), find largest other class
    uniq, counts = np.unique(pred, return_counts=True)
    fg_classes = [(u, c) for u, c in zip(uniq, counts) if u != 0]
    if not fg_classes:
        return np.zeros_like(pred, dtype=np.uint8)

    target_class = max(fg_classes, key=lambda x: x[1])[0]
    binary = (pred == target_class).astype(np.uint8) * 255

    # Clean mask with morphology
    kernel = np.ones((3,3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return binary

def to_png_bytes(arr):
    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()

# ---------------- STREAMLIT APP ----------------
st.set_page_config(page_title="VisionAI Auto Binary Mask", layout="centered")
st.title("VisionAI – Best Binary Mask")
st.caption("Automatically selects the main object class and produces binary mask (white=object, black=background).")

uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded:
    pil = Image.open(uploaded).convert("RGB")
    st.image(pil, caption="Uploaded Image", use_column_width=True)

    model = load_model()
    logits = forward_image(model, pil)

    binary = auto_binary_mask(logits)

    st.subheader("Final Binary Mask")
    st.image(binary, clamp=True, channels="GRAY")

    st.download_button("⬇ Download Mask (PNG)",
        data=to_png_bytes(binary),
        file_name="binary_mask.png",
        mime="image/png")
else:
    st.info("Upload an image to start.")











