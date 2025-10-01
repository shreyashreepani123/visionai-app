# app.py
# VisionAI – Best Binary Mask Extraction
# Converts COCO multi-class predictions into clean binary mask (object=white, background=black)

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
TARGET_CLASS = 1  # COCO class id for 'person' (purple in your mask)

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

def make_binary_mask(logits, target_class=1):
    pred = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
    binary = (pred == target_class).astype(np.uint8) * 255

    # Post-process (remove noise, smooth edges)
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
st.set_page_config(page_title="VisionAI Best Mask", layout="centered")
st.title("VisionAI Best Binary Masking")
st.caption("Extracts target objects as binary mask (white = object, black = background).")

uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded:
    pil = Image.open(uploaded).convert("RGB")
    st.image(pil, caption="Uploaded Image", use_column_width=True)

    model = load_model()
    logits = forward_image(model, pil)

    binary = make_binary_mask(logits, TARGET_CLASS)

    st.subheader("Binary Mask (object = white, background = black)")
    st.image(binary, clamp=True, channels="GRAY")

    st.download_button("⬇ Download Mask (PNG)",
        data=to_png_bytes(binary),
        file_name="binary_mask.png",
        mime="image/png")
else:
    st.info("Upload an image to start.")










