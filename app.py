# app.py
# VisionAI – Human Segmentation (humans white, background black)

import os
import requests
import numpy as np
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

def build_model(num_classes=91):  # COCO has 91 classes
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

# ---------------- STREAMLIT APP ----------------
st.set_page_config(page_title="VisionAI Human Segmentation", layout="centered")
st.title("VisionAI – Human Segmentation")
st.caption("Humans will appear in white, background in black.")

uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded:
    pil = Image.open(uploaded).convert("RGB")
    st.image(pil, caption="Uploaded Image", use_column_width=True)

    model = load_model()
    logits = forward_image(model, pil)

    # Prediction mask
    pred = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

    # ⚠️ For COCO, 'person' class ID = 1
    target_class_id = 1

    # Create binary mask: 1 for humans, 0 for everything else
    binary_mask = np.where(pred == target_class_id, 1, 0).astype(np.uint8)

    # ✅ Invert mask so humans = white, background = black
    binary_mask = 1 - binary_mask

    # Convert to image
    mask_img = Image.fromarray((binary_mask * 255).astype(np.uint8))
    st.image(mask_img, caption="Final Binary Mask", use_column_width=True)

else:
    st.info("Upload an image to start.")












