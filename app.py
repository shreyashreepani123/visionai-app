import os
import cv2
import numpy as np
import requests
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50

import streamlit as st

# ---------------- CONFIG ----------------
MODEL_URL = "https://github.com/shreyashreepani123/visionai-app/releases/download/v1.1/checkpoint.pth"
CHECKPOINT_PATH = "checkpoint.pth"
NUM_CLASSES = 91   # must match training
DEVICE = torch.device("cpu")

# ---------------- Download Checkpoint ----------------
def ensure_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        return
    st.write("⬇️ Downloading checkpoint...")
    r = requests.get(MODEL_URL, timeout=300, allow_redirects=True)
    r.raise_for_status()
    with open(CHECKPOINT_PATH, "wb") as f:
        f.write(r.content)
    st.write("✅ Download complete!")

@st.cache_resource
def load_model():
    ensure_checkpoint()
    # Important: must match training setup!
    model = deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE).eval()
    return model

# ---------------- Transform (match training!) ----------------
transform = T.Compose([
    T.Resize((256, 256)),  # same as training
    T.ToTensor()           # no normalization
])

def predict(model, pil_img):
    w, h = pil_img.size
    inp = transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(inp)["out"]  # [1,C,H,W]
        probs = torch.softmax(out, dim=1).squeeze(0).cpu().numpy()
        pred = np.argmax(probs, axis=0).astype(np.uint8)
    return pred

def colorize_mask(mask, num_classes=NUM_CLASSES):
    rng = np.random.RandomState(12345)
    palette = rng.randint(0, 256, size=(num_classes, 3), dtype=np.uint8)
    palette[0] = np.array([0, 0, 0], dtype=np.uint8)  # background black
    mask = np.clip(mask, 0, palette.shape[0]-1)
    return palette[mask]

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="VisionAI Segmentation (Fixed)", layout="centered")
st.title("🔍 VisionAI Segmentation Demo (Correct Training Setup)")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    pil = Image.open(uploaded).convert("RGB")
    img_np = np.array(pil)

    st.subheader("Uploaded Image")
    st.image(img_np, use_column_width=True)

    model = load_model()
    pred = predict(model, pil)

    # Binary mask (foreground vs background)
    binary = (pred != 0).astype(np.uint8) * 255
    color_mask = colorize_mask(pred)

    st.subheader("Binary Mask")
    st.image(binary, use_column_width=True)

    st.subheader("Colorized Segmentation")
    st.image(color_mask, use_column_width=True)













