import os
import requests
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50

import streamlit as st

# ---------------- CONFIG ----------------
MODEL_URL = "https://github.com/shreyashreepani123/visionai-app/releases/download/v1.1/checkpoint.pth"
CHECKPOINT_PATH = "checkpoint.pth"
NUM_CLASSES = 91  # must match training
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

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    ensure_checkpoint()
    # must match training
    model = deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    if "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # load weights, ignore mismatches
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        st.warning(f"⚠️ Missing keys in checkpoint: {missing}")
    if unexpected:
        st.warning(f"⚠️ Unexpected keys in checkpoint: {unexpected}")

    model.to(DEVICE).eval()
    return model

# ---------------- Transform (match training!) ----------------
transform = T.Compose([
    T.Resize((256, 256)),  # same as training
    T.ToTensor()           # no normalization (training didn’t use it)
])

def predict(model, pil_img):
    inp = transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(inp)["out"]  # [1, C, H, W]
        pred = torch.argmax(out, dim=1).squeeze(0).cpu().numpy()
    return pred

def colorize_mask(mask, num_classes=NUM_CLASSES):
    rng = np.random.RandomState(12345)
    palette = rng.randint(0, 256, size=(num_classes, 3), dtype=np.uint8)
    palette[0] = np.array([0, 0, 0], dtype=np.uint8)  # background black
    return palette[mask]

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="VisionAI Segmentation", layout="centered")
st.title("🔍 VisionAI Segmentation Demo (Fixed)")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    pil = Image.open(uploaded).convert("RGB")
    st.subheader("Uploaded Image")
    st.image(pil, use_column_width=True)

    model = load_model()
    pred = predict(model, pil)

    # Binary mask
    binary = (pred != 0).astype(np.uint8) * 255
    color_mask = colorize_mask(pred)

    st.subheader("Binary Mask")
    st.image(binary, use_column_width=True)

    st.subheader("Colorized Segmentation")
    st.image(color_mask, use_column_width=True)














