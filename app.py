import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models.segmentation as segmodels
from PIL import Image
import requests
import streamlit as st
import numpy as np
import cv2
from io import BytesIO

# ---------------- CONFIG ----------------
MODEL_URL = "https://github.com/shreyashreepani123/visionai-app/releases/download/v1.1/checkpoint.pth"
CHECKPOINT_PATH = "checkpoint.pth"
NUM_CLASSES = 91   # Must match training
DEVICE = torch.device("cpu")
IMAGE_SIZE = 256


# ---------------- DOWNLOAD CHECKPOINT ----------------
def ensure_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        return
    st.write("⬇️ Downloading checkpoint…")
    r = requests.get(MODEL_URL, allow_redirects=True, timeout=300)
    with open(CHECKPOINT_PATH, "wb") as f:
        f.write(r.content)
    st.write("✅ Download complete!")


@st.cache_resource
def load_model():
    ensure_checkpoint()
    model = segmodels.deeplabv3_resnet50(weights=None)
    model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state_dict, strict=False)

    model.to(DEVICE).eval()
    return model


# ---------------- TRANSFORMS ----------------
# ⚠️ IMPORTANT: same as training (resize only, no normalization)
transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor()
])


def postprocess_to_original(logits, orig_h, orig_w):
    """Resize logits back to original size, take argmax."""
    up = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    pred = up.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)
    return pred


def clean_mask(mask):
    """Remove noise and fill small holes in mask."""
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


# ---------------- STREAMLIT APP ----------------
st.set_page_config(page_title="VisionAI Segmentation", layout="centered")
st.title("🔍 VisionAI Segmentation Demo (Your Trained Model)")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    image_pil = Image.open(uploaded).convert("RGB")
    orig_w, orig_h = image_pil.size
    image_np = np.array(image_pil)

    # Show uploaded image
    st.subheader("Uploaded Image")
    st.image(image_np, use_column_width=True)

    # Load model
    model = load_model()

    # Inference
    with torch.no_grad():
        inp = transform(image_pil).unsqueeze(0).to(DEVICE)
        out = model(inp)
        logits = out["out"]
        pred_classes = postprocess_to_original(logits, orig_h, orig_w)

    # Background = class 0
    background_index = 0

    # ---------------- BINARY MASK ----------------
    binary = (pred_classes != background_index).astype(np.uint8) * 255
    binary = clean_mask(binary)

    # ---------------- COLOR MASK ----------------
    color_mask = np.zeros_like(image_np)
    mask_area = pred_classes != background_index
    color_mask[mask_area] = image_np[mask_area]

    # ---------------- DISPLAY ----------------
    st.subheader("Binary Mask (Objects vs Background)")
    st.image(binary, use_column_width=True)

    st.download_button(
        "⬇ Download Binary Mask (PNG)",
        data=BytesIO(cv2.imencode(".png", binary)[1].tobytes()),
        file_name="binary_mask.png",
        mime="image/png",
    )

    st.subheader("Color Mask (Original Colors on Black)")
    st.image(color_mask, use_column_width=True)

    st.download_button(
        "⬇ Download Color Mask (PNG)",
        data=BytesIO(cv2.imencode(".png", cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))[1].tobytes()),
        file_name="color_mask.png",
        mime="image/png",
    )













