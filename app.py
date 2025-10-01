import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models.segmentation as segmodels
from PIL import Image
import streamlit as st
import numpy as np
import cv2
from io import BytesIO
import requests

# ---------------- CONFIG ----------------
DEVICE = torch.device("cpu")
IMAGE_SIZE = 512
CONF_THRESH = 0.5
MODEL_URL = "https://github.com/shreyashreepani123/visionai-app/releases/download/v1.1/checkpoint.pth"
CHECKPOINT_PATH = "checkpoint.pth"


# ---------------- LOAD CHECKPOINT ----------------
def ensure_checkpoint():
    if not os.path.exists(CHECKPOINT_PATH):
        st.info("Downloading VisionAI checkpoint weights...")
        r = requests.get(MODEL_URL, timeout=300)
        r.raise_for_status()
        with open(CHECKPOINT_PATH, "wb") as f:
            f.write(r.content)


@st.cache_resource
def load_model():
    ensure_checkpoint()
    # Load empty DeepLabv3
    model = segmodels.deeplabv3_resnet101(weights=None, num_classes=91)

    # Load checkpoint
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    new_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_sd, strict=False)

    model.to(DEVICE).eval()
    return model


# ---------------- TRANSFORMS ----------------
transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
])


# ---------------- MASK PROCESSING ----------------
def get_clean_masks(logits, orig_h, orig_w, image_np, conf_thresh=0.5):
    probs = torch.softmax(logits, dim=1)
    up = F.interpolate(probs, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    probs_np = up.squeeze(0).cpu().numpy()

    pred_classes = np.argmax(probs_np, axis=0)
    max_conf = np.max(probs_np, axis=0)

    binary_mask = ((pred_classes != 0) & (max_conf > conf_thresh)).astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    min_size = 1000
    new_mask = np.zeros_like(binary_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            new_mask[labels == i] = 255
    binary_mask = new_mask

    color_mask = np.zeros_like(image_np)
    color_mask[binary_mask == 255] = image_np[binary_mask == 255]

    return binary_mask, color_mask


# ---------------- STREAMLIT APP ----------------
st.set_page_config(page_title="VisionAI Segmentation", layout="centered")
st.title("🌍 VisionAI: Ultra Accurate Image Segmentation")
st.write("✔ Model weights loaded from **VisionAI checkpoint**.")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

conf_thresh = st.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.05)

if uploaded is not None:
    image_pil = Image.open(uploaded).convert("RGB")
    orig_w, orig_h = image_pil.size
    image_np = np.array(image_pil)

    st.subheader("Uploaded Image")
    st.image(image_np, use_column_width=True)

    model = load_model()

    with torch.no_grad():
        inp = transform(image_pil).unsqueeze(0).to(DEVICE)
        out = model(inp)
        logits = out["out"]

    binary_mask, color_mask = get_clean_masks(logits, orig_h, orig_w, image_np, conf_thresh)

    st.subheader("Binary Mask (All Objects)")
    st.image(binary_mask, use_column_width=True)
    st.download_button("⬇ Download Binary Mask",
                       data=BytesIO(cv2.imencode(".png", binary_mask)[1].tobytes()),
                       file_name="binary_mask.png",
                       mime="image/png")

    st.subheader("Color Masking (Objects on Black Background)")
    st.image(color_mask, use_column_width=True)
    st.download_button("⬇ Download Color Mask",
                       data=BytesIO(cv2.imencode(".png", cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))[1].tobytes()),
                       file_name="color_mask.png",
                       mime="image/png")





