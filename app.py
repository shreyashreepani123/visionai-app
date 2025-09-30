import os
import cv2
import numpy as np
import requests
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models.segmentation as segmodels

import streamlit as st

# ---------------- CONFIG ----------------
MODEL_URL = "https://github.com/shreyashreepani123/visionai-app/releases/download/v1.1/checkpoint.pth"
CHECKPOINT_PATH = "checkpoint.pth"
NUM_CLASSES = 91           # <- adjust this if your training had fewer classes
DEVICE = torch.device("cpu")


# ---------------- Utils ----------------
def ensure_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        return
    st.write("⬇️ Downloading checkpoint from GitHub release...")
    r = requests.get(MODEL_URL, timeout=300, allow_redirects=True)
    r.raise_for_status()
    with open(CHECKPOINT_PATH, "wb") as f:
        f.write(r.content)
    st.write("✅ Download complete!")


@st.cache_resource
def load_model():
    ensure_checkpoint()
    model = segmodels.deeplabv3_resnet50(weights=None, num_classes=NUM_CLASSES)
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE).eval()
    return model


# torchvision transform
_t = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
])


def softmax_probs(model, pil_img: Image.Image) -> np.ndarray:
    """Return per-class probabilities upsampled to original HxW."""
    w, h = pil_img.size
    x = _t(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x)["out"]                      # [1,C,h',w']
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        probs = torch.softmax(out, dim=1).squeeze(0).cpu().numpy()  # [C,H,W]
    return probs


def visualize_raw_predictions(probs: np.ndarray) -> np.ndarray:
    """Color map for argmax predictions (for debugging)."""
    pred_classes = np.argmax(probs, axis=0).astype(np.uint8)
    color_map = (pred_classes * 15) % 255
    return color_map


def safe_refine(image_rgb: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """
    Safe refinement:
      - Keep low threshold foreground to avoid empty results.
      - Falls back to argmax if mask is empty.
    """
    H, W, _ = image_rgb.shape
    arg = probs.argmax(0)
    bg_idx = int(np.bincount(arg.flatten()).argmax())

    maxp = probs.max(0)
    fg = (arg != bg_idx) & (maxp > 0.2)

    if fg.sum() < 50:   # if almost empty, fallback
        fg = (arg != bg_idx)

    mask = (fg.astype(np.uint8) * 255)
    return mask


def color_mask_from_binary(image_rgb: np.ndarray, binary_255: np.ndarray) -> np.ndarray:
    out = image_rgb.copy()
    out[binary_255 == 0] = 0
    return out


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="VisionAI Segmentation (Safe Mode)", layout="centered")
st.title("🔍 VisionAI Segmentation Demo (with Safe Refinement)")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    pil = Image.open(uploaded).convert("RGB")
    img = np.array(pil)

    st.subheader("Uploaded Image")
    st.image(img, use_column_width=True)

    model = load_model()
    probs = softmax_probs(model, pil)

    # --- Raw model output ---
    st.subheader("Raw Prediction (Argmax Classes)")
    raw_vis = visualize_raw_predictions(probs)
    st.image(raw_vis, caption="Raw Argmax (each color = a class)", use_column_width=True)

    # --- Safe refinement ---
    binary = safe_refine(img, probs)
    color_mask = color_mask_from_binary(img, binary)

    st.subheader("Binary Mask (Safe Refined)")
    st.image(binary, use_column_width=True)

    st.download_button(
        "⬇️ Download Binary Mask (PNG)",
        data=cv2.imencode(".png", binary)[1].tobytes(),
        file_name="binary_mask.png",
        mime="image/png",
    )

    st.subheader("Color Mask (Objects on Black Background)")
    st.image(color_mask, use_column_width=True)

    st.download_button(
        "⬇️ Download Color Mask (PNG)",
        data=cv2.imencode(".png", cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))[1].tobytes(),
        file_name="color_mask.png",
        mime="image/png",
    )










