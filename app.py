import os
import torch
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


# ---------------- DOWNLOAD MODEL ----------------
def ensure_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        return
    r = requests.get(MODEL_URL, allow_redirects=True, timeout=300)
    with open(CHECKPOINT_PATH, "wb") as f:
        f.write(r.content)


@st.cache_resource
def load_model():
    ensure_checkpoint()
    model = segmodels.deeplabv3_resnet50(weights=None, num_classes=NUM_CLASSES)
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE).eval()
    return model


# ---------------- TRANSFORMS ----------------
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
])


def predict(model, pil_img):
    """Single inference without filtering too much"""
    w, h = pil_img.size
    inp = transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(inp)["out"]
    out_up = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
    probs = torch.softmax(out_up, dim=1).squeeze(0).cpu().numpy()
    pred_classes = np.argmax(probs, axis=0)
    return pred_classes, probs


def clean_mask(mask):
    """Refine mask with morphology"""
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


# ---------------- STREAMLIT APP ----------------
st.set_page_config(page_title="VisionAI Segmentation", layout="centered")
st.title("🔍 VisionAI Segmentation Demo (Fixed)")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    image_pil = Image.open(uploaded).convert("RGB")
    image_np = np.array(image_pil)

    # Show uploaded image
    st.subheader("Uploaded Image")
    st.image(image_np, use_column_width=True)

    # Load model
    model = load_model()

    # Inference
    pred_classes, probs = predict(model, image_pil)

    # Detect background = most common class
    background_index = int(np.bincount(pred_classes.flatten()).argmax())

    # ---------------- BINARY MASK ----------------
    binary = (pred_classes != background_index).astype(np.uint8) * 255
    binary = clean_mask(binary)

    # ---------------- COLOR MASK ----------------
    color_mask = np.zeros_like(image_np)
    mask_area = (pred_classes != background_index)
    color_mask[mask_area] = image_np[mask_area]

    # ---------------- DISPLAY ----------------
    st.subheader("Binary Mask (Fixed)")
    st.image(binary, use_column_width=True)

    st.download_button(
        "⬇ Download Binary Mask (PNG)",
        data=BytesIO(cv2.imencode(".png", binary)[1].tobytes()),
        file_name="binary_mask.png",
        mime="image/png",
    )

    st.subheader("Color Mask (Fixed)")
    st.image(color_mask, use_column_width=True)

    st.download_button(
        "⬇ Download Color Mask (PNG)",
        data=BytesIO(cv2.imencode(".png", cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))[1].tobytes()),
        file_name="color_mask.png",
        mime="image/png",
    )



