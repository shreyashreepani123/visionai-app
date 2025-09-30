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

# ⚠️ Update this based on your training
NUM_CLASSES = 2   # Use 2 if foreground/background, 21 for VOC, 91 for COCO
BACKBONE = "resnet50"  # change to "resnet101" if trained with ResNet101

DEVICE = torch.device("cpu")
CONF_THRESH = 0.6


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
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")

    # Detect which key holds weights
    if "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    # Select backbone
    if BACKBONE == "resnet50":
        model = segmodels.deeplabv3_resnet50(weights=None, num_classes=NUM_CLASSES)
    elif BACKBONE == "resnet101":
        model = segmodels.deeplabv3_resnet101(weights=None, num_classes=NUM_CLASSES)
    else:
        raise ValueError("Unsupported backbone. Use resnet50 or resnet101.")

    # Load weights safely
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE).eval()
    return model


# ---------------- TRANSFORMS ----------------
base_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
])


def predict_with_tta(model, image_pil):
    """Test-time augmentation: original + flipped + scaled"""
    w, h = image_pil.size
    img_tensor = base_transform(image_pil).unsqueeze(0).to(DEVICE)

    preds = []

    # Normal
    with torch.no_grad():
        preds.append(model(img_tensor)["out"])

    # Horizontal flip
    with torch.no_grad():
        out = model(torch.flip(img_tensor, dims=[3]))["out"]
        preds.append(torch.flip(out, dims=[3]))

    # Scale (downsize + upsize)
    scaled = T.Resize((h // 2, w // 2))(image_pil)
    scaled_tensor = base_transform(scaled).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(scaled_tensor)["out"]
        out_up = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        preds.append(out_up)

    # Average predictions
    avg_logits = torch.mean(torch.stack(preds), dim=0)
    return avg_logits


def clean_mask(mask):
    """Remove noise + fill gaps"""
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Remove very small blobs
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    new_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 500:
            new_mask[labels == i] = 255
    return new_mask


# ---------------- STREAMLIT APP ----------------
st.set_page_config(page_title="VisionAI Segmentation", layout="centered")
st.title("🔍 VisionAI Segmentation Demo (Ultra Accurate)")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    image_pil = Image.open(uploaded).convert("RGB")
    image_np = np.array(image_pil)
    h, w = image_np.shape[:2]

    # Show uploaded image
    st.subheader("Uploaded Image")
    st.image(image_np, use_column_width=True)

    # Load model
    model = load_model()

    # Inference with TTA
    logits = predict_with_tta(model, image_pil)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred_classes = np.argmax(probs, axis=0)
    max_conf = np.max(probs, axis=0)

    # Binary mask: use confidence
    binary = ((pred_classes != 0) & (max_conf > CONF_THRESH)).astype(np.uint8) * 255
    binary = clean_mask(binary)

    # Color mask
    color_mask = np.zeros_like(image_np)
    color_mask[binary == 255] = image_np[binary == 255]

    # ---------------- DISPLAY ----------------
    st.subheader("Binary Mask (Refined)")
    st.image(binary, use_column_width=True)

    st.download_button(
        "⬇️ Download Binary Mask (PNG)",
        data=BytesIO(cv2.imencode(".png", binary)[1].tobytes()),
        file_name="binary_mask.png",
        mime="image/png",
    )

    st.subheader("Color Mask (Refined Objects)")
    st.image(color_mask, use_column_width=True)

    st.download_button(
        "⬇️ Download Color Mask (PNG)",
        data=BytesIO(cv2.imencode(".png", cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))[1].tobytes()),
        file_name="color_mask.png",
        mime="image/png",
    )



