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
NUM_CLASSES = 91   # COCO dataset
DEVICE = torch.device("cpu")
IMAGE_SIZE = 256


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
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
])


# ---------------- POSTPROCESSING ----------------
def get_clean_masks(logits, orig_h, orig_w, image_np, conf_thresh=0.5):
    # Get class probabilities
    probs = torch.softmax(logits, dim=1)  # [1, C, H, W]
    up = F.interpolate(probs, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    probs_np = up.squeeze(0).cpu().numpy()  # [C, H, W]

    # Predicted class = highest probability
    pred_classes = np.argmax(probs_np, axis=0)
    max_conf = np.max(probs_np, axis=0)

    # ---------------- BINARY MASK ----------------
    # All non-background classes with high confidence
    binary_mask = ((pred_classes != 0) & (max_conf > conf_thresh)).astype(np.uint8) * 255

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # Keep largest connected components (remove tiny blobs)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    min_size = 500  # pixels to keep
    new_mask = np.zeros_like(binary_mask)
    for i in range(1, num_labels):  # skip background
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            new_mask[labels == i] = 255
    binary_mask = new_mask

    # ---------------- COLOR MASK ----------------
    color_mask = np.zeros_like(image_np)
    color_mask[binary_mask == 255] = image_np[binary_mask == 255]

    return binary_mask, color_mask


# ---------------- STREAMLIT APP ----------------
st.set_page_config(page_title="VisionAI Segmentation", layout="centered")
st.title("🔍 VisionAI Segmentation Demo")
st.write("Upload an image to see binary and color masking results. Model runs on CPU.")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Confidence threshold slider
conf_thresh = st.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.05)

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

    # Get cleaned masks
    binary_mask, color_mask = get_clean_masks(logits, orig_h, orig_w, image_np, conf_thresh)

    # ---------------- DISPLAY RESULTS ----------------
    st.subheader("Binary Mask (All Objects)")
    st.image(binary_mask, use_column_width=True)

    st.download_button(
        "⬇️ Download Binary Mask (PNG)",
        data=BytesIO(cv2.imencode(".png", binary_mask)[1].tobytes()),
        file_name="binary_mask.png",
        mime="image/png",
    )

    st.subheader("Color Masking (Objects on Black Background)")
    st.image(color_mask, use_column_width=True)

    st.download_button(
        "⬇️ Download Color Mask (PNG)",
        data=BytesIO(cv2.imencode(".png", cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))[1].tobytes()),
        file_name="color_mask.png",
        mime="image/png",
    )










