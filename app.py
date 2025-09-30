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

MODEL_URL = "https://github.com/shreyashreepani123/visionai-app/releases/download/v1.1/checkpoint.pth"
CHECKPOINT_PATH = "checkpoint.pth"
NUM_CLASSES = 91   # your training classes
BACKGROUND_CLASS = 0  # <-- change if your dataset uses another background id (e.g. 255)
DEVICE = torch.device("cpu")
IMAGE_SIZE = 256


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
    state_dict = ckpt.get("model_state", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE).eval()
    return model


transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
])


def decode_segmap(pred, n_classes=NUM_CLASSES):
    """Create a colored segmentation map."""
    label_colors = np.random.randint(0, 255, size=(n_classes, 3), dtype=np.uint8)
    r = np.zeros_like(pred).astype(np.uint8)
    g = np.zeros_like(pred).astype(np.uint8)
    b = np.zeros_like(pred).astype(np.uint8)
    for l in range(0, n_classes):
        idx = pred == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    return np.stack([r, g, b], axis=2)


def postprocess_to_original(logits, orig_h, orig_w):
    up = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    pred = up.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)
    return pred


st.set_page_config(page_title="VisionAI Segmentation", layout="centered")

st.title("🔍 VisionAI Segmentation Demo")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    image_pil = Image.open(uploaded).convert("RGB")
    orig_w, orig_h = image_pil.size
    image_np = np.array(image_pil)

    model = load_model()

    with torch.no_grad():
        inp = transform(image_pil).unsqueeze(0).to(DEVICE)
        out = model(inp)
        logits = out["out"]
        pred_classes = postprocess_to_original(logits, orig_h, orig_w)

    # Binary mask (everything != background)
    binary = (pred_classes != BACKGROUND_CLASS).astype(np.uint8) * 255

    # Colored segmentation map
    color_mask = decode_segmap(pred_classes, n_classes=NUM_CLASSES)

    # Overlay on original
    overlay = cv2.addWeighted(image_np, 0.6, color_mask, 0.4, 0)

    # Show results
    st.subheader("Binary Mask")
    st.image(binary, use_column_width=True)

    st.subheader("Colored Mask (per class)")
    st.image(color_mask, use_column_width=True)

    st.subheader("Overlay with Original Image")
    st.image(overlay, use_column_width=True)






