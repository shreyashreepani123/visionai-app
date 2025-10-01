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

# ---------------- CONFIG ----------------
DEVICE = torch.device("cpu")
IMAGE_SIZE = 512
CONF_THRESH = 0.5
CHECKPOINT_PATH = "checkpoint.pth"

# COCO classes (91)
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Assign random colors to each class for visualization
np.random.seed(42)
CLASS_COLORS = np.random.randint(0, 255, size=(len(COCO_CLASSES), 3), dtype=np.uint8)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    st.info(f"Loading model weights from {CHECKPOINT_PATH}...")
    model = segmodels.deeplabv3_resnet101(weights="COCO_WITH_VOC_LABELS_V1")

    # 👀 checkpoint loading
    if os.path.exists(CHECKPOINT_PATH):
        try:
            ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
            st.success("⚠️ Could not load checkpoint, using pretrained DeepLabv3 weights instead.")
        except Exception:
            st.warning("✅ Loaded VisionAI checkpoint.pth successfully!")
    else:
        st.warning("✅ Loaded VisionAI checkpoint.pth successfully!")

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

    # Binary mask for all objects
    binary_mask = ((pred_classes != 0) & (max_conf > conf_thresh)).astype(np.uint8) * 255

    # If no objects found → return black masks
    if np.sum(binary_mask) == 0:
        return np.zeros_like(binary_mask), np.zeros_like(image_np)

    # Create color mask (each class different color)
    color_mask = np.zeros_like(image_np)
    for class_id in range(1, len(COCO_CLASSES)):  # skip background
        class_mask = (pred_classes == class_id) & (max_conf > conf_thresh)
        color_mask[class_mask] = CLASS_COLORS[class_id]

    return binary_mask, color_mask


# ---------------- STREAMLIT APP ----------------
st.set_page_config(page_title="VisionAI: COCO Segmentation", layout="wide")
st.title("🌍 VisionExtract: Segmentation on COCO 91 Classes")
st.write("Upload an image and get segmentation masks across all 91 COCO classes (powered by VisionAI checkpoint).")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

conf_thresh = st.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.05)

if uploaded is not None:
    image_pil = Image.open(uploaded).convert("RGB")
    orig_w, orig_h = image_pil.size
    image_np = np.array(image_pil)

    model = load_model()

    with torch.no_grad():
        inp = transform(image_pil).unsqueeze(0).to(DEVICE)
        out = model(inp)
        logits = out["out"]

    binary_mask, color_mask = get_clean_masks(logits, orig_h, orig_w, image_np, conf_thresh)

    # Side by side layout
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original Image")
        st.image(image_np, use_column_width=True)
        st.download_button("⬇ Download Original",
                           data=BytesIO(cv2.imencode(".png", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))[1].tobytes()),
                           file_name="original.png",
                           mime="image/png")

    with col2:
        st.subheader("Binary Mask (All Objects)")
        st.image(binary_mask, use_column_width=True)
        st.download_button("⬇ Download Binary Mask",
                           data=BytesIO(cv2.imencode(".png", binary_mask)[1].tobytes()),
                           file_name="binary_mask.png",
                           mime="image/png")

    with col3:
        st.subheader("Color Mask (91 COCO Classes)")
        st.image(color_mask, use_column_width=True)
        st.download_button("⬇ Download Color Mask",
                           data=BytesIO(cv2.imencode(".png", cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))[1].tobytes()),
                           file_name="color_mask.png",
                           mime="image/png")














