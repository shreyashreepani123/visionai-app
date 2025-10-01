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

# ---------------- STREAMLIT CONFIG (must be first) ----------------
st.set_page_config(page_title="VisionAI: Image Segmentation", layout="wide")

# ---------------- CUSTOM CSS FOR BEAUTIFUL UI ----------------
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            color: #ffffff;
        }
        .stApp {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            font-family: 'Segoe UI', sans-serif;
        }
        h1 {
            text-align: center;
            color: #00eaff;
            font-size: 48px !important;
            text-shadow: 2px 2px 15px rgba(0, 234, 255, 0.8);
        }
        h2, h3 {
            color: #00ffd5 !important;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(10px);
        }
        .stDownloadButton button {
            background-color: #00eaff;
            color: black;
            border-radius: 10px;
            font-weight: bold;
            border: none;
            padding: 8px 20px;
        }
        .stDownloadButton button:hover {
            background-color: #00bcd4;
            color: white;
        }
        .stSlider > div > div > div > div {
            background: #00eaff;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------- CONFIG ----------------
DEVICE = torch.device("cpu")
IMAGE_SIZE = 512
CONF_THRESH = 0.5
CHECKPOINT_PATH = "checkpoint.pth"

@st.cache_resource
def load_model():
    st.info(f"Loading model weights from {CHECKPOINT_PATH}...") 
    model = segmodels.deeplabv3_resnet101(weights="COCO_WITH_VOC_LABELS_V1")

    if os.path.exists(CHECKPOINT_PATH):
        try:
            ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
            model.load_state_dict(ckpt, strict=False)
            st.success("✅ Loaded VisionAI checkpoint.pth successfully!")
        except Exception:
            st.warning("⚠ Could not load checkpoint, using pretrained DeepLabv3 weights instead.")
    else:
        st.warning("⚠ Checkpoint not found, using pretrained DeepLabv3 weights instead.")
    
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


# ---------------- APP HEADER ----------------
st.markdown("<h1>🌌 VisionExtract: Next-Gen Image Segmentation</h1>", unsafe_allow_html=True)
st.write("<p style='text-align:center; font-size:18px;'>Upload an image and experience <b>cutting-edge AI segmentation</b> with stunning visuals 🚀</p>", unsafe_allow_html=True)

# ---------------- DEMO SECTION ----------------
st.markdown("<h2>✨ Demo Preview</h2>", unsafe_allow_html=True)

# Load a demo image from URL
demo_url = "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg"
demo_img = Image.open(requests.get(demo_url, stream=True).raw).convert("RGB")
demo_np = np.array(demo_img)
orig_w, orig_h = demo_img.size

model = load_model()
with torch.no_grad():
    inp = transform(demo_img).unsqueeze(0).to(DEVICE)
    out = model(inp)
    logits = out["out"]

demo_binary, demo_color = get_clean_masks(logits, orig_h, orig_w, demo_np, conf_thresh=0.5)

demo_col1, demo_col2, demo_col3 = st.columns(3)

with demo_col1:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.image(demo_np, caption="🌌 Demo Input", use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
with demo_col2:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.image(demo_binary, caption="⚡ Binary Mask", use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
with demo_col3:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.image(demo_color, caption="🎨 Color Mask", use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)


# ---------------- UPLOAD + INFERENCE ----------------
uploaded = st.file_uploader("📤 Upload your image (JPG/PNG)", type=["jpg", "jpeg", "png"])
conf_thresh = st.slider("🎚 Confidence Threshold", 0.1, 0.9, 0.5, 0.05)

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

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("📸 Original Image")
        st.image(image_np, use_column_width=True)
        st.download_button("⬇ Download Original",
                           data=BytesIO(cv2.imencode(".png", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))[1].tobytes()),
                           file_name="original.png",
                           mime="image/png")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("⚫ Binary Mask")
        st.image(binary_mask, use_column_width=True)
        st.download_button("⬇ Download Binary Mask",
                           data=BytesIO(cv2.imencode(".png", binary_mask)[1].tobytes()),
                           file_name="binary_mask.png",
                           mime="image/png")
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("🎨 Color Mask")
        st.image(color_mask, use_column_width=True)
        st.download_button("⬇ Download Color Mask",
                           data=BytesIO(cv2.imencode(".png", cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))[1].tobytes()),
                           file_name="color_mask.png",
                           mime="image/png")
        st.markdown("</div>", unsafe_allow_html=True)
