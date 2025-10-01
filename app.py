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

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(page_title="🌌 VisionExtract AI", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
    <style>
        /* Cosmic Gradient */
        .stApp {
            background: radial-gradient(circle at top left, #0f2027, #203a43, #2c5364);
            color: #ffffff;
            font-family: 'Segoe UI', sans-serif;
        }
        h1 {
            text-align: center;
            color: #00eaff;
            font-size: 54px !important;
            text-shadow: 0px 0px 25px rgba(0,234,255,0.9);
            margin-bottom: 15px;
        }
        h2 {
            color: #ffd166 !important;
            text-shadow: 0px 0px 12px rgba(255,209,102,0.9);
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.08);
            border-radius: 15px;
            padding: 18px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(12px);
        }
        .stButton>button {
            background: linear-gradient(90deg, #00eaff, #00bcd4);
            color: black;
            font-weight: bold;
            border-radius: 10px;
            border: none;
            padding: 12px 24px;
            font-size: 18px;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #ff9800, #ff5722);
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------- CONFIG ----------------
DEVICE = torch.device("cpu")
IMAGE_SIZE = 512
CHECKPOINT_PATH = "checkpoint.pth"

@st.cache_resource
def load_model():
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

transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
])

def get_clean_masks(logits, orig_h, orig_w, image_np, conf_thresh=0.5):
    probs = torch.softmax(logits, dim=1)
    up = torch.nn.functional.interpolate(probs, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    probs_np = up.squeeze(0).cpu().numpy()

    pred_classes = np.argmax(probs_np, axis=0)
    max_conf = np.max(probs_np, axis=0)
    binary_mask = ((pred_classes != 0) & (max_conf > conf_thresh)).astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    color_mask = np.zeros_like(image_np)
    color_mask[binary_mask == 255] = image_np[binary_mask == 255]
    return binary_mask, color_mask

# ---------------- APP FLOW ----------------
st.markdown("<h1>🌌 VisionExtract: Next-Gen Image Segmentation</h1>", unsafe_allow_html=True)

# Sidebar navigation
page = st.sidebar.radio("🔹 Navigate", ["🏠 Home", "✨ Demo", "ℹ️ How it Works", "📊 Results", "📤 Upload Your Own"])

# ---------------- HOME ----------------
if page == "🏠 Home":
    st.markdown("<h2>Welcome to VisionExtract AI</h2>", unsafe_allow_html=True)
    st.write("🚀 This tool brings **cutting-edge segmentation** trained on the **COCO dataset** to your browser with stunning UI.")

# ---------------- DEMO ----------------
elif page == "✨ Demo":
    st.markdown("<h2>✨ Demo Image</h2>", unsafe_allow_html=True)
    demo_url = "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg"
    demo_img = Image.open(requests.get(demo_url, stream=True).raw).convert("RGB")
    st.image(demo_img, caption="Demo Input Image", use_column_width=True)

# ---------------- HOW IT WORKS ----------------
elif page == "ℹ️ How it Works":
    st.markdown("<h2>⚡ How the Tool Works</h2>", unsafe_allow_html=True)
    st.markdown("""
    - Upload any image or try the demo  
    - Automatically segment **all COCO classes** with AI precision  
    - Remove or replace backgrounds easily  
    - Highlight edges with stylish overlays  
    """)
    st.info("Click **Results** in the sidebar to see segmentation in action!")

# ---------------- RESULTS ----------------
elif page == "📊 Results":
    st.markdown("<h2>📊 AI Segmentation Results</h2>", unsafe_allow_html=True)

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

    col1, col2, col3 = st.columns(3)
    with col1: st.image(demo_np, caption="Original Image", use_column_width=True)
    with col2: st.image(demo_binary, caption="Binary Mask", use_column_width=True)
    with col3: st.image(demo_color, caption="Color Mask", use_column_width=True)

# ---------------- UPLOAD ----------------
elif page == "📤 Upload Your Own":
    st.markdown("<h2>📤 Upload Your Own Image</h2>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload JPG/PNG", type=["jpg", "jpeg", "png"])
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
            st.subheader("📸 Original Image")
            st.image(image_np, use_column_width=True)
        with col2:
            st.subheader("⚫ Binary Mask")
            st.image(binary_mask, use_column_width=True)
        with col3:
            st.subheader("🎨 Color Mask")
            st.image(color_mask, use_column_width=True)

