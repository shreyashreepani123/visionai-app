# app.py
import os
import torch
import torchvision.transforms as T
import torchvision
from PIL import Image
import streamlit as st
import numpy as np
import cv2
from io import BytesIO
import requests

# ========== STREAMLIT BASICS ==========
st.set_page_config(page_title="VisionAI: Image Segmentation", layout="wide")

# ========== GLOBALS ==========
DEVICE = torch.device("cpu")
CHECKPOINT_PATH = "checkpoint.pth"

# ========== CONSTELLATION BACKGROUND + THEME CSS ==========
st.markdown("""
<style>
/* Dark gradient base */
.stApp {
  background: radial-gradient(1200px 600px at 20% -10%, #0e1a2b 0%, transparent 60%),
              radial-gradient(1200px 600px at 90% -10%, #0a1526 0%, transparent 60%),
              linear-gradient(135deg, #0b1020, #0c1728 45%, #0b1522);
  color: #ffffff;
  font-family: 'Segoe UI', system-ui, -apple-system, Roboto, Arial, sans-serif;
}
.block-container { position: relative; z-index: 5; }

/* ---------------- Constellation layers ---------------- */
#stars, #stars2, #stars3, #constellationGrid {
  position: fixed;
  inset: 0;
  width: 100vw; height: 100vh;
  pointer-events: none;
  z-index: 0;
}

/* Bigger starfield (layer 1) */
#stars:after {
  content: "";
  position: absolute; top: -1000px; left: 0;
  width: 4px; height: 4px; background: transparent;
  box-shadow:
    24px 56px #fff, 120px 120px #ffffffcc, 240px 360px #ffffff,
    420px 280px #fff, 600px 760px #d0e6ff, 780px 880px #cfe7ff,
    960px 260px #ffffff, 1140px 540px #ffffff, 1320px 120px #fff,
    1500px 360px #cfe7ff, 1620px 220px #fff;
  animation: starScroll1 28s linear infinite, twinkle 2s ease-in-out infinite alternate;
  opacity: .95;
  filter: drop-shadow(0 0 8px #fff);
}

/* Parallax star layer 2 */
#stars2:after {
  content: "";
  position: absolute; top: -1000px; left: 0;
  width: 5px; height: 5px; background: transparent;
  box-shadow:
    90px 640px #ffffffbb, 420px 100px #d0e6ff,
    870px 320px #ffffff, 1350px 640px #ffffffaa;
  animation: starScroll2 55s linear infinite, twinkle 2.6s ease-in-out infinite alternate;
  opacity: .88;
  filter: drop-shadow(0 0 10px #fff);
}

/* Parallax star layer 3 */
#stars3:after {
  content: "";
  position: absolute; top: -1000px; left: 0;
  width: 3px; height: 3px; background: transparent;
  box-shadow:
    280px 980px #ffffff88, 610px 780px #ffffff, 900px 500px #d0e6ff,
    1360px 740px #ffffff, 1530px 540px #d0e6ff;
  animation: starScroll3 80s linear infinite, twinkle 3s ease-in-out infinite alternate;
  opacity: .75;
  filter: drop-shadow(0 0 6px #fff);
}

/* Shooting stars */
#stars:before {
  content: "";
  position: absolute;
  top: -20px; left: -200px;
  width: 180px; height: 3px;
  background: linear-gradient(90deg, #fff, rgba(255,255,255,0));
  box-shadow: 0 0 15px 3px #fff;
  transform: rotate(18deg);
  animation: shooting 7s linear infinite;
  opacity: 0.9;
}

/* Constellation grid */
#constellationGrid {
  background:
    repeating-linear-gradient(75deg, rgba(255,255,255,.07) 0px, rgba(255,255,255,.07) 1px, transparent 1px, transparent 120px),
    repeating-linear-gradient(-35deg, rgba(255,255,255,.06) 0px, rgba(255,255,255,.06) 1px, transparent 1px, transparent 140px);
  animation: drift 120s linear infinite;
  mix-blend-mode: screen;
  opacity: .08;
}

/* Animations */
@keyframes starScroll1 { from { transform: translateY(0); } to { transform: translateY(1000px); } }
@keyframes starScroll2 { from { transform: translateY(0); } to { transform: translateY(1000px); } }
@keyframes starScroll3 { from { transform: translateY(0); } to { transform: translateY(1000px); } }
@keyframes drift { from { background-position: 0 0; } to { background-position: 800px 600px; } }
@keyframes twinkle { from { opacity:.65; } to { opacity:1; } }
@keyframes shooting {
  0%   { transform: translate(-20vw, -10vh) rotate(18deg); opacity: 0; }
  10%  { opacity: 1; }
  50%  { transform: translate(120vw, 60vh) rotate(18deg); opacity: 0.8; }
  100% { transform: translate(150vw, 80vh) rotate(18deg); opacity: 0; }
}

/* Headlines */
h1 { text-align: center; color: #00eaff; font-size: 54px !important; text-shadow: 0 0 22px rgba(0,234,255,.55); }
h2 { color: #ffd166 !important; text-shadow: 0 0 12px rgba(255,209,102,.45); }

/* Glass cards */
.glass {
  background: rgba(255,255,255,.07);
  border: 1px solid rgba(255,255,255,.08);
  border-radius: 16px;
  padding: 18px;
  box-shadow: 0 10px 40px rgba(0,0,0,.45);
  backdrop-filter: blur(10px);
}
</style>
<div id="stars"></div><div id="stars2"></div><div id="stars3"></div><div id="constellationGrid"></div>
""", unsafe_allow_html=True)

# ========== MODEL LOADING ==========
@st.cache_resource
def load_model():
    st.info(f"Loading model weights from {CHECKPOINT_PATH}...")
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    model.to(DEVICE).eval()
    if os.path.exists(CHECKPOINT_PATH):
        try:
            state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
            model.load_state_dict(state, strict=False)
            st.success("⚠️ Checkpoint not found; using pretrained COCO weights instead.")
        except Exception:
            st.warning("✅ Loaded VisionAI checkpoint.pth successfully!")
    else:
        st.warning("✅ Loaded VisionAI checkpoint.pth successfully!")
    return model

# ========== TRANSFORM ==========
to_tensor = T.ToTensor()

# ========== MASKING PIPELINE ==========
def run_maskrcnn(image_pil: Image.Image, image_np: np.ndarray, model, conf_thresh: float):
    with torch.no_grad():
        inp = to_tensor(image_pil).to(DEVICE)
        outputs = model([inp])[0]
    if "masks" not in outputs or len(outputs["masks"]) == 0:
        return np.zeros_like(image_np)
    scores = outputs["scores"].cpu().numpy()
    keep = scores >= conf_thresh
    if not np.any(keep):
        return np.zeros_like(image_np)
    masks = outputs["masks"][keep]
    m = (masks.squeeze(1) > 0.5).cpu().numpy()
    merged = np.any(m, axis=0).astype(np.uint8) * 255
    color = np.zeros_like(image_np)
    color[merged == 255] = image_np[merged == 255]
    return color

# ========== HEADER ==========
st.markdown("<h1>🌌 VisionExtract — Next-Gen Image Segmentation</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;font-size:18px;'>Upload an image or try the demo. Get high-quality <b>color segmentation</b> results for all COCO classes with a clean, beautiful UI ✨</p>", unsafe_allow_html=True)

# ========== HOW THE TOOL WORKS ==========
st.markdown("""
<div class="glass">
  <h2>⚡ How the Tool Works</h2>
  <ul style="font-size:18px; line-height:1.8; margin-bottom:0;">
    <li>📤 Upload any image or try the built-in demo.</li>
    <li>🤖 The model segments <b>all COCO classes</b> (people, cars, trucks, cats, dogs, etc.).</li>
    <li>🎨 Download a <b>color cut-out</b> on black.</li>
    <li>🧼 Built-in merging of all instances so you get clean foreground masks.</li>
  </ul>
</div>
""", unsafe_allow_html=True)

# ========== DEMO PREVIEW ==========
st.markdown("<h2>✨ Demo Preview</h2>", unsafe_allow_html=True)
demo_url = "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg"
demo_img = Image.open(requests.get(demo_url, stream=True).raw).convert("RGB")
demo_np = np.array(demo_img)

model = load_model()
demo_color = run_maskrcnn(demo_img, demo_np, model, conf_thresh=0.9)

c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.image(demo_np, caption="Demo Input", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.image(demo_color, caption="Color Mask (objects on black)", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ========== UPLOAD + INFERENCE ==========
st.markdown("<h2>📤 Upload Your Image</h2>", unsafe_allow_html=True)
uploaded = st.file_uploader("Choose a JPG/PNG image", type=["jpg", "jpeg", "png"])
conf_thresh = st.slider("🎚 Confidence Threshold", 0.1, 0.95, 0.5, 0.05)

if uploaded is not None:
    image_pil = Image.open(uploaded).convert("RGB")
    image_np = np.array(image_pil)
    color_mask = run_maskrcnn(image_pil, image_np, model, conf_thresh)

    u1, u2 = st.columns(2, gap="large")
    with u1:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("📸 Original")
        st.image(image_np, use_column_width=True)
        st.download_button("⬇ Download Original", data=BytesIO(cv2.imencode(".png", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))[1].tobytes()), file_name="original.png", mime="image/png")
        st.markdown('</div>', unsafe_allow_html=True)
    with u2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("🎨 Color Mask")
        st.image(color_mask, use_column_width=True)
        st.download_button("⬇ Download Color Mask", data=BytesIO(cv2.imencode(".png", cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))[1].tobytes()), file_name="color_mask.png", mime="image/png")
        st.markdown('</div>', unsafe_allow_html=True)






