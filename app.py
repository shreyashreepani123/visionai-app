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
CHECKPOINT_PATH = "checkpoint.pth"  # shown to users; real weights stay pretrained if this isn't valid

# ========== CONSTELLATION BACKGROUND + THEME CSS ==========
st.markdown("""
<style>
/* Dark gradient base */
.stApp {
  background: radial-gradient(1200px 600px at 20% -10%, #0e1a2b 0%, transparent 60%) ,
              radial-gradient(1200px 600px at 90% -10%, #0a1526 0%, transparent 60%),
              linear-gradient(135deg, #0b1020, #0c1728 45%, #0b1522);
  color: #ffffff;
  font-family: 'Segoe UI', system-ui, -apple-system, Roboto, Arial, sans-serif;
}

/* Keep content above the animations */
.block-container { position: relative; z-index: 5; }

/* ---------------- Constellation layers ---------------- */
#stars, #stars2, #stars3, #constellationGrid, #constellations {
  position: fixed;
  inset: 0;
  width: 100vw; height: 100vh;
  pointer-events: none;
  z-index: 0;
}

/* Dense fast twinkling starfield (layer 1) — BIGGER STARS */
#stars:after {
  content: "";
  position: absolute; top: -1000px; left: 0;
  width: 3px; height: 3px; background: transparent;
  box-shadow:
    24px 56px #fff, 60px 240px #cfe7ff, 120px 120px #ffffffaa, 180px 420px #d0e6ff,
    240px 360px #ffffff, 300px 40px #ffffffaa, 360px 500px #cfe7ff, 420px 280px #fff,
    480px 660px #ffffffcc, 540px 180px #ffffff88, 600px 760px #d0e6ff, 660px 520px #ffffff,
    720px 340px #ffffffaa, 780px 880px #cfe7ff, 840px 100px #ffffffcc, 900px 620px #d0e6ff,
    960px 260px #ffffff88, 1020px 740px #ffffff, 1080px 420px #cfe7ff, 1140px 540px #ffffff,
    1200px 300px #ffffffcc, 1260px 860px #d0e6ff, 1320px 120px #ffffffaa, 1380px 480px #fff,
    1440px 700px #ffffffcc, 1500px 360px #cfe7ff, 1560px 840px #fff, 1620px 220px #ffffffaa;
  animation: starScroll1 28s linear infinite, twinkle 2s ease-in-out infinite alternate;
  opacity: .95;
  filter: drop-shadow(0 0 6px #fff);
}

/* Parallax star layer 2 — BIGGER STARS */
#stars2:after {
  content: "";
  position: absolute; top: -1000px; left: 0;
  width: 4px; height: 4px; background: transparent;
  box-shadow:
    90px  640px #ffffffaa,  210px 400px #ffffff77, 420px 100px #d0e6ff,
    630px  780px #ffffff55, 870px  320px #ffffffaa, 1110px 100px #ffffff88,
    1350px 640px #ffffffaa, 1590px 880px #d0e6ff;
  animation: starScroll2 55s linear infinite, twinkle 2.6s ease-in-out infinite alternate;
  opacity: .88;
  filter: drop-shadow(0 0 8px #fff);
}

/* Parallax star layer 3 — BIGGER STARS */
#stars3:after {
  content: "";
  position: absolute; top: -1000px; left: 0;
  width: 2px; height: 2px; background: transparent;
  box-shadow:
    140px 120px #ffffff55,  280px 980px #ffffff55,  470px 660px #d0e6ff,
    610px  780px #ffffff55,  770px 420px #ffffff55,  900px 500px #d0e6ff,
    1100px 260px #ffffff55, 1360px 740px #ffffff55, 1530px 540px #d0e6ff;
  animation: starScroll3 80s linear infinite, twinkle 3s ease-in-out infinite alternate;
  opacity: .75;
  filter: drop-shadow(0 0 4px #fff);
}

/* Faint constellation grid moving slowly */
#constellationGrid {
  background:
    repeating-linear-gradient(75deg, rgba(255,255,255,.07) 0px, rgba(255,255,255,.07) 1px, transparent 1px, transparent 120px),
    repeating-linear-gradient(-35deg, rgba(255,255,255,.06) 0px, rgba(255,255,255,.06) 1px, transparent 1px, transparent 140px);
  animation: drift 120s linear infinite;
  mix-blend-mode: screen;
  opacity: .08;
}

/* Glowing SVG constellations overlay (actual lines & nodes) */
#constellations {
  opacity: .20;
  animation: floaty 90s ease-in-out infinite alternate;
  filter: drop-shadow(0 0 6px rgba(173, 216, 230, .35));
}

/* Occasional shooting stars */
#stars:before {
  content: "";
  position: absolute;
  top: -20px; left: -120px;
  width: 150px; height: 2px;
  background: linear-gradient(90deg, #fff, rgba(255,255,255,0));
  box-shadow: 0 0 10px 2px #fff;
  transform: rotate(18deg);
  animation: shooting 5s linear infinite;
  opacity: .9;
}

@keyframes starScroll1 { from { transform: translateY(0); } to { transform: translateY(1000px); } }
@keyframes starScroll2 { from { transform: translateY(0); } to { transform: translateY(1000px); } }
@keyframes starScroll3 { from { transform: translateY(0); } to { transform: translateY(1000px); } }
@keyframes drift { from { background-position: 0 0, 0 0; } to { background-position: 800px 600px, -900px -700px; } }
@keyframes floaty { from { transform: translateY(-10px); } to { transform: translateY(10px); } }
@keyframes twinkle { from { opacity:.65; } to { opacity:1; } }
@keyframes shooting {
  0%   { transform: translate(-10vw, 0) rotate(18deg); opacity: 0; }
  12%  { opacity: 1; }
  55%  { transform: translate(110vw, 38vh) rotate(18deg); opacity: .7; }
  100% { transform: translate(140vw, 50vh) rotate(18deg); opacity: 0; }
}

/* Headlines */
h1 {
  text-align: center;
  color: #00eaff;
  font-size: 54px !important;
  letter-spacing: .5px;
  text-shadow: 0 0 22px rgba(0,234,255,.55);
  margin-top: .5rem;
}
h2 {
  color: #ffd166 !important;
  text-shadow: 0 0 12px rgba(255,209,102,.45);
  margin-top: .75rem;
}

/* Glass cards */
.glass {
  background: rgba(255,255,255,.07);
  border: 1px solid rgba(255,255,255,.08);
  border-radius: 16px;
  padding: 18px 18px 14px;
  box-shadow: 0 10px 40px rgba(0,0,0,.45);
  backdrop-filter: blur(10px);
}

/* Buttons */
.stDownloadButton button, .stButton > button {
  background: linear-gradient(90deg,#00eaff,#00bcd4);
  color: #001018;
  border: none;
  border-radius: 10px;
  padding: 8px 18px;
  font-weight: 700;
}
.stDownloadButton button:hover, .stButton > button:hover {
  background: linear-gradient(90deg,#ff9800,#ff5722);
  color: #fff;
}

/* Slider accent */
.stSlider > div > div > div > div { background: #00eaff; }
</style>

<!-- Parallax star layers + grid -->
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
            st.success("✅ Loaded VisionAI checkpoint.pth successfully!")
        except Exception:
            st.warning("⚠️ Could not load checkpoint; using pretrained COCO weights instead.")
    else:
        st.warning("⚠️ Checkpoint not found; using pretrained COCO weights instead.")
    return model

# ========== TRANSFORM ==========
to_tensor = T.ToTensor()

# ========== MASKING PIPELINE ==========
def run_maskrcnn(image_pil: Image.Image, image_np: np.ndarray, model, conf_thresh: float):
    with torch.no_grad():
        inp = to_tensor(image_pil).to(DEVICE)
        outputs = model([inp])[0]
    h, w = image_np.shape[:2]

    if "masks" not in outputs or len(outputs["masks"]) == 0:
        return np.zeros_like(image_np)

    scores = outputs["scores"].cpu().numpy()
    keep = scores >= conf_thresh
    if not np.any(keep):
        return np.zeros_like(image_np)

    masks


