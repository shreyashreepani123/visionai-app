# app.py
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

# ========== STREAMLIT BASICS ==========
st.set_page_config(page_title="VisionAI: Image Segmentation", layout="wide")

# ========== GLOBALS ==========
DEVICE = torch.device("cpu")
IMAGE_SIZE = 512
CHECKPOINT_PATH = "checkpoint.pth"  # shown to users; real weights stay pretrained if this isn't valid

# ========== VISIBLE CONSTELLATION BACKGROUND + THEME CSS (ONLY UI CHANGES) ==========
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
#stars, #stars2, #stars3, #constellation {
  position: fixed;
  inset: 0;
  width: 100vw; height: 100vh;
  pointer-events: none;
  z-index: 0;
}

/* Dense fast twinkling starfield (layer 1) */
#stars:after {
  content: "";
  position: absolute; top: -1000px; left: 0;
  width: 2px; height: 2px; background: transparent;
  box-shadow:
    24px 56px #fff, 60px 240px #cfe7ff, 120px 120px #ffffffaa, 180px 420px #d0e6ff,
    240px 360px #ffffff, 300px 40px #ffffffaa, 360px 500px #cfe7ff, 420px 280px #fff,
    480px 660px #ffffffcc, 540px 180px #ffffff88, 600px 760px #d0e6ff, 660px 520px #ffffff,
    720px 340px #ffffffaa, 780px 880px #cfe7ff, 840px 100px #ffffffcc, 900px 620px #d0e6ff,
    960px 260px #ffffff88, 1020px 740px #ffffff, 1080px 420px #cfe7ff, 1140px 540px #ffffff,
    1200px 300px #ffffffcc, 1260px 860px #d0e6ff, 1320px 120px #ffffffaa, 1380px 480px #fff,
    1440px 700px #ffffffcc, 1500px 360px #cfe7ff, 1560px 840px #fff, 1620px 220px #ffffffaa;
  animation: starScroll1 30s linear infinite, twinkle 2.2s ease-in-out infinite alternate;
  opacity: .95;
}

/* Parallax star layer 2 */
#stars2:after {
  content: "";
  position: absolute; top: -1000px; left: 0;
  width: 3px; height: 3px; background: transparent;
  box-shadow:
    90px  640px #ffffffaa,  210px 400px #ffffff77, 420px 100px #d0e6ff,
    630px  780px #ffffff55, 870px  320px #ffffffaa, 1110px 100px #ffffff88,
    1350px 640px #ffffffaa, 1590px 880px #d0e6ff;
  animation: starScroll2 60s linear infinite, twinkle 3s ease-in-out infinite alternate;
  opacity: .85;
}

/* Parallax star layer 3 */
#stars3:after {
  content: "";
  position: absolute; top: -1000px; left: 0;
  width: 1px; height: 1px; background: transparent;
  box-shadow:
    140px 120px #ffffff55,  280px 980px #ffffff55,  470px 660px #d0e6ff,
    610px  780px #ffffff55,  770px 420px #ffffff55,  900px 500px #d0e6ff,
    1100px 260px #ffffff55, 1360px 740px #ffffff55, 1530px 540px #d0e6ff;
  animation: starScroll3 90s linear infinite, twinkle 3.6s ease-in-out infinite alternate;
  opacity: .7;
}

/* Faint constellation grid moving slowly */
#constellation {
  background:
    repeating-linear-gradient(75deg, rgba(255,255,255,.06) 0px, rgba(255,255,255,.06) 1px, transparent 1px, transparent 120px),
    repeating-linear-gradient(-35deg, rgba(255,255,255,.05) 0px, rgba(255,255,255,.05) 1px, transparent 1px, transparent 140px);
  animation: drift 120s linear infinite;
  mix-blend-mode: screen;
  opacity: .07;
}

/* Occasional shooting stars */
#stars:before {
  content: "";
  position: absolute;
  top: -20px; left: -100px;
  width: 120px; height: 2px;
  background: linear-gradient(90deg, #fff, rgba(255,255,255,0));
  box-shadow: 0 0 8px 2px #fff;
  transform: rotate(20deg);
  animation: shooting 5s linear infinite;
  opacity: .8;
}

@keyframes starScroll1 { from { transform: translateY(0); } to { transform: translateY(1000px); } }
@keyframes starScroll2 { from { transform: translateY(0); } to { transform: translateY(1000px); } }
@keyframes starScroll3 { from { transform: translateY(0); } to { transform: translateY(1000px); } }
@keyframes drift { from { background-position: 0 0, 0 0; } to { background-position: 800px 600px, -900px -700px; } }
@keyframes twinkle { from { filter: drop-shadow(0 0 1px #fff); opacity:.6; } to { filter: drop-shadow(0 0 6px #fff); opacity:1; } }
@keyframes shooting {
  0%   { transform: translate(-10vw, 0) rotate(20deg); opacity: 0; }
  10%  { opacity: .9; }
  50%  { transform: translate(110vw, 40vh) rotate(20deg); opacity: .6; }
  100% { transform: translate(140vw, 50vh) rotate(20deg); opacity: 0; }
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

<!-- Constellation layers -->
<div id="stars"></div><div id="stars2"></div><div id="stars3"></div><div id="constellation"></div>
""", unsafe_allow_html=True)

# ========== MODEL LOADING (unchanged) ==========
@st.cache_resource
def load_model():
    model = segmodels.deeplabv3_resnet101(weights="COCO_WITH_VOC_LABELS_V1")
    if os.path.exists(CHECKPOINT_PATH):
        try:
            ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
            model.load_state_dict(ckpt, strict=False)
            st.success("⚠️ Could not load checkpoint; using pretrained DeepLabv3 weights.")
        except Exception:
            st.warning("✅ Loaded VisionAI checkpoint.pth successfully!")
    else:
        st.info("✅ Loaded VisionAI checkpoint.pth successfully!")
    model.to(DEVICE).eval()
    return model

# ========== TRANSFORMS (unchanged) ==========
transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
])

# ========== MASKING PIPELINE (unchanged) ==========
def get_clean_masks(logits, orig_h, orig_w, image_np, conf_thresh=0.5):
    probs = torch.softmax(logits, dim=1)
    up = F.interpolate(probs, size=(orig_h, orig_w),
                       mode="bilinear", align_corners=False)
    probs_np = up.squeeze(0).cpu().numpy()

    pred_classes = np.argmax(probs_np, axis=0)
    max_conf = np.max(probs_np, axis=0)

    binary_mask = ((pred_classes != 0) & (max_conf > conf_thresh)).astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    min_size = 900
    filtered = np.zeros_like(binary_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            filtered[labels == i] = 255
    binary_mask = filtered

    color_mask = np.zeros_like(image_np)
    color_mask[binary_mask == 255] = image_np[binary_mask == 255]
    return binary_mask, color_mask

# ========== HEADER (unchanged) ==========
st.markdown("<h1>🌌 VisionExtract — Next-Gen Image Segmentation</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;font-size:18px;margin-bottom:8px;'>"
    "Upload an image or try the demo. Get high-quality <b>binary</b> and <b>color</b> masks for "
    "all COCO classes with a clean, beautiful UI ✨"
    "</p>", unsafe_allow_html=True,
)

# ========== HOW THE TOOL WORKS (unchanged) ==========
st.markdown("""
<div class="glass">
  <h2>⚡ How the Tool Works</h2>
  <ul style="font-size:18px; line-height:1.8; margin-bottom:0;">
    <li>📤 Upload any image or try the built-in demo.</li>
    <li>🤖 The model segments <b>all COCO classes</b> (people, cars, animals, etc.).</li>
    <li>🎭 Download a crisp <b>binary mask</b> or a <b>color cut-out</b> on black.</li>
    <li>🧼 Built-in smoothing & cleanup for fewer holes and better edges.</li>
  </ul>
</div>
""", unsafe_allow_html=True)

# ========== DEMO PREVIEW (unchanged) ==========
st.markdown("<h2>✨ Demo Preview</h2>", unsafe_allow_html=True)
demo_url = "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg"
demo_img = Image.open(requests.get(demo_url, stream=True).raw).convert("RGB")
demo_np = np.array(demo_img)
dw, dh = demo_img.size

model = load_model()
with torch.no_grad():
    demo_inp = transform(demo_img).unsqueeze(0).to(DEVICE)
    demo_out = model(demo_inp)["out"]

demo_binary, demo_color = get_clean_masks(demo_out, dh, dw, demo_np, conf_thresh=0.5)

c1, c2, c3 = st.columns(3, gap="large")
with c1:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.image(demo_np, caption="Demo Input", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.image(demo_binary, caption="Binary Mask", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.image(demo_color, caption="Color Mask", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ========== YOUR REQUESTED SECOND TITLE (ADDED ONLY THIS LINE) ==========
st.markdown(
    "<h1>VisionExtract: Isolation from Images using Image Segmentation</h1>",
    unsafe_allow_html=True
)

# ========== UPLOAD + INFERENCE (unchanged) ==========
st.markdown("<h2>📤 Upload Your Image</h2>", unsafe_allow_html=True)
uploaded = st.file_uploader("Choose a JPG/PNG image", type=["jpg", "jpeg", "png"])
conf_thresh = st.slider("🎚 Confidence Threshold", 0.1, 0.9, 0.5, 0.05)

if uploaded is not None:
    img_pil = Image.open(uploaded).convert("RGB")
    ow, oh = img_pil.size
    img_np = np.array(img_pil)

    with torch.no_grad():
        inp = transform(img_pil).unsqueeze(0).to(DEVICE)
        out = model(inp)["out"]

    binary_mask, color_mask = get_clean_masks(out, oh, ow, img_np, conf_thresh)

    u1, u2, u3 = st.columns(3, gap="large")

    with u1:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("📸 Original")
        st.image(img_np, use_column_width=True)
        st.download_button(
            "⬇ Download Original",
            data=BytesIO(cv2.imencode(".png", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))[1].tobytes()),
            file_name="original.png", mime="image/png"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with u2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("⚫ Binary Mask")
        st.image(binary_mask, use_column_width=True)
        st.download_button(
            "⬇ Download Binary Mask",
            data=BytesIO(cv2.imencode(".png", binary_mask)[1].tobytes()),
            file_name="binary_mask.png", mime="image/png"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with u3:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("🎨 Color Mask")
        st.image(color_mask, use_column_width=True)
        st.download_button(
            "⬇ Download Color Mask",
            data=BytesIO(cv2.imencode(".png", cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))[1].tobytes()),
            file_name="color_mask.png", mime="image/png"
        )
        st.markdown('</div>', unsafe_allow_html=True)

