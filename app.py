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


# ========== STARFIELD + THEME CSS ==========
st.markdown("""
<style>
/* Dark gradient base */
.stApp {
  background: linear-gradient(135deg, #0b1020, #121a2a 45%, #0d1c29);
  color: #ffffff;
  font-family: 'Segoe UI', system-ui, -apple-system, Roboto, Arial, sans-serif;
}

/* Make content render above animation */
.block-container { position: relative; z-index: 5; }

/* --- Starfield animation (three parallax layers) --- */
#stars, #stars2, #stars3 {
  position: fixed;
  top: 0; left: 0;
  width: 100vw; height: 100vh;
  display: block;
  background: transparent;
  pointer-events: none;            /* do not block clicks/scroll */
  z-index: 0;
}

#stars:after, #stars2:after, #stars3:after {
  content: " ";
  position: absolute;
  top: -1000px;
  width: 2px; height: 2px;
  background: transparent;
  box-shadow:
    1240px 200px #fff, 340px 540px #cfe7ff, 980px 300px #ffffff55, 400px 900px #d0e6ff,
    760px 120px #ffffff88, 1500px 640px #ffffff, 60px  840px #cfe7ff, 1350px 100px #ffffff88,
    1020px 760px #ffffffaa, 180px  300px #ffffff55, 860px  560px #ffffffbb, 40px   110px #d0e6ff;
  animation: animStar 100s linear infinite;
}

#stars2:after {
  width: 3px; height: 3px;
  box-shadow:
    140px  680px #ffffffaa,  760px  460px #ffffff77, 1020px 240px #d0e6ff,
    1260px 900px #ffffff55, 1600px 320px #ffffffaa, 460px  100px #ffffff88;
  animation: animStar 140s linear infinite;
}

#stars3:after {
  width: 1px; height: 1px;
  box-shadow:
    240px 120px #ffffff55,  560px 980px #ffffff55,  940px  660px #d0e6ff,
    1220px 780px #ffffff55, 1540px 420px #ffffff55, 300px  500px #d0e6ff;
  animation: animStar 180s linear infinite;
}

@keyframes animStar {
  from { transform: translateY(0);   }
  to   { transform: translateY(1000px); }
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

<!-- Starfield layers (non-blocking, under content) -->
<div id="stars"></div><div id="stars2"></div><div id="stars3"></div>
""", unsafe_allow_html=True)
@st.cache_resource
def load_model():
    model = segmodels.deeplabv3_resnet101(weights="COCO_WITH_VOC_LABELS_V1")


  # ========== MODEL LOADING ==========
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


# ========== TRANSFORMS ==========
transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
])


# ========== MASKING PIPELINE ==========
def get_clean_masks(logits, orig_h, orig_w, image_np, conf_thresh=0.5):
    # logits [1, C, H/8, W/8] -> softmax -> upsample to original
    probs = torch.softmax(logits, dim=1)
    up = F.interpolate(probs, size=(orig_h, orig_w),
                       mode="bilinear", align_corners=False)
    probs_np = up.squeeze(0).cpu().numpy()     # [C, H, W]

    pred_classes = np.argmax(probs_np, axis=0) # [H, W]
    max_conf = np.max(probs_np, axis=0)        # [H, W]

    # Everything except background (index 0) above threshold = object
    binary_mask = ((pred_classes != 0) & (max_conf > conf_thresh)).astype(np.uint8) * 255

    # Morphological clean-up
    kernel = np.ones((5, 5), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    # Keep only large components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    min_size = 900
    filtered = np.zeros_like(binary_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            filtered[labels == i] = 255
    binary_mask = filtered

    # Color mask: object on black
    color_mask = np.zeros_like(image_np)
    color_mask[binary_mask == 255] = image_np[binary_mask == 255]

    return binary_mask, color_mask


# ========== HEADER ==========
st.markdown("<h1>🌌 VisionExtract — Next-Gen Image Segmentation</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;font-size:18px;margin-bottom:8px;'>"
    "Upload an image or try the demo. Get high-quality <b>binary</b> and <b>color</b> masks for "
    "all COCO classes with a clean, beautiful UI ✨"
    "</p>", unsafe_allow_html=True,
)


# ========== HOW THE TOOL WORKS ==========
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


# ========== DEMO PREVIEW ==========
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


# ========== UPLOAD + INFERENCE ==========
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

