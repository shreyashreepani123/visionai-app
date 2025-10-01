import os
import io
import base64
import requests
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50
import streamlit as st

# ---------------- CONFIG ----------------
MODEL_URL = "https://github.com/shreyashreepani123/visionai-app/releases/download/v1.1/checkpoint.pth"
CHECKPOINT_PATH = "checkpoint.pth"
DEVICE = torch.device("cpu")
IMAGE_SIZE = 512
FALLBACK_NUM_CLASSES = 91   # COCO default

# ---------------- UTIL: checkpoint ----------------
def ensure_checkpoint():
    if os.path.exists(CHECKPOINT_PATH) and os.path.getsize(CHECKPOINT_PATH) > 0:
        return
    st.info("Downloading model checkpoint…")
    r = requests.get(MODEL_URL, timeout=300, allow_redirects=True)
    r.raise_for_status()
    with open(CHECKPOINT_PATH, "wb") as f:
        f.write(r.content)

def strip_module_prefix(state_dict):
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_sd[k[len("module."):]] = v
        else:
            new_sd[k] = v
    return new_sd

def detect_num_classes(state_dict, default_nc=FALLBACK_NUM_CLASSES):
    key = "classifier.4.weight"
    if key in state_dict:
        return int(state_dict[key].shape[0])
    return default_nc

# ---------------- MODEL LOADER ----------------
@st.cache_resource
def load_model():
    ensure_checkpoint()
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    state_dict = strip_module_prefix(state_dict)
    num_classes = detect_num_classes(state_dict, FALLBACK_NUM_CLASSES)

    model = deeplabv3_resnet50(weights=None, num_classes=num_classes)
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE).eval()
    return model, num_classes

# ---------------- TRANSFORMS ----------------
to_tensor = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),   # no ImageNet normalization
])

# ---------------- INFERENCE HELPERS ----------------
def run_logits(model, pil_img):
    w0, h0 = pil_img.size
    x = to_tensor(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)["out"]
        logits = F.interpolate(logits, size=(h0, w0), mode="bilinear", align_corners=False)

    return logits.squeeze(0)

def get_person_mask_from_logits(logits, person_class=1, conf_thresh=0.5):
    """
    Extract binary mask for the person class (COCO = class 1).
    """
    probs = torch.softmax(logits, dim=0)  # (C,H,W)
    if person_class >= probs.shape[0]:
        raise ValueError(f"Person class {person_class} not found in model outputs")

    person_prob = probs[person_class].cpu().numpy()
    mask = (person_prob > conf_thresh).astype(np.uint8) * 255
    return mask

def extract_transparent(orig_rgb, mask):
    rgba = np.dstack([orig_rgb, mask])
    return rgba

def add_download_button(img_np, filename):
    pil = Image.fromarray(img_np)
    buf = io.BytesIO()
    fmt = "PNG" if filename.endswith(".png") else "JPEG"
    pil.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    st.markdown(
        f'<a href="data:file/{fmt.lower()};base64,{b64}" download="{filename}">📥 Download {filename}</a>',
        unsafe_allow_html=True,
    )

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="VisionAI Person Segmentation", layout="wide")
st.title("🖼️ VisionAI - Person Segmentation")

uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded is None:
    st.info("⬆️ Upload an image to start.")
else:
    image_pil = Image.open(uploaded).convert("RGB")
    orig_rgb = np.array(image_pil)

    with st.spinner("Running inference..."):
        model, num_classes = load_model()
        logits = run_logits(model, image_pil)

        # Binary mask for "person" only
        binary_mask = get_person_mask_from_logits(logits, person_class=1, conf_thresh=0.5)

        # Morphological cleanup
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN,  np.ones((5,5),np.uint8))

        # Transparent cutout
        cutout = extract_transparent(orig_rgb, binary_mask)

    # ---------------- Results ----------------
    st.subheader("Original Image")
    st.image(orig_rgb, use_column_width=True)

    st.subheader("Person Binary Mask")
    st.image(binary_mask, use_column_width=True)
    add_download_button(binary_mask, "person_mask.png")

    st.subheader("Transparent Cutout (alpha)")
    st.image(cutout, use_column_width=True)
    add_download_button(cutout, "person_cutout.png")







