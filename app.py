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
FALLBACK_NUM_CLASSES = 91

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
    T.ToTensor(),   # no ImageNet normalization (your training used raw tensor)
])

# ---------------- INFERENCE HELPERS ----------------
def run_tta_logits(model, pil_img, scales=(0.75, 1.0, 1.25), do_flip=True):
    w0, h0 = pil_img.size
    acc = None
    for s in scales:
        size = (int(IMAGE_SIZE * s), int(IMAGE_SIZE * s))
        resized = pil_img.resize(size, Image.BILINEAR)
        x = T.ToTensor()(resized).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(x)["out"]
            if do_flip:
                xf = torch.flip(x, dims=[3])
                logits_f = model(xf)["out"]
                logits_f = torch.flip(logits_f, dims=[3])
                logits += logits_f
            logits = F.interpolate(logits, size=(IMAGE_SIZE, IMAGE_SIZE),
                                   mode="bilinear", align_corners=False)

        acc = logits if acc is None else acc + logits

    logits_full = F.interpolate(acc, size=(h0, w0), mode="bilinear", align_corners=False)
    return logits_full.squeeze(0)

def get_binary_mask_from_logits(logits, conf_thresh=0.6):
    probs = torch.softmax(logits, dim=0)
    fg_probs, _ = probs[1:].max(0)  # skip background
    fg_probs = fg_probs.cpu().numpy()
    mask = (fg_probs > conf_thresh).astype(np.uint8) * 255
    return mask

def refine_mask_with_grabcut(image_rgb, init_mask):
    try:
        h, w = init_mask.shape
        gc_mask = np.where(init_mask > 0, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype('uint8')

        kernel = np.ones((5,5), np.uint8)
        sure_fg = cv2.erode((init_mask>0).astype(np.uint8)*255, kernel, 2)
        sure_bg = cv2.dilate((init_mask==0).astype(np.uint8)*255, kernel, 2)
        gc_mask[sure_fg>0] = cv2.GC_FGD
        gc_mask[sure_bg>0] = cv2.GC_BGD

        bgdModel = np.zeros((1,65), np.float64)
        fgdModel = np.zeros((1,65), np.float64)
        cv2.grabCut(image_rgb, gc_mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

        refined = np.where((gc_mask==cv2.GC_FGD) | (gc_mask==cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        return refined
    except Exception:
        return init_mask

def colorize_multiclass(mask, num_classes):
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    rng = np.random.RandomState(999)
    palette = rng.randint(0, 256, size=(num_classes, 3), dtype=np.uint8)
    palette[0] = np.array([0, 0, 0], np.uint8)
    for cls in np.unique(mask):
        out[mask == cls] = palette[int(cls) % num_classes]
    return out

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
st.set_page_config(page_title="VisionAI Improved Extraction", layout="wide")
st.title("🖼️ VisionAI Segmentation with Better Extraction")

uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded is None:
    st.info("⬆️ Upload an image to start. The app will auto-download your checkpoint.")
else:
    image_pil = Image.open(uploaded).convert("RGB")
    orig_rgb = np.array(image_pil)

    with st.spinner("Running inference..."):
        model, num_classes = load_model()
        logits = run_tta_logits(model, image_pil)

        # Binary mask (softmax + threshold)
        binary_mask = get_binary_mask_from_logits(logits, conf_thresh=0.6)

        # Morphological cleanup
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN,  np.ones((5,5),np.uint8))

        # GrabCut refinement
        binary_mask = refine_mask_with_grabcut(orig_rgb, binary_mask)

        # Multi-class argmax for visualization
        pred_classes = torch.argmax(logits, dim=0).cpu().numpy().astype(np.int32)
        color_multiclass = colorize_multiclass(pred_classes, num_classes)

        # Transparent cutout
        cutout = extract_transparent(orig_rgb, binary_mask)

    # ---------------- Results ----------------
    st.subheader("Original")
    st.image(orig_rgb, use_column_width=True)

    st.subheader("Binary Mask")
    st.image(binary_mask, use_column_width=True)
    add_download_button(binary_mask, "binary_mask.png")

    st.subheader("Colored Multi-class Mask")
    st.image(color_multiclass, use_column_width=True)
    add_download_button(color_multiclass, "colored_mask.png")

    st.subheader("Transparent Cutout (alpha)")
    st.image(cutout, use_column_width=True)
    add_download_button(cutout, "cutout.png")









