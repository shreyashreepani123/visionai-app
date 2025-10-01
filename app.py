# app.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.segmentation as segmodels
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2
import streamlit as st
from io import BytesIO
import requests

# ---------------- CONFIG ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 512                # larger -> usually better mask quality (slower)
CONF_THRESH = 0.5               # confidence threshold for binary mask
MIN_COMPONENT_AREA = 1500       # remove tiny connected components
MODEL_URL = "https://github.com/shreyashreepani123/visionai-app/releases/download/v1.1/checkpoint.pth"
CHECKPOINT_PATH = "checkpoint.pth"

# ---------------- UTILITIES ----------------
def ensure_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        return
    st.info("Downloading checkpoint...")
    r = requests.get(MODEL_URL, timeout=300)
    r.raise_for_status()
    with open(CHECKPOINT_PATH, "wb") as f:
        f.write(r.content)
    st.success("Downloaded checkpoint.")

def detect_num_classes_from_state(state_dict):
    # Try to find classifier weights in common keys
    for k in ("classifier.4.weight", "aux_classifier.4.weight", "classifier.classifier.4.weight"):
        if k in state_dict:
            return int(state_dict[k].shape[0])
    # fallback: try to find any weight with classifier.4 in key
    for k in state_dict:
        if "classifier.4.weight" in k:
            return int(state_dict[k].shape[0])
    # default fallback (common number for COCO-style training may be 91)
    return None

def build_deeplab(num_classes, aux_loss=True):
    # create model architecture matching typical training setup
    model = segmodels.deeplabv3_resnet101(weights=None, aux_loss=aux_loss)
    # replace classifier output channels to num_classes
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    if getattr(model, "aux_classifier", None) is not None:
        model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model

def safe_load_state(model, state_dict):
    model_state = model.state_dict()
    loaded = {}
    missing = []
    mismatched = []
    for k,v in state_dict.items():
        if k in model_state:
            if v.shape == model_state[k].shape:
                loaded[k] = v
            else:
                mismatched.append((k, v.shape, model_state[k].shape))
        else:
            # ignore keys not present
            pass
    model_state.update(loaded)
    model.load_state_dict(model_state, strict=False)
    # compute missing keys for diagnostic
    for k in model_state:
        if k not in loaded:
            missing.append(k)
    return loaded.keys(), missing, mismatched

# ---------------- TRANSFORM ----------------
transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
])

# ---------------- POSTPROCESS & REFINEMENT ----------------
def refine_with_grabcut(image_rgb, initial_binary_mask):
    """
    Try to refine the binary mask with OpenCV grabCut using initial mask as seeds.
    If any OpenCV error occurs, return original mask.
    initial_binary_mask: uint8 0/255
    """
    try:
        img = image_rgb.copy()
        h,w = initial_binary_mask.shape
        mask_gc = np.zeros((h,w), np.uint8)
        # probable bg by default
        mask_gc[:] = cv2.GC_PR_BGD
        # sure FG: initial mask pixels
        mask_gc[initial_binary_mask == 255] = cv2.GC_FGD
        # pad border as background
        pad = 3
        mask_gc[:pad,:] = cv2.GC_BGD
        mask_gc[-pad:,:] = cv2.GC_BGD
        mask_gc[:,:pad] = cv2.GC_BGD
        mask_gc[:,-pad:] = cv2.GC_BGD

        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        # run grabCut
        cv2.grabCut(img, mask_gc, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        # final mask: sure+probable foreground
        refined = np.where((mask_gc==cv2.GC_FGD) | (mask_gc==cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        return refined
    except Exception:
        return initial_binary_mask

def postprocess_logits_to_masks(logits, orig_h, orig_w, image_np, conf_thresh=0.5):
    # logits: torch tensor [1, C, H, W]
    probs = torch.softmax(logits, dim=1)
    up = F.interpolate(probs, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    probs_np = up.squeeze(0).cpu().numpy()   # shape (C, H, W)
    pred_classes = np.argmax(probs_np, axis=0)
    max_conf = np.max(probs_np, axis=0)

    # binary foreground mask: any class != 0 and confidence > threshold
    binary = ((pred_classes != 0) & (max_conf > conf_thresh)).astype(np.uint8) * 255

    # morphological cleaning
    kernel = np.ones((5,5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # remove small connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean = np.zeros_like(binary)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= MIN_COMPONENT_AREA:
            clean[labels == i] = 255

    # try GrabCut refinement (optional, can be slow)
    refined = refine_with_grabcut(image_np, clean)

    # color mask: keep original colors where refined==255
    color_mask = np.zeros_like(image_np)
    color_mask[refined==255] = image_np[refined==255]

    # multi-class colorized mask (simple palette)
    c_mask = pred_classes
    colorized = colorize_label_map(c_mask)

    return refined, color_mask, colorized, pred_classes, max_conf

def colorize_label_map(label_map):
    # label_map: HxW ints
    rng = np.random.RandomState(12345)
    max_label = int(label_map.max())
    palette = rng.randint(0,256,size=(max_label+1,3),dtype=np.uint8)
    palette[0] = np.array([0,0,0], np.uint8)  # background black
    H,W = label_map.shape
    out = np.zeros((H,W,3), dtype=np.uint8)
    for lbl in range(0, max_label+1):
        out[label_map==lbl] = palette[lbl]
    return out

# ---------------- STREAMLIT APP ----------------
st.set_page_config(page_title="VisionAI Segmentation (Your checkpoint)", layout="centered")
st.title("VisionAI Segmentation — using your checkpoint")
st.markdown(
    "This app **loads and uses only your provided checkpoint** (no external pretrained weights are loaded at runtime). "
    "If loading reports missing or mismatched layers, the app will attempt a best-effort safe load and continue."
)

uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
conf_thresh = st.slider("Confidence threshold for foreground", 0.05, 0.95, float(CONF_THRESH), 0.05)
use_grabcut = st.checkbox("Use GrabCut refinement (slower)", value=True)
resize_size = st.selectbox("Input size (larger = slower but usually better)", [256, 320, 384, 512, 640], index=3)
IMAGE_SIZE = int(resize_size)

# load model once
@st.cache_resource
def load_model_from_checkpoint():
    ensure_checkpoint()
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    # detect classes
    num_cls = detect_num_classes_from_state(state)
    if num_cls is None:
        # fallback to 91 if unknown (common for COCO)
        num_cls = 91
    # build model
    model = build_deeplab(num_classes=num_cls, aux_loss=True)
    # safe load
    loaded_keys, missing, mismatched = safe_load_state(model, state)
    model.to(DEVICE).eval()
    return model, num_cls, list(missing), list(mismatched)

# show diagnostics
try:
    model, detected_num_classes, missing_keys, mismatched_keys = load_model_from_checkpoint()
    st.success(f"Model loaded from checkpoint. Detected num_classes = {detected_num_classes}")
    if missing_keys:
        st.warning(f"{len(missing_keys)} layer(s) from model were not present in checkpoint (will use defaults).")
    if mismatched_keys:
        st.warning(f"{len(mismatched_keys)} layer(s) had shape mismatches and were skipped (examples shown).")
        st.write(mismatched_keys[:5])
except Exception as e:
    st.error("Failed to load checkpoint: " + str(e))
    st.stop()

if uploaded is not None:
    image_pil = Image.open(uploaded).convert("RGB")
    orig_w, orig_h = image_pil.size
    image_np = np.array(image_pil)

    st.subheader("Original Image")
    st.image(image_np, use_column_width=True)

    # transform with chosen size
    tfm = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])
    inp = tfm(image_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(inp)
        logits = out["out"]

    # postprocess (upsample back to original)
    refined, color_mask, colorized, pred_classes, max_conf = postprocess_logits_to_masks(logits, orig_h, orig_w, image_np, conf_thresh)

    st.subheader("Binary Mask (objects vs background)")
    st.image(refined, use_column_width=True)
    st.download_button("Download Binary Mask (PNG)",
                       data=BytesIO(cv2.imencode(".png", refined)[1].tobytes()),
                       file_name="binary_mask.png",
                       mime="image/png")

    st.subheader("Color Mask (original colors on black background)")
    st.image(color_mask, use_column_width=True)
    buf = cv2.imencode(".png", cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))[1].tobytes()
    st.download_button("Download Color Mask (PNG)", data=BytesIO(buf), file_name="color_mask.png", mime="image/png")

    st.subheader("Colored Multi-class Mask (visualized)")
    st.image(colorized, use_column_width=True)
    st.download_button("Download Colored Multi-class Mask", data=BytesIO(cv2.imencode(".png", cv2.cvtColor(colorized, cv2.COLOR_RGB2BGR))[1].tobytes()), file_name="multiclass_mask.png", mime="image/png")






