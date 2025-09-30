import os
import io
import cv2
import time
import torch
import numpy as np
from io import BytesIO
from PIL import Image
import requests
import streamlit as st
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models.segmentation as segmodels

# -------------------------- Config --------------------------
MODEL_URL        = "https://github.com/shreyashreepani123/visionai-app/releases/download/v1.1/checkpoint.pth"
CHECKPOINT_PATH  = "checkpoint.pth"
DEVICE           = torch.device("cpu")     # Streamlit Cloud: CPU-only
IMAGE_SIZE       = 256                     # matches your training
TTA_SCALES       = [0.85, 1.0, 1.20]       # Test-time augmentation (multi-scale)
TTA_HFLIP        = True                    # horizontal flip TTA
MIN_OBJ_PIXELS   = 0.0015                  # remove tiny blobs (as fraction of image pixels)

# Mirror training: NO normalization, NEAREST resize (you trained with A.Resize(..., interpolation=Image.NEAREST))
to_tensor_no_norm = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=T.InterpolationMode.NEAREST),
    T.ToTensor(),                          # 0..1 floats, no mean/std
])

# ---------------------- Utility funcs -----------------------
def ensure_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        return
    st.write("📥 Downloading model weights…")
    r = requests.get(MODEL_URL, allow_redirects=True, timeout=300)
    r.raise_for_status()
    with open(CHECKPOINT_PATH, "wb") as f:
        f.write(r.content)
    st.write("✅ Downloaded model checkpoint.")

def detect_num_classes_from_state(state_dict, fallback=91):
    # Look for classifier head shape to infer classes
    for k, v in state_dict.items():
        if k.endswith("classifier.4.weight"):
            return int(v.shape[0])
    return fallback

def build_model(num_classes, state_dict):
    # Your training used DeepLabV3-ResNet50
    model = segmodels.deeplabv3_resnet50(weights=None, aux_loss=True)
    # Replace heads with correct class count
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    if getattr(model, "aux_classifier", None) is not None:
        model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    # Many checkpoints don’t include aux head; filter unknown keys
    filtered = {k: v for k, v in state_dict.items() if not k.startswith("aux_classifier")}
    missing_unexpected = model.load_state_dict(filtered, strict=False)
    # (Don’t crash on leftover keys)
    if missing_unexpected.missing_keys or missing_unexpected.unexpected_keys:
        st.caption(f"ℹ️ load_state_dict info — missing: {len(missing_unexpected.missing_keys)}, "
                   f"unexpected: {len(missing_unexpected.unexpected_keys)}")

    model.to(DEVICE).eval()
    return model

@st.cache_resource
def load_model():
    ensure_checkpoint()
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    num_classes = detect_num_classes_from_state(state, fallback=91)
    st.caption(f"Detected num_classes = **{num_classes}** from checkpoint.")
    return build_model(num_classes, state), num_classes

def upsample_logits(logits, h, w):
    return F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)

def predict_tta(model, pil_img, num_classes):
    """
    Multi-scale + horizontal flip TTA (averages class probabilities)
    """
    orig_w, orig_h = pil_img.size
    prob_accum = None

    for s in TTA_SCALES:
        # scale with NEAREST (match training), but we want square 256 -> consistent behavior
        sz = int(round(IMAGE_SIZE * s))
        resize_t = T.Compose([
            T.Resize((sz, sz), interpolation=T.InterpolationMode.NEAREST),
            T.ToTensor(),
        ])

        # normal pass
        x = resize_t(pil_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(x)["out"]                      # [1, C, h', w']
            out = upsample_logits(out, orig_h, orig_w) # back to original size
            prob = out.softmax(dim=1)                  # [1, C, H, W]
        prob_accum = prob if prob_accum is None else (prob_accum + prob)

        # horizontal flip pass (if enabled)
        if TTA_HFLIP:
            x_flip = T.functional.hflip(x)
            with torch.no_grad():
                out_f = model(x_flip)["out"]
                out_f = upsample_logits(out_f, orig_h, orig_w)
                # unflip back
                out_f = torch.flip(out_f, dims=[3])
                prob_f = out_f.softmax(dim=1)
            prob_accum += prob_f

    prob_mean = prob_accum / (len(TTA_SCALES) * (2 if TTA_HFLIP else 1))
    pred = prob_mean.argmax(1).squeeze(0).cpu().numpy().astype(np.int32)
    return pred, prob_mean.squeeze(0).cpu().numpy()  # pred, per-class probs [C,H,W] (for confidence if needed)

def compute_background_index(pred_classes):
    # In your dataset the background is the most frequent id in the prediction
    flat = pred_classes.reshape(-1)
    bg = int(np.bincount(flat).argmax())
    return bg

def remove_tiny_regions(binary, min_frac=MIN_OBJ_PIXELS):
    """Remove tiny components from a 0/255 binary mask."""
    H, W = binary.shape
    min_pixels = max(1, int(min_frac * H * W))
    _, labels, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), connectivity=8)
    out = np.zeros_like(binary)
    for i in range(1, stats.shape[0]):  # 0 is background
        if stats[i, cv2.CC_STAT_AREA] >= min_pixels:
            out[labels == i] = 255
    return out

def refine_edges(binary):
    """Light morphological smooth (close->open) and fast guided-like edge smooth."""
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=1)
    return binary

def make_color_mask(image_rgb, binary255):
    out = np.zeros_like(image_rgb)
    out[binary255 == 255] = image_rgb[binary255 == 255]
    return out

def to_png_bytes(img_rgb_or_gray):
    if img_rgb_or_gray.ndim == 2:
        enc = cv2.imencode(".png", img_rgb_or_gray)[1]
    else:
        enc = cv2.imencode(".png", cv2.cvtColor(img_rgb_or_gray, cv2.COLOR_RGB2BGR))[1]
    return enc.tobytes()

# -------------------------- App UI --------------------------
st.set_page_config(page_title="VisionAI Segmentation (Your Weights)", layout="centered")
st.title("🔍 VisionAI Segmentation — using **your** checkpoint")

st.write("- Preproc matches training (**NEAREST resize**, **no normalization**).")
st.write("- Inference uses **TTA (multi-scale + flip)** for better masks.")
st.write("- Produces **Binary** (white vs black) and **Color Mask** (original colors on black).")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    # Load and show
    img_pil = Image.open(uploaded).convert("RGB")
    img_np  = np.array(img_pil)
    H, W    = img_np.shape[:2]

    st.subheader("Uploaded Image")
    st.image(img_np, use_column_width=True)

    # Load model
    model, NUM_CLASSES = load_model()

    # Predict (with TTA)
    with st.spinner("Running segmentation…"):
        t0 = time.time()
        pred_classes, per_class_probs = predict_tta(model, img_pil, NUM_CLASSES)
        elapsed = time.time() - t0

    # Background detection
    bg_idx = compute_background_index(pred_classes)

    # Binary mask
    binary255 = ((pred_classes != bg_idx).astype(np.uint8) * 255)
    binary255 = remove_tiny_regions(binary255, min_frac=MIN_OBJ_PIXELS)
    binary255 = refine_edges(binary255)

    # Color mask (show original colors only on objects)
    color_mask = make_color_mask(img_np, binary255)

    # ---- Display & downloads ----
    st.subheader("Binary Mask (Objects vs Background)")
    st.image(binary255, use_column_width=True)
    st.download_button(
        "⬇️ Download Binary Mask (PNG)",
        data=to_png_bytes(binary255),
        file_name="binary_mask.png",
        mime="image/png",
    )

    st.subheader("Color Mask (Original Colors on Black Background)")
    st.image(color_mask, use_column_width=True)
    st.download_button(
        "⬇️ Download Color Mask (PNG)",
        data=to_png_bytes(color_mask),
        file_name="color_mask.png",
        mime="image/png",
    )

    # Some quick introspection for confidence
    st.caption(f"Inference time: **{elapsed:.2f}s** • Pred unique classes: {np.unique(pred_classes)[:10]}… "
               f"• Auto background id: **{bg_idx}**")
else:
    st.info("Upload a JPG/PNG to run segmentation.")












