import os
import io
import cv2
import numpy as np
import requests
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models.segmentation as segmodels

import streamlit as st

# ---------------- CONFIG ----------------
MODEL_URL = "https://github.com/shreyashreepani123/visionai-app/releases/download/v1.1/checkpoint.pth"
CHECKPOINT_PATH = "checkpoint.pth"
NUM_CLASSES = 91           # <- must match your training
DEVICE = torch.device("cpu")


# ---------------- Utils ----------------
def ensure_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        return
    r = requests.get(MODEL_URL, timeout=300, allow_redirects=True)
    r.raise_for_status()
    with open(CHECKPOINT_PATH, "wb") as f:
        f.write(r.content)


@st.cache_resource
def load_model():
    ensure_checkpoint()
    model = segmodels.deeplabv3_resnet50(weights=None, num_classes=NUM_CLASSES)
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE).eval()
    return model


# torchvision transform (no resize -> keep native res; upsample logits instead)
_t = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
])


def softmax_probs(model, pil_img: Image.Image) -> np.ndarray:
    """Return per-class probabilities upsampled to original HxW."""
    w, h = pil_img.size
    x = _t(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(x)["out"]                      # [1,C,h',w']
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        probs = torch.softmax(out, dim=1).squeeze(0).cpu().numpy()  # [C,H,W]
    return probs


def biggest_component(mask: np.ndarray, min_keep_ratio=0.002) -> np.ndarray:
    """Keep only the biggest foreground blob; drop tiny noise."""
    H, W = mask.shape
    min_keep = max(1, int(min_keep_ratio * H * W))
    num, lab = cv2.connectedComponents(mask.astype(np.uint8))
    if num <= 1:
        return mask

    sizes = [(lab == i).sum() for i in range(1, num)]
    if not sizes:
        return mask
    i_big = 1 + int(np.argmax(sizes))
    cleaned = (lab == i_big).astype(np.uint8)
    if cleaned.sum() < min_keep:
        return mask
    return cleaned


def prob_guided_refine(image_rgb: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """
    Strong fallback refinement that uses:
      - background index = mode of argmax
      - dynamic foreground threshold using percentiles
      - morphology (close/open), biggest component
    Produces a clean binary mask (0/255).
    """
    H, W, _ = image_rgb.shape
    arg = probs.argmax(0).astype(np.int32)
    bg_idx = int(np.bincount(arg.flatten()).argmax())

    maxp = probs.max(0)  # [H,W]

    # dynamic threshold: keep confident foreground where it is NOT background
    fg0 = (arg != bg_idx)
    if fg0.mean() < 0.0005:     # almost empty -> trust probabilities directly
        thr = max(0.5, float(np.percentile(maxp, 98)))
        fg = (maxp >= thr).astype(np.uint8)
    else:
        thr_hi = float(np.percentile(maxp[fg0], 70))
        thr_lo = max(0.15, float(np.percentile(maxp[~fg0], 90)))
        thr = max(thr_lo, min(0.9, thr_hi))
        fg = (fg0 & (maxp >= thr)).astype(np.uint8)

    # morphology to densify & remove holes
    k3 = np.ones((3, 3), np.uint8)
    k5 = np.ones((5, 5), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k5, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k3, iterations=1)

    # keep largest blob if many speckles
    if fg.sum() > 0:
        fg = biggest_component(fg)

    return (fg * 255).astype(np.uint8)


def try_grabcut(image_rgb: np.ndarray, probs: np.ndarray) -> np.ndarray | None:
    """
    Edge-aware refinement with GrabCut.
    Returns a binary 0/255 mask or None if unsafe / error.
    """
    H, W, _ = image_rgb.shape
    arg = probs.argmax(0).astype(np.int32)
    bg_idx = int(np.bincount(arg.flatten()).argmax())
    maxp = probs.max(0)

    # Build a robust mask for GC_INIT_WITH_MASK:
    # default: probable background
    mask = np.full((H, W), cv2.GC_PR_BGD, dtype=np.uint8)

    # sure background = very low prob
    mask[maxp < 0.1] = cv2.GC_BGD

    # probable fg where not background and reasonably confident
    pr_fg = (arg != bg_idx) & (maxp >= 0.35)
    mask[pr_fg] = cv2.GC_PR_FGD

    # sure foreground = eroded core of confident area
    sure_fg = (arg != bg_idx) & (maxp >= 0.65)
    if sure_fg.any():
        core = cv2.erode(sure_fg.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
        mask[core.astype(bool)] = cv2.GC_FGD

    # Safety checks (GrabCut fails if all same label or extreme imbalance)
    n_fg = int(((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).sum())
    n_bg = int(((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD)).sum())
    if n_fg < 50 or n_bg < 50:   # too little information -> skip
        return None

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Try GC with mask; if it throws, we return None
    try:
        cv2.grabCut(image_rgb, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
        gc = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        return gc
    except cv2.error:
        return None


def color_mask_from_binary(image_rgb: np.ndarray, binary_255: np.ndarray) -> np.ndarray:
    out = image_rgb.copy()
    out[binary_255 == 0] = 0
    return out


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="VisionAI Segmentation (Refined, No-CRF)", layout="centered")
st.title("🔍 VisionAI Segmentation — Best-Quality (Your Weights Only)")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    pil = Image.open(uploaded).convert("RGB")
    img = np.array(pil)

    st.subheader("Uploaded Image")
    st.image(img, use_column_width=True)

    model = load_model()
    probs = softmax_probs(model, pil)  # [C,H,W]

    # 1) Try edge-aware GrabCut (safe)
    gc_mask = try_grabcut(img, probs)

    # 2) If not available, use strong probability-guided refinement
    if gc_mask is None:
        binary = prob_guided_refine(img, probs)
        used = "Probability-guided refinement"
    else:
        binary = gc_mask
        used = "GrabCut refinement"

    st.caption(f"Refinement used: **{used}**")

    # Color mask
    color_mask = color_mask_from_binary(img, binary)

    # Display
    st.subheader("Binary Mask (Refined)")
    st.image(binary, use_column_width=True)

    st.download_button(
        "⬇️ Download Binary Mask (PNG)",
        data=cv2.imencode(".png", binary)[1].tobytes(),
        file_name="binary_mask.png",
        mime="image/png",
    )

    st.subheader("Color Mask (Objects on Black Background, Refined)")
    st.image(color_mask, use_column_width=True)

    st.download_button(
        "⬇️ Download Color Mask (PNG)",
        data=cv2.imencode(".png", cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))[1].tobytes(),
        file_name="color_mask.png",
        mime="image/png",
    )











