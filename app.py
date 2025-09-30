import os
import cv2
import numpy as np
import requests
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models.segmentation as segmodels

import streamlit as st

# DenseCRF
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

# ---------------- CONFIG ----------------
MODEL_URL = "https://github.com/shreyashreepani123/visionai-app/releases/download/v1.1/checkpoint.pth"
CHECKPOINT_PATH = "checkpoint.pth"
NUM_CLASSES = 91          # change to your model's number of classes
DEVICE = torch.device("cpu")

# ---------------- Utils ----------------
def ensure_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        return
    st.write("⬇️ Downloading checkpoint from GitHub release...")
    r = requests.get(MODEL_URL, timeout=300, allow_redirects=True)
    r.raise_for_status()
    with open(CHECKPOINT_PATH, "wb") as f:
        f.write(r.content)
    st.write("✅ Download complete!")


@st.cache_resource
def load_model():
    ensure_checkpoint()
    model = segmodels.deeplabv3_resnet50(weights=None, num_classes=NUM_CLASSES)
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE).eval()
    return model


# torchvision transform
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


# ---------- CRF + Morphology Refinement ----------
def apply_dense_crf(image: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """
    Refine segmentation with DenseCRF.
    image: RGB image
    probs: [C, H, W] softmax probabilities
    """
    H, W = image.shape[:2]
    n_classes = probs.shape[0]

    d = dcrf.DenseCRF2D(W, H, n_classes)

    # Unary from softmax
    U = unary_from_softmax(probs)
    d.setUnaryEnergy(U)

    # Pairwise terms
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=50, srgb=13, rgbim=image, compat=10)

    Q = d.inference(5)  # iterations
    refined = np.array(Q).reshape((n_classes, H, W))
    return refined


def enhance_mask(image: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Combines CRF + morphology to clean mask."""
    crf_probs = apply_dense_crf(image, probs)
    argmax_map = np.argmax(crf_probs, axis=0).astype(np.uint8)

    # Background = most common class
    bg_idx = int(np.bincount(argmax_map.flatten()).argmax())
    mask = (argmax_map != bg_idx).astype(np.uint8) * 255

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # fill holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # remove noise

    return mask


def color_mask_from_binary(image_rgb: np.ndarray, binary_255: np.ndarray) -> np.ndarray:
    out = image_rgb.copy()
    out[binary_255 == 0] = 0
    return out


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="VisionAI Segmentation (Enhanced)", layout="centered")
st.title("🔍 VisionAI Segmentation Demo (Enhanced with CRF + Morphology)")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    pil = Image.open(uploaded).convert("RGB")
    img = np.array(pil)

    st.subheader("Uploaded Image")
    st.image(img, use_column_width=True)

    model = load_model()
    probs = softmax_probs(model, pil)

    # --- Enhanced refinement ---
    binary = enhance_mask(img, probs)
    color_mask = color_mask_from_binary(img, binary)

    st.subheader("Binary Mask (Enhanced)")
    st.image(binary, use_column_width=True)

    st.download_button(
        "⬇️ Download Binary Mask (PNG)",
        data=cv2.imencode(".png", binary)[1].tobytes(),
        file_name="binary_mask.png",
        mime="image/png",
    )

    st.subheader("Color Mask (Objects on Black Background)")
    st.image(color_mask, use_column_width=True)

    st.download_button(
        "⬇️ Download Color Mask (PNG)",
        data=cv2.imencode(".png", cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))[1].tobytes(),
        file_name="color_mask.png",
        mime="image/png",
    )












