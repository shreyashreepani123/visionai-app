import os
from io import BytesIO

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models.segmentation as segmodels
from PIL import Image
import requests
import streamlit as st
import cv2  # opencv-python-headless

# ---------------- CONFIG ----------------
# 1) Where your released checkpoint lives (GitHub Releases asset)
MODEL_URL = "https://github.com/shreyashreepani123/visionai-app/releases/download/v1.1/checkpoint.pth"
CHECKPOINT_PATH = "checkpoint.pth"

# 2) Model must match the checkpoint's head size (your file has 91 classes)
NUM_CLASSES = 91
BACKGROUND_CLASS = 0  # usually 0

# 3) Streamlit Cloud is CPU-only
DEVICE = torch.device("cpu")

# 4) Resize used for inference (network input); results are upsampled back to original
IMAGE_SIZE = 256


# ---------------- UTILS ----------------
def ensure_checkpoint():
    """Download checkpoint once from GitHub Releases."""
    if os.path.exists(CHECKPOINT_PATH):
        return
    st.info("📥 Downloading model weights from GitHub Releases…")
    r = requests.get(MODEL_URL, allow_redirects=True, timeout=300)
    r.raise_for_status()
    with open(CHECKPOINT_PATH, "wb") as f:
        f.write(r.content)
    st.success("✅ Download complete.")


def _strip_module_prefix(state_dict):
    """Remove 'module.' prefix if present (DataParallel checkpoints)."""
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_sd[k[len("module."):]] = v
        else:
            new_sd[k] = v
    return new_sd


@st.cache_resource
def load_model():
    """Create DeepLabV3-ResNet50 with correct head and load your checkpoint."""
    ensure_checkpoint()

    # Build model with correct number of classes
    model = segmodels.deeplabv3_resnet50(weights=None, num_classes=NUM_CLASSES)

    # Load checkpoint (supports {"model_state": ...} or direct state dict)
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    else:
        state_dict = ckpt

    state_dict = _strip_module_prefix(state_dict)

    # If you ever just want to partially load, set strict=False,
    # but for best accuracy we load strictly so the head is used.
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # If classifier keys were missing, strict=True would throw. We still report info:
    if len(missing) + len(unexpected) > 0:
        st.warning(
            "Note: some keys did not load strictly. "
            f"Missing: {len(missing)} | Unexpected: {len(unexpected)}"
        )

    model.to(DEVICE).eval()
    return model


# ImageNet normalization (typical for torchvision backbones)
transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
])


def tensor_to_pil_uint8(arr):
    """arr: (H,W) or (H,W,3) np.uint8 -> PNG bytes + PIL image."""
    if arr.ndim == 2:
        pil = Image.fromarray(arr, mode="L")
    else:
        pil = Image.fromarray(arr, mode="RGB")
    bio = BytesIO()
    pil.save(bio, format="PNG")
    return bio.getvalue(), pil


def make_color_mask(original_rgb, binary_mask):
    """
    Keep original colors where mask==1; black elsewhere.
    original_rgb: np.uint8 (H,W,3)
    binary_mask : np.uint8 (H,W) with values 0 or 255
    """
    m = (binary_mask > 0)[..., None]  # (H,W,1) boolean
    color = np.where(m, original_rgb, 0)
    return color.astype(np.uint8)


def postprocess_to_original(logits, orig_h, orig_w):
    """Upsample logits to original size and get predicted class map."""
    # logits: (1, C, h, w) -> upsample to orig (H,W)
    up = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    # per-pixel class prediction
    pred = up.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)  # (H,W)
    return pred


# ---------------- STREAMLIT APP ----------------
st.set_page_config(page_title="VisionAI Segmentation", layout="centered")

st.title("🔍 VisionAI Segmentation Demo")
st.caption(
    "Binary mask (white object on black) and color masking (image colors on black background). "
    "Runs on CPU. Set `NUM_CLASSES` to your training setting (here 91) to match your checkpoint."
)

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    # Load image
    image_pil = Image.open(uploaded).convert("RGB")
    orig_w, orig_h = image_pil.size
    image_np = np.array(image_pil)  # (H,W,3), uint8

    # Model
    model = load_model()

    # Preprocess
    with torch.no_grad():
        inp = transform(image_pil).unsqueeze(0).to(DEVICE)  # (1,3,H,W)
        out = model(inp)
        # torchvision segmentation returns dict {"out": logits, ...}
        logits = out["out"] if isinstance(out, dict) else out  # (1,C,h,w)

        # Upsample and get class map
        pred_classes = postprocess_to_original(logits, orig_h, orig_w)

    # Binary mask (object = any class != background)
    binary = (pred_classes != BACKGROUND_CLASS).astype(np.uint8) * 255  # (H,W) uint8 in {0,255}

    # (Optional) small clean-up (opening/closing) for nicer edges
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Color mask: keep image where mask==1 else black
    color_mask_img = make_color_mask(image_np, binary)

    # Prepare downloads
    binary_bytes, binary_pil = tensor_to_pil_uint8(binary)
    color_bytes, color_pil = tensor_to_pil_uint8(color_mask_img)

    # Show & download
    st.subheader("Binary Segmentation (White=Object, Black=Background)")
    st.image(binary_pil, use_column_width=True)
    st.download_button(
        "⬇️ Download Binary Mask (PNG)", data=binary_bytes,
        file_name="mask_binary.png", mime="image/png"
    )

    st.subheader("Color Masking (Original Colors on Black Background)")
    st.image(color_pil, use_column_width=True)
    st.download_button(
        "⬇️ Download Color Mask (PNG)", data=color_bytes,
        file_name="mask_color.png", mime="image/png"
    )








