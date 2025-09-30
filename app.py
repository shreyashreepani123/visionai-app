# app.py
import os
import io
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from albumentations import Compose, Resize
from albumentations.pytorch import ToTensorV2
from torchvision.models.segmentation import deeplabv3_resnet50
import gdown

# ------------------ CONFIG ------------------
# 1) Put your public Google Drive file ID here (JUST THE ID, not the whole URL)
FILE_ID = "1BD-iZ8b_37vxXCWRNwk-CtxBvXgG97V3"  # e.g. 1Bd-iZ8b_37vxXCWRNwk-CtxBVxgG97V3
CHECKPOINT_PATH = "checkpoint.pth"

# 2) CPU only (Streamlit Cloud has no GPU)
DEVICE = torch.device("cpu")

# 3) Input size for inference (model runs at 256 then we resize mask back)
IMAGE_SIZE = 256

# ------------------ HELPERS ------------------
def image_download_button(img_pil: Image.Image, filename: str, label: str):
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    st.download_button(label, buf.getvalue(), file_name=filename, mime="image/png")


def ensure_checkpoint():
    """
    Ensure checkpoint.pth exists locally. If not, download from Google Drive using the FILE_ID.
    Also validate that a .pth is actually present after download.
    """
    if os.path.exists(CHECKPOINT_PATH) and os.path.getsize(CHECKPOINT_PATH) > 0:
        return

    if not FILE_ID or FILE_ID.strip() in ("", "PUT_YOUR_FILE_ID_HERE"):
        raise RuntimeError("Please set FILE_ID at the top of app.py to your Google Drive file ID.")

    url = f"https://drive.google.com/uc?id={FILE_ID}"
    st.info("⬇️ Downloading model weights from Google Drive...")
    try:
        gdown.download(url, CHECKPOINT_PATH, quiet=False)
    except Exception as e:
        raise RuntimeError(f"gdown failed to download model: {e}")

    # If the file didn't come down as 'checkpoint.pth', try to find any .pth and rename it
    if not os.path.exists(CHECKPOINT_PATH) or os.path.getsize(CHECKPOINT_PATH) == 0:
        for f in os.listdir("."):
            if f.lower().endswith(".pth"):
                try:
                    os.replace(f, CHECKPOINT_PATH)
                    break
                except Exception:
                    pass

    if not os.path.exists(CHECKPOINT_PATH) or os.path.getsize(CHECKPOINT_PATH) == 0:
        raise FileNotFoundError(
            "Download finished, but 'checkpoint.pth' not found or empty. "
            "Check Drive sharing and FILE_ID."
        )

    st.success("✅ Download complete.")


def detect_num_classes(state_dict: dict, fallback: int = 2) -> int:
    """
    Try to infer num_classes from DeepLabV3 head weight shape.
    """
    keys = [
        "classifier.4.weight",
        "module.classifier.4.weight",
    ]
    for k in keys:
        if k in state_dict:
            return int(state_dict[k].shape[0])
    return fallback


@st.cache_resource
def load_model() -> torch.nn.Module:
    """
    Load DeepLabV3-ResNet50 model with weights from checkpoint.pth (CPU).
    """
    ensure_checkpoint()

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    # Support both raw state_dict and wrapped dicts
    state_dict = ckpt.get("model_state", ckpt.get("state_dict", ckpt))

    # Remove a possible 'module.' prefix if the model was saved with DataParallel
    norm_sd = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            norm_sd[k[len("module."):]] = v
        else:
            norm_sd[k] = v

    num_classes = detect_num_classes(norm_sd, fallback=2)

    model = deeplabv3_resnet50(weights=None, aux_loss=True)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    if getattr(model, "aux_classifier", None) is not None:
        model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    load_res = model.load_state_dict(norm_sd, strict=False)
    print("load_state_dict:", load_res)  # will show missing/unexpected keys

    model.to(DEVICE).eval()
    return model


# Albumentations transform
transform = Compose([
    Resize(IMAGE_SIZE, IMAGE_SIZE, interpolation=1),  # 1 == cv2.INTER_NEAREST
    ToTensorV2()
])


def predict_mask(model: torch.nn.Module, pil_img: Image.Image) -> np.ndarray:
    """
    Run segmentation and return a HxW uint8 mask (labels).
    """
    orig_w, orig_h = pil_img.size
    resized = pil_img.resize((IMAGE_SIZE, IMAGE_SIZE))
    augmented = transform(image=np.array(resized))
    x = augmented["image"].unsqueeze(0).float().to(DEVICE)  # [1,3,H,W]

    with torch.no_grad():
        out = model(x)["out"]  # [1, num_classes, H, W]
        preds = torch.argmax(out, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # Resize mask back to original size using nearest to preserve labels
    mask = Image.fromarray(preds).resize((orig_w, orig_h), Image.NEAREST)
    return np.array(mask).astype(np.uint8)


# ------------------ UI ------------------
st.set_page_config(page_title="VisionAI Segmentation", page_icon="🔍", layout="centered")
st.title("🔍 VisionAI Segmentation Demo")
st.caption("Upload an image to see binary and color masking results. Model runs on CPU.")

# Load model (download checkpoint first)
try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    pil_img = Image.open(uploaded).convert("RGB")
    st.image(pil_img, caption="Original Image", use_column_width=True)

    with st.spinner("Running segmentation..."):
        mask = predict_mask(model, pil_img)

    # Binary mask (foreground white, background black)
    bin_mask = (mask != 0).astype(np.uint8) * 255
    bin_mask_pil = Image.fromarray(bin_mask)

    # Color mask (foreground retains original colors, background black)
    orig_np = np.array(pil_img).astype(np.uint8)
    color_np = orig_np.copy()
    color_np[mask == 0] = 0
    color_mask_pil = Image.fromarray(color_np)

    st.subheader("Results")
    c1, c2 = st.columns(2)
    with c1:
        st.image(bin_mask_pil, caption="Binary Mask (B/W)", use_column_width=True)
        image_download_button(bin_mask_pil, "binary_mask.png", "⬇️ Download Binary Mask")
    with c2:
        st.image(color_mask_pil, caption="Color Mask (Background Black)", use_column_width=True)
        image_download_button(color_mask_pil, "color_mask.png", "⬇️ Download Color Mask")
else:
    st.info("👉 Upload a JPG/PNG image to get started.")




