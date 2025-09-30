import os
import io
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet50
import albumentations as A
from albumentations.pytorch import ToTensorV2
import gdown

# ------------------ CONFIG ------------------
# 1) Put your public Google Drive file ID here
FILE_ID = "1BD-iZ8b_37vxXCWRNwk-CtxBvXgG97V3"  # <- e.g. '1Bd-iZ8b_37vxXCWRNwk-CtxBvXgG97V3'
CHECKPOINT_PATH = "checkpoint.pth"

# 2) Force CPU on Streamlit Cloud (no GPU available)
DEVICE = torch.device("cpu")

# 3) Inference image size (model will be resized to this then back to original)
IMAGE_SIZE = 256

# ------------------ UTIL: Download checkpoint ------------------
def ensure_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        return
    if not FILE_ID or FILE_ID.strip() == "YOUR_DRIVE_FILE_ID":
        raise RuntimeError("Please set FILE_ID in app.py to your Google Drive file ID.")
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    st.write("⬇️ Downloading model weights from Google Drive...")
    gdown.download(url, CHECKPOINT_PATH, quiet=False)
    st.write("✅ Download complete.")

# ------------------ detect num_classes from state_dict ------------------
def detect_num_classes(state_dict: dict, fallback: int = 2) -> int:
    # DeeplabV3 classifier head weight has shape [num_classes, 256, 1, 1]
    if "classifier.4.weight" in state_dict:
        return int(state_dict["classifier.4.weight"].shape[0])
    return fallback

# ------------------ load model ------------------
@st.cache_resource
def load_model() -> torch.nn.Module:
    ensure_checkpoint()
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = ckpt.get("model_state", ckpt)  # support raw state_dict or wrapped
    num_classes = detect_num_classes(state_dict, fallback=2)

    model = deeplabv3_resnet50(weights=None, aux_loss=True)
    # replace classifier heads to match num_classes
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    if getattr(model, "aux_classifier", None) is not None:
        model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    # load (ignore aux keys if absent)
    filtered = {k: v for k, v in state_dict.items() if not k.startswith("aux_classifier")}
    load_res = model.load_state_dict(filtered, strict=False)
    print("load_state_dict:", load_res)

    model.to(DEVICE).eval()
    return model

# ------------------ transforms ------------------
transform = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE, interpolation=1),
    ToTensorV2()
])

def predict_mask(model: torch.nn.Module, pil_img: Image.Image) -> np.ndarray:
    orig_w, orig_h = pil_img.size
    resized = pil_img.resize((IMAGE_SIZE, IMAGE_SIZE))
    augmented = transform(image=np.array(resized))
    x = augmented["image"].unsqueeze(0).float()  # [1,3,H,W]
    with torch.no_grad():
        out = model(x)["out"]          # [1, num_classes, H, W]
        preds = torch.argmax(out, 1)   # [1,H,W]
    pred_small = preds.squeeze(0).cpu().numpy().astype(np.uint8)
    # Resize back to original size using nearest to keep labels intact
    pred_full = Image.fromarray(pred_small).resize((orig_w, orig_h), Image.NEAREST)
    return np.array(pred_full).astype(np.uint8)

def image_download_button(img_pil: Image.Image, filename: str, label: str):
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    st.download_button(label, buf.getvalue(), file_name=filename, mime="image/png")

# ------------------ UI ------------------
st.set_page_config(page_title="VisionAI Segmentation", page_icon="🔍", layout="centered")
st.title("🔍 VisionAI Segmentation Demo")
st.caption("Upload an image to see binary and color masking results. Model runs on CPU.")

try:
    model = load_model()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    pil_img = Image.open(uploaded_file).convert("RGB")
    st.image(pil_img, caption="Original Image", use_column_width=True)

    with st.spinner("Running segmentation..."):
        pred_mask = predict_mask(model, pil_img)

    # Binary mask (foreground white, background black)
    bin_mask = (pred_mask != 0).astype(np.uint8) * 255
    bin_mask_pil = Image.fromarray(bin_mask)

    # Color mask (background black, original colors for foreground)
    orig_np = np.array(pil_img).astype(np.uint8)
    color_np = orig_np.copy()
    color_np[pred_mask == 0] = 0
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




