import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet50
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import io
import gdown   # for downloading model from Google Drive

# ---------- CONFIG ----------
# Replace with your actual Google Drive file ID for checkpoint.pth
CHECKPOINT_ID = "YOUR_GOOGLE_DRIVE_FILE_ID"
CHECKPOINT_PATH = "checkpoint.pth"

if not os.path.exists(CHECKPOINT_PATH):
    url = f"https://drive.google.com/uc?id={CHECKPOINT_ID}"
    gdown.download(url, CHECKPOINT_PATH, quiet=False)

IMAGE_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- detect num_classes ----------
def detect_num_classes(sd):
    if "classifier.4.weight" in sd:
        return int(sd["classifier.4.weight"].shape[0])
    return 2

# ---------- load model ----------
@st.cache_resource
def load_model():
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = ckpt.get("model_state", ckpt)
    num_classes = detect_num_classes(state_dict)
    model = deeplabv3_resnet50(weights=None, aux_loss=True)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    if getattr(model, "aux_classifier", None) is not None:
        model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    filtered = {k: v for k, v in state_dict.items() if not k.startswith("aux_classifier")}
    model.load_state_dict(filtered, strict=False)
    return model.to(DEVICE).eval()

model = load_model()
transform = A.Compose([A.Resize(IMAGE_SIZE, IMAGE_SIZE, interpolation=1), ToTensorV2()])

def predict_mask(pil_img):
    orig_w, orig_h = pil_img.size
    resized = pil_img.resize((IMAGE_SIZE, IMAGE_SIZE))
    augmented = transform(image=np.array(resized))
    img_tensor = augmented["image"].unsqueeze(0).to(DEVICE).float()
    with torch.no_grad():
        out = model(img_tensor)["out"]
        preds = torch.argmax(out, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    pred_full = Image.fromarray(preds).resize((orig_w, orig_h), resample=Image.NEAREST)
    return np.array(pred_full).astype(np.uint8)

def image_download_button(img_pil, filename, label):
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    st.download_button(label, buf.getvalue(), file_name=filename, mime="image/png")

# ---------- Streamlit UI ----------
st.title("🔍 VisionAI Segmentation Demo")
st.write("Upload an image to see binary and color masking results.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    pil_img = Image.open(uploaded_file).convert("RGB")
    st.image(pil_img, caption="Original Image", use_column_width=True)
    pred_mask = predict_mask(pil_img)
    orig_np = np.array(pil_img).astype(np.uint8)

    # Binary mask
    bin_mask = (pred_mask != 0).astype(np.uint8) * 255
    bin_mask_pil = Image.fromarray(bin_mask)

    # Color mask
    color_mask = orig_np.copy()
    color_mask[pred_mask == 0] = 0
    color_mask_pil = Image.fromarray(color_mask)

    st.subheader("Results")
    col1, col2 = st.columns(2)
    with col1:
        st.image(bin_mask_pil, caption="Binary Mask", use_column_width=True)
        image_download_button(bin_mask_pil, "binary_mask.png", "⬇️ Download Binary Mask")
    with col2:
        st.image(color_mask_pil, caption="Color Mask", use_column_width=True)
        image_download_button(color_mask_pil, "color_mask.png", "⬇️ Download Color Mask")
