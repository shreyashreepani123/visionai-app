import os
import torch
import torchvision.transforms as T
from PIL import Image
import streamlit as st
import requests

import torchvision.models.segmentation as models

# ---------------- CONFIG ----------------
# 1) Model download from GitHub Release
CHECKPOINT_PATH = "checkpoint.pth"
MODEL_URL = "https://github.com/shreyashreepani123/visionai-app/releases/download/v1.1/checkpoint.pth"

# 2) Force CPU on Streamlit Cloud (no GPU available)
DEVICE = torch.device("cpu")

# 3) Inference image size (model will be resized to this then back to original)
IMAGE_SIZE = 256


# ---------------- UTIL: Download checkpoint ----------------
def ensure_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        return
    st.write("📥 Downloading model weights from GitHub Releases...")
    r = requests.get(MODEL_URL, allow_redirects=True)
    open(CHECKPOINT_PATH, 'wb').write(r.content)
    st.write("✅ Download complete.")


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    ensure_checkpoint()
    # Use DeepLabV3 with ResNet50 backbone (matches backbone keys in checkpoint)
    model = models.deeplabv3_resnet50(weights=None, num_classes=21)  # change num_classes if needed
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(DEVICE)
    model.eval()
    return model


# ---------------- IMAGE TRANSFORMS ----------------
transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
])


# ---------------- STREAMLIT APP ----------------
st.set_page_config(page_title="VisionAI: Image Segmentation Demo", layout="centered")

st.title("🔍 VisionAI Segmentation Demo")
st.write("Upload an image to see segmentation results. Model runs on CPU.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model = load_model()

    # Preprocess
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)["out"][0]   # DeepLab outputs dict with "out"

    # Take argmax across classes for segmentation map
    mask = output.argmax(0).cpu().numpy()

    st.image(mask, caption="Predicted Segmentation Mask", use_column_width=True)







