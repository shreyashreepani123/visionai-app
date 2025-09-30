import os
import torch
import torchvision.transforms as T
from PIL import Image
import streamlit as st
import requests
import numpy as np
import torchvision.models.segmentation as models

# ---------------- CONFIG ----------------
CHECKPOINT_PATH = "checkpoint.pth"
MODEL_URL = "https://github.com/shreyashreepani123/visionai-app/releases/download/v1.1/checkpoint.pth"
DEVICE = torch.device("cpu")
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
    model = models.deeplabv3_resnet50(weights=None, num_classes=21)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    # Flexible key handling
    if "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    st.write("⚠️ Missing keys:", missing)
    st.write("⚠️ Unexpected keys:", unexpected)

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
        output = model(img_tensor)["out"][0]   # DeepLab output

    # Segmentation prediction
    mask = output.argmax(0).cpu().numpy()

    # --- Binary mask (object=white, background=black) ---
    binary_mask = (mask > 0).astype(np.uint8) * 255  # 0=background, 255=object

    # --- Colored masking (object keeps real colors, background black) ---
    img_resized = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_np = np.array(img_resized)
    color_mask = img_np.copy()
    color_mask[mask == 0] = [0, 0, 0]  # set background to black

    # Show results
    st.image(binary_mask, caption="Binary Segmentation (White=Object, Black=Background)", use_column_width=True)
    st.image(color_mask, caption="Colored Masking (Object in Original Colors, Background Black)", use_column_width=True)







