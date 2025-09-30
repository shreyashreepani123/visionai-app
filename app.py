import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import streamlit as st
import gdown

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
    gdown.download(MODEL_URL, CHECKPOINT_PATH, quiet=False)
    st.write("✅ Download complete.")


# ---------------- MODEL ----------------
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.enc1 = nn.Conv2d(3, 64, 3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dec1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x1 = F.relu(self.enc1(x))
        x2 = self.pool(x1)
        x2 = F.relu(self.enc2(x2))
        x3 = self.dec1(x2)
        x = torch.sigmoid(self.out(x3))
        return x


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    ensure_checkpoint()
    model = SimpleUNet().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE)["model_state"])
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
st.write("Upload an image to see binary and color masking results. Model runs on CPU.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model = load_model()

    # Preprocess
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)[0][0].cpu()

    # Convert to binary mask
    mask = (output > 0.5).float()

    # Show results
    st.image(mask.numpy(), caption="Predicted Mask", use_column_width=True)








