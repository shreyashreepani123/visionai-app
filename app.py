import streamlit as st
import torch
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image
import io
import base64

# ---------------- CONFIG ----------------
CHECKPOINT_PATH = "checkpoint.pth"
NUM_CLASSES = 91  # COCO style

DEVICE = torch.device("cpu")
IMAGE_SIZE = 512  # upscale for better visualization


# ---------------- MODEL LOADER ----------------
@st.cache_resource
def load_model():
    # Load your checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    state_dict = checkpoint["model_state"]

    # Load a pretrained DeepLabv3 and adjust for your classes
    from torchvision.models.segmentation import deeplabv3_resnet50
    model = deeplabv3_resnet50(weights=None, num_classes=NUM_CLASSES)
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE).eval()
    return model


# ---------------- HELPERS ----------------
def preprocess(img: Image.Image):
    transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
    ])
    return transform(img).unsqueeze(0)


def decode_segmap(mask, binary=False):
    """Convert mask into color image"""
    if binary:
        mask = (mask > 0).astype(np.uint8) * 255
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    else:
        # Random color map for each class
        h, w = mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        np.random.seed(42)  # fixed colors
        colors = np.random.randint(0, 255, size=(NUM_CLASSES, 3), dtype=np.uint8)
        for cls_id in np.unique(mask):
            if cls_id == 0:  # background
                continue
            color_mask[mask == cls_id] = colors[cls_id]
        return color_mask


def add_download_button(result_img, filename="segmentation.png"):
    """Adds a download button for the segmented image"""
    pil_img = Image.fromarray(result_img)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    byte_im = buf.getvalue()

    b64 = base64.b64encode(byte_im).decode()
    href = f'<a href="data:file/png;base64,{b64}" download="{filename}">📥 Download result</a>'
    st.markdown(href, unsafe_allow_html=True)


# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="VisionAI Segmentation", layout="wide")
st.title("🔍 VisionAI Segmentation Demo")
st.write("Upload an image to see **binary** and **colored multi-class** segmentation. Model runs on CPU.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model = load_model()
    tensor = preprocess(image).to(DEVICE)

    with torch.no_grad():
        output = model(tensor)["out"]
        pred = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    # Binary mask
    binary_mask = decode_segmap(pred, binary=True)
    # Colored segmentation
    color_mask = decode_segmap(pred, binary=False)

    st.subheader("Binary Segmentation")
    st.image(binary_mask, use_column_width=True)
    add_download_button(binary_mask, filename="binary_segmentation.png")

    st.subheader("Colored Multi-class Segmentation")
    st.image(color_mask, use_column_width=True)
    add_download_button(color_mask, filename="colored_segmentation.png")

else:
    st.info("⬆️ Please upload an image to start.")











