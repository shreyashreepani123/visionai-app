import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models.segmentation as segmodels
from PIL import Image
import requests
import streamlit as st
import numpy as np
import cv2
from io import BytesIO

# ---------------- CONFIG ----------------
MODEL_URL = "https://github.com/shreyashreepani123/visionai-app/releases/download/v1.1/checkpoint.pth"
CHECKPOINT_PATH = "checkpoint.pth"
NUM_CLASSES = 91   # must match training
DEVICE = torch.device("cpu")


# ---------------- DOWNLOAD MODEL ----------------
def ensure_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        return
    r = requests.get(MODEL_URL, allow_redirects=True, timeout=300)
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


# ---------------- TRANSFORMS ----------------
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
])


def predict(model, pil_img):
    """Return raw probability maps"""
    w, h = pil_img.size
    inp = transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(inp)["out"]
    out_up = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
    probs = torch.softmax(out_up, dim=1).squeeze(0).cpu().numpy()
    return probs


def refine_with_grabcut(image_np, probs):
    """Use GrabCut guided by model mask"""
    pred_classes = np.argmax(probs, axis=0).astype(np.uint8)

    # Initial mask for GrabCut
    mask = np.where(pred_classes > 0, cv2.GC_PR_FGD, cv2.GC_BGD).astype("uint8")

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Run GrabCut refinement
    cv2.grabCut(image_np, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)

    final_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")
    return final_mask


# ---------------- STREAMLIT APP ----------------
st.set_page_config(page_title="VisionAI Segmentation", layout="centered")
st.title("🔍 VisionAI Segmentation Demo (GrabCut Enhanced)")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    image_pil = Image.open(uploaded).convert("RGB")
    image_np = np.array(image_pil)

    # Show uploaded image
    st.subheader("Uploaded Image")
    st.image(image_np, use_column_width=True)

    # Load model
    model = load_model()

    # Inference (get probabilities)
    probs = predict(model, image_pil)

    # Refinement with GrabCut
    binary = refine_with_grabcut(image_np, probs)

    # Color mask overlay
    color_mask = image_np.copy()
    color_mask[binary == 0] = 0  # keep only foreground

    # ---------------- DISPLAY ----------------
    st.subheader("Binary Mask (GrabCut Refined)")
    st.image(binary, use_column_width=True)

    st.subheader("Color Mask (GrabCut Refined)")
    st.image(color_mask, use_column_width=True)

    st.download_button(
        "⬇ Download Binary Mask (PNG)",
        data=BytesIO(cv2.imencode(".png", binary)[1].tobytes()),
        file_name="binary_mask.png",
        mime="image/png",
    )

    st.download_button(
        "⬇ Download Color Mask (PNG)",
        data=BytesIO(cv2.imencode(".png", cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))[1].tobytes()),
        file_name="color_mask.png",
        mime="image/png",
    )









