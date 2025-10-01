import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models.segmentation as segmodels
import numpy as np
import cv2
from PIL import Image
import streamlit as st

# ---------------- CONFIG ----------------
CHECKPOINT_PATH = "checkpoint.pth"   # Your uploaded model file
DEVICE = torch.device("cpu")
IMAGE_SIZE = 256

# COCO color palette (91 classes, person=red, car=blue, etc.)
# Taken from COCO API (shortened here for clarity)
COCO_COLORS = np.array([
    [0, 0, 0],        # background
    [220, 20, 60],    # person
    [119, 11, 32],    # bicycle
    [0, 0, 142],      # car
    [0, 0, 230],      # motorcycle
    [106, 0, 228],    # airplane
    [0, 60, 100],     # bus
    [0, 80, 100],     # train
    [0, 0, 70],       # truck
    [0, 0, 192],      # boat
    # ... extend to 91
], dtype=np.uint8)


# ---------------- MODEL ----------------
def build_deeplab(num_classes=91):
    model = segmodels.deeplabv3_resnet50(weights=None, aux_loss=False)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model

@st.cache_resource
def load_model():
    model = build_deeplab(num_classes=91)
    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state["model_state"], strict=False)
    model.to(DEVICE).eval()
    return model

# ---------------- TRANSFORM ----------------
transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# ---------------- POSTPROCESS ----------------
def postprocess_logits(logits, orig_h, orig_w, image_np):
    probs = torch.softmax(logits, dim=1)  # (1, 91, h, w)
    up = F.interpolate(probs, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    mask = up.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)  # (H, W)

    # Binary mask
    binary = (mask > 0).astype(np.uint8) * 255

    # Color mask with original image colors
    color_mask = np.zeros_like(image_np)
    color_mask[mask > 0] = image_np[mask > 0]

    # Multi-class mask with fixed COCO colors
    class_colored = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    for cls_id in np.unique(mask):
        if cls_id < len(COCO_COLORS):
            class_colored[mask == cls_id] = COCO_COLORS[cls_id]

    return binary, color_mask, class_colored

# ---------------- STREAMLIT APP ----------------
st.title("VisionAI Segmentation (COCO 91 classes)")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    orig_h, orig_w = image_np.shape[:2]

    st.image(image, caption="Uploaded Image", use_column_width=True)

    model = load_model()

    # Preprocess
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        logits = model(input_tensor)["out"]

    # Postprocess
    binary, color_mask, class_colored = postprocess_logits(logits, orig_h, orig_w, image_np)

    # Display
    st.subheader("Binary Mask (objects vs background)")
    st.image(binary, clamp=True, channels="GRAY")

    st.subheader("Color Mask (original image colors)")
    st.image(color_mask, clamp=True)

    st.subheader("Multi-class Mask (COCO colors)")
    st.image(class_colored, clamp=True)

    # Download buttons
    cv2.imwrite("binary_mask.png", binary)
    cv2.imwrite("color_mask.png", cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
    cv2.imwrite("class_mask.png", cv2.cvtColor(class_colored, cv2.COLOR_RGB2BGR))

    with open("binary_mask.png", "rb") as f:
        st.download_button("⬇ Download Binary Mask", f, file_name="binary_mask.png")
    with open("color_mask.png", "rb") as f:
        st.download_button("⬇ Download Color Mask", f, file_name="color_mask.png")
    with open("class_mask.png", "rb") as f:
        st.download_button("⬇ Download Multi-class Mask", f, file_name="class_mask.png")







