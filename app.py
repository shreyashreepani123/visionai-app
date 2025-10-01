import os
import io
import time
import numpy as np
import cv2
from PIL import Image
import streamlit as st

import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.model_zoo import get_config_file, get_checkpoint_url
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

# ----------------- CONFIG -----------------
st.set_page_config(page_title="COCO Panoptic Segmentation", layout="wide")
st.title("🧠 COCO Panoptic Segmentation (Detectron2)")
st.caption("Panoptic FPN — all COCO classes (80 things + 53 stuff).")

# Random color palette
def build_palette(n=256, seed=42):
    rng = np.random.RandomState(seed)
    pal = rng.randint(0, 255, (n, 3), dtype=np.uint8)
    pal[0] = [0, 0, 0]  # background black
    return pal

PALETTE = build_palette()

# ----------------- LOAD MODEL -----------------
@st.cache_resource
def load_model(thresh=0.5):
    cfg = get_cfg()
    config_path = "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"
    cfg.merge_from_file(get_config_file(config_path))
    cfg.MODEL.WEIGHTS = get_checkpoint_url(config_path)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh
    predictor = DefaultPredictor(cfg)
    return predictor

# ----------------- MASKING -----------------
def run_panoptic(predictor, img_bgr):
    outputs = predictor(img_bgr)
    pan_map, seg_info = outputs["panoptic_seg"]
    return pan_map.cpu().numpy(), seg_info

def make_masks(pan_map, seg_info):
    h, w = pan_map.shape
    binary = np.zeros((h, w), np.uint8)
    color = np.zeros((h, w, 3), np.uint8)
    for seg in seg_info:
        seg_id = seg["id"]
        cid = seg["category_id"]
        mask = pan_map == seg_id
        binary[mask] = 255
        color[mask] = PALETTE[(cid + 1) % len(PALETTE)]
    return binary, color

def overlay_legend(seg_info):
    id2name = {c["id"]: c["name"] for c in COCO_CATEGORIES}
    present = sorted({id2name[s["category_id"]] for s in seg_info if s["category_id"] in id2name})
    if not present:
        return None
    h = 25 * len(present) + 10
    w = 300
    legend = np.full((h, w, 3), 255, np.uint8)
    y = 20
    for name in present:
        cid = next(c["id"] for c in COCO_CATEGORIES if c["name"] == name)
        col = tuple(int(x) for x in PALETTE[(cid + 1) % len(PALETTE)][::-1])
        cv2.rectangle(legend, (10, y - 12), (30, y + 2), col, -1)
        cv2.putText(legend, name, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        y += 25
    return cv2.cvtColor(legend, cv2.COLOR_BGR2RGB)

def to_bytes(arr):
    if arr.ndim == 2:
        pil = Image.fromarray(arr)
    else:
        pil = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()

# ----------------- STREAMLIT -----------------
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
thresh = st.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.05)

if uploaded:
    pil = Image.open(uploaded).convert("RGB")
    rgb = np.array(pil)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    predictor = load_model(thresh)

    t0 = time.time()
    pan_map, seg_info = run_panoptic(predictor, bgr)
    dt = time.time() - t0

    binary, color = make_masks(pan_map, seg_info)
    legend = overlay_legend(seg_info)

    st.write(f"Inference: {dt:.2f}s on {'GPU' if torch.cuda.is_available() else 'CPU'}")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Original")
        st.image(rgb, use_column_width=True)
        st.download_button("⬇ Download", to_bytes(rgb), "original.png")
    with c2:
        st.subheader("Binary Mask")
        st.image(binary, use_column_width=True, clamp=True)
        st.download_button("⬇ Download", to_bytes(binary), "binary.png")
    with c3:
        st.subheader("Color Mask")
        st.image(color, use_column_width=True)
        st.download_button("⬇ Download", to_bytes(color), "color.png")

    if legend is not None:
        st.subheader("Legend")
        st.image(legend, use_column_width=False)

else:
    st.info("Upload a JPG/PNG to run segmentation.")











