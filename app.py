# app.py
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
from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from detectron2.utils.visualizer import ColorMode

# ----------------------------- UI SETUP -----------------------------
st.set_page_config(page_title="COCO Panoptic Segmentation (Detectron2)", layout="wide")
st.title("🧠 COCO Panoptic Segmentation (Detectron2)")
st.caption("Panoptic FPN — all COCO classes (things + stuff). CPU-compatible. No fake checkpoints.")

# --------------------------- CONFIG / MODEL -------------------------
@st.cache_resource
def load_predictor(score_thresh: float = 0.5):
    """
    Load Detectron2 Panoptic FPN model on CPU (or GPU if available).
    """
    cfg = get_cfg()
    # Panoptic FPN (things + stuff)
    from detectron2.model_zoo import get_config_file, get_checkpoint_url
    config_path = "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"
    cfg.merge_from_file(get_config_file(config_path))
    cfg.MODEL.WEIGHTS = get_checkpoint_url(config_path)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.SEM_SEG_HEAD.SCORE_THRESH_TEST = score_thresh
    predictor = DefaultPredictor(cfg)

    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0]) if len(cfg.DATASETS.TEST) else MetadataCatalog.get("coco_2017_val_panoptic")
    return predictor, metadata

def pil_to_cv2(pil_img: Image.Image):
    arr = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def cv2_to_pil(arr_bgr: np.ndarray):
    rgb = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def to_png_bytes(arr_rgb_or_gray: np.ndarray) -> bytes:
    if arr_rgb_or_gray.ndim == 2:
        pil = Image.fromarray(arr_rgb_or_gray)
    else:
        pil = Image.fromarray(arr_rgb_or_gray)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()

def build_color_palette(n: int, seed: int = 42):
    rng = np.random.RandomState(seed)
    pal = rng.randint(0, 255, size=(n, 3), dtype=np.uint8)
    pal[0] = np.array([0, 0, 0], dtype=np.uint8)  # background black
    return pal

# 133 categories in COCO panoptic (80 thing + 53 stuff). We’ll build a palette large enough.
PALETTE = build_color_palette(256)

# ----------------------------- INFERENCE ----------------------------
def run_panoptic(predictor: DefaultPredictor, image_bgr: np.ndarray):
    """
    Returns:
      panoptic_map: HxW int map (segment ids)
      segments_info: list of dicts (id, category_id, isthing, area, etc.)
    """
    outputs = predictor(image_bgr)
    # For Panoptic FPN, predictor returns:
    # outputs["panoptic_seg"] = (panoptic_map (H,W), segments_info (list))
    panoptic_map, segments_info = outputs["panoptic_seg"]
    panoptic_map = panoptic_map.to("cpu").numpy().astype(np.int32)
    return panoptic_map, segments_info

def make_binary_and_color_masks(panoptic_map: np.ndarray, segments_info: list, image_rgb: np.ndarray):
    """
    Build:
      - Binary mask (objects vs background): white for ANY segment, black for background
      - Multi-class colored mask: each segment colored by its category id
    """
    H, W = panoptic_map.shape
    binary = np.zeros((H, W), dtype=np.uint8)
    color = np.zeros((H, W, 3), dtype=np.uint8)

    # If no segments, return all black
    if len(segments_info) == 0:
        return binary, color

    for seg in segments_info:
        seg_id = seg["id"]
        cat_id = seg["category_id"]  # 0..132 COCO panoptic taxonomy
        mask = (panoptic_map == seg_id)
        if mask.any():
            binary[mask] = 255
            color[mask] = PALETTE[(cat_id + 1) % len(PALETTE)]  # shift so background stays 0

    # Morphological cleanup for binary (optional and light)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return binary, color

def overlay_legend(color_mask: np.ndarray, segments_info: list):
    """
    Draw a small legend image listing class names present in the mask.
    """
    # Map COCO category id -> name
    id_to_name = {c["id"]: c["name"] for c in COCO_CATEGORIES}
    # Collect present categories
    present = []
    seen = set()
    for s in segments_info:
        cid = s["category_id"]
        if cid not in seen:
            seen.add(cid)
            present.append(id_to_name.get(cid, f"id_{cid}"))
    present = sorted(present)

    if not present:
        return None

    # Build a simple legend image
    h = 22 * len(present) + 12
    w = 360
    legend = np.full((h, w, 3), 255, dtype=np.uint8)
    y = 10
    for s in segments_info:
        cid = s["category_id"]
        name = id_to_name.get(cid, f"id_{cid}")
        if name in present:
            color = (int(PALETTE[(cid + 1) % len(PALETTE)][2]),
                     int(PALETTE[(cid + 1) % len(PALETTE)][1]),
                     int(PALETTE[(cid + 1) % len(PALETTE)][0]))
            cv2.rectangle(legend, (10, y), (30, y + 14), color, -1)
            cv2.putText(legend, name, (40, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            y += 22
            present.remove(name)
    return legend

# ------------------------------ UI ----------------------------------
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
score_thresh = st.slider("Confidence threshold", 0.1, 0.9, 0.5, 0.05)

if uploaded is not None:
    pil = Image.open(uploaded).convert("RGB")
    rgb = np.array(pil)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    predictor, metadata = load_predictor(score_thresh)
    t0 = time.time()
    pan_map, seg_info = run_panoptic(predictor, bgr)
    dt = time.time() - t0

    binary, color = make_binary_and_color_masks(pan_map, seg_info, rgb)
    legend = overlay_legend(color, seg_info)

    st.write(f"Inference: **{dt:.2f}s** on {'GPU' if torch.cuda.is_available() else 'CPU'}")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Original")
        st.image(rgb, use_column_width=True)
        st.download_button("⬇️ Download Original",
                           data=to_png_bytes(rgb),
                           file_name="original.png",
                           mime="image/png")
    with c2:
        st.subheader("Binary (objects vs background)")
        st.image(binary, use_column_width=True, clamp=True)
        st.download_button("⬇️ Download Binary Mask",
                           data=to_png_bytes(binary),
                           file_name="binary_mask.png",
                           mime="image/png")
    with c3:
        st.subheader("Multi-class (COCO panoptic)")
        st.image(color, use_column_width=True)
        st.download_button("⬇️ Download Color Mask",
                           data=to_png_bytes(color),
                           file_name="color_mask.png",
                           mime="image/png")

    if legend is not None:
        st.subheader("Legend (present classes)")
        st.image(cv2.cvtColor(legend, cv2.COLOR_BGR2RGB), use_column_width=False)

else:
    st.info("Upload a JPG/PNG image to start.")














