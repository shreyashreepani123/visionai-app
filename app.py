# app.py
import os
import io
import base64
import requests
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50
import streamlit as st

# ------------- CONFIG -------------
MODEL_URL = "https://github.com/shreyashreepani123/visionai-app/releases/download/v1.1/checkpoint.pth"
CHECKPOINT_PATH = "checkpoint.pth"          # downloaded here if missing
DEVICE = torch.device("cpu")                # Streamlit Cloud CPU
IMAGE_SIZE = 512                            # inference size
FALLBACK_NUM_CLASSES = 91                   # COCO-style

# ------------- UTIL: ensure weights -------------
def ensure_checkpoint():
    if os.path.exists(CHECKPOINT_PATH) and os.path.getsize(CHECKPOINT_PATH) > 0:
        return
    st.info("Downloading checkpoint…")
    r = requests.get(MODEL_URL, timeout=300, allow_redirects=True)
    r.raise_for_status()
    with open(CHECKPOINT_PATH, "wb") as f:
        f.write(r.content)

# ------------- UTIL: state_dict helpers ----------
def strip_module_prefix(state_dict):
    """remove 'module.' that appears when trained with DataParallel/DDP."""
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_sd[k[len("module."):]] = v
        else:
            new_sd[k] = v
    return new_sd

def detect_num_classes(state_dict, default_nc=FALLBACK_NUM_CLASSES):
    # Torchvision head is classifier[4] (Conv2d) with out_channels = num_classes
    key = "classifier.4.weight"
    if key in state_dict:
        return int(state_dict[key].shape[0])
    return default_nc

# ------------- MODEL LOADER (cached) -------------
@st.cache_resource
def load_model():
    ensure_checkpoint()
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state_dict = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    state_dict = strip_module_prefix(state_dict)
    num_classes = detect_num_classes(state_dict, FALLBACK_NUM_CLASSES)

    model = deeplabv3_resnet50(weights=None, num_classes=num_classes)
    # Aux classifier safety: ignore unexpected aux keys
    load_res = model.load_state_dict(state_dict, strict=False)
    # (optional) print to console: st.write(load_res)
    model.to(DEVICE).eval()
    return model, num_classes

# ------------- PRE/POST PROCESS -------------
to_tensor = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),                  # your training used ToTensorV2 => no mean/std normalization
])

def run_tta_logits(model, pil_img, scales=(0.75, 1.0, 1.25), do_flip=True):
    """Multi-scale + optional horizontal flip TTA; accumulates logits."""
    w0, h0 = pil_img.size
    acc = None
    for s in scales:
        # resize *from original* each time to avoid cumulative resampling
        size = (int(IMAGE_SIZE * s), int(IMAGE_SIZE * s))
        resized = pil_img.resize(size, Image.BILINEAR)
        x = T.ToTensor()(resized).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = model(x)["out"]                              # (1, C, h, w)
            logits = out

            if do_flip:
                xf = torch.flip(x, dims=[3])                  # horizontal flip
                out_f = model(xf)["out"]
                out_f = torch.flip(out_f, dims=[3])           # unflip logits back
                logits = logits + out_f

            # resize logits back to IMAGE_SIZE for consistent accumulation
            logits = F.interpolate(logits, size=(IMAGE_SIZE, IMAGE_SIZE),
                                   mode="bilinear", align_corners=False)

        acc = logits if acc is None else acc + logits

    # upsample logits to original image size
    logits_full = F.interpolate(acc, size=(h0, w0), mode="bilinear", align_corners=False)
    return logits_full.squeeze(0)  # (C, H, W)

def logits_to_pred_classes(logits_chw):
    pred = torch.argmax(logits_chw, dim=0)     # (H, W) long
    return pred.cpu().numpy().astype(np.int32)

def get_binary_mask_from_multiclass(pred_classes):
    # Treat the most-frequent label as background (robust autodetect)
    bg = int(np.bincount(pred_classes.reshape(-1)).argmax())
    binary = (pred_classes != bg).astype(np.uint8) * 255
    return binary, bg

def refine_binary_mask(binary, k_close=5, k_open=3, min_area=200):
    """Morphology + small component removal."""
    mask = binary.copy()
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k_close, k_close), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((k_open,  k_open),  np.uint8))

    # remove small blobs
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, n):  # skip background 0
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 255
    return out

def grabcut_refine(image_rgb, binary_mask):
    """Safe GrabCut using seeded sure-foreground/background."""
    try:
        h, w = binary_mask.shape
        gc_mask = np.where(binary_mask > 0, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype('uint8')
        # seeds
        kernel = np.ones((5,5), np.uint8)
        sure_fg = cv2.erode((binary_mask>0).astype(np.uint8)*255, kernel, 2)
        sure_bg = cv2.dilate((binary_mask==0).astype(np.uint8)*255, kernel, 2)
        gc_mask[sure_fg>0] = cv2.GC_FGD
        gc_mask[sure_bg>0] = cv2.GC_BGD

        bgdModel = np.zeros((1,65), np.float64)
        fgdModel = np.zeros((1,65), np.float64)
        cv2.grabCut(image_rgb, gc_mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

        refined = np.where((gc_mask==cv2.GC_FGD) | (gc_mask==cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        return refined
    except Exception:
        # if anything goes wrong, just return original
        return binary_mask

def colorize_multiclass(mask, num_classes):
    """Generate a stable color palette and colorize mask (H, W)->(H,W,3)."""
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    rng = np.random.RandomState(999)
    palette = rng.randint(0, 256, size=(num_classes, 3), dtype=np.uint8)
    palette[0] = np.array([0, 0, 0], np.uint8)     # background stays black
    for cls in np.unique(mask):
        out[mask == cls] = palette[int(cls) % num_classes]
    return out

def apply_color_on_black(orig_rgb, binary_mask):
    res = np.zeros_like(orig_rgb)
    res[binary_mask > 0] = orig_rgb[binary_mask > 0]
    return res

def add_download_button(img_np, filename):
    pil = Image.fromarray(img_np)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    st.markdown(
        f'<a href="data:file/png;base64,{b64}" download="{filename}">📥 Download {filename}</a>',
        unsafe_allow_html=True,
    )

# ------------- UI -------------
st.set_page_config(page_title="VisionAI Segmentation (DeepLabv3)", layout="wide")
st.title("🔍 VisionAI Segmentation (DeepLabv3 + your weights)")

colL, colR = st.columns([2,1])
with colR:
    st.markdown("**Inference options**")
    use_tta = st.checkbox("Use TTA (multi-scale + flip)", value=True)
    refine_opt = st.checkbox("Refine with morphology", value=True)
    refine_grabcut = st.checkbox("Extra refine with GrabCut (slower)", value=False)

uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded is None:
    st.info("⬆️ Upload an image to start. The app will auto-download your `checkpoint.pth` if missing.")
else:
    # Show input
    image_pil = Image.open(uploaded).convert("RGB")
    orig_rgb = np.array(image_pil)

    with st.spinner("Loading model & running inference…"):
        model, num_classes = load_model()
        if use_tta:
            logits = run_tta_logits(model, image_pil, scales=(0.75,1.0,1.25), do_flip=True)
        else:
            # simple single-scale path
            x = to_tensor(image_pil).to(DEVICE).unsqueeze(0)
            with torch.no_grad():
                out = model(x)["out"]
            logits = F.interpolate(out, size=orig_rgb.shape[:2], mode="bilinear", align_corners=False).squeeze(0)

        pred_classes = logits_to_pred_classes(logits)          # (H, W)

        # Binary (objects vs background) by auto background detection
        binary_raw, bg_idx = get_binary_mask_from_multiclass(pred_classes)

        # Refinements
        binary_ref = binary_raw
        if refine_opt:
            binary_ref = refine_binary_mask(binary_ref, k_close=7, k_open=3, min_area=300)
        if refine_grabcut:
            binary_ref = grabcut_refine(orig_rgb, binary_ref)

        # Outputs
        color_multiclass = colorize_multiclass(pred_classes, num_classes)
        color_objects_on_black = apply_color_on_black(orig_rgb, binary_ref)

    with colL:
        st.subheader("Original")
        st.image(orig_rgb, use_column_width=True)

        st.subheader("Binary Mask (objects vs background)")
        st.image(binary_ref, use_column_width=True)
        add_download_button(binary_ref, "binary_mask.png")

        st.subheader("Color Mask (multi-class)")
        st.image(color_multiclass, use_column_width=True)
        add_download_button(color_multiclass, "colored_mask.png")

        st.subheader("Color Mask (original colors on black)")
        st.image(color_objects_on_black, use_column_width=True)
        add_download_button(color_objects_on_black, "objects_on_black.png")

    with colR:
        st.markdown("**Details**")
        st.write(f"- Detected **num_classes** from weights: `{num_classes}`")
        st.write(f"- Autodetected background index: `{bg_idx}`")
        st.write("- If you still see imperfect masks, enable **GrabCut**. It’s slower but can improve edges.")











