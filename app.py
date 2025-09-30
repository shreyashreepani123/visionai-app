import cv2
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

def apply_dense_crf(image: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """
    Refine segmentation with DenseCRF.
    image: RGB image
    probs: [C, H, W] softmax probabilities
    """
    H, W = image.shape[:2]
    n_classes = probs.shape[0]

    d = dcrf.DenseCRF2D(W, H, n_classes)

    # Use softmax outputs as unary potentials
    U = unary_from_softmax(probs)
    d.setUnaryEnergy(U)

    # Add pairwise terms (bilateral + spatial)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=50, srgb=13, rgbim=image, compat=10)

    Q = d.inference(5)  # number of iterations
    refined = np.array(Q).reshape((n_classes, H, W))
    return refined


def enhance_mask(image: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """
    Combines CRF + morphology to clean mask.
    """
    crf_probs = apply_dense_crf(image, probs)
    argmax_map = np.argmax(crf_probs, axis=0).astype(np.uint8)

    # Assume background = most common class
    bg_idx = int(np.bincount(argmax_map.flatten()).argmax())
    mask = (argmax_map != bg_idx).astype(np.uint8) * 255

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # fill holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # remove noise

    return mask



def color_mask_from_binary(image_rgb: np.ndarray, binary_255: np.ndarray) -> np.ndarray:
    out = image_rgb.copy()
    out[binary_255 == 0] = 0
    return out


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="VisionAI Segmentation (Safe Mode)", layout="centered")
st.title("🔍 VisionAI Segmentation Demo (with Safe Refinement)")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    pil = Image.open(uploaded).convert("RGB")
    img = np.array(pil)

    st.subheader("Uploaded Image")
    st.image(img, use_column_width=True)

    model = load_model()
    probs = softmax_probs(model, pil)

    # --- Raw model output ---
    st.subheader("Raw Prediction (Argmax Classes)")
    raw_vis = visualize_raw_predictions(probs)
    st.image(raw_vis, caption="Raw Argmax (each color = a class)", use_column_width=True)

    # --- Safe refinement ---
    binary = safe_refine(img, probs)
    color_mask = color_mask_from_binary(img, binary)

    st.subheader("Binary Mask (Safe Refined)")
    st.image(binary, use_column_width=True)

    st.download_button(
        "⬇️ Download Binary Mask (PNG)",
        data=cv2.imencode(".png", binary)[1].tobytes(),
        file_name="binary_mask.png",
        mime="image/png",
    )

    st.subheader("Color Mask (Objects on Black Background)")
    st.image(color_mask, use_column_width=True)

    st.download_button(
        "⬇️ Download Color Mask (PNG)",
        data=cv2.imencode(".png", cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))[1].tobytes(),
        file_name="color_mask.png",
        mime="image/png",
    )











