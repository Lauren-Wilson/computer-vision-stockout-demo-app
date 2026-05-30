import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
import time

# ── Path constants ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_images")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

CFG_PATH     = os.path.join(MODELS_DIR, "yolov4-custom.cfg")
WEIGHTS_PATH = os.path.join(MODELS_DIR, "yolov4-custom_last.weights")
NAMES_PATH   = os.path.join(MODELS_DIR, "obj.names")

SAMPLE_IMAGES = {
    "Dressing Shelf":      "dressing_shelf.JPG",
    "Cereal Shelf":        "cereal_shelf.jpg",
    "Personal Care Shelf": "personal_care_shelf.jpg",
    "Soda Shelf":          "soda_shelf.jpg",
    "Spices Shelf":        "spices_shelf.jpg",
}

BOX_COLOR   = (0, 0, 220)   # BGR red-ish
LABEL_COLOR = (255, 255, 255)
NMS_THRESHOLD = 0.40

# ── Download Model Files ──────────────────────────────────────────────────────────────
import os
import gdown


os.makedirs(MODELS_DIR, exist_ok=True)

FILES = {
    "yolov4-custom.cfg": "1__ZcAVCAg7Wxv57gkQ_IjTkXDhvyJdwD",
    "yolov4-custom_last.weights": "1VeOYC7HDj4mhw2vap6yKgBuQxV-IcLz6",
    "obj.names": "1WJylgWOkEIWe1zRFs96WRM0ZIdVodvcs" 
}

for filename, file_id in FILES.items():
    output_path = os.path.join(MODELS_DIR, filename)

    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)


# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading YOLOv4 model…")
def load_model():
    """Load YOLOv4 model from Darknet artifacts. Cached across sessions."""
    missing = [p for p in (CFG_PATH, WEIGHTS_PATH, NAMES_PATH) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Required model file(s) not found:\n" +
            "\n".join(f"  • {p}" for p in missing)
        )

    net = cv2.dnn.readNetFromDarknet(CFG_PATH, WEIGHTS_PATH)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    with open(NAMES_PATH, "r") as f:
        class_names = [line.strip() for line in f if line.strip()]

    output_layers = net.getUnconnectedOutLayersNames()
    return net, class_names, output_layers


# ── Inference ──────────────────────────────────────────────────────────────────
def run_inference(image_bgr: np.ndarray,
                  net, class_names, output_layers,
                  conf_thresh: float, nms_thresh: float = NMS_THRESHOLD):
    """
    Run YOLOv4 forward pass and return annotated image + detection list.

    Returns
    -------
    annotated : np.ndarray  BGR image with boxes drawn
    detections : list of dict  {label, confidence, box}
    """
    h, w = image_bgr.shape[:2]

    blob = cv2.dnn.blobFromImage(
        image_bgr, scalefactor=1 / 255.0,
        size=(416, 416), swapRB=True, crop=False
    )
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence < conf_thresh:
                continue

            cx, cy, bw, bh = detection[:4]
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            boxes.append([x1, y1, int(bw * w), int(bh * h)])
            confidences.append(confidence)
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thresh, nms_thresh)

    detections = []
    annotated = image_bgr.copy()

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, bw, bh = boxes[i]
            label = class_names[class_ids[i]] if class_ids[i] < len(class_names) else "UNKNOWN"
            conf  = confidences[i]

            detections.append({"label": label, "confidence": conf,
                                "box": (x, y, bw, bh)})

            # Draw box
            cv2.rectangle(annotated, (x, y), (x + bw, y + bh), BOX_COLOR, 2)

            # Draw label badge
            text = f"{label}  {conf:.0%}"
            font_scale = 2.0
            font_thickness = 3
            text_pad_x = 8
            text_pad_y = 6
            (tw, th), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            badge_y1 = max(y - th - baseline - (text_pad_y * 2), 0)
            cv2.rectangle(annotated,
                          (x, badge_y1),
                      (x + tw + (text_pad_x * 2), badge_y1 + th + baseline + (text_pad_y * 2)),
                          BOX_COLOR, cv2.FILLED)
            cv2.putText(annotated, text,
                    (x + text_pad_x, badge_y1 + th + text_pad_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, LABEL_COLOR, font_thickness, cv2.LINE_AA)

    return annotated, detections


# ── Image helpers ──────────────────────────────────────────────────────────────
def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_sample_image(filename: str) -> np.ndarray:
    path = os.path.join(SAMPLE_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Sample image not found: {path}")
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"OpenCV could not decode image: {path}")
    return img


# ── Page layout ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retail Shelf Stockout Detection",
    page_icon="🛒",
    layout="wide",
)

st.title("🛒 Retail Shelf Stockout Detection")
st.markdown(
    """
    This demo runs inference with a **custom-trained YOLOv4 model** that detects
    **empty shelf space** in retail store images.  
    Select a sample image or upload your own, then adjust the confidence threshold
    to update the prediction live.
    """
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Detection Settings")
    conf_thresh = st.slider(
        "Confidence Threshold", min_value=0.10, max_value=0.95,
        value=0.35, step=0.05,
        help="Minimum confidence score to keep a detection."
    )

    st.divider()
    st.subheader("ℹ️ About this Demo")
    st.markdown(
        """
        In 2021, this archived computer vision project explored using YOLOv4 object detection to identify empty retail shelf space across real-world grocery and retail environments.

        The model was originally developed as part of an award-winning capstone project and trained using manually annotated shelf images labeled with LabelIMG in YOLO format.

        This demo does not retrain the model. It loads preserved YOLOv4 model artifacts and performs inference on the selected images.
        """
    )

    st.divider()
    st.subheader("📌 What this Project Demonstrates")
    st.markdown(
        """
        • Custom YOLOv4 object detection workflow  
        • Manual image annotation and dataset preparation  
        • Darknet model configuration and OpenCV DNN inference  
        • Non-Maximum Suppression for bounding box filtering  
        • End-to-end computer vision deployment in an interactive application  
        • Detection across varied retail shelf layouts and product densities  
        """
    )

    st.divider()
    st.subheader("📘 Want the Full Story?")
    st.markdown(
        """
        Read the full case study to explore:
        - dataset preparation
        - annotation workflow
        - YOLO configuration tuning
        - model comparisons
        - project challenges
        - lessons learned
        - what I would improve today 
        """
    )

# ── Model loading (with graceful error) ───────────────────────────────────────
try:
    net, class_names, output_layers = load_model()
    model_ok = True
except FileNotFoundError as e:
    st.error(
        f"**Model files missing.**\n\n{e}\n\n"
        "Place `yolov4-custom.cfg`, `yolov4-custom_last.weights`, and `obj.names` "
        "inside the `models/` directory, then refresh the page."
    )
    model_ok = False
except Exception as e:
    st.error(f"**Failed to load model:** {e}")
    model_ok = False

# ── Image source selection ────────────────────────────────────────────────────
st.subheader("📂 Select an Image")

tab_sample, tab_upload = st.tabs(["Sample Images", "Upload Your Own"])

image_bgr = None
image_source_label = ""

with tab_sample:
    sample_label = st.selectbox(
        "Choose a sample shelf image", list(SAMPLE_IMAGES.keys())
    )
    if st.button("Load Sample Image", key="load_sample"):
        try:
            image_bgr = load_sample_image(SAMPLE_IMAGES[sample_label])
            st.session_state["image_bgr"] = image_bgr
            st.session_state["image_label"] = sample_label
        except FileNotFoundError as e:
            st.warning(str(e))
        except Exception as e:
            st.error(f"Could not load sample image: {e}")

with tab_upload:
    uploaded = st.file_uploader(
        "Upload a JPG or PNG image", type=["jpg", "jpeg", "png"]
    )
    if uploaded is not None:
        try:
            pil_img = Image.open(uploaded)
            # Basic validation
            if pil_img.format not in ("JPEG", "PNG", None):
                st.error("Please upload a valid JPG or PNG file.")
            else:
                image_bgr = pil_to_bgr(pil_img)
                st.session_state["image_bgr"] = image_bgr
                st.session_state["image_label"] = uploaded.name
        except Exception as e:
            st.error(f"Could not read uploaded image: {e}")

# Restore from session if available
if image_bgr is None and "image_bgr" in st.session_state:
    image_bgr = st.session_state["image_bgr"]

# ── Preview + detection ───────────────────────────────────────────────────────
if image_bgr is not None:
    label = st.session_state.get("image_label", "Image")

    col_orig, col_pred = st.columns(2)

    with col_orig:
        st.markdown("**Original Image**")
        st.image(bgr_to_rgb(image_bgr), use_container_width=True,
                 caption=label)

    st.button("🔍 Run Detection", type="primary", disabled=not model_ok)

    if model_ok:
        with st.spinner("Running YOLOv4 inference…"):
            t0 = time.time()
            try:
                annotated, detections = run_inference(
                    image_bgr, net, class_names, output_layers,
                    conf_thresh
                )
                elapsed = time.time() - t0
            except Exception as e:
                st.error(f"Inference failed: {e}")
                st.stop()

        # Save to session so result persists across reruns
        st.session_state["annotated"] = annotated
        st.session_state["detections"] = detections
        st.session_state["elapsed"] = elapsed

        st.caption("Results refresh automatically whenever the confidence threshold changes.")

    # Render results if they exist for the current image
    if "annotated" in st.session_state:
        annotated  = st.session_state["annotated"]
        detections = st.session_state["detections"]
        elapsed    = st.session_state.get("elapsed", 0)

        with col_pred:
            st.markdown("**Prediction — Detected Empty Shelves**")
            st.image(bgr_to_rgb(annotated), use_container_width=True,
                     caption=f"Inference time: {elapsed:.2f}s")

        st.divider()

        if len(detections) == 0:
            st.info(
                "No detections found above the current confidence threshold. "
                "Try lowering the **Confidence Threshold** in the sidebar."
            )
        else:
            n = len(detections)
            st.success(f"**{n} empty shelf region{'s' if n != 1 else ''} detected.**")

            with st.expander("Detection Details", expanded=True):
                for i, d in enumerate(detections, start=1):
                    st.markdown(
                        f"**Detection {i}** — "
                        f"Label: **{d['label']}** | "
                        f"Confidence: **{d['confidence']:.1%}**"
                    )

            # Download annotated image
            result_rgb = bgr_to_rgb(annotated)
            buf = io.BytesIO()
            Image.fromarray(result_rgb).save(buf, format="PNG")
            st.download_button(
                label="⬇️ Download Annotated Image",
                data=buf.getvalue(),
                file_name="stockout_detection_result.png",
                mime="image/png",
            )
else:
    st.info("Load a sample image or upload your own to get started.")
