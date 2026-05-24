# Retail Shelf Stockout Detection — Portfolio Demo

> **Archived model inference demo.** This app loads pre-trained YOLOv4 artifacts
> from a capstone project and runs forward-pass inference only. No model training
> or fine-tuning occurs here.

---

## What the Project Does

This Streamlit application demonstrates an end-to-end computer vision inference
pipeline for detecting **empty shelf space** (stockouts) in retail store images.

A custom YOLOv4 object detector was trained on annotated retail shelf imagery.
The model learns to identify regions where products are missing from shelves and
draws bounding boxes labeled **"EMPTY"** around those areas.

This demo lets visitors:
- Select from a set of sample shelf images
- Upload their own JPG/PNG shelf photo
- Adjust confidence and NMS thresholds via sidebar sliders
- Run YOLOv4 inference with a single button click
- View the original and annotated output side by side
- Download the annotated result

---

## Connecting to the Portfolio Case Study

This app is the interactive companion to a data science capstone project covering:

- Dataset collection and labeling for retail shelf imagery
- Darknet/YOLOv4 custom training pipeline
- Evaluation using mAP (mean Average Precision)
- Deployment of the trained model via a lightweight Streamlit UI

The capstone case study documents the full ML lifecycle from raw images to a
working detector. This demo isolates and showcases the inference stage in a
clean, portfolio-ready format.

---

## Quick Start (Local)

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd computer-vision-stockout-demo-app
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add model artifacts

Place the following files in the `models/` directory:

| File | Description |
|------|-------------|
| `yolov4-custom.cfg` | Darknet network configuration |
| `yolov4-custom_last.weights` | Trained model weights |
| `obj.names` | Class label file (one label per line) |

> These files are **not** committed to the repository due to size. Obtain them
> from the project's model storage location.

### 5. Add sample images (optional)

Place any of the following in `sample_images/`:

```
dressing_shelf.jpg
cereal_shelf.jpg
personal_care_shelf.jpg
soda_shelf.jpg
spices_shelf.jpg
```

The app works without sample images — you can always upload your own.

### 6. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Folder Structure

```
computer-vision-stockout-demo-app/
├── app.py                          # Streamlit application entry point
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .gitignore
├── models/
│   ├── yolov4-custom.cfg           # YOLOv4 network config (Darknet format)
│   ├── yolov4-custom_last.weights  # Trained weights (not committed)
│   └── obj.names                   # Class labels
├── sample_images/
│   ├── dressing_shelf.jpg
│   ├── cereal_shelf.jpg
│   ├── personal_care_shelf.jpg
│   ├── soda_shelf.jpg
│   └── spices_shelf.jpg
└── outputs/
    └── .gitkeep                    # Placeholder; downloaded results go here
```

---

## Required Model Artifacts

The app uses the OpenCV DNN module to load the model:

```python
net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
```

Inference uses a 416×416 input blob with scale factor `1/255.0`.
Non-Maximum Suppression is applied via `cv2.dnn.NMSBoxes`.

Both the `.cfg` and `.weights` files must be present and compatible (same
architecture and number of classes) for the app to start successfully.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI framework |
| `opencv-python-headless` | DNN inference + image processing |
| `numpy` | Array operations |
| `pillow` | Image decoding for uploaded files |

---

## Notes

- This app is intended as an **archived inference demo** and does not support
  retraining, fine-tuning, or dataset management.
- GPU acceleration is not configured; inference runs on CPU.
- Inference speed will depend on the host machine's CPU.
