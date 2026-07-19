"""
FastAPI backend for the digit recognizer.

Endpoints:
    GET  /health            -> simple health check
    POST /predict           -> upload a canvas image, get back predicted digit,
                                confidence scores, a preview of what the model
                                actually saw, and a Grad-CAM heatmap
    POST /feedback          -> submit a correction (stores image + true label for future retraining)
    GET  /feedback/count    -> how many corrections have been collected so far

Run:
    uvicorn main:app --reload --port 8000
"""

import base64
import io
import os
import sqlite3
import uuid
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms

from model import DigitCNN

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Digit Recognizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your frontend's origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

FEEDBACK_DIR = "feedback_images"
DB_PATH = "feedback.db"
os.makedirs(FEEDBACK_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

device = torch.device("cpu")
model = DigitCNN().to(device)

MODEL_PATH = "digit_cnn.pt"
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        f"'{MODEL_PATH}' not found. Run `python train.py` first to create it."
    )

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

normalize = transforms.Normalize((0.1307,), (0.3081,))


# ---------------------------------------------------------------------------
# Preprocessing — replicates MNIST-style centering so canvas drawings
# look like the data the model was trained on.
# ---------------------------------------------------------------------------

def preprocess(image: Image.Image):
    """Returns (model_input_tensor, raw_28x28_uint8_array). The raw array is
    what we show the user as "what the model actually saw" — it's also the
    base image the Grad-CAM heatmap gets overlaid on."""
    arr = np.array(image.convert("L"))  # grayscale

    coords = np.argwhere(arr > 20)
    if coords.size == 0:
        canvas = np.zeros((28, 28), dtype=np.float32)
    else:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        cropped = Image.fromarray(arr[y0:y1 + 1, x0:x1 + 1])

        w, h = cropped.size
        scale = 20 / max(w, h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cropped.resize((new_w, new_h))

        canvas_img = Image.new("L", (28, 28), 0)
        offset = ((28 - new_w) // 2, (28 - new_h) // 2)
        canvas_img.paste(resized, offset)
        canvas = np.array(canvas_img, dtype=np.float32)

    raw_28x28 = canvas.astype(np.uint8)

    tensor = torch.tensor(canvas / 255.0).unsqueeze(0)  # (1, 28, 28)
    tensor = normalize(tensor)
    return tensor.unsqueeze(0), raw_28x28  # (1, 1, 28, 28), (28, 28)


# ---------------------------------------------------------------------------
# Visualization helpers — Grad-CAM heatmap + base64 image encoding so the
# frontend can render "what the model saw" and "where it focused".
# ---------------------------------------------------------------------------

def array_to_base64_png(arr: np.ndarray, upscale_to: int = 140) -> str:
    """Converts a numpy image array (grayscale or RGB) to a base64 PNG string,
    upscaled with nearest-neighbor so the 28x28 source stays crisp/pixelated
    rather than blurry."""
    mode = "L" if arr.ndim == 2 else "RGB"
    img = Image.fromarray(arr.astype(np.uint8), mode=mode)
    img = img.resize((upscale_to, upscale_to), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# Node visualization helpers — produces a simplified "input pixels -> hidden
# nodes -> output nodes" view of the network with REAL activation values
# (not just decorative), for the animated node diagram in the frontend.
# ---------------------------------------------------------------------------

INPUT_GRID_SIZE = 6    # 6x6 = 36 input nodes shown (downsampled from 28x28)
HIDDEN_NODE_COUNT = 12  # 128 fc1 units pooled down to 12 groups for display


def get_input_node_values(raw_28x28: np.ndarray) -> list:
    """Downsamples the 28x28 preprocessed image to a small grid so it can be
    shown as individual input nodes rather than 784 tiny dots."""
    img = Image.fromarray(raw_28x28)
    small = img.resize((INPUT_GRID_SIZE, INPUT_GRID_SIZE), Image.BILINEAR)
    arr = np.array(small, dtype=np.float32) / 255.0
    return arr.flatten().tolist()


def get_hidden_node_values(fc1_raw_output: torch.Tensor) -> list:
    """Pools the 128 fc1 activations (after ReLU) down to a handful of
    representative nodes for display, normalized to [0, 1]."""
    relu_out = torch.clamp(fc1_raw_output, min=0).detach().numpy().flatten()  # (128,)
    groups = np.array_split(relu_out, HIDDEN_NODE_COUNT)
    pooled = np.array([g.mean() for g in groups])
    if pooled.max() > 0:
        pooled = pooled / pooled.max()
    return pooled.tolist()


def compute_gradcam(input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
    """Runs Grad-CAM against model.conv2 (the last conv layer) to produce a
    28x28 heatmap showing which pixels most influenced the prediction.
    Returns a float array in [0, 1]."""
    activations = {}
    gradients = {}

    def forward_hook(module, inp, out):
        activations["value"] = out

    def backward_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0]

    fwd_handle = model.conv2.register_forward_hook(forward_hook)
    bwd_handle = model.conv2.register_full_backward_hook(backward_hook)

    model.zero_grad()
    logits = model(input_tensor)  # grad-enabled forward pass
    score = logits[0, class_idx]
    score.backward()

    fwd_handle.remove()
    bwd_handle.remove()

    acts = activations["value"][0]      # (64, 14, 14)
    grads = gradients["value"][0]       # (64, 14, 14)

    weights = grads.mean(dim=(1, 2))    # (64,) — global average pooled gradients
    cam = torch.zeros(acts.shape[1:], dtype=torch.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = F.relu(cam)
    cam = cam.detach().numpy()

    if cam.max() > 0:
        cam = cam / cam.max()

    # Upsample 14x14 -> 28x28 to match the input image resolution
    cam_img = Image.fromarray((cam * 255).astype(np.uint8))
    cam_img = cam_img.resize((28, 28), Image.BILINEAR)
    return np.array(cam_img, dtype=np.float32) / 255.0


def make_gradcam_overlay(raw_28x28: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """Blends the Grad-CAM heatmap (red = high influence) over the grayscale
    digit image, returning an RGB uint8 array."""
    base = np.stack([raw_28x28] * 3, axis=-1).astype(np.float32)  # (28,28,3)

    heat = np.zeros_like(base)
    heat[..., 0] = cam * 255  # red channel encodes influence strength

    alpha = 0.55
    overlay = base * (1 - alpha * cam[..., None]) + heat * (alpha * cam[..., None])
    return np.clip(overlay, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Feedback storage (SQLite — zero setup, swap for MSSQL later if you want)
# ---------------------------------------------------------------------------

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id TEXT PRIMARY KEY,
            image_path TEXT NOT NULL,
            predicted_digit INTEGER NOT NULL,
            true_digit INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


init_db()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read uploaded image")

    tensor, raw_28x28 = preprocess(image)
    tensor = tensor.to(device)

    # Capture the fc1 (hidden layer) activations for the node visualization
    # by hooking it during the same forward pass used for the prediction.
    fc1_capture = {}

    def fc1_hook(module, inp, out):
        fc1_capture["value"] = out

    fc1_handle = model.fc1.register_forward_hook(fc1_hook)

    # Grad-CAM needs gradients, so this forward pass runs outside no_grad.
    # It's a single 28x28 image, so the extra cost is negligible.
    logits = model(tensor)
    fc1_handle.remove()

    probs = F.softmax(logits, dim=1).squeeze().detach().tolist()
    prediction = int(np.argmax(probs))

    cam = compute_gradcam(tensor, prediction)
    overlay = make_gradcam_overlay(raw_28x28, cam)

    return {
        "prediction": prediction,
        "probabilities": probs,
        "preprocessed_image": array_to_base64_png(raw_28x28),
        "gradcam_image": array_to_base64_png(overlay),
        "viz_input": get_input_node_values(raw_28x28),
        "viz_hidden": get_hidden_node_values(fc1_capture["value"]),
    }


@app.post("/feedback")
async def feedback(
    file: UploadFile = File(...),
    predicted_digit: int = Form(...),
    true_digit: int = Form(...),
):
    """
    Call this when the user corrects a wrong prediction. Stores the original
    image plus both labels so you can build a growing dataset of real,
    hard examples to retrain on later.
    """
    if not (0 <= true_digit <= 9):
        raise HTTPException(status_code=400, detail="true_digit must be between 0 and 9")

    contents = await file.read()
    feedback_id = str(uuid.uuid4())
    image_path = os.path.join(FEEDBACK_DIR, f"{feedback_id}.png")

    with open(image_path, "wb") as f:
        f.write(contents)

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO feedback (id, image_path, predicted_digit, true_digit, created_at) VALUES (?, ?, ?, ?, ?)",
        (feedback_id, image_path, predicted_digit, true_digit, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()

    return {"status": "saved", "id": feedback_id}


@app.get("/feedback/count")
def feedback_count():
    conn = sqlite3.connect(DB_PATH)
    count = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    conn.close()
    return {"count": count}
