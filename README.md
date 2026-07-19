# Digit Recognizer — React + FastAPI + CNN

A hand-drawn digit recognizer. Draw a digit (0-9) on a canvas in the browser,
and a CNN trained on MNIST predicts what you drew, along with a live confidence
breakdown for all 10 digits. Includes a feedback loop that stores corrections
so you can retrain on real, hard examples later.

## Tech stack

- **Frontend:** React (Vite) — canvas drawing, confidence bar chart, feedback UI
- **Backend:** FastAPI — image preprocessing + CNN inference
- **Model:** PyTorch CNN trained on MNIST with data augmentation
- **Feedback storage:** SQLite + saved images (swap for MSSQL/Postgres later if you want)

## Project structure

```
digit-recognizer/
├── backend/
│   ├── model.py            CNN architecture
│   ├── train.py            Trains the model on MNIST, saves digit_cnn.pth
│   ├── main.py              FastAPI app: /predict, /feedback, /health
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   ├── index.css
│   │   └── components/
│   │       ├── DigitCanvas.jsx       drawing canvas
│   │       ├── ConfidenceBars.jsx    live probability bars
│   │       └── FeedbackWidget.jsx    correct/incorrect feedback UI
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
└── README.md
```

## Prerequisites

- Python 3.9+
- Node.js 18+
- pip and npm installed

## Step-by-step setup

### 1. Backend — install dependencies

```bash
cd digit-recognizer/backend
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Backend — train the model

This downloads MNIST automatically (~10 min first time due to download,
then a few minutes to train on CPU, faster on GPU).

```bash
python train.py
```

You should see output like:

```
Epoch 1/10 | loss: 245.31 | test accuracy: 97.10%
...
Epoch 10/10 | loss: 42.18 | test accuracy: 99.05%

Saved trained model to digit_cnn.pth
```

This creates `digit_cnn.pth` in the `backend/` folder — the API loads this file
at startup, so this step must be done before running the server.

### 3. Backend — start the API server

```bash
uvicorn main:app --reload --port 8000
```

Leave this terminal running. Verify it's up by visiting:
`http://localhost:8000/health` → should return `{"status": "ok"}`

You can also see interactive API docs at `http://localhost:8000/docs`.

### 4. Frontend — install dependencies

Open a **new terminal**:

```bash
cd digit-recognizer/frontend
npm install
```

### 5. Frontend — run the dev server

```bash
npm run dev
```

Open the URL it prints (typically `http://localhost:5173`).

### 6. Use it

1. Draw a digit on the black canvas with your mouse (or finger, on touch devices).
2. Click **Predict**.
3. See the predicted digit and a live confidence bar for all 10 classes.
4. Click **Yes** if it's correct, or **No** and pick the actual digit — this
   gets saved to `backend/feedback_images/` and logged in `backend/feedback.db`
   for future retraining.

## Node visualization

Below the confidence bars, you'll see an animated network diagram:

- **Input pixels** — the preprocessed 28x28 image downsampled to a 6x6 grid of nodes, brightness = pixel intensity.
- **Hidden layer** — the model's actual 128 `fc1` activations, pooled down to 12 representative nodes, brightness = activation strength.
- **Output (0-9)** — the real softmax probabilities, with the predicted digit highlighted in green.

Every time you click Predict, the diagram re-animates left to right, giving a
sense of the signal flowing through the network. Node brightness reflects
real values pulled from the model on every request. Edge brightness is a
simplified visual cue (based on the connected nodes' activation) rather than
the literal learned weight matrix — showing true per-connection weights for
every edge wouldn't be readable at this scale, so this favors intuition over
literal precision. Implemented in `NeuralNetVisual.jsx`, fed by the
`viz_input` and `viz_hidden` fields the backend now returns from `/predict`
(captured via a forward hook on `model.fc1` in `main.py`).

## Visualizations

After clicking **Predict**, you'll now see two extra images alongside the
confidence bars:

- **"What the model sees"** — the actual 28x28 cropped, centered, normalized
  image that gets fed into the CNN. Handy for debugging odd predictions —
  if this looks wrong, the issue is in preprocessing, not the model.
- **"Where it focused" (Grad-CAM)** — a heatmap overlay showing which pixels
  most influenced the prediction, computed from the gradients flowing back
  through the last convolutional layer. Brighter red = more influence.

Both are computed server-side in `main.py` (`compute_gradcam`,
`make_gradcam_overlay`, `array_to_base64_png`) and sent back as base64 PNGs
in the `/predict` response (`preprocessed_image`, `gradcam_image` fields), so
no extra frontend libraries are needed.

## How the preprocessing works

Canvas drawings don't look like MNIST images by default — MNIST digits are
tightly cropped and centered. The backend replicates that:

1. Finds the bounding box of the drawn strokes.
2. Crops to that box.
3. Resizes so the longest side is 20px (keeping aspect ratio).
4. Pastes it centered into a blank 28x28 frame.
5. Normalizes pixel values to match the training normalization.

Skipping this step is the most common reason canvas-based digit recognizers
perform worse than their reported test accuracy suggests.

## Retraining with feedback data (optional next step)

Once you've collected a batch of corrections in `backend/feedback_images/` +
`feedback.db`, you can write a small script to:

1. Load the stored images + `true_digit` labels from SQLite.
2. Combine them with the original MNIST training set.
3. Re-run `train.py`-style training on the combined dataset.
4. Replace `digit_cnn.pth` with the improved model.

This turns the project from a static demo into a self-improving system —
a good talking point if you're presenting this in an interview or on your
resume.

## Troubleshooting

| Problem | Fix |
|---|---|
| `RuntimeError: 'digit_cnn.pth' not found` | Run `python train.py` before starting `uvicorn` |
| Frontend shows "Could not reach backend" | Make sure `uvicorn` is running on port 8000 |
| CORS errors in browser console | Confirm `CORSMiddleware` in `main.py` is present (it is, by default) |
| Predictions look wrong even after training | Check your canvas stroke width isn't too thin/thick — 18px works well for a 280x280 canvas |
| `pip install` fails on torch | Visit https://pytorch.org/get-started/locally/ for a platform-specific install command |

## Ideas to extend this project

- Support multi-digit recognition (segment a whole handwritten number with OpenCV, classify each digit)
- Deploy the model in-browser with TensorFlow.js to remove the backend dependency entirely
- Swap SQLite for MSSQL/Postgres for the feedback store
- Package the frontend as a PWA so it works offline
- Add a scheduled retraining job that folds in feedback data automatically
