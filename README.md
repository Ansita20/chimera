# CHIMERA

A visual anomaly detection system. It learns what a scene normally looks like and flags anything unexpected.

- Learns normality from images, videos or a live webcam
- Localizes anomalies with heatmaps and bounding boxes
- Tracks anomalous regions over time and logs them as events with snapshots
- Estimates uncertainty for every prediction
- Explains *why* each region was flagged: unfamiliar appearance, failed reconstruction, unexpected motion, colour/texture shift, persistence
- Dashboard with live view, single-image inspection, an event log, reports and model benchmarks

> CHIMERA detects that something **deviates from normal**. It does not identify **what** the object is, because recognizing unknown objects is a separate problem.

## Detectors

| Detector | Idea | Good at |
|---|---|---|
| Feature embedding (PatchCore-style) | Pretrained ResNet patch features + nearest-neighbour memory bank | Subtle, static, local defects |
| Autoencoder | Reconstruction error (L1 + SSIM) | Large structural changes |
| Temporal | U-Net predicts the next frame | Unexpected motion, sudden appearances |

Uncertainty comes from Monte-Carlo dropout (autoencoder, temporal) or memory-bank disagreement (embedding), combined with how close the score is to the threshold. Thresholds are calibrated on held-out normal data, so a score of `1.0×` means "at threshold" for every model.

## Stack

Python · PyTorch · OpenCV · FastAPI · React + TypeScript · PostgreSQL · Docker

## Quick start

**Docker** (PostgreSQL + API + dashboard):

```bash
docker compose up --build
# dashboard: http://localhost:8080   API docs: http://localhost:8000/docs
```

**Local** (SQLite):

```bash
# backend
cd backend
python -m venv ../.venv && ../.venv/bin/pip install -r requirements-dev.txt
../.venv/bin/uvicorn chimera.api.app:app --port 8000

# frontend (second terminal)
cd frontend
npm install && npx vite          # http://localhost:5173
```

## Usage

1. Create a **scene** in the top bar (one camera view or product type).
2. Teach it normality:
   - **Live** → *Learn normality* records webcam footage and trains all three detectors, or
   - **Setup** → upload 30–200 normal images or a normal video, then *Train*.
3. **Live** → *Detect anomalies*, or **Inspect** → upload a test image.
4. Review **Events** and **Reports**.

## Benchmarks

Compares the three detectors on [MVTec AD 2](https://www.mvtec.com/company/research/datasets/mvtec-ad-2) (still-image anomalies) and MSAD (video anomalies):

```bash
cd backend
python scripts/benchmark.py mvtec_ad2 --root /path/to/mvtec_ad2 --category all --save-db
python scripts/benchmark.py msad --root /path/to/msad --labels /path/to/labels.json --save-db
```

Metrics: image/pixel AUROC, AU-PRO, F1 for MVTec AD 2; frame AUROC and F1 for MSAD; plus training time and latency. Results appear in the **Research** tab. See `backend/chimera/datasets.py` for the expected folder layouts.

## Configuration

Settings live in `backend/chimera/config.py` and can be overridden with `CHIMERA_*` env vars, e.g.:

| Variable | Default | |
|---|---|---|
| `CHIMERA_DATABASE_URL` | SQLite in `data/` | PostgreSQL URL in Docker |
| `CHIMERA_DEVICE` | `auto` (CUDA, else CPU) | `mps` is opt-in; it crashes on some macOS/PyTorch versions |
| `CHIMERA_BACKBONE` | `resnet18` | `wide_resnet50_2` is more accurate but slower |
| `CHIMERA_AE_EPOCHS` / `CHIMERA_AE_MIN_STEPS` | `40` / `1500` | Autoencoder training length |
| `CHIMERA_TEMPORAL_EPOCHS` / `CHIMERA_TEMPORAL_MIN_STEPS` | `25` / `600` | Temporal training length |

Training runs on CPU unless you have CUDA. The embedding detector fits in seconds; the autoencoder and temporal models take a few minutes.

## Tests

```bash
cd backend && ../.venv/bin/python -m pytest -q
```
