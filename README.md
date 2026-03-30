# Art or Algo

CMU **05-318** — *Art or Algorithm*: distinguishing human creativity from AI ingenuity (image classifier + web app).

## Environment

```bash
chmod +x setup_env.sh
./setup_env.sh          # macOS / Linux: venv + pip install + checks data/train & data/val
# Windows: setup_env.bat
```

Python deps live in **`requirements.txt`** (`torch`, `torchvision`, `timm`, `pandas`, `opencv-python`, `matplotlib`, …).

## Dataset

[Kaggle: AI art VS Human art](https://www.kaggle.com/datasets/hassnainzaidi/ai-art-vs-human-art) (`hassnainzaidi/ai-art-vs-human-art`, CC0-1.0).

```bash
scripts/download_dataset.sh
python3 scripts/split_train_eval.py   # → data/train and data/val
```

See `data/README.md` for layout and Kaggle auth.

## Training (master runner)

```bash
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
python3 run_train.py
```

- **Checkpoint:** `models/final_vit.pth` (best validation accuracy during the run).
- **Metrics:** `models/metrics.json` — **final_accuracy**, per-class **precision** / **recall**, **confusion_matrix** `[[TN, FP],[FN, TP]]`-style rows=true / cols=pred, **history**, early-stop metadata.

Defaults: up to **20** epochs, minimum **10** before early stopping, patience **4**. Override with `--epochs-max`, `--epochs-min`, `--early-stop-patience`.

Legacy entry point: `python3 -m ml.train` (writes `models/best_vit.pth`).

## Failure analysis (HCI)

High-confidence wrong predictions on the validation set → `server/public/failures/` + `failures.json`:

```bash
python3 ml/analyze_failures.py --checkpoint models/final_vit.pth --conf-threshold 0.9
```

## Inference (Node.js backend)

One JSON object on **stdout** (for `child_process` / Express):

```bash
python3 -m ml.inference --image path/to/file.png --checkpoint models/final_vit.pth
```

Fields include **`prediction`**, **`confidence`**, **`probabilities`**, **`attention_map_base64`** (PNG), and **`ok`**.

```python
from ml.inference import load_model, predict_json_for_backend
session = load_model("models/final_vit.pth")
out = predict_json_for_backend(session, "upload.png")
```

**Attention Rollout** uses all encoder layers by default; use `--rollout-mode last_layer` for the final block only.

## Tests

```bash
python3 -m unittest tests.test_inference_cli
```

## GitHub Pages + API

GitHub Pages only serves **static** files. This repo’s workflow (`.github/workflows/deploy-pages.yml`) builds the Vue app with `VITE_BASE_PATH=/Art-or-Algo/` (change if your repo name differs).

1. **Repository → Settings → Pages**: Source = **GitHub Actions**.
2. **Repository → Settings → Variables**: add `VITE_API_BASE` = your deployed Express URL (e.g. `https://art-or-algo-api.onrender.com`) so the **classifier** can reach `POST /api/classify`. Leave it empty to rely on baked-in `metrics.json` + `failures/failures.json` only (metrics + failure *text* work; images need either committed thumbnails or the API).
3. After training / `analyze_failures`, commit **`models/metrics.json`** and **`server/public/failures/failures.json`** so CI can copy them into the static build. Failure **images** stay gitignored by default—run `analyze_failures` on the API host or add `-f` git add for a small curated set if you want them on Pages without a backend.

**Full stack locally:** `npm start` (builds client, serves `client/dist` + `/api` on port 3000). Set `PYTHON_BIN` if not using `.venv`.
