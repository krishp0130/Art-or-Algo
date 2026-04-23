# Art or Algo

CMU **05-318** — *Art or Algorithm*: distinguishing human creativity from AI ingenuity (image classifier + web app).

## Open-Source Code, Changes, and Original Work

### Open-source libraries and pretrained weights used

| Component | Source | License |
|-----------|--------|---------|
| **ViT-B/16 pretrained model** | `torchvision.models.vit_b_16(weights=IMAGENET1K_V1)` | BSD-3-Clause |
| **PyTorch / torchvision** | [pytorch.org](https://pytorch.org) | BSD-3-Clause |
| **timm** (PyTorch Image Models) | [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) | Apache-2.0 |
| **Vue 3 + Vite** | [vuejs.org](https://vuejs.org) / [vite.dev](https://vite.dev) | MIT |
| **Chart.js / vue-chartjs** | [chartjs.org](https://www.chartjs.org) | MIT |
| **Express / multer / cors** | [expressjs.com](https://expressjs.com) | MIT |
| **Dataset** | [Kaggle: AI Art vs Human Art](https://www.kaggle.com/datasets/hassnainzaidi/ai-art-vs-human-art) (hassnainzaidi) | CC0-1.0 |

No application-level starter code or boilerplate template was used. The Vue project was scaffolded with `npm create vite@latest` (empty template), and all component code was written from scratch.

### Changes made to imported code

- **ViT-B/16 classification head replaced:** the stock 1000-class ImageNet head was swapped for `LayerNorm(768) → Linear(768, 2)` for binary AI-vs-human classification.
- **Partial fine-tuning:** 9 of 12 encoder blocks frozen; only the last 3 blocks + encoder LayerNorm + new head are trainable (21.3 M of 85.8 M parameters).
- **MPS bug workaround:** disabled `non_blocking=True` on Apple Silicon MPS transfers to fix a race condition that caused non-deterministic evaluation (see `ml/train.py`, `ml/inference.py`, `ml/analyze_failures.py`).

### New code implemented (all original)

| File / Directory | What it does |
|------------------|--------------|
| `run_train.py` | Master training runner with AdamW differential LR, cosine schedule with warmup, label smoothing, early stopping, and rich `metrics.json` output |
| `ml/train.py` | Model construction, training loop, evaluation, and per-class metrics |
| `ml/dataset.py` | Custom dataset/dataloader with augmentation pipeline (RandomResizedCrop, ColorJitter, RandomErasing, etc.) |
| `ml/inference.py` | CLI + library inference with **Attention Rollout** heatmap generation (extracts attention matrices from all 12 encoder layers, multiplies through residual connections, produces a 224×224 overlay) |
| `ml/analyze_failures.py` | High-confidence failure extraction with heuristic explanations for the failure gallery |
| `server/src/index.js` | Express backend: image upload, Python subprocess bridge for inference, static serving, metrics/failures API |
| `client/src/components/Classifier.vue` | Guess-first classifier flow with staged disclosure, attention heatmap overlay, agreement/disagreement framing |
| `client/src/components/MetricsDashboard.vue` | Interactive Chart.js training curves and summary stats |
| `client/src/components/FailureGallery.vue` | Grid of high-confidence misclassifications with explanations |
| `client/src/components/AboutProject.vue` | Full project documentation page |
| `client/src/components/ImageUpload.vue` | Accessible drag-and-drop upload with keyboard support |
| `client/src/App.vue`, `client/src/style.css` | Gallery-themed UI: Fraunces/Source Sans typography, warm parchment palette, progressive disclosure layout |
| `scripts/split_train_eval.py` | Stratified train/val split script |
| `scripts/sync-static-assets.mjs` | Copies metrics + failure data into `client/public` for static GitHub Pages builds |
| `.github/workflows/deploy-pages.yml` | CI/CD for GitHub Pages deployment |

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
