# Art or Algo

CMU **05-318** — *Art or Algorithm*: distinguishing human creativity from AI ingenuity (image classifier + web app).

## Dataset

[Kaggle: AI art VS Human art](https://www.kaggle.com/datasets/hassnainzaidi/ai-art-vs-human-art) (`hassnainzaidi/ai-art-vs-human-art`, CC0-1.0).

Images live under **`data/Art/`** (`AiArtData/` vs `RealArt/`). The raw images are **not** tracked in git (~480MB); clone the repo, then run:

```bash
scripts/download_dataset.sh
python3 scripts/split_train_eval.py   # stratified train/eval under data/train & data/eval
```

See `data/README.md` for layout, split details, and Kaggle auth notes.

## Training (ViT-B/16)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-ml.txt
python3 -m ml.train
```

- **Checkpoint:** `models/best_vit.pth` (best validation accuracy; includes `model_state_dict`, `class_to_idx`, epoch).
- **Metrics (dashboard):** `models/metrics.json` — per-epoch `train_loss`, `val_loss`, `train_acc`, `val_acc`, plus `best_*` and `hyperparams`.

Optional flags: `--epochs`, `--batch-size`, `--lr`, `--data-root`, `--device cuda|mps|cpu`, `--checkpoint`, `--metrics-out`.
