# Dataset: AI art vs human art

Source: [AI art VS Human art (Kaggle)](https://www.kaggle.com/datasets/hassnainzaidi/ai-art-vs-human-art) · `hassnainzaidi/ai-art-vs-human-art` · license **CC0-1.0**.

## Layout (after download)

```text
data/
  Art/
    AiArtData/    # AI-generated artwork images
    RealArt/      # Human-created artwork images
```

## Train / validation split

After `Art/` exists, build stratified **train** and **val** sets (default **80% / 20%** per class, seed **42**). Files are **hard-linked** into `train/` and `val/` so disk usage stays the same.

```text
data/
  train/
    ai/           # linked from Art/AiArtData
    human/        # linked from Art/RealArt
  val/
    ai/
    human/
```

```bash
python3 scripts/split_train_eval.py
# optional: --train-ratio 0.85 --seed 123 --mode copy|symlink|link
```

If you still have `data/eval` from an older layout, rename it:

```bash
mv data/eval data/val
```

Re-download raw Kaggle data from the project root:

```bash
scripts/download_dataset.sh
```

Requires the [Kaggle API credentials](https://www.kaggle.com/docs/api#authentication) (`~/.kaggle/kaggle.json` or `KAGGLE_USERNAME` + `KAGGLE_KEY`).
