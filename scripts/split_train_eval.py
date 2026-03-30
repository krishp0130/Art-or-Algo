#!/usr/bin/env python3
"""Split Kaggle art images into train / validation folders (stratified by class).

Uses hard links by default so files are not duplicated on disk. Falls back to
copy if linking fails (e.g. cross-device).
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

# Source folder names from Kaggle zip
SRC_AI = "AiArtData"
SRC_HUMAN = "RealArt"


def list_images(folder: Path) -> list[Path]:
    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def place(
    paths: list[Path],
    dest_dir: Path,
    mode: str,
) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src in paths:
        dst = dest_dir / src.name
        if dst.exists():
            dst.unlink()
        if mode == "copy":
            shutil.copy2(src, dst)
        elif mode == "symlink":
            os.symlink(src.resolve(), dst)
        else:
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Path to data/ (contains Art/)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of each class assigned to train (rest is val)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        choices=("link", "copy", "symlink"),
        default="link",
        help="How to populate train/eval (default: hard link)",
    )
    args = parser.parse_args()

    if not 0 < args.train_ratio < 1:
        raise SystemExit("--train-ratio must be between 0 and 1 (exclusive)")

    art = args.data_root / "Art"
    ai_src = art / SRC_AI
    human_src = art / SRC_HUMAN
    if not ai_src.is_dir() or not human_src.is_dir():
        raise SystemExit(
            f"Expected {ai_src} and {human_src}. Run scripts/download_dataset.sh first."
        )

    random.seed(args.seed)
    train_root = args.data_root / "train"
    val_root = args.data_root / "val"

    for split_root in (train_root, val_root):
        if split_root.exists():
            shutil.rmtree(split_root)

    splits = [
        ("ai", ai_src, train_root / "ai", val_root / "ai"),
        ("human", human_src, train_root / "human", val_root / "human"),
    ]

    summary: list[tuple[str, int, int]] = []
    for _label, src_dir, train_dir, val_dir in splits:
        files = list_images(src_dir)
        random.shuffle(files)
        n_train = int(len(files) * args.train_ratio)
        train_files = files[:n_train]
        val_files = files[n_train:]
        place(train_files, train_dir, args.mode)
        place(val_files, val_dir, args.mode)
        summary.append((_label, len(train_files), len(val_files)))

    print("Split complete (per class):")
    for label, n_tr, n_val in summary:
        print(f"  {label:6}  train={n_tr:4}  val={n_val:4}")
    print(f"  train dir: {train_root}")
    print(f"  val dir:   {val_root}")


if __name__ == "__main__":
    main()
