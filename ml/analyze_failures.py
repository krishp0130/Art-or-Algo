#!/usr/bin/env python3
"""Scan the validation set for high-confidence mistakes; copy samples + HCI-oriented notes."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.dataset import vit_eval_transforms
from ml.inference import load_model
from torchvision import datasets


def _heuristic_notes(
    path: Path,
    true_label: str,
    pred_label: str,
) -> list[str]:
    notes: list[str] = []
    try:
        pil = Image.open(path).convert("RGB")
    except OSError:
        return [
            "Could not decode image for texture analysis; unusual encoding or a corrupt file may affect both you and the model."
        ]

    rgb = np.asarray(pil, dtype=np.float32)
    h, w = rgb.shape[:2]
    gray = (
        0.299 * rgb[:, :, 0]
        + 0.587 * rgb[:, :, 1]
        + 0.114 * rgb[:, :, 2]
    )
    if gray.shape[0] >= 3 and gray.shape[1] >= 3:
        g = gray
        lap = (
            -4.0 * g[1:-1, 1:-1]
            + g[:-2, 1:-1]
            + g[2:, 1:-1]
            + g[1:-1, :-2]
            + g[1:-1, 2:]
        )
        lap_var = float(np.var(lap))
    else:
        lap_var = 0.0
    ar = w / max(h, 1)
    color_rich = float(rgb.reshape(-1, 3).std(axis=0).mean())

    notes.append(
        f"Geometry: {w}×{h}px, aspect ratio {ar:.2f}. "
        f"Laplacian variance ≈ {lap_var:.1f} (proxy for fine detail / sharpness)."
    )
    notes.append(
        f"Color spread (mean BGR channel std) ≈ {color_rich:.1f} — richer palettes can correlate with "
        f"either photographic captures or heavily stylized AI outputs."
    )

    if lap_var < 80.0:
        notes.append(
            "HCI angle: very smooth, low-detail regions are a cue the model often associates with "
            "generative art; minimalist human work can be unfairly pulled toward 'AI'."
        )
    if lap_var > 300.0:
        notes.append(
            "HCI angle: strong micro-texture can look 'physical' or photographic; synthetic images "
            "that mimic film grain or canvas can be mislabeled as human."
        )

    if true_label == "human" and pred_label == "ai":
        notes.append(
            "Possible failure story: clean vector-like composition, flat shading, or digital poster "
            "aesthetics overlap the training distribution of AI samples."
        )
    else:
        notes.append(
            "Possible failure story: brushy strokes, noise, or traditional-media mimicry in AI "
            "outputs overlap human-art texture statistics the model relied on."
        )

    return notes


def _safe_stem(name: str, max_len: int = 72) -> str:
    s = re.sub(r"[^\w.\-]+", "_", name, flags=re.UNICODE)
    return s[:max_len]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=_ROOT / "models" / "final_vit.pth",
    )
    parser.add_argument("--data-root", type=Path, default=_ROOT / "data")
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=None,
        help="Validation root (default: data-root/val)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_ROOT / "server" / "public" / "failures",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.9,
        help="Wrong prediction must exceed this softmax confidence on the predicted class",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    ckpt = args.checkpoint
    if not ckpt.is_file():
        alt = _ROOT / "models" / "best_vit.pth"
        if alt.is_file():
            ckpt = alt
        else:
            print(f"ERROR: No checkpoint at {args.checkpoint}", file=sys.stderr)
            sys.exit(1)

    val_root = args.val_dir or (args.data_root / "val")
    if not val_root.is_dir():
        print(f"ERROR: Missing validation directory {val_root}", file=sys.stderr)
        sys.exit(1)

    session = load_model(ckpt, device=args.device if args.device != "auto" else None)
    session.model.eval()
    device = session.device

    val_ds = datasets.ImageFolder(
        str(val_root),
        transform=vit_eval_transforms(size=224),
    )
    class_names = val_ds.classes
    idx_to_name = {i: class_names[i] for i in range(len(class_names))}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for old in args.out_dir.glob("*"):
        if old.is_file() and old.name not in {".gitkeep"}:
            old.unlink()

    rows: list[dict] = []
    failure_idx = 0
    conf_t = args.conf_threshold
    loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    with torch.no_grad():
        base = 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            logits = session.model(xb)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            for j in range(xb.size(0)):
                i = base + j
                path_str, y_true = val_ds.samples[i]
                path = Path(path_str)
                pred = int(preds[j].item())
                p_pred = float(probs[j, pred].item())
                if pred == y_true or p_pred < conf_t:
                    continue

                true_name = idx_to_name[y_true]
                pred_name = idx_to_name[pred]
                prob_map = {
                    class_names[k]: float(probs[j, k].item())
                    for k in range(len(class_names))
                }

                ext = path.suffix.lower() or ".png"
                fname = (
                    f"{failure_idx:03d}_true{true_name}_pred{pred_name}_"
                    f"p{p_pred:.3f}_{_safe_stem(path.name)}{ext}"
                )
                dest = args.out_dir / fname
                shutil.copy2(path, dest)
                failure_idx += 1

                hints = _heuristic_notes(path, true_name, pred_name)
                rows.append(
                    {
                        "filename": fname,
                        "public_url_path": f"/failures/{fname}",
                        "true_label": true_name,
                        "predicted_label": pred_name,
                        "confidence_in_predicted_class": round(p_pred, 6),
                        "probabilities": prob_map,
                        "why_the_model_may_have_struggled": hints,
                    }
                )
            base += xb.size(0)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(ckpt.resolve()),
        "validation_dir": str(val_root.resolve()),
        "confidence_threshold_wrong_class": conf_t,
        "failure_count": len(rows),
        "failures": rows,
    }
    out_json = args.out_dir / "failures.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    if rows:
        pd.DataFrame(rows).to_csv(args.out_dir / "failures_summary.csv", index=False)

    print(f"Wrote {len(rows)} high-confidence failures -> {args.out_dir}")
    print(f"Manifest: {out_json}")


if __name__ == "__main__":
    main()
