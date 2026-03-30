#!/usr/bin/env python3
"""Master ViT-B/16 training runner: early stopping, final_vit.pth, rich metrics.json."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.dataset import build_dataloaders
from ml.train import (
    build_model,
    evaluate,
    pick_device,
    train_one_epoch,
)


def confusion_and_prf(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict:
    """Rows = true class, cols = predicted class (same order as ``class_names``)."""
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    def pr_recall(true_idx: int) -> tuple[float, float]:
        tp = cm[true_idx, true_idx]
        fp = cm[:, true_idx].sum() - tp
        fn = cm[true_idx, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return float(prec), float(rec)

    precision = {}
    recall = {}
    for i, name in enumerate(class_names):
        p_i, r_i = pr_recall(i)
        precision[name] = round(p_i, 6)
        recall[name] = round(r_i, 6)

    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    return {
        "final_accuracy": round(acc, 6),
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
    }


@torch.no_grad()
def collect_val_predictions(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys: list[int] = []
    ps: list[int] = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        pred = logits.argmax(dim=1)
        ys.extend(targets.tolist())
        ps.extend(pred.cpu().tolist())
    return np.array(ys, dtype=int), np.array(ps, dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=_ROOT / "data")
    parser.add_argument("--epochs-max", type=int, default=20, help="Maximum epochs (10–20 typical)")
    parser.add_argument("--epochs-min", type=int, default=10, help="Minimum epochs before early stop")
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=4,
        help="Stop if val acc does not improve for this many epochs (after --epochs-min)",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-4,
        help="Minimum val-acc improvement to reset patience",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=_ROOT / "models" / "final_vit.pth",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=_ROOT / "models" / "metrics.json",
    )
    args = parser.parse_args()
    if args.epochs_max < args.epochs_min:
        parser.error("--epochs-max must be >= --epochs-min")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = pick_device(args.device if args.device != "auto" else None)
    train_dir = args.data_root / "train"
    val_dir = args.data_root / "val"
    train_loader, val_loader, data_meta = build_dataloaders(
        train_dir,
        val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    num_classes = int(data_meta["num_classes"])
    class_names = list(data_meta["class_names"])
    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs_max, 1)
    )

    use_amp = device.type == "cuda"
    scaler = (
        torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)

    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    patience_left = args.early_stop_patience
    t0 = time.perf_counter()
    stopped_early = False

    for epoch in range(1, args.epochs_max + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, use_amp
        )
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(tr_loss, 6),
                "val_loss": round(va_loss, 6),
                "train_acc": round(tr_acc, 6),
                "val_acc": round(va_acc, 6),
            }
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs_max}  "
            f"train_loss={tr_loss:.4f} acc={tr_acc:.4f}  "
            f"val_loss={va_loss:.4f} acc={va_acc:.4f}"
        )

        improved = (va_acc - best_val_acc) > args.early_stop_min_delta or (
            abs(va_acc - best_val_acc) <= args.early_stop_min_delta
            and va_loss < best_val_loss
        )
        if improved:
            best_val_acc = va_acc
            best_val_loss = va_loss
            best_epoch = epoch
            payload = {
                "model": "vit_b_16",
                "weights_backbone": "IMAGENET1K_V1",
                "epoch": epoch,
                "val_acc": va_acc,
                "val_loss": va_loss,
                "class_to_idx": data_meta["class_to_idx"],
                "idx_to_class": {
                    str(k): v for k, v in data_meta["idx_to_class"].items()
                },
                "model_state_dict": model.state_dict(),
            }
            torch.save(payload, args.checkpoint)
            print(f"  saved new best -> {args.checkpoint}")
            patience_left = args.early_stop_patience
        elif epoch >= args.epochs_min:
            patience_left -= 1
            print(f"  no val improvement ({patience_left} patience left)")

        if epoch >= args.epochs_min and patience_left <= 0:
            stopped_early = True
            print("Early stopping.")
            break

    elapsed = time.perf_counter() - t0

    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    y_true, y_pred = collect_val_predictions(model, val_loader, device)
    report = confusion_and_prf(y_true, y_pred, class_names)

    metrics = {
        "model": "vit_b_16",
        "dataset": "hassnainzaidi/ai-art-vs-human-art",
        "train_dir": str(train_dir.resolve()),
        "val_dir": str(val_dir.resolve()),
        "num_classes": num_classes,
        "epochs_ran": len(history),
        "epochs_max": args.epochs_max,
        "stopped_early": stopped_early,
        "best_epoch": best_epoch,
        "best_val_acc": round(best_val_acc, 6),
        "best_val_loss": round(best_val_loss, 6),
        "train_seconds": round(elapsed, 2),
        "final_accuracy": report["final_accuracy"],
        "precision": report["precision"],
        "recall": report["recall"],
        "confusion_matrix": report["confusion_matrix"],
        "class_names": report["class_names"],
        "hyperparams": {
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "device": str(device),
            "early_stop_patience": args.early_stop_patience,
        },
        "checkpoint": str(args.checkpoint.resolve()),
        "history": history,
    }
    args.metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote metrics -> {args.metrics_out}")


if __name__ == "__main__":
    main()
