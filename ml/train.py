#!/usr/bin/env python3
"""Fine-tune torchvision ViT-B/16 for binary AI vs human art classification."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import ViT_B_16_Weights, vit_b_16

# Repo root on disk (supports `python ml/train.py` from project root)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.dataset import build_dataloaders


def pick_device(prefer: str | None) -> torch.device:
    if prefer and prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(num_classes: int) -> nn.Module:
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights)
    dim = model.hidden_dim
    model.heads = nn.Sequential(
        nn.LayerNorm(dim, eps=1e-6),
        nn.Linear(dim, num_classes),
    )
    return model


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        total_loss += loss.item() * images.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == targets).sum().item()
        total += images.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    use_amp: bool,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp and scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == targets).sum().item()
        total += images.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=_ROOT / "data")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=_ROOT / "models" / "best_vit.pth",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=_ROOT / "models" / "metrics.json",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = pick_device(args.device if args.device != "auto" else None)
    train_dir = args.data_root / "train"
    eval_dir = args.data_root / "eval"
    train_loader, eval_loader, data_meta = build_dataloaders(
        train_dir,
        eval_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    num_classes = int(data_meta["num_classes"])
    if num_classes != 2:
        print(
            f"Warning: expected 2 classes (ai/human), found {num_classes}: {data_meta['class_names']}"
        )

    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

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
    t0 = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, use_amp
        )
        va_loss, va_acc = evaluate(model, eval_loader, criterion, device)
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
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"train_loss={tr_loss:.4f} acc={tr_acc:.4f}  "
            f"val_loss={va_loss:.4f} acc={va_acc:.4f}"
        )

        if va_acc > best_val_acc or (
            va_acc == best_val_acc and va_loss < best_val_loss
        ):
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
                "idx_to_class": {str(k): v for k, v in data_meta["idx_to_class"].items()},
                "model_state_dict": model.state_dict(),
            }
            torch.save(payload, args.checkpoint)
            print(f"  saved new best -> {args.checkpoint}")

    elapsed = time.perf_counter() - t0
    metrics = {
        "model": "vit_b_16",
        "dataset": "hassnainzaidi/ai-art-vs-human-art",
        "train_dir": str(train_dir.resolve()),
        "eval_dir": str(eval_dir.resolve()),
        "num_classes": num_classes,
        "class_names": data_meta["class_names"],
        "class_to_idx": data_meta["class_to_idx"],
        "epochs_ran": args.epochs,
        "best_epoch": best_epoch,
        "best_val_acc": round(best_val_acc, 6),
        "best_val_loss": round(best_val_loss, 6),
        "train_seconds": round(elapsed, 2),
        "hyperparams": {
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "device": str(device),
        },
        "checkpoint": str(args.checkpoint.resolve()),
        "history": history,
    }
    args.metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote metrics -> {args.metrics_out}")


if __name__ == "__main__":
    main()
