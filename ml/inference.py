"""Load fine-tuned ViT-B/16, run classification, Attention Rollout, and heatmap overlays."""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image
from torchvision.models import VisionTransformer

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ml.dataset import vit_eval_transforms
from ml.train import build_model, pick_device


RolloutMode = Literal["rollout", "last_layer"]


@dataclass
class LoadedViT:
    model: VisionTransformer
    device: torch.device
    class_to_idx: dict[str, int]
    idx_to_class: dict[int, str]
    num_classes: int


def _ensure_rgb(img: Image.Image) -> Image.Image:
    return img.convert("RGB")


def load_model(
    checkpoint_path: str | Path,
    device: str | torch.device | None = None,
) -> LoadedViT:
    """Load ``best_vit.pth`` (or compatible) and return a handle for inference."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    class_to_idx_raw = ckpt.get("class_to_idx") or {}
    class_to_idx = {str(k): int(v) for k, v in class_to_idx_raw.items()}
    num_classes = len(class_to_idx)

    model = build_model(num_classes)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    if device is None:
        dev = pick_device(None)
    elif isinstance(device, torch.device):
        dev = device
    else:
        dev = pick_device(device if device != "auto" else None)
    model.to(dev)
    model.eval()

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return LoadedViT(
        model=model,
        device=dev,
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        num_classes=num_classes,
    )


def preprocess_image(
    image: Image.Image | str | Path,
    *,
    size: int = 224,
) -> tuple[torch.Tensor, Image.Image]:
    """Return a normalized batch ``(1, 3, H, W)`` for the ViT and the RGB ``PIL.Image`` (resized)."""
    if isinstance(image, (str, Path)):
        pil = _ensure_rgb(Image.open(image))
    else:
        pil = _ensure_rgb(image)

    t = vit_eval_transforms(size=size)(pil)
    return t.unsqueeze(0), pil


def preprocess_original_for_overlay(image: Image.Image | str | Path) -> Image.Image:
    """RGB PIL of the **original** resolution (for heatmap overlay in web previews)."""
    if isinstance(image, (str, Path)):
        return _ensure_rgb(Image.open(image))
    return _ensure_rgb(image)


def predict(
    session: LoadedViT,
    image: Image.Image | str | Path,
    *,
    size: int = 224,
) -> dict[str, Any]:
    """Return class name/id, confidence (max softmax), and per-class probabilities."""
    x, _ = preprocess_image(image, size=size)
    x = x.to(session.device, non_blocking=True)

    with torch.no_grad():
        logits = session.model(x)
        probs = torch.softmax(logits, dim=1)[0]

    conf, pred_id = probs.max(dim=0)
    pred_id_int = int(pred_id.item())
    p_list = probs.detach().cpu().tolist()
    names = [session.idx_to_class[i] for i in range(len(p_list))]
    return {
        "label_id": pred_id_int,
        "label": session.idx_to_class[pred_id_int],
        "confidence": float(conf.item()),
        "probabilities": {names[i]: float(p_list[i]) for i in range(len(names))},
    }


@contextmanager
def _capture_encoder_self_attention(model: VisionTransformer):
    """Temporarily patch ViT encoder blocks to expose MHA weights (torchvision defaults to ``need_weights=False``)."""
    storage: list[torch.Tensor] = []
    blocks = list(model.encoder.layers)
    originals: list[Any] = []
    for block in blocks:
        orig_forward = block.forward

        def patched_forward(inp: torch.Tensor, b=block) -> torch.Tensor:
            x = b.ln_1(inp)
            x_out, attn_w = b.self_attention(
                x,
                x,
                x,
                need_weights=True,
                average_attn_weights=True,
            )
            storage.append(attn_w.detach())
            x = b.dropout(x_out)
            x = x + inp
            y = b.ln_2(x)
            y = b.mlp(y)
            return x + y

        block.forward = patched_forward
        originals.append((block, orig_forward))
    try:
        yield storage
    finally:
        for block, orig_forward in originals:
            block.forward = orig_forward


def _encoder_forward_tokens(model: VisionTransformer, x: torch.Tensor) -> torch.Tensor:
    """Run patch stem + class token + encoder (ln + blocks) like ``VisionTransformer.forward`` without ``heads``."""
    patches = model._process_input(x)
    n = patches.shape[0]
    cls = model.class_token.expand(n, -1, -1)
    tokens = torch.cat([cls, patches], dim=1)
    tokens = tokens + model.encoder.pos_embedding
    tokens = model.encoder.dropout(tokens)
    tokens = model.encoder.layers(tokens)
    tokens = model.encoder.ln(tokens)
    return tokens


def attention_rollout_tensor(
    attentions: list[torch.Tensor],
    *,
    mode: RolloutMode = "rollout",
) -> torch.Tensor:
    """Combine self-attention maps into CLS→patch importance (length ``num_patches``)."""
    if not attentions:
        raise ValueError("No attention maps captured.")

    if mode == "last_layer":
        attn = attentions[-1][0]
        seq = attn.shape[0]
        eye = torch.eye(seq, device=attn.device, dtype=attn.dtype)
        a = attn + eye
        a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        mask = a[0, 1:]
    else:
        mats = [m[0] for m in attentions]
        seq_len = mats[0].shape[0]
        eye = torch.eye(seq_len, device=mats[0].device, dtype=mats[0].dtype)
        result = torch.eye(seq_len, device=mats[0].device, dtype=mats[0].dtype)
        for a in mats:
            a_ = a + eye
            a_ = a_ / a_.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            result = a_ @ result
        mask = result[0, 1:]

    mask = mask - mask.min()
    denom = mask.max().clamp_min(1e-8)
    mask = mask / denom
    return mask


def patch_grid_side(model: VisionTransformer) -> int:
    n_patch = model.image_size // model.patch_size
    return n_patch


def attention_heatmap_2d(
    session: LoadedViT,
    image: Image.Image | str | Path,
    *,
    size: int = 224,
    rollout_mode: RolloutMode = "rollout",
) -> np.ndarray:
    """2D float map ``(H, W)`` in ``[0, 1]`` on the ViT input grid (``size``×``size``)."""
    x, _ = preprocess_image(image, size=size)
    x = x.to(session.device, non_blocking=True)

    with torch.no_grad():
        with _capture_encoder_self_attention(session.model) as attn_list:
            _ = _encoder_forward_tokens(session.model, x)

    vec = attention_rollout_tensor(attn_list, mode=rollout_mode)
    side = patch_grid_side(session.model)
    if vec.numel() != side * side:
        raise RuntimeError(
            f"Patch count mismatch: got {vec.numel()} values, expected {side * side}."
        )
    patch_map = vec.reshape(side, side).detach().float().cpu().numpy()

    t = torch.from_numpy(patch_map)[None, None, ...]
    up = F.interpolate(t, size=(size, size), mode="bicubic", align_corners=False)
    return up[0, 0].numpy()


def overlay_attention_heatmap(
    original: Image.Image,
    heatmap_hw: np.ndarray,
    *,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> Image.Image:
    """Blend a colormapped attention map over the **original** RGB image."""
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1].")

    base = np.asarray(_ensure_rgb(original), dtype=np.float32) / 255.0
    h, w = base.shape[:2]

    hm = np.clip(heatmap_hw.astype(np.float32), 0.0, 1.0)
    if hm.shape != (h, w):
        pil_hm = Image.fromarray(np.uint8(hm * 255))
        pil_hm = pil_hm.resize((w, h), Image.Resampling.BICUBIC)
        hm = np.asarray(pil_hm, dtype=np.float32) / 255.0

    cmap = colormaps[colormap]
    color = cmap(hm)[..., :3].astype(np.float32)

    out = (1.0 - alpha) * base + alpha * color
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def heatmap_to_buffer(
    image: Image.Image,
    *,
    format: str = "PNG",
) -> bytes:
    """Encode a PIL image as raw bytes (e.g. ``PNG`` / ``JPEG``)."""
    buf = io.BytesIO()
    image.save(buf, format=format)
    return buf.getvalue()


def heatmap_to_base64(
    image: Image.Image,
    *,
    format: str = "PNG",
) -> str:
    """Encode a PIL image as a base64 string (no data URL prefix)."""
    return base64.b64encode(heatmap_to_buffer(image, format=format)).decode("ascii")


def heatmap_data_url(
    image: Image.Image,
    *,
    format: str = "PNG",
) -> str:
    """``data:image/png;base64,...`` suitable for HTML ``<img src>``."""
    mime = "image/png" if format.upper() == "PNG" else f"image/{format.lower()}"
    b64 = heatmap_to_base64(image, format=format)
    return f"data:{mime};base64,{b64}"


def explain_prediction(
    session: LoadedViT,
    image: Image.Image | str | Path,
    *,
    size: int = 224,
    rollout_mode: RolloutMode = "rollout",
    overlay_alpha: float = 0.45,
    image_format: str = "PNG",
    original_for_overlay: Image.Image | None = None,
) -> dict[str, Any]:
    """One-call helper: prediction + heatmap buffer/base64 for the web Tier."""
    pred = predict(session, image, size=size)
    hm2d = attention_heatmap_2d(
        session, image, size=size, rollout_mode=rollout_mode
    )
    orig = (
        preprocess_original_for_overlay(original_for_overlay)
        if original_for_overlay is not None
        else preprocess_original_for_overlay(image)
    )
    overlay = overlay_attention_heatmap(orig, hm2d, alpha=overlay_alpha)
    buff = heatmap_to_buffer(overlay, format=image_format)
    b64 = base64.b64encode(buff).decode("ascii")
    return {
        **pred,
        "heatmap_png_base64": b64,
        "heatmap_data_url": heatmap_data_url(overlay, format=image_format),
        "rollout_mode": rollout_mode,
    }


def predict_json_for_backend(
    session: LoadedViT,
    image: Image.Image | str | Path,
    *,
    size: int = 224,
    rollout_mode: RolloutMode = "rollout",
    overlay_alpha: float = 0.45,
    original_for_overlay: Image.Image | None = None,
) -> dict[str, Any]:
    """Structured dict for Node: prediction, confidence, probabilities, attention map (base64 PNG)."""
    pred = predict(session, image, size=size)
    hm2d = attention_heatmap_2d(
        session, image, size=size, rollout_mode=rollout_mode
    )
    orig = (
        preprocess_original_for_overlay(original_for_overlay)
        if original_for_overlay is not None
        else preprocess_original_for_overlay(image)
    )
    overlay = overlay_attention_heatmap(orig, hm2d, alpha=overlay_alpha)
    b64 = heatmap_to_base64(overlay, format="PNG")
    return {
        "prediction": pred["label"],
        "confidence": pred["confidence"],
        "label_id": pred["label_id"],
        "probabilities": pred["probabilities"],
        "attention_map_base64": b64,
    }


def _default_checkpoint() -> Path:
    final_p = _ROOT / "models" / "final_vit.pth"
    best_p = _ROOT / "models" / "best_vit.pth"
    if final_p.is_file():
        return final_p
    if best_p.is_file():
        return best_p
    return final_p


def main_cli() -> None:
    """Print a single JSON object on stdout for ``child_process`` / Express bridges."""
    parser = argparse.ArgumentParser(description="ViT inference JSON for Node.js backend.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--rollout-mode",
        choices=("rollout", "last_layer"),
        default="rollout",
    )
    args = parser.parse_args()

    ckpt = args.checkpoint or _default_checkpoint()
    if not ckpt.is_file():
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": f"Checkpoint not found: {ckpt}. Train with run_train.py first.",
                }
            ),
            flush=True,
        )
        sys.exit(1)
    if not args.image.is_file():
        print(
            json.dumps({"ok": False, "error": f"Image not found: {args.image}"}),
            flush=True,
        )
        sys.exit(1)

    session = load_model(
        ckpt,
        device=args.device if args.device != "auto" else None,
    )
    out = predict_json_for_backend(
        session,
        args.image,
        rollout_mode=args.rollout_mode,
    )
    out["ok"] = True
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main_cli()
