#!/usr/bin/env python3
"""
Run image classification + attention heatmap for Art or Algorithm.

Usage:
  python3 inference.py <image_path> <heatmap_output.png>

Prints a single JSON object on stdout (last line must be valid JSON):
  {
    "prediction": "human" | "ai",
    "confidence": <float 0..1>,
    "heatmapData": <optional base64 or null>
  }

Replace the stub logic below with your real model, Grad-CAM, etc.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# 1×1 PNG (stdlib only) as a stand-in heatmap asset
_MINIMAL_PNG = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
    b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: inference.py <image_path> <heatmap_output.png>", file=sys.stderr)
        sys.exit(2)

    image_path = Path(sys.argv[1])
    heatmap_path = Path(sys.argv[2])

    if not image_path.is_file():
        print(f"Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    heatmap_path.parent.mkdir(parents=True, exist_ok=True)
    heatmap_path.write_bytes(_MINIMAL_PNG)

    out = {
        "prediction": "human",
        "confidence": 0.5,
        "heatmapData": None,
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
