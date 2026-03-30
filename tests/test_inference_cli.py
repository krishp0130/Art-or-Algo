"""Smoke-test inference CLI JSON for Node integration (requires checkpoint + val image)."""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


class TestInferenceCLI(unittest.TestCase):
    def test_stdout_is_single_json_object(self) -> None:
        ckpt = ROOT / "models" / "final_vit.pth"
        if not ckpt.is_file():
            ckpt = ROOT / "models" / "best_vit.pth"
        if not ckpt.is_file():
            self.skipTest("No checkpoint; train with run_train.py first.")

        candidates = [
            ROOT / "data" / "val" / "ai",
            ROOT / "data" / "val" / "human",
        ]
        img: Path | None = None
        for d in candidates:
            if d.is_dir():
                for p in d.iterdir():
                    if p.is_file() and p.suffix.lower() in {
                        ".png",
                        ".jpg",
                        ".jpeg",
                        ".webp",
                    }:
                        img = p
                        break
            if img:
                break
        if img is None:
            self.skipTest("No validation images; run split_train_eval.py.")

        cmd = [
            PY,
            "-m",
            "ml.inference",
            "--image",
            str(img),
            "--checkpoint",
            str(ckpt),
            "--device",
            "cpu",
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        out = json.loads(proc.stdout.strip())
        self.assertTrue(out.get("ok"))
        self.assertIn(out["prediction"], ("ai", "human"))
        self.assertGreaterEqual(float(out["confidence"]), 0.0)
        self.assertLessEqual(float(out["confidence"]), 1.0)
        self.assertIn("attention_map_base64", out)
        self.assertGreater(len(out["attention_map_base64"]), 100)


if __name__ == "__main__":
    unittest.main()
