#!/usr/bin/env bash
# Create venv, install Python deps, verify Kaggle split under data/train and data/val.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "==> Project root: $ROOT"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found. Install Python 3.10+ and retry."
  exit 1
fi

if [[ ! -d .venv ]]; then
  echo "==> Creating virtual environment (.venv) ..."
  python3 -m venv .venv
else
  echo "==> Using existing .venv"
fi
# shellcheck source=/dev/null
source .venv/bin/activate

echo "==> Upgrading pip ..."
python -m pip install --upgrade pip

echo "==> Installing requirements.txt ..."
pip install -r requirements.txt

check_split() {
  local name="$1"
  local d="$ROOT/data/$name"
  local missing=0
  for sub in ai human; do
    if [[ ! -d "$d/$sub" ]]; then
      echo "  MISSING: $d/$sub"
      missing=1
      continue
    fi
    local n
    n="$(find "$d/$sub" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.webp' -o -iname '*.gif' -o -iname '*.bmp' \) | wc -l | tr -d ' ')"
    if [[ "$n" -eq 0 ]]; then
      echo "  WARNING: $d/$sub has no image files."
      missing=1
    else
      echo "  OK: $d/$sub ($n images)"
    fi
  done
  return "$missing"
}

echo "==> Verifying dataset layout (data/train, data/val) ..."
ok=0
if [[ ! -d "$ROOT/data/train" ]] || [[ ! -d "$ROOT/data/val" ]]; then
  echo "ERROR: Expected data/train and data/val."
  echo "  1) scripts/download_dataset.sh"
  echo "  2) python3 scripts/split_train_eval.py"
  if [[ -d "$ROOT/data/eval" ]] && [[ ! -d "$ROOT/data/val" ]]; then
    echo "  Hint: you have data/eval — rename with:  mv data/eval data/val"
  fi
  exit 1
fi

check_split train || ok=1
check_split val || ok=1

if [[ "$ok" -ne 0 ]]; then
  echo "ERROR: Dataset check failed. Fix folders above or re-run the split script."
  exit 1
fi

echo "==> Environment ready. Activate with:"
echo "    source .venv/bin/activate"
