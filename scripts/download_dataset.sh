#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/data"
cd "$ROOT"
exec kaggle datasets download -d hassnainzaidi/ai-art-vs-human-art -p "$ROOT/data" --unzip
