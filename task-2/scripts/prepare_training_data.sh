#!/usr/bin/env bash
# Подготовка данных для QLoRA (общая для gemma и translate).
#
#   ./scripts/prepare_training_data.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo ">>> download ru-wikipedia (skip if exists)" >&2
if [[ ! -f data/ru-wikipedia.jsonl ]]; then
  python -u download_ru_wikipedia.py
fi

echo ">>> mono CLM (весь ab-mono + ru-wikipedia + parallel RU)" >&2
python -u scripts/prepare_mono.py --ru-wikipedia --parallel-ru

echo ">>> parallel SFT + dev holdout" >&2
python -u scripts/prepare_parallel.py --val-n 500
python -u scripts/prepare_dev.py

echo "done. data/processed/{mono,parallel,dev.*}" >&2
