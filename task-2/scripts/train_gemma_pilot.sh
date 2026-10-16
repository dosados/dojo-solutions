#!/usr/bin/env bash
# Полный pilot-пайплайн QLoRA для Gemma-4-E2B (task-2).
# CLM (mono ru+ab) → SFT (parallel ru→ab), аналог translate.
#
# Требуется:
#   submission/weights/     — python download_weights.py
#   data/processed/         — ./scripts/prepare_training_data.sh
#
#   ./scripts/train_gemma_pilot.sh           # CLM + SFT
#   ./scripts/train_gemma_pilot.sh clm       # только stage 1
#   ./scripts/train_gemma_pilot.sh sft       # только stage 2
#   ./scripts/train_gemma_pilot.sh smoke     # 10 steps
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -d submission/weights ]]; then
  echo "error: submission/weights/ missing — run: python download_weights.py" >&2
  exit 1
fi

STAGE="${1:-all}"

run() {
  echo ">>> $*" >&2
  python -u "$@"
}

case "$STAGE" in
  smoke)
    run scripts/train_qlora.py --config configs/gemma/clm_smoke.yaml
    run scripts/train_qlora.py --config configs/gemma/sft_smoke.yaml
    ;;
  clm)
    run scripts/train_qlora.py --config configs/gemma/clm_pilot.yaml
    ;;
  sft)
    run scripts/train_qlora.py --config configs/gemma/sft_pilot.yaml
    ;;
  all)
    run scripts/train_qlora.py --config configs/gemma/clm_pilot.yaml
    run scripts/train_qlora.py --config configs/gemma/sft_pilot.yaml
    ;;
  *)
    echo "usage: $0 [all|clm|sft|smoke]" >&2
    exit 1
    ;;
esac

echo "done ($STAGE)." >&2
echo "  validate: python -u scripts/validate_lora.py --model gemma --adapter runs/lora/gemma/sft_pilot/final --limit 50" >&2
echo "  merge:    python -u scripts/merge_lora.py --model gemma --adapter runs/lora/gemma/sft_pilot/final --out submission/weights --backup --device cpu" >&2
echo "  zip:      ./scripts/zip-submission.sh" >&2
