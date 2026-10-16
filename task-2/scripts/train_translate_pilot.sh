#!/usr/bin/env bash
# Полный pilot-пайплайн QLoRA для TranslateGemma-4B (task-2).
#
#   ./scripts/train_translate_pilot.sh           # CLM + SFT
#   ./scripts/train_translate_pilot.sh clm|sft|smoke
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

STAGE="${1:-all}"

run() {
  echo ">>> $*" >&2
  python -u "$@"
}

case "$STAGE" in
  smoke)
    run scripts/train_qlora.py --config configs/translate/clm_smoke.yaml
    run scripts/train_qlora.py --config configs/translate/sft_smoke.yaml
    ;;
  clm)
    run scripts/train_qlora.py --config configs/translate/clm_pilot.yaml
    ;;
  sft)
    run scripts/train_qlora.py --config configs/translate/sft_pilot.yaml
    ;;
  all)
    run scripts/train_qlora.py --config configs/translate/clm_pilot.yaml
    run scripts/train_qlora.py --config configs/translate/sft_pilot.yaml
    ;;
  *)
    echo "usage: $0 [all|clm|sft|smoke]" >&2
    exit 1
    ;;
esac

echo "done ($STAGE). validate:" >&2
echo "  python -u scripts/validate_lora.py --model translate --adapter runs/lora/translate/sft_pilot/final --limit 50" >&2
