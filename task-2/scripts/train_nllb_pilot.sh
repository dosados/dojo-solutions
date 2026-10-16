#!/usr/bin/env bash
# Pilot QLoRA для NLLB-200-3.3B (task-2).
#
#   ./scripts/train_nllb_pilot.sh           # CLM + SFT
#   ./scripts/train_nllb_pilot.sh clm|sft|smoke
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
    run scripts/train_qlora_nllb.py --config configs/nllb/clm_smoke.yaml
    run scripts/train_qlora_nllb.py --config configs/nllb/sft_smoke.yaml
    ;;
  clm)
    run scripts/train_qlora_nllb.py --config configs/nllb/clm_pilot.yaml
    ;;
  sft)
    run scripts/train_qlora_nllb.py --config configs/nllb/sft_pilot.yaml
    ;;
  all)
    run scripts/train_qlora_nllb.py --config configs/nllb/clm_pilot.yaml
    run scripts/train_qlora_nllb.py --config configs/nllb/sft_pilot.yaml
    ;;
  *)
    echo "usage: $0 [all|clm|sft|smoke]" >&2
    exit 1
    ;;
esac

echo "done ($STAGE). validate:" >&2
echo "  python -u scripts/validate_lora.py --model nllb --adapter runs/lora/nllb/sft_pilot/final --limit 50" >&2
