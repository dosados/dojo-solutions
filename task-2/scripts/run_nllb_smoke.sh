#!/usr/bin/env bash
# Smoke: CLM (10 шагов) → SFT (10 шагов) → dev inference base + LoRA.
#
#   ./scripts/run_nllb_smoke.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

run() {
  echo ">>> $*" >&2
  python -u "$@"
}

run scripts/train_qlora_nllb.py --config configs/nllb/clm_smoke.yaml --no-auto-resume
run scripts/train_qlora_nllb.py --config configs/nllb/sft_smoke.yaml --no-auto-resume

run scripts/run_local_eval.py --model nllb --limit 10 --quantize 4bit --gpu-only --skip-bleu
run scripts/validate_lora.py --model nllb --adapter runs/lora/nllb/sft_smoke/final --limit 10

echo "smoke ok. pilot:" >&2
echo "  ./scripts/train_nllb_pilot.sh" >&2
