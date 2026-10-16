#!/usr/bin/env bash
# Два отдельных процесса Python: CLM pilot → SFT pilot (Gemma-4-E2B).
# После завершения первого процесса VRAM освобождается, затем стартует второй.
#
# Требуется (один раз):
#   conda activate ml-env
#   python download_weights.py
#   ./scripts/prepare_training_data.sh
#
# Запуск:
#   cd task-2
#   conda activate ml-env
#   ./scripts/run_gemma_pilot_sequential.sh
#
# Логи (опционально):
#   ./scripts/run_gemma_pilot_sequential.sh 2>&1 | tee logs/gemma_pilot_$(date +%Y%m%d_%H%M).log
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CLM_CONFIG="configs/gemma/clm_pilot.yaml"
SFT_CONFIG="configs/gemma/sft_pilot.yaml"
CLM_ADAPTER="runs/lora/gemma/clm_pilot/final"
SFT_ADAPTER="runs/lora/gemma/sft_pilot/final"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2
}

die() {
  log "ERROR: $*"
  exit 1
}

[[ -d submission/weights ]] || die "submission/weights/ missing — run: python download_weights.py"
[[ -f data/processed/mono/train.jsonl ]] || die "mono data missing — run: ./scripts/prepare_training_data.sh"
[[ -f data/processed/parallel/train.jsonl ]] || die "parallel data missing — run: ./scripts/prepare_training_data.sh"

if ! command -v python >/dev/null 2>&1; then
  die "python not in PATH — conda activate ml-env"
fi

log "=== Stage 1/2: CLM (mono ru+ab) ==="
log "config: $CLM_CONFIG"
log "starting new process …"

python -u scripts/train_qlora.py --config "$CLM_CONFIG"
CLM_EXIT=$?
if [[ "$CLM_EXIT" -ne 0 ]]; then
  die "CLM failed with exit code $CLM_EXIT"
fi

[[ -d "$CLM_ADAPTER" ]] || die "CLM adapter not found: $CLM_ADAPTER"

log "CLM done. GPU memory should be free before SFT."
sleep 3

log "=== Stage 2/2: SFT (parallel ru→ab) ==="
log "config: $SFT_CONFIG"
log "adapter_init: $CLM_ADAPTER"
log "starting new process …"

python -u scripts/train_qlora.py --config "$SFT_CONFIG"
SFT_EXIT=$?
if [[ "$SFT_EXIT" -ne 0 ]]; then
  die "SFT failed with exit code $SFT_EXIT"
fi

[[ -d "$SFT_ADAPTER" ]] || die "SFT adapter not found: $SFT_ADAPTER"

log "=== All training finished ==="
log "Adapter: $SFT_ADAPTER"
log ""
log "Next steps:"
log "  python -u scripts/validate_lora.py --model gemma --adapter $SFT_ADAPTER --limit 50"
log "  python -u scripts/merge_lora.py --model gemma --adapter $SFT_ADAPTER \\"
log "    --out submission/weights --backup --device cpu"
log "  ./scripts/zip-submission.sh"
