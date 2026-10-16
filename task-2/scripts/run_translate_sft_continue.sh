#!/usr/bin/env bash
# Продолжение SFT TranslateGemma: база weights_original + LoRA sft_r24_full_v1 → новый sft_r24_cont_v2.
#
# Не изменяет:
#   submission_translate/weights          (merged для контеста)
#   runs/lora/translate/sft_r24_full_v1/  (исходный адаптер r=24)
#
# Использование (из task-2/):
#   conda activate ml-env
#   ./scripts/run_translate_sft_continue.sh          # только проверки + команда
#   ./scripts/run_translate_sft_continue.sh run    # подготовить parallel + обучение
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG=configs/translate/sft_r24_cont_v2.yaml
BASE=submission_translate/weights_original
ADAPTER=runs/lora/translate/sft_r24_full_v1/final
TRAIN_JSONL=data/processed/parallel/train.jsonl
VAL_JSONL=data/processed/parallel/val.jsonl

die() { echo "error: $*" >&2; exit 1; }

[[ -d "$BASE" ]] || die "нет $BASE — скачайте веса: python download_weights_translate.py (или восстановите weights_original)"
[[ -f "$BASE/model.safetensors.index.json" || -f "$BASE/model.safetensors" ]] \
  || die "$BASE не похож на каталог HF-весов (ожидаются shard или single safetensors)"
[[ -d "$ADAPTER" ]] || die "нет $ADAPTER — нужен завершённый прогон sft_r24_full_v1"

echo "=== TranslateGemma SFT continue (r=24) ===" >&2
echo "  base (read-only):  $BASE" >&2
echo "  adapter_init:      $ADAPTER" >&2
echo "  output:            runs/lora/translate/sft_r24_cont_v2/" >&2
echo "  merged weights:    submission_translate/weights (не используются)" >&2
echo "" >&2

prepare_data() {
  if [[ -f "$TRAIN_JSONL" && -f "$VAL_JSONL" ]]; then
    echo "parallel jsonl уже есть: $TRAIN_JSONL" >&2
    return 0
  fi
  echo "prepare_parallel …" >&2
  python -u scripts/prepare_parallel.py --val-n 500
}

run_train() {
  prepare_data
  python -u scripts/train_qlora.py --config "$CONFIG" --no-auto-resume
}

case "${1:-}" in
  run)
    run_train
    ;;
  ""|check)
    prepare_data 2>/dev/null || true
    echo "Команда обучения:" >&2
    echo "  python -u scripts/train_qlora.py --config $CONFIG --no-auto-resume" >&2
    ;;
  *)
    die "usage: $0 [check|run]"
    ;;
esac
