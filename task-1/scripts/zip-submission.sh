#!/usr/bin/env bash
# Создаёт task-1/task-1-submission.zip из содержимого submission/ (плоский корень zip).
#
# По умолчанию zip -0 (STORE, без сжатия) — в 5–20× быстрее, чем zip -9.
# model.safetensors уже сжат; выигрыш по размеру от -9 обычно <5%.
#
#   ./scripts/zip-submission.sh           # быстро (рекомендуется)
#   ./scripts/zip-submission.sh --compress  # меньше файл, долго
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUB="$ROOT/submission"
OUT="$ROOT/task-1-submission.zip"

ZIP_LEVEL=0
if [[ "${1:-}" == "--compress" ]]; then
  ZIP_LEVEL=6
elif [[ -n "${1:-}" ]]; then
  echo "usage: $0 [--compress]" >&2
  exit 1
fi

if [[ ! -f "$SUB/Dockerfile" || ! -f "$SUB/solution.py" ]]; then
  echo "error: missing submission/Dockerfile or submission/solution.py" >&2
  exit 1
fi
if [[ ! -d "$SUB/weights" ]]; then
  echo "warning: submission/weights/ not found — run download_weights.py first" >&2
fi

# Исключения: не раздувать архив и не тратить время на мусор
EXCLUDE=(
  "*.git*"
  "*__MACOSX*"
  "*__pycache__*"
  "*.pyc"
  "weights/.cache/*"
  "weights/.cache/**"
)

rm -f "$OUT"
echo "packing $SUB → $OUT (zip -$ZIP_LEVEL) …"
SECONDS=0
(
  cd "$SUB"
  ZIP_ARGS=(-r -"$ZIP_LEVEL" "$OUT" .)
  for pat in "${EXCLUDE[@]}"; do
    ZIP_ARGS+=(-x "$pat")
  done
  zip "${ZIP_ARGS[@]}"
)
echo "created: $OUT ($(du -h "$OUT" | cut -f1)) in ${SECONDS}s"
