#!/usr/bin/env bash
# Создаёт task-2/task-2-submission.zip из содержимого submission/ (плоский корень zip).
#
#   ./scripts/zip-submission.sh           # быстро (zip -0)
#   ./scripts/zip-submission.sh --compress
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUB="$ROOT/submission"
OUT="$ROOT/task-2-submission.zip"

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

EXCLUDE=(
  "*.git*"
  "*__MACOSX*"
  "*__pycache__*"
  "*.pyc"
  "weights_backup/*"
  "weights_original/*"
  "weights_merged/*"
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
