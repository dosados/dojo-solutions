#!/usr/bin/env bash
# Упаковка submission_translate/ для контеста (плоский корень zip).
# В архив: только Dockerfile, solution.py, weights/ — без backup/merged/README и т.п.
#
#   ./scripts/zip-submission-translate.sh           # быстро (zip -0)
#   ./scripts/zip-submission-translate.sh --compress
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUB="$ROOT/submission_translate"
OUT="$ROOT/task-2-translategemma-submission.zip"

ZIP_LEVEL=0
if [[ "${1:-}" == "--compress" ]]; then
  ZIP_LEVEL=6
elif [[ -n "${1:-}" ]]; then
  echo "usage: $0 [--compress]" >&2
  exit 1
fi

if [[ ! -f "$SUB/Dockerfile" || ! -f "$SUB/solution.py" ]]; then
  echo "error: missing submission_translate/Dockerfile or solution.py" >&2
  exit 1
fi
if [[ ! -d "$SUB/weights" ]]; then
  echo "error: submission_translate/weights/ not found — merge or download_weights_translate.py first" >&2
  exit 1
fi
if [[ ! -f "$SUB/weights/config.json" ]]; then
  echo "error: submission_translate/weights/config.json missing — weights look incomplete" >&2
  exit 1
fi

# Только то, что нужно контейнеру; явный список вместо zip всего каталога
INCLUDE=(Dockerfile solution.py weights/)

EXCLUDE=(
  "weights/.cache/*"
  "weights/.cache/**"
  "weights/__pycache__/*"
  "weights/**/__pycache__/*"
)

rm -f "$OUT"
echo "packing $SUB → $OUT (zip -$ZIP_LEVEL, ${INCLUDE[*]}) …"
SECONDS=0
(
  cd "$SUB"
  ZIP_ARGS=(-r -"$ZIP_LEVEL" "$OUT" "${INCLUDE[@]}")
  for pat in "${EXCLUDE[@]}"; do
    ZIP_ARGS+=(-x "$pat")
  done
  zip "${ZIP_ARGS[@]}"
)
echo "created: $OUT ($(du -h "$OUT" | cut -f1)) in ${SECONDS}s"
