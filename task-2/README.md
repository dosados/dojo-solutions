# Task 2 — RU→Abkhaz

## Структура

```
task-2/
├── submission/          # ← zip и Docker: Dockerfile, solution.py, weights/
├── download_weights.py  # скачивает веса в submission/weights/
├── examples/            # example-input.pickle, example-output.json
├── scripts/             # zip-submission.sh
└── docs/                # SPECS (условие контеста)
```

**Правило:** меняете только `submission/` (и веса через `download_weights.py`). Всё остальное в `task-2/` на контест не влияет.

## Быстрый старт

```bash
cd task-2
conda run -n ml-env python download_weights.py

cd submission
docker build -t task-2-solution .
mkdir -p ../out
docker run --rm --gpus all \
  -v "$PWD/../examples/example-input.pickle":/workspace/input.pickle \
  -v "$PWD/../out/output.json":/workspace/output.json \
  task-2-solution
cat ../out/output.json
```

Оригинальный архив организаторов: `ru-ab-gemma-4-baseline-submission/`.

## Документация

- [docs/SPECS.md](docs/SPECS.md) — условие (не редактировать)
