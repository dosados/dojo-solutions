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
conda activate ml-env
python download_weights_translate.py   # или download_weights.py для Gemma-4-E2B

# Локальная оценка TranslateGemma (dev из параллельного корпуса + BLEU)
python -u scripts/run_local_eval.py --prepare-dev --limit 20 --show-text
python -u scripts/run_local_eval.py --prepare-dev --show-text
```

Оригинальный архив организаторов: `ru-ab-gemma-4-baseline-submission/`.

## Документация

- [docs/SPECS.md](docs/SPECS.md) — условие (не редактировать)
