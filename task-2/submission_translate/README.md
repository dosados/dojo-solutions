# TranslateGemma-4B — сдача на контест (task-2)

Вариант бейзлайна с [google/translategemma-4b-it](https://huggingface.co/google/translategemma-4b-it) вместо Gemma-4-E2B. Формат zip/Docker тот же: `Dockerfile`, `solution.py`, `weights/`.

## Содержимое

| Файл / папка | Назначение |
|--------------|------------|
| `Dockerfile` | Образ для проверки (тот же базовый образ, что у `submission/`) |
| `solution.py` | Точка входа (`ENTRYPOINT`), перевод `ru` → `ab` |
| `weights/` | Веса TranslateGemma-4B (~8 GB), см. `../download_weights_translate.py` |

## Подготовка весов

Из `task-2/` (нужен доступ к Hugging Face и принятая лицензия Gemma):

```bash
conda activate ml-env
python download_weights_translate.py
```

## Сборка и smoke test

Из `task-2/submission_translate/`:

```bash
docker build -t task-2-translategemma .
mkdir -p ../out
touch ../out/output.json
docker run --rm --gpus all \
  -v "$PWD/../examples/example-input.pickle":/workspace/input.pickle \
  -v "$PWD/../out/output.json":/workspace/output.json \
  task-2-translategemma
cat ../out/output.json
```

## Упаковка в zip

Корень архива = содержимое **этой** папки (без обёртки `submission_translate/`):

```bash
cd task-2/submission_translate
zip -r ../task-2-translategemma-submission.zip .
```

Не включайте `__MACOSX`, не делайте двойную вложенность папок.

## Локальная оценка (dev + BLEU)

Из `task-2/`:

```bash
conda activate ml-env
python -u scripts/run_local_eval.py --prepare-dev --limit 20 --show-text
python -u scripts/run_local_eval.py --prepare-dev --show-text
```

Скрипт берёт выборку из `data/ab-ru-parallel.csv`, гоняет `submission_translate/solution.py`, считает macro BLEU через `task-1/eval`.
