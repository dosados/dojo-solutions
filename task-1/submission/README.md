# Сдача на контест (task-1)

В этой папке — **только** то, что попадает в zip и в Docker-образ. Вспомогательные скрипты, документация и примеры лежат на уровень выше: `task-1/`.

## Содержимое

| Файл / папка | Назначение |
|--------------|------------|
| `Dockerfile` | Образ для проверки |
| `solution.py` | Точка входа (`ENTRYPOINT`) |
| `weights/` | Веса модели (~10 GB), см. `../download_weights.py` |

## Подготовка весов

Из `task-1/`:

```bash
conda run -n ml-env python download_weights.py
```

## Сборка и smoke test

Из `task-1/submission/`:

```bash
docker build -t task-1-solution .
mkdir -p ../out
docker run --rm --gpus all \
  -v "$PWD/../examples/example-input.pickle":/workspace/input.pickle \
  -v "$PWD/../out/dev_output.json":/workspace/output.json \
  task-1-solution
cat ../out/dev_output.json
```

Локально без копирования pickle в `submission/`:

```bash
conda run -n ml-env python solution.py \
  --input ../data/processed/self/dev.input.pickle \
  --output ../out/dev_output.json \
  --progress
```

Или из `task-1/`:

```bash
docker build -t task-1-solution ./submission
```

## Упаковка в zip

Корень архива = содержимое **этой** папки (без обёртки `submission/`):

```bash
cd task-1
./scripts/zip-submission.sh
```

Быстро (без сжатия, ~минуты вместо десятков минут на 10 GB):

```bash
./scripts/zip-submission.sh          # zip -0, STORE
./scripts/zip-submission.sh --compress  # меньше архив, намного дольше
```

Вручную: `cd submission && zip -0 -r ../task-1-submission.zip . -x "weights/.cache/*"`

Не включайте `__MACOSX`, не делайте двойную вложенность `submission/submission/…`.
