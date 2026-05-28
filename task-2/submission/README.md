# Сдача на контест (task-2)

В этой папке — **только** то, что попадает в zip и в Docker-образ. Вспомогательные скрипты, документация и примеры лежат на уровень выше: `task-2/`.

## Содержимое

| Файл / папка | Назначение |
|--------------|------------|
| `Dockerfile` | Образ для проверки |
| `solution.py` | Точка входа (`ENTRYPOINT`) |
| `weights/` | Веса модели (~10 GB), см. `../download_weights.py` |

## Подготовка весов

Из `task-2/`:

```bash
conda run -n ml-env python download_weights.py
```

## Сборка и smoke test

Из `task-2/submission/`:

```bash
docker build -t task-2-solution .
mkdir -p ../out
touch ../out/output.json
docker run --rm --gpus all \
  -v "$PWD/../examples/example-input.pickle":/workspace/input.pickle \
  -v "$PWD/../out/output.json":/workspace/output.json \
  task-2-solution
cat ../out/output.json
```

Или из `task-2/`:

```bash
docker build -t task-2-solution ./submission
```

## Упаковка в zip

Корень архива = содержимое **этой** папки (без обёртки `submission/`):

```bash
cd task-2
./scripts/zip-submission.sh
```

Не включайте `__MACOSX`, не делайте двойную вложенность `submission/submission/…`.
