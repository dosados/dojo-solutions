# Task 1 — EN→RU (абзацы)

## Структура

```
task-1/
├── submission/          # ← zip и Docker: Dockerfile, solution.py, weights/
├── download_weights.py  # скачивает веса в submission/weights/
├── examples/            # example-input.pickle, example-output.json
├── scripts/             # zip-submission.sh и др.
└── docs/                # SPECS, BLEU, стратегии, roadmap
```

**Правило:** меняете только `submission/` (и веса через `download_weights.py`). Всё остальное в `task-1/` на контест не влияет.

## Быстрый старт

```bash
cd task-1
conda run -n ml-env python download_weights.py

cd submission
docker build -t task-1-solution .
mkdir -p ../out
docker run --rm --gpus all \
  -v "$PWD/../examples/example-input.pickle":/workspace/input.pickle \
  -v "$PWD/../out":/workspace/out \
  task-1-solution
```

## Данные и локальная оценка BLEU

```bash
conda run -n ml-env pip install -r requirements-eval.txt

# OPUS (если ещё нет): scripts/download_opus100.py
# Конвертация self_data + opus → data/processed/
conda run -n ml-env python scripts/convert_datasets.py

# Прогон бейзлайна на абзачном dev (статьи 1–4)
# ~8GB VRAM (RTX 5070 Laptop): 4-bit, иначе OOM на полной bf16-модели
cd submission && conda run -n ml-env python -u solution.py \
  --device cuda --quantize 4bit \
  --input ../data/processed/self/dev.input.pickle \
  --output ../out/dev_output.json \
  --progress
cd .. && conda run -n ml-env python scripts/eval_bleu.py \
  --hyp out/dev_output.json \
  --ref data/processed/self/dev.jsonl --worst 5 --show-text
```

- [docs/DATASETS.md](docs/DATASETS.md) — какие корпуса скачать
- [docs/BLEU.md](docs/BLEU.md) — как считается метрика на контесте
- Модуль: `task-1/eval/`

## LoRA (три этапа: train+val → merge → zip)

```bash
conda activate ml-env
cd task-1
pip install -r requirements-train.txt

# 1) обучение + BLEU
python -u scripts/lora_train_validate.py --config configs/lora_pilot.yaml

# 2) merge в submission/weights (отдельно)
python -u scripts/merge_lora.py --adapter runs/lora/pilot_10k/final --device cpu --backup

# 3) zip для контеста (отдельно)
./scripts/zip-submission.sh
```

Подробно: [docs/TRAINING.md](docs/TRAINING.md).

## Документация

- [docs/SPECS.md](docs/SPECS.md) — условие
- [docs/SOLUTION_STRATEGIES.md](docs/SOLUTION_STRATEGIES.md) — подходы
- [docs/IMPLEMENTATION_ROADMAP.md](docs/IMPLEMENTATION_ROADMAP.md) — план работ
- [docs/TRAINING.md](docs/TRAINING.md) — QLoRA, merge, zip
