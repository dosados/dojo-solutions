# NLLB-200-3.3B — сдача на контест (task-2)

Вариант с [facebook/nllb-200-3.3B](https://huggingface.co/facebook/nllb-200-3.3B): encoder-decoder MT, `rus_Cyrl` → `abk_Cyrl`. Формат zip/Docker как у `submission_translate/`.

## Содержимое

| Файл / папка | Назначение |
|--------------|------------|
| `Dockerfile` | Образ для проверки |
| `solution.py` | Точка входа, батчевый greedy decode |
| `weights/` | Веса NLLB (~7 GB bf16 merged), см. `../download_weights_nllb.py` |

## Подготовка весов

Из `task-2/`:

```bash
conda activate ml-env
python download_weights_nllb.py
```

После QLoRA — merge и замена `weights/`:

```bash
python -u scripts/merge_lora.py --model nllb \
  --adapter runs/lora/nllb/sft_pilot/final \
  --out submission_nllb/weights --backup --device cpu
```

## QLoRA (те же данные, что TranslateGemma)

```bash
./scripts/prepare_training_data.sh

# Smoke (проверено: CLM + SFT + inference)
./scripts/run_nllb_smoke.sh

# Pilot
./scripts/train_nllb_pilot.sh
```

Локальная оценка:

```bash
python -u scripts/run_local_eval.py --model nllb --limit 50 --quantize 4bit --gpu-only
python -u scripts/validate_lora.py --model nllb --adapter runs/lora/nllb/sft_pilot/final --limit 50
```

Подробно: [../docs/TRAINING.md](../docs/TRAINING.md) (раздел NLLB).

## Сборка и smoke test

Из `task-2/submission_nllb/`:

```bash
docker build -t task-2-nllb .
mkdir -p ../out
touch ../out/output.json
docker run --rm --gpus all \
  -v "$PWD/../examples/example-input.pickle":/workspace/input.pickle \
  -v "$PWD/../out/output.json":/workspace/output.json \
  task-2-nllb
cat ../out/output.json
```

## Упаковка в zip

```bash
cd task-2/submission_nllb
zip -r ../task-2-nllb-submission.zip .
```

## Локальная оценка

```bash
cd task-2
python -u scripts/validate_lora.py --model nllb \
  --adapter runs/lora/nllb/sft_pilot/final --limit 50
```
