# QLoRA: обучение и посылка

Три **отдельных** этапа — не смешивать обучение/метрики с упаковкой zip.

| Этап | Скрипт | Что делает |
|------|--------|------------|
| **1. Train + validate** | `lora_train_validate.py` | QLoRA, BLEU baseline vs adapter |
| **2. Merge** | `merge_lora.py` | Адаптер → полный `model.safetensors` |
| **3. Package** | `zip-submission.sh` | `task-1-submission.zip` для контеста |

Локально: **4-bit QLoRA** при обучении и проверке адаптера. На контест: **merged** bf16 в `submission/weights/` (без `peft` в Docker).

---

## 0. Зависимости и данные

```bash
conda activate ml-env
cd task-1
pip install -r requirements-train.txt
pip install -r requirements-eval.txt

python scripts/convert_datasets.py
python download_weights.py
```

---

## GPU (~8 GB)

Обучение грузит модель в **4-bit целиком на `cuda:0`** (как `solution.py --device cuda --quantize 4bit`), без `device_map=auto`.

Если OOM на ~8GB: `low_vram: true` (включает `LowVRAMSFTTrainer` + `logits_to_keep`), `max_seq_length: 256`, `max_src_chars: 400`, `loss_type: nll` (не `chunked_nll` — он OOM на matmul lm_head). В логе должно быть `tokenize train`, не `Tokenizing train dataset` (это TRL без `skip_prepare_dataset`). Закройте другие процессы на GPU.

---

## Этап 1 — обучение + валидация

Проверка на 8 GB (15 шагов, ~30 с):

```bash
python -u scripts/train_lora.py --config configs/lora_smoke.yaml
```

Пилот (10k):

```bash
python -u scripts/lora_train_validate.py --config configs/lora_pilot.yaml
```

Только BLEU по уже обученному адаптеру:

```bash
python -u scripts/lora_train_validate.py --config configs/lora_pilot.yaml --skip-train
```

Отдельно (если нужно):

```bash
python -u scripts/train_lora.py --config configs/lora_pilot.yaml
python -u scripts/validate_lora.py --adapter runs/lora/pilot_10k/final --max-samples 300 --progress
```

Результаты: `runs/lora/<run>/final/`, метрики — `out/lora_val/scores.json`.

Рабочий прогон после удачного пилота:

```bash
python -u scripts/lora_train_validate.py --config configs/lora_100k.yaml
```

---

## Этап 2 — merge в веса посылки

Только если на этапе 1 LoRA лучше baseline на OPUS val.

```bash
python -u scripts/merge_lora.py \
  --adapter runs/lora/pilot_10k/final \
  --device cpu --backup
```

Опционально проверить merged:

```bash
python -u scripts/validate_lora.py --merged --max-samples 300 --skip-baseline --progress
```

Docker smoke:

```bash
cd submission && docker build -t task-1-solution .
```

---

## Этап 3 — zip (отправка на контест)

**Только когда веса и smoke готовы** — отдельно от train/val:

```bash
./scripts/zip-submission.sh
# → task-1-submission.zip
```

Быстрая упаковка без сжатия (по умолчанию); `./scripts/zip-submission.sh --compress` — меньше файл, дольше.

---

## Локальный инференс с адаптером (до merge)

```bash
cd submission
python -u solution.py \
  --device cuda --quantize 4bit \
  --lora-dir ../runs/lora/pilot_10k/final \
  --input ../data/processed/self/dev.input.pickle \
  --output ../out/dev_lora.json --progress
```

---

## Критерии перехода между этапами

| Переход | Условие |
|---------|---------|
| 1 → 2 (merge) | OPUS val: LoRA > baseline в `out/lora_val/scores.json` |
| 2 → 3 (zip) | merge ok, Docker smoke на `examples/` |
| Контест | Не подбирать по self/Пушкину |

---

## Скрипты

```
configs/lora_pilot.yaml
configs/lora_100k.yaml
scripts/train_lora.py
scripts/lora_train_validate.py   # этап 1
scripts/validate_lora.py
scripts/merge_lora.py            # этап 2
scripts/zip-submission.sh        # этап 3
```
