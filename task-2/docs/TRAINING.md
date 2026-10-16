# Task-2 — QLoRA (два этапа)

Три линии моделей:

- **TranslateGemma** (`submission_translate/`) — causal LM + chat-template
- **Gemma-4-E2B** (`submission/`) — causal LM
- **NLLB-200-3.3B** (`submission_nllb/`) — encoder-decoder seq2seq (`scripts/train_qlora_nllb.py`)

Два этапа для TranslateGemma и Gemma:

1. **CLM** — next-token prediction на смеси абхазского + русского моноязычного текста.
2. **SFT** — перевод RU→AB на параллельном корпусе (тот же формат, что в `solution.py`).

Обучение: **4-bit QLoRA**, модель целиком на `cuda:0` (`device_map={"": 0}`).
Merge для контеста: **bf16** полные веса (`scripts/merge_lora.py`).

## Зависимости

```bash
cd task-2
conda activate ml-env
pip install -r requirements-train.txt
```

Веса (один раз):

```bash
python download_weights_translate.py   # TranslateGemma
python download_weights.py             # Gemma-4-E2B
python download_weights_nllb.py        # NLLB-3.3B (~7 GB bf16)
```

## Данные

Один раз (общие для gemma и translate):

```bash
./scripts/prepare_training_data.sh
```

### Stage 1 — моноязычная смесь

Абхазский: `data/ab-mono.txt` (конкурсный корпус).

Русский:

| Источник | Как получить | Зачем |
|----------|--------------|-------|
| **misterkirill/ru-wikipedia** | `python -u download_ru_wikipedia.py` | ~69k коротких фрагментов Wikipedia RU |
| **RU из параллельного csv** | `--parallel-ru` | In-domain к стилю перевода |

```bash
python -u download_ru_wikipedia.py
python -u scripts/prepare_mono.py --ru-wikipedia --parallel-ru
```

Выход: `data/processed/mono/train.jsonl` (`text`, `lang`). При обучении CLM в yaml задана смесь **60% ab / 40% ru** (`mono_lang_weights`).

### Stage 2 — параллельный корпус

```bash
python -u scripts/prepare_parallel.py --val-n 500
```

По умолчанию: до 4096 символов RU, до 20 предложений (без урезания pilot-лимитов 120/3).
Pilot-обучение: **2 эпохи CLM**, **3 эпохи SFT** (`num_train_epochs` в `*_pilot.yaml`).

Выход: `data/processed/parallel/train.jsonl`, `val.jsonl` (`src`, `ref`).

Dev для BLEU (как раньше):

```bash
python -u scripts/prepare_dev.py
```

## Обучение

### Gemma-4-E2B (рекомендуется для контеста — быстрее)

~2B параметров (inference), но архитектура Gemma4 multimodal — **тяжелее TranslateGemma на 8GB VRAM даже в 4-bit QLoRA**.
CLM (loss на **всех** токенах): `max_seq_length: 128` для gemma (VRAM 8GB). SFT (только перевод): `256`. TranslateGemma pilot: `max_seq_length: 1024`.

Промпт SFT совпадает с `submission/solution.py` (`train/prompts.py`).

Веса: `python download_weights.py` (обязательно в `submission/weights/`).

```bash
# Smoke (10 шагов)
python -u scripts/train_qlora.py --config configs/gemma/clm_smoke.yaml
python -u scripts/train_qlora.py --config configs/gemma/sft_smoke.yaml

# Pilot (полный корпус, без max_train_samples) — два отдельных процесса:
./scripts/run_gemma_pilot_sequential.sh
```

Адаптеры: `runs/lora/gemma/clm_pilot/final`, `runs/lora/gemma/sft_pilot/final`.

### TranslateGemma — продолжение SFT (r=24, без перезаписи весов)

Если `submission_translate/weights` уже **merged**, а актуальный LoRA — `runs/lora/translate/sft_r24_full_v1/final`:

| Что | Путь | Перезапись |
|-----|------|------------|
| База для QLoRA | `submission_translate/weights_original` | нет |
| Старт LoRA | `runs/lora/translate/sft_r24_full_v1/final` | нет (только чтение) |
| Новый адаптер | `runs/lora/translate/sft_r24_cont_v2/final` | новый каталог |
| Merged для zip | `submission_translate/weights` | не трогаем при train |

Конфиг: `configs/translate/sft_r24_cont_v2.yaml` (`model_dir: weights_original`, `adapter_init: sft_r24_full_v1/final`, `run_name: sft_r24_cont_v2`, `auto_resume: false`).

```bash
# Данные (если ещё нет data/processed/parallel/*.jsonl)
python -u scripts/prepare_parallel.py --val-n 500

# Проверки + команда
./scripts/run_translate_sft_continue.sh

# Обучение
./scripts/run_translate_sft_continue.sh run
# или:
python -u scripts/train_qlora.py --config configs/translate/sft_r24_cont_v2.yaml --no-auto-resume
```

После обучения merge в **отдельный** каталог (не затирать текущий `weights` без `--backup`):

```bash
python -u scripts/merge_lora.py --model translate \
  --adapter runs/lora/translate/sft_r24_cont_v2/final \
  --out submission_translate/weights_merged_contest_8p7_bleu \
  --device cpu
# затем вручную скопировать/заменить weights для zip, если метрика лучше
```

### TranslateGemma-4B

```bash
# Smoke (10 шагов)
python -u scripts/train_qlora.py --config configs/translate/clm_smoke.yaml
python -u scripts/train_qlora.py --config configs/translate/sft_smoke.yaml

# Pilot
python -u scripts/train_qlora.py --config configs/translate/clm_pilot.yaml
python -u scripts/train_qlora.py --config configs/translate/sft_pilot.yaml
```

`sft_*` конфиги указывают `adapter_init` на `final/` предыдущего CLM-прогона.

### NLLB-200-3.3B

Те же `data/processed/mono` и `data/processed/parallel`. Stage 1 — **same-language reconstruction** (ru→ru, ab→ab), stage 2 — parallel `rus_Cyrl`→`abk_Cyrl`. Скрипт: `train_qlora_nllb.py` + `Seq2SeqTrainer` (не TRL SFT). База: `facebook/nllb-200-3.3B`.

```bash
python -u scripts/train_qlora_nllb.py --config configs/nllb/clm_smoke.yaml
python -u scripts/train_qlora_nllb.py --config configs/nllb/sft_smoke.yaml

python -u scripts/train_qlora_nllb.py --config configs/nllb/clm_pilot.yaml
python -u scripts/train_qlora_nllb.py --config configs/nllb/sft_pilot.yaml

# или
./scripts/train_nllb_pilot.sh
```

Адаптеры: `runs/lora/nllb/clm_pilot/final`, `runs/lora/nllb/sft_pilot/final`.
3.3B в 4-bit на ~8 GB: pilot — `per_device_train_batch_size: 1`, `gradient_accumulation_steps: 8`, `max_seq_length: 384`. При OOM — `cpu_offload` в yaml или seq 256.

## Валидация и merge

```bash
# Gemma
python -u scripts/validate_lora.py --model gemma \
  --adapter runs/lora/gemma/sft_pilot/final --limit 50

python -u scripts/merge_lora.py --model gemma \
  --adapter runs/lora/gemma/sft_pilot/final \
  --out submission/weights --backup --device cpu

./scripts/zip-submission.sh

# TranslateGemma
python -u scripts/validate_lora.py --model translate \
  --adapter runs/lora/translate/sft_pilot/final --limit 50

python -u scripts/merge_lora.py --model translate \
  --adapter runs/lora/translate/sft_pilot/final \
  --out submission_translate/weights --backup --device cpu

./scripts/zip-submission-translate.sh

# NLLB (~7 GB merged)
python -u scripts/validate_lora.py --model nllb \
  --adapter runs/lora/nllb/sft_pilot/final --limit 50

python -u scripts/merge_lora.py --model nllb \
  --adapter runs/lora/nllb/sft_pilot/final \
  --out submission_nllb/weights --backup --device cpu
```

На GPU ≤8 GiB merge bf16 идёт на CPU (`--device auto` по умолчанию). NLLB-3.3B merge ~7 GB, нужно ~16 GiB RAM.

## Конфиги

| Файл | Этап | Модель |
|------|------|--------|
| `configs/translate/clm_pilot.yaml` | CLM | TranslateGemma |
| `configs/translate/sft_pilot.yaml` | SFT | TranslateGemma |
| `configs/gemma/clm_pilot.yaml` | CLM | Gemma-4-E2B (max_seq 256) |
| `configs/gemma/sft_pilot.yaml` | SFT | Gemma-4-E2B (max_seq 256) |
| `configs/nllb/clm_pilot.yaml` | mono CLM | NLLB-3.3B |
| `configs/nllb/sft_pilot.yaml` | SFT | NLLB-3.3B |

Ключевые поля: `stage`, `model_type`, `adapter_init`, `max_seq_length`, `mono_lang_weights` (ab 0.6 / ru 0.4), `lora_r` / `lora_alpha` (pilot: 24 / 48). `max_train_samples` в pilot не задан — берётся весь отфильтрованный корпус.

### Чекпоинты и resume (не 10 GB!)

При обучении сохраняется **только LoRA-адаптер** (~60–90 MiB на checkpoint), не полная модель. **10 GB** — это merged bf16 для контеста (`merge_lora.py`).

| Поле | Значение pilot |
|------|----------------|
| `save_steps` | 200 |
| `save_total_limit` | 2 (старые checkpoint удаляются) |
| `save_only_model` | true (без optimizer.pt — меньше VRAM при save) |
| `auto_resume` | true (продолжить с `checkpoint-N`, если прогон оборвался) |
| `save_best_eval` | true → лучший `eval_loss` в `{run}/best_eval/` (merge, не resume) |

```bash
# Продолжить после OOM / Ctrl+C (автоматически, если нет final/):
python -u scripts/train_qlora.py --config configs/translate/clm_pilot.yaml

# Явно с checkpoint:
python -u scripts/train_qlora.py --config configs/translate/clm_pilot.yaml --resume latest

# Новый прогон с нуля:
rm -rf runs/lora/translate/clm_pilot/checkpoint-* runs/lora/translate/clm_pilot/final
python -u scripts/train_qlora.py --config configs/translate/clm_pilot.yaml --no-auto-resume
```

При OOM на 8 GB раскомментируй в `*_pilot.yaml`:

```yaml
cpu_offload: true
gpu_max_memory_gib: 6.5
cpu_max_memory_gib: 32
```

Часть слоёв уйдёт на CPU — медленнее, но стабильнее на длинных `max_seq_length`.

## Рекомендация по русскому монокорпусу

**Основной:** `misterkirill/ru-wikipedia` — скачать один раз:

```bash
python -u download_ru_wikipedia.py   # → data/ru-wikipedia.jsonl (~26 MB)
```

Короткие фрагменты статей (~69k после фильтра длины), хорошо подходят к формату теста (~78 символов).

**Добавить:** `--parallel-ru` — in-domain RU из параллельного корпуса.
