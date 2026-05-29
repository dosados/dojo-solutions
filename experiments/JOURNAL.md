# Журнал экспериментов

Хронологический лог: что пробовали, как запускали, что получилось. Новые записи — **только в конец файла**. Старые не редактировать.

Формат записи см. `.cursor/rules/experiments-journal.mdc`.

---

## 2026-05-28 — setup — структура репозитория

**Что пробовали:** Перенесли бейзлайн task-1 в корень `task-1/` (Dockerfile, solution.py, …). Настроили `.cursor/rules/` и этот журнал.

**Как работал код:** N/A (организация репо).

**Результат:** Готова структура для локальных прогонов и docker smoke test.

**Вывод:** Следующий шаг — прогон бейзлайна на `example-input.pickle` и при необходимости первая сдача на контест.

---

## 2026-05-28 — layout — каталог submission/

**Что пробовали:** Артефакты сдачи перенесены в `task-1/submission/` (Dockerfile, solution.py, weights/). Вспомогательное — `examples/`, `download_weights.py`, `docs/`, `scripts/zip-submission.sh`.

**Как работал код:** `download_weights.py` пишет в `submission/weights/`; zip = содержимое `submission/`.

**Результат:** Удобная упаковка без смешения с train/eval скриптами.

**Вывод:** Сборка: `docker build ./submission`; zip: `./scripts/zip-submission.sh`.

---

## 2026-05-28 — task-1 — eval BLEU + prepare_data

**Что пробовали:** Модуль `task-1/eval/` (macro `sentence_bleu`), CLI `scripts/eval_bleu.py`, `scripts/prepare_data.py`, `docs/DATASETS.md`.

**Как работал код:** `conda run -n ml-env python scripts/eval_bleu.py --sanity-ref --ref examples/example-output.json --ref-key translation` — macro BLEU 100.00 на 6 примерах.

**Результат:** Локальная метрика совпадает с ожиданием; sacrebleu 2.6.0.

**Вывод:** Скачать OPUS-100 через `prepare_data.py`, прогон `solution.py`, оценка `--hyp submission/output.json --ref data/opus100_val.jsonl`.

---

## 2026-05-28 — task-1 — convert_datasets

**Что пробовали:** `scripts/convert_datasets.py` → `data/processed/`: self (139 абзацев, dev/heldout) + opus100 train/val/test JSONL и `.input.pickle`.

**Как работал код:** `conda run -n ml-env python scripts/convert_datasets.py` — self: 117 dev + 22 heldout (article-5); opus: 1M train, 2k val, 2k test.

**Результат:** manifest.json в `data/processed/`.

**Вывод:** Валидация бейзлайна на `self/dev.jsonl`; LoRA — `opus100/train.jsonl`.

---

## 2026-05-28 — task-1 — baseline — macro BLEU на self/dev

**Что пробовали:** Бейзлайн без дообучения: `submission/solution.py` (Gemma-4-E2B-it, `GENERAL_PROMPT` из шаблона), инференс → `out/dev_output.json`; оценка против ручных эталонов `data/processed/self/dev.jsonl` (117 абзацев, статьи 1–4).

**Как работал код:**
```bash
# инференс (флаги device/quantize — как в успешном локальном прогоне)
cd task-1/submission && python -u solution.py \
  --input ../data/processed/self/dev.input.pickle \
  --output ../out/dev_output.json

cd task-1 && conda run -n ml-env python scripts/eval_bleu.py \
  --hyp out/dev_output.json \
  --ref data/processed/self/dev.jsonl \
  --worst 10 --show-text
```

**Результат:** sacrebleu 2.6.0; **macro BLEU = 53.88** (min 9.26, max 100.00, n=117). Худшие rid: 14 (9.26, типы ML-обучения), 102/108/116 (Пушкин, архаичный ref), 58 (Python community), 51/52 (короткие заголовки). Это **не** балл контеста (69 скрытых абзацев, другая длина/домен).

**Вывод:** Зафиксирована нижняя планка на paragraph-dev; улучшения (промпт, постобработка, LoRA) сравнивать с **53.88**. Held-out: `self/heldout.jsonl` (22, article-5) — не трогать до финальной проверки. Контестный скор — отдельной записью после zip.

---

## 2026-05-28 — task-1 — baseline v0 — worst-10 на self/dev (детально)

**Что пробовали:** Зафиксированы худшие примеры бейзлайна v0 (`GENERAL_PROMPT`, `max_new_tokens=1024`) перед фазой 1 (промпт BLEU + few-shot + postprocess).

**Как работал код:**
```bash
conda run -n ml-env python scripts/eval_bleu.py \
  --hyp out/dev_output.json \
  --ref data/processed/self/dev.jsonl \
  --worst 10 --show-text
```

**Результат:** macro BLEU **53.88** (n=117, min 9.26, max 100.00). Worst-10:

| rid | BLEU | Тип ошибки |
|-----|------|------------|
| 14 | 9.26 | ML: «Для обучения ИИ могут использоваться…» vs ref «Чтобы обучить искусственный интеллект, можно использовать разные методы…» |
| 102 | 11.32 | Пушкин: современный стиль vs архаика ref («сестрою», «Царскосельский лицей») |
| 108 | 12.22 | Пушкин/театр: «Воришка» vs «Похититель», другая лексика цитаты |
| 58 | 12.75 | Python: «сообщество» vs ref «комьюнити», иные обороты |
| 109 | 14.85 | Пушкин: «притчи»/«Ангел» vs ref «басни»/«Генриада» |
| 116 | 15.85 | Пушкин: «в детские годы» vs «в детских летах» |
| 51 | 17.97 | Заголовок: «что хорошо в Python» vs «чем хорош Python» |
| 110 | 18.11 | Пушкин: перефраз стихотворного фрагмента |
| 52 | 19.36 | «множество»/«сосредоточимся» vs «массу»/«остановимся на ключевых из них» |
| 107 | 22.21 | Пушкин: «придумывать комедии» vs «импровизировать комедии» |

**Вывод:** Быстрый выигрыш — few-shot на IT-абзацах (rid 14, 51, 52, 58) и BLEU-инструкция (ближе к ref, без лишнего перефраза). Пушкин (102–110, 116, 107) — потолок BLEU на self/dev из‑за архаичного ref при современном переводе модели; не оптимизировать промпт только под эти rid. Следующий прогон — `prompts.py` + `postprocess.py` + динамический `max_new_tokens` (cap 2048).

---

## 2026-05-28 — task-1 — inference v1 — промпт BLEU + few-shot + postprocess

**Что пробовали:** Фаза 1 в `submission/`: `prompts.py` (инструкция под BLEU, 3 few-shot из dev article-1–2), `postprocess.py` (преамбулы, «Русский:», thinking/fences), `solution.py` — динамический `max_new_tokens` (512–2048 от длины `src`, default cap 1536), постобработка после `parse_response`.

**Как работал код:**
```bash
cd task-1/submission && python -u solution.py \
  --device cuda --quantize 4bit \
  --input ../data/processed/self/dev.input.pickle \
  --output ../out/dev_output_v1.json --progress

python scripts/eval_bleu.py \
  --hyp out/dev_output_v1.json \
  --ref data/processed/self/dev.jsonl --worst 10 --show-text
```

**Результат:** sacrebleu 2.6.0; **macro BLEU = 57.12** (было **53.88**, **+3.24**), n=117, min 10.18, max 100.00.

Ушли из worst-10 (улучшились): **rid=14** (было 9.26, ML), **rid=52** (было 19.36). **rid=51** без изменений (17.97): hyp всё ещё «что хорошо в Python» vs ref «чем хорош Python». **rid=58** чуть лучше (12.75→16.84), но «сообщество» vs «комьюнити» остаётся.

Worst-10 v1:

| rid | BLEU | Комментарий |
|-----|------|-------------|
| 109 | 10.18 | Пушкин: «притчи»/«Ангелиада» vs «басни»/«Генриада» |
| 102 | 15.23 | Пушкин: архаика ref («сестрою», «Царскосельский») |
| 58 | 16.84 | Python: «сообщество» vs «комьюнити» |
| 108 | 17.56 | Пушкин/театр: «Воришка» vs «Похититель» |
| 51 | 17.97 | Заголовок (без изменения vs v0) |
| 116 | 18.17 | Пушкин: «в детские годы» vs «в детских летах» |
| 107 | 19.14 | Пушкин: «придумывать» vs «импровизировать» |
| 110 | 23.85 | Пушкин: имена/стих (Мэджорн vs Матюрена) |
| 106 | 26.05 | Пушкин: обороты ref vs современный hyp |
| 115 | 26.07 | Пушкин: «задние обложки» vs «затылки сафьяновых фолиантов» |

**Вывод:** Фаза 1 окупилась (+3.24 BLEU); few-shot сработал на длинных IT-абзацах (14, 52), не на коротком заголовке 51 и не на loanword 58. **8/10** худших — article Пушкин (архаичный ref): дальнейший выигрыш промптом на self/dev ограничен. Следующее: (1) глоссарий IT + few-shot rid=58; (2) отдельная метрика по article-2 vs article-3; (3) OPUS paragraph-val для выбора перед контестом; (4) LoRA на OPUS — основной рычаг после сдачи v1 на контест.

---

## 2026-05-28 — task-1 — QLoRA pipeline (скрипты)

**Что пробовали:** Пайплайн: `train_lora.py`, `lora_train_validate.py` (этап 1), `merge_lora.py` (этап 2), `zip-submission.sh` (этап 3); конфиги `lora_pilot.yaml` / `lora_100k.yaml`; `solution.py` — `--lora-dir`.

**Как работал код:** `docs/TRAINING.md`; `pip install -r requirements-train.txt`; этап 1: `python -u scripts/lora_train_validate.py --config configs/lora_pilot.yaml`.

**Результат:** _(прогон обучения — следующая запись)_

**Вывод:** Train OPUS предложения; merge → `submission/weights/model.safetensors`; valid по OPUS val subset, не self/Пушкин для early stop.

---

## 2026-05-28 — task-1 — QLoRA pilot_10k — self/dev: регрессия −20 BLEU

**Что пробовали:** Пилот QLoRA (`configs/lora_pilot.yaml`): 10k предложений OPUS, `max_seq_length=256`, короткий train-промпт (`train/prompts_train.py`, без few-shot). Обучение ~49 мин / 1250 шагов на 8 GB (Legion), адаптер `runs/lora/pilot_10k/final`. Валидация на **целевом** корпусе: 117 абзацев `self/dev`, тот же инференс, что для контеста — полный `prompts.py` (BLEU-инструкция + 3 few-shot) + `postprocess.py`, 4-bit, динамический `max_new_tokens`.

**Как работал код:**
```bash
# train (успешно, без OOM после LowVRAMSFTTrainer + tokenize_sft)
python -u scripts/lora_train_validate.py --config configs/lora_pilot.yaml

# сравнение baseline vs adapter на абзацах
python -u scripts/validate_lora.py \
  --adapter runs/lora/pilot_10k/final \
  --input data/processed/self/dev.input.pickle \
  --ref data/processed/self/dev.jsonl \
  --max-samples 0 \
  --out-dir out/lora_val_self_dev \
  --progress --worst 10
```
Справка: OPUS val 300 предложений давала baseline macro BLEU **~26.5** (другой домен и шум в ref) — не сравнивать с self/dev.

**Результат:** `out/lora_val_self_dev/scores.json`

| | macro BLEU | n | Δ vs baseline |
|--|------------|---|----------------|
| baseline (v1 prompt, без LoRA) | **56.86** | 117 | — |
| + LoRA pilot_10k | **36.48** | 117 | **−20.38** |

Референс inference v1 без LoRA в журнале: **57.12** — текущий baseline 56.86 согласуется (тот же промпт, мелкий разброс прогона).

Worst LoRA (примеры обвала): rid=58 BLEU **0.00**, rid=42 **0.18**, rid=31 **0.44** — много IT/абзацев, где baseline был нормальным. Worst baseline на том же прогоне — по-прежнему Пушкин (rid 109, 108, 102…), не совпадает с worst LoRA → модель с адаптером **сменила режим перевода**, а не «чуть подправила» v1.

Train loss на пилоте падал (шаг 1 ~121 → ~15), то есть обучение технически шло; деградация на self/dev — не артефакт сломанного чекпоинта, а **несовпадение train/eval**.

**Вывод:** Для **посылки на контест этот адаптер не использовать** (не merge, не `--lora-dir`). Это **провал гипотезы пилота** в текущем виде, но **объяснимый**, не «LoRA мёртв»:

1. **Prompt mismatch:** SFT на коротком «Переведи… / Английский / Русский», инференс на длинном few-shot — адаптер учился под другой вход.
2. **Domain/format:** OPUS **предложения** (укороченные пары, `max_src_chars=400`) vs **абзацы** self/dev.
3. **Длина:** train `max_seq_length=256` vs длинные абзацы при generate.
4. Пилот не замерял BLEU на OPUS с **тем же** промптом, что train — роста на целевой метрике не показано.

**Решение:** **Переходим к другому** в смысле **этого прогона** — оставляем **inference v1 (prompt + postprocess)** как основную линию для сдачи; пилотный LoRA **закрыть**. Повторять LoRA имеет смысл только как **новый эксперимент** (не дотюнинг пилота): выровнять train-промпт с `build_translation_prompt`, paragraph-level данные или длинные сегменты, 50–100k+, val на OPUS с тем же промптом; желательно GPU 16 GB+ (Colab). До этого — не тратить merge/попытки контеста на `pilot_10k`.

---

## 2026-05-29 — task-1 — beam search в solution.py

**Что пробовали:** Beam search через `transformers.generate`: `--num-beams` (default **4**), `--length-penalty` (default 1.0), `early_stopping=True` при `num_beams > 1`. `--num-beams 1` — прежний greedy.

**Как работал код:**
```bash
cd task-1/submission && python -u solution.py \
  --device cuda --quantize 4bit \
  --input ../data/processed/self/dev.input.pickle \
  --output ../out/dev_output_beam4.json --progress

cd .. && python scripts/eval_bleu.py \
  --hyp out/dev_output_beam4.json \
  --ref data/processed/self/dev.jsonl --worst 10 --show-text
```
Сравнение с inference v1 greedy: `dev_output_v1.json`, macro **57.12** (запись 2026-05-28).

**Результат:** sacrebleu 2.6.0; **macro BLEU = 56.42** (n=117, min 13.48, max 100.00). Относительно greedy v1: **−0.70** BLEU. Worst-10 по-прежнему доминирует Пушкин (rid 109, 102, 107, 108, 110, 116, 115, 106) + IT rid 58, 51.

| rid | BLEU (beam4) | Комментарий |
|-----|--------------|-------------|
| 109 | 13.48 | Пушкин: «притчи»/«Ангериаду» vs «басни»/«Генриада» |
| 102 | 15.23 | Пушкин: «Царское Село Лицей» vs «Царскосельский лицей» |
| 107 | 16.17 | Пушкин: «придумывать комедии» vs «импровизировать» |
| 58 | 16.84 | Python: «сообщество» vs «комьюнити» |
| 110 | 17.42 | Пушкин: стих, имена |
| 108 | 17.51 | Пушкин: «Вор» vs «Похититель» |
| 51 | 17.97 | Заголовок (без изменения vs v1) |
| 116 | 19.08 | Пушкин: «в детских годах» vs «в детских летах» |
| 115 | 24.72 | Пушкин: «задние части фолионов» vs «затылки сафьяновых фолиантов» |
| 106 | 25.79 | Пушкин: обороты |

**Вывод:** Beam 4 **не окупился** на self/dev: медленнее (~4×), BLEU ниже greedy v1. Для сдачи и основной линии — **greedy** (`--num-beams 1`, default в `solution.py`); beam — опциональный флаг (`--num-beams 4`) для редких подборов.
