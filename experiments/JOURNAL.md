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

---

## 2026-05-29 — task-1 — contest submit — inference v1 (prompt + few-shot + postprocess)

**Что пробовали:** Первая посылка на контест с модификацией относительно шаблонного бейзлайна: `prompts.py` (BLEU-инструкция + 3 few-shot), `postprocess.py`, динамический `max_new_tokens` в `solution.py`. Zip пересобран из `submission/` (`./scripts/zip-submission.sh`).

**Как работал код:** Загрузка архива на contest.yandex.ru (task-1, Docker + `weights/`). Локально на `self/dev` тот же пайплайн давал macro BLEU **57.12** vs бейзлайна **53.88** (запись 2026-05-28).

**Результат:** На **закрытом тесте контеста** (69 абзацев) итоговая метрика **ниже**, чем у исходного бейзлайна без доработки промпта. Локальный прирост **+3.24** на `self/dev` **не перенёсся** на чекер.

**Вывод:** Шаг «inference v1 (промпт + few-shot + postprocess)» как стратегия для контеста **неэффективен** — откатить ориентир на посылку: не считать v1 улучшением до подтверждения на скрытом тесте; для следующих попыток — другие рычаги (минимальный/оригинальный промпт, LoRA с выровненным train/eval, глоссарий/постобработка точечно с A/B на OPUS или held-out). Не оптимизировать дальше длину промпта и few-shot только по `self/dev` без проверки на контесте.

---

## 2026-05-30 — task-2 — baseline: Gemma-4-E2B vs TranslateGemma-4B (dev, n=50)

**Что пробовали:** Сравнение двух базовых моделей без дообучения на dev-подмножестве task-2 (RU→Abkhaz): `submission/` (Gemma-4-E2B-it) и `submission_translate/` (TranslateGemma-4B). Цель — выбрать базу для QLoRA и зафиксировать стартовую планку BLEU до CLM/SFT.

**Как работал код:**
```bash
cd task-2
python -u scripts/compare_models.py --limit 50 --gpu-only --rerun
```
Окружение: `ml-env`, GPU Legion 7.5 GiB — bf16 не влезает, авто **4-bit NF4** (bitsandbytes). Референс: `data/processed/dev.jsonl`. Артефакты: `out/gemma_dev_output.json`, `out/translategemma_dev_output.json`, `out/compare_models.json`.

| Модель | Время | Скорость |
|--------|-------|----------|
| Gemma-4-E2B | 59.1 s | 1.18 s/example |
| TranslateGemma-4B | 108.7 s | 2.17 s/example |

**Результат:** macro BLEU на 50 примерах (sacrebleu, тот же скрипт, что `eval_bleu.py`):

| Модель | macro BLEU | n | min | max |
|--------|------------|---|-----|-----|
| Gemma-4-E2B | **4.44** | 50 | 0.00 | 43.21 |
| TranslateGemma-4B | **4.49** | 50 | 0.00 | 45.23 |

Δ (translate − gemma): **+0.04** → формально чуть лучше TranslateGemma-4B; разница в пределах шума на n=50. Оба скора близки к «копировать русский» (~5.2 по SPECS) — модели почти не переводят на абхазский out of the box.

**Вывод:** Для QLoRA оставляем **TranslateGemma-4B** как основную линию (специализированная MT-модель, marginal +0.04; chat-template с `source_lang`/`target_lang` уже в inference). Gemma-4-E2B — запасной baseline. После SFT (`configs/translate/sft_pilot.yaml`, init от `clm_pilot/final`) повторить `compare_models.py` или `validate_lora.py` на том же dev; ориентир контеста — strong ~17.6 BLEU. TranslateGemma медленнее ~1.8× на 4-bit — учитывать лимит 30 min / 515 запросов при финальной упаковке.

---

## 2026-05-31 — task-2 — contest submit — TranslateGemma QLoRA (CLM + SFT)

**Что пробовали:** Первая успешная посылка на контест task-2 (RU→Abkhaz): **TranslateGemma-4B-it** после двухэтапного QLoRA — **CLM** на монокорпусе (`configs/translate/clm_pilot.yaml`) → **SFT** на параллельном корпусе (`configs/translate/sft_pilot.yaml`, init от `runs/lora/translate/clm_pilot/final`). Адаптер: `runs/lora/translate/sft_pilot/final`. Для контеста: merge в bf16 → `submission_translate/weights/`, zip из `submission_translate/` (`Dockerfile`, `solution.py`, `weights/`). Inference: chat-template `ru`→`ab`, батчинг, `max_new_tokens=128`, `attn_implementation=sdpa`.

**Как работал код:** Обучение — `scripts/train_qlora.py` (4-bit QLoRA). Merge — `scripts/merge_lora.py --model translate --adapter runs/lora/translate/sft_pilot/final --out submission_translate/weights`. Локально до посылки `validate_lora.py` на dev (n=50) давал macro BLEU **~21** (sacrebleu). Загрузка `task-2-translategemma-submission.zip` на contest.yandex.ru. Ориентиры из SPECS: «копировать русский» ≈ **5.2**, strong ≈ **17.6** BLEU.

**Результат:** На **закрытом тесте контеста** (515 примеров) итоговая метрика **8.7 BLEU**. Посылка прошла по времени (после ускорения inference). Существенный разрыв с локальным dev (~21): скрытый тест / train-dev mismatch, как в task-1.

**Вывод:** Двухэтапный QLoRA на TranslateGemma **лучше бейзлайна** (4.5 → 8.7 vs ~5.2 «копировать русский»), но **далеко от strong ~17.6** и от локальной оценки. Не считать dev BLEU предсказателем контеста без проверки на held-out ближе к домену теста. Следующие рычаги: больше/лучше SFT-данные, выравнивание train/inference, постобработка, Gemma-line или Marian fine-tune; повторная калибровка dev-сплита.

---

## 2026-06-02 — task-1 — TranslateGemma baseline EN→RU (self/dev, n=20)

**Что пробовали:** Вторая линия сдачи: `submission_translate/` (TranslateGemma-4B-it, `en`→`ru`), те же ускорения, что в task-2 — батчинг по длине, `sdpa`, `torch.compile`, динамический `max_new_tokens`. Локальные скрипты: `scripts/local_inference.py`, `run_local_eval.py`, `compare_models.py`, `bench_inference.py`. Сравнение с Gemma-4-E2B (prompt v1 + postprocess) на `data/processed/self/dev` (первые 20 абзацев).

**Как работал код:**
```bash
cd task-1
conda run -n ml-env python download_weights_translate.py   # веса только сюда, не из task-2
conda run -n ml-env python -u scripts/compare_models.py --limit 20 --gpu-only --quantize 4bit --rerun
conda run -n ml-env python -u scripts/bench_inference.py --limit 10 --gpu-only --quantize 4bit
```
Окружение: `ml-env`, GPU ~7.5 GiB, 4-bit NF4. Первый прогон ошибочно использовал веса из task-2 (другая задача, LoRA RU→ab) — BLEU **5.44**; повтор с `download_weights_translate.py` → базовая EN→RU модель.

**Результат:**

| Модель | macro BLEU (n=20) | s/paragraph (4bit) |
|--------|-------------------|---------------------|
| Gemma-4-E2B (v1) | **64.60** | 2.32 |
| TranslateGemma-4B (base) | **35.50** | 1.55 (batch=4) |

Бенчмарк n=10: Gemma **2.00** s/para; Translate seq **2.12** (1.06×); Translate batched **1.68** (0.84× vs Gemma).

Δ (translate − gemma): **−29.10** BLEU → без дообучения TranslateGemma **хуже** chat-Gemma с промптом v1 на абзацном dev.

**Вывод:** Инфраструктура готова; для контеста task-1 базовый TranslateGemma **не заменяет** текущую Gemma-линию по BLEU. Имеет смысл пробовать QLoRA EN→RU на OPUS/абзацах. Веса — только `task-1/download_weights_translate.py` → `submission_translate/weights/`, не из task-2.

---

## 2026-06-04 — task-1 — agentic prompt tuning (16 итераций, Gemma)

**Что пробовали:** 16 итераций промпта вручную: документ `task-1/docs/PROMPT_TUNING.md`, полный лог `task-1/out/prompt_tune/prompt_tuning_16_iterations.md`, активный промпт `submission/prompt_tune_active.py`, прогон по одному `scripts/run_prompt_iteration.py --iter N`. Ориентир — OPUS val n=300; self/dev n=117 — sanity. Без пакетного прогона всех вариантов в одном скрипте.

**Как работал код:**
```bash
cd task-1
conda run -n ml-env python -u scripts/run_prompt_iteration.py --iter 6 --gpu-only --quantize 4bit
```
4-bit, postprocess on. Лучший: **#06/#15** — EN-инструкция + 4 rules + `English:`/`Russian:`.

**Результат:** OPUS macro BLEU **28.42** (было **26.97** на бейзлайне организаторов, **+1.45**). self **55.20** (v1 ~57 — на абзацах sentence-промпт слабее). В код: `prompts.build_opus_en_prompt`, `build_contest_en_prompt`; `--prompt opus_en|contest_en`.

**Вывод:** Для OPUS/предложений — **EN rules без few-shot из статей**. Few-shot и длинные RU-антипримеры на OPUS не помогли. Для контеста (абзацы) — отдельно сравнить `contest_en` vs `v1`; не слепо переносить sentence-промпт. Постпроцессинг по локальной метрике без него не нужен для лучшего промпта; в submission оставить опциональный флаг, но выключить по умолчанию.

---

## 2026-06-04 — task-2 — TranslateGemma prompt tuning (8 итераций)

**Что пробовали:** 8 вариантов prompt для `submission_translate/weights` (см. `task-2/docs/PROMPT_TUNING_TRANSLATE.md`): текущий bundled English template, полный русский перевод этого template, короткие русские инструкции, no-copy вариант, natural-style, минимальные метки `Русский:/Абхазский:`, явное `аԥсуа`, короткий English control. Ориентир — `data/processed/dev` первые 100 примеров.

**Как работал код:**
```bash
cd task-2
conda run -n ml-env python -u scripts/run_translate_prompt_iteration.py --iter N --limit 100 --gpu-only --quantize 4bit
```
Во время iter01 найден bug в batched decode: при left padding нельзя резать output по индивидуальной непадденной длине prompt; нужно резать по общей padded длине `inputs["input_ids"].shape[-1]`. Исправлено в `submission_translate/solution.py` и runner.

Проверка фикса: baseline на первых 40 dev с `batch_size=40` vs `batch_size=1` дал BLEU **22.36** vs **22.23**, prompt-артефактов (`model`, `Abkhazian:`, `Русский:`) в обоих output **0**. До фикса batched output содержал такие фрагменты prompt.

**Результат:**

| iter | prompt | BLEU n=100 |
|---:|---|---:|
| 01 | bundled English template | **21.55** |
| 02 | русский перевод bundled template | 21.47 |
| 03 | короткая русская инструкция | 20.27 |
| 04 | русская инструкция + no-copy | 19.99 |
| 05 | natural corpus style | 18.94 |
| 06 | `Русский:/Абхазский:` labels | 20.73 |
| 07 | русская инструкция + `аԥсуа` | 20.33 |
| 08 | короткий English control | 20.09 |

**Вывод:** Русский перевод системного prompt почти догнал baseline, но не улучшил его. Для отправки оставить bundled English TranslateGemma template, потому что адаптер обучался через structured TranslateGemma message format; сильнее всего повлияло исправление batched decode, его обязательно включить в submission.

---

## 2026-06-04 — task-2 — TranslateGemma beam search для посылки

**Что пробовали:** На текущих merged `submission_translate/weights` проверили deterministic decode для RU→AB: greedy vs beam search с `num_beams=2/3/4` и `length_penalty=0.6/0.8/1.0/1.2/1.4`. Prompt оставлен bundled TranslateGemma template; batched decode — с корректным срезом по padded input length.

**Как работал код:**
```bash
cd task-2
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ml-env
# одноразовые Python grid-прогоны на data/processed/dev.input.pickle / dev.jsonl
```
Локальная карта 8GB: `beam=3` при `batch=32` упал с CUDA OOM, поэтому full-dev проверка `beam=3` шла с `batch=16`. На контестной L4 24GB это ожидаемо должно влезть с текущим `BATCH_SIZE=40`, но время/VRAM подтвердит только checker.

**Результат:** Full local dev `n=500`, `max_new_tokens=128`:

| decode | macro BLEU | elapsed |
|---|---:|---:|
| greedy | 25.154 | 128.2s |
| beam=2, lp=0.8 | 26.065 | 205.3s |
| beam=3, lp=0.6 | 26.326 | 325.5s |
| beam=3, lp=0.8 | **26.401** | 327.3s |
| beam=3, lp=1.0 | 26.308 | 326.5s |

На первых 100 dev `beam=4, lp=0.8` дал 23.282 BLEU против 23.403 у `beam=2, lp=0.6`; full-dev для beam=4 не гоняли из-за скорости и отсутствия сигнала улучшения.

**Вывод:** Для следующей посылки выбран `num_beams=3`, `length_penalty=0.8`, `early_stopping=True`, `do_sample=False`. Локальный прирост к greedy: **+1.25 BLEU** на full dev. `task-2/submission_translate/solution.py` обновлен под этот decode; следующий шаг — собрать `task-2-translategemma-submission.zip` и отправить в contest.

---

## 2026-10-16 — task-1 / task-2 — оформление исследовательского репозитория

**Что пробовали:** Сохранить текущие реализации в воспроизводимом интерфейсе для исследования. Добавлены `dojo.py` (download/infer/evaluate/train/pack), документация архитектуры, ресурсов, воспроизведения и результатов, JSON-примеры и зависимости по режимам. Место 24/500 указано со слов автора отдельно от метрик посылок. Вспомогательный загрузчик данных и merge Task 1 восстановлены из `62e1c30`; исходные алгоритмы генерации не редактировались при оформлении. Существовавшие локальные изменения решений и конфигов включены в публикацию.

**Как работал код:** CPU-only `python -m unittest discover -s tests -v` (12 тестов), dry-run маршрутов, реальная оценка через `dojo.py evaluate` на двух синтетических совпадающих парах с переставленными rid. Три trainer entrypoints успешно загружены до `--help`; в локальном окружении были предупреждения NVML и необязательных C++ extensions. Выполнены проверка синтаксиса, путей в основной документации и состава Git. GitHub CI запускает CPU-only проверки. Веса, данные, generated tokenizers/checkpoint metadata исключены из текущего Git-дерева; файлы на диске сохранены.

**Результат:** 12 тестов прошли, sanity macro BLEU 100.00. Новые GPU-метрики, конкурсные посылки и полное воспроизведение обучения не запускались. Прямые URL конкурсных RU–AB корпусов в сохранённых материалах отсутствуют; ограничение явно документировано. Обученные веса не опубликованы; загрузчик получает исходные Hub-модели.

**Вывод:** Код готов для клонирования и исследования через единый интерфейс. Для восстановления конкретного обученного состояния нужны сохранённый адаптер/merged-веса либо повторное обучение на исходных данных; один clone это состояние не восстанавливает.

---

## 2026-10-16 — уточнение результата участия и удаление настроек редактора

По уточнению автора, всего было **320 участников**, итог — **24 / 320 (top 7,5%)**. Ранее указанное в записи об оформлении число 500 участников ошибочно; размер локальных dev-выборок (500 примеров) не меняется. Каталог настроек Cursor удалён, действующие инструкции перенесены в `AGENTS.md`. Исторические упоминания редактора в журнале относятся к прежней структуре проекта.
