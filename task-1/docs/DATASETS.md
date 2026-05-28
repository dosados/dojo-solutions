# Датасеты EN→RU для dev и обучения (task-1)

Официальный тест — **69 закрытых абзацев** (~162 слова). Локально нужны свои параллельные тексты: **macro BLEU** на val и большой **train** для LoRA / prompt tuning.

Формат после `scripts/prepare_data.py`: JSONL, одна строка = один пример:

```json
{"rid": 0, "src": "English ...", "ref": "Русский эталон ..."}
```

Оценка: [scripts/eval_bleu.py](../scripts/eval_bleu.py) + [BLEU.md](BLEU.md).

---

## Рекомендация (практичный стек)

| Роль | Датасет | Размер | Зачем |
|------|---------|--------|--------|
| **Train (большой)** | [Helsinki-NLP/opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100) `en-ru` | ~1M предложений | LoRA / SFT; домены: книги, Европарламент, UN, … |
| **Val (абзацы)** | OPUS-100 val + склейка 5–8 предложений | 300–500 «псевдо-абзацев» | BLEU ближе к формату контеста |
| **Val (sentence, быстро)** | [facebook/flores](https://huggingface.co/datasets/facebook/flores) `dev` / `devtest` | ~997 / ~1012 | Быстрый бенчмарк, **не** заменяет абзачный val |

Команды:

```bash
cd task-1

# 1) Большой train + val 3000 предложений
conda run -n ml-env pip install sacrebleu datasets
conda run -n ml-env python scripts/prepare_data.py --dataset opus100 --val-size 3000

# 2) Val как псевдо-абзацы (5 предложений) — для macro BLEU под task-1
conda run -n ml-env python scripts/prepare_data.py --dataset opus100 \
  --val-size 2000 --paragraph-sentences 5 --paragraph-val-only

# 3) Отдельно FLORES devtest — sentence-level ориентир
conda run -n ml-env python scripts/prepare_data.py --dataset flores --split devtest
```

Файлы появятся в `task-1/data/` (в `.gitignore`).

---

## Другие источники (если нужен ещё объём)

| Датасет | Плюсы | Минусы |
|---------|--------|--------|
| **ParaCrawl** (en-ru) | Очень большой | Шумный; нужна фильтрация по score |
| **WikiMatrix** | Много wiki-пар | Качество неоднородное |
| **WMT News Commentary** | Новостной стиль | Меньше объём, чем OPUS-100 |
| **UN parallel corpus** | Официальный стиль | Другой домен vs IT/веб |

Для **абзацев как в тесте** идеален document-level корпус (склейка по статье/URL). OPUS-100 без метаданных документа — компромисс: склейка **подряд идущих** предложений в val (`--paragraph-sentences`).

---

## Разбиение train / val

- **Не** оценивать на train — только на `*_val.jsonl`.
- Для opus100 скрипт делает случайный split (`--seed`, `--val-size`).
- FLORES `devtest` — фиксированный; не смешивать с train opus без dedup (возможен overlap по текстам).

Для finetune:

- `data/opus100_train.jsonl` — обучение (предложения или свои абзацы)
- `data/opus100_val.jsonl` — ранняя остановка / выбор чекпоинта по macro BLEU

---

## Цикл работы

```bash
# Подготовить input для модели
conda run -n ml-env python scripts/jsonl_to_pickle.py data/opus100_val.jsonl -o submission/input.pickle

# Прогон (из submission/)
conda run -n ml-env python solution.py

# BLEU
conda run -n ml-env python scripts/eval_bleu.py \
  --hyp submission/output.json \
  --ref data/opus100_val.jsonl \
  --worst 10 --show-text
```

---

## Ограничения

- Локальный BLEU **не равен** баллу на 69 скрытых абзацах.
- FLORES val — короткие предложения (~21 слово); OPUS paragraph-val — лучший компромисс без ручной разметки.
- Перед zip на контест — smoke на `examples/example-input.pickle` (без ref).
