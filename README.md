# Dojo · Machine Translation

**24-е место среди 320 участников · top 7,5%**

Моё решение соревнования по машинному переводу в Yandex Contest.

Две задачи: перевод английских абзацев на русский и перевод с русского на абхазский. Для них я сравнивал Gemma, TranslateGemma и NLLB, подбирал промпты, дообучал модели через QLoRA и собирал Docker-контейнеры для запуска без сети.

## Задачи

| | Task 1 · EN → RU | Task 2 · RU → AB |
|---|---|---|
| Цель | Сохранить смысл, стиль и терминологию целого абзаца | Научить модель переводу на язык с малым количеством данных |
| Закрытый тест | 69 абзацев, в среднем 846 символов | 515 коротких текстов, в среднем 78 символов |
| Метрика | Среднее sentence BLEU через sacrebleu | Та же метрика |
| Лимит инференса | 45 минут | 30 минут |
| Среда проверки | NVIDIA L4 24 ГБ, 16 ГБ RAM, 4 vCPU; образ <20 GB; без сети | Та же среда |

Оригинальные условия: [Task 1](task-1/docs/SPECS.md), [Task 2](task-2/docs/SPECS.md).

## Что реализовано

- **Gemma-4-E2B**: промпты для EN→RU и RU→AB, эксперименты с few-shot и постобработкой, QLoRA-ветка для EN→RU.
- **TranslateGemma-4B**: структурированный формат перевода; для RU→AB — два этапа QLoRA: адаптация на моноязычном тексте (CLM), затем обучение на параллельном корпусе (SFT).
- **NLLB-200-3.3B**: отдельная encoder–decoder линия для RU→AB с языковыми кодами и принудительным начальным токеном целевого языка.
- **Инференс**: группировка по длине, батчинг, SDPA и beam search там, где они применены в коде сдачи; отдельные инструменты сравнения скорости и качества.
- Скрипты подготовки данных, обучения, оценки BLEU, объединения адаптеров с базовой моделью и сборки Docker/ZIP.

## Что в репозитории

CLI для локального повтора моих запусков и расчёта BLEU. Внутри — код конкурсных решений, конфиги обучения, скрипты подготовки данных и журнал экспериментов.

`dojo.py` вызывает нужное решение из `task-1/` или `task-2/`. Режим выбирается через `--task` и `--model`:

| Задача | Направление | Модели |
|---|---|---|
| `1` | EN → RU | `gemma`, `translate` |
| `2` | RU → AB | `gemma`, `translate`, `nllb` |

Веса и корпуса скачиваются отдельно. `download` загружает базовые модели; мои дообученные веса не опубликованы.

## Установка

Python 3.10+, Linux. Для обучения нужна NVIDIA GPU с CUDA.

```bash
git clone https://github.com/dosados/dojo-solutions.git
cd dojo-solutions
conda create -n dojo python=3.10 -y
conda activate dojo
python -m pip install -r requirements/inference.txt
```

Для оценки установите `requirements/evaluation.txt`, для обучения — `requirements/training.txt`. Если модель требует доступа Hugging Face, выполните `hf auth login`.

## Команды CLI

```bash
# Скачать модель и перевести примеры
python dojo.py download --task 1 --model gemma
python -u dojo.py infer --task 1 --model gemma \
  --input examples/en-ru.json --output out/en-ru.json

# Посчитать macro BLEU по своим переводам и эталонам
python dojo.py evaluate --hyp out/predictions.json --ref data/references.jsonl

# Обучить адаптер по конфигу; сначала подготовьте данные
python -u dojo.py train --task 2 --model translate \
  --config task-2/configs/translate/clm_pilot.yaml

# Собрать конкурсный ZIP с локальными весами
python dojo.py pack --task 1 --model gemma --output out/task1.zip
```

Вход для `infer`: JSON/JSONL с `rid, src`; выход: JSON с `rid, translation`. Для `evaluate` нужны эталоны с `rid, ref`. Пути в примерах оценки замените на свои.

`infer --dry-run` проверяет вход без загрузки модели. `--weights PATH` задаёт каталог весов для `download`, `infer` и `pack`. Все параметры: `python dojo.py <команда> --help`.

[Архитектура](docs/ARCHITECTURE.md) · [Запуски](docs/REPRODUCING.md) · [Ресурсы](docs/RESOURCES.md) · [Результаты](docs/RESULTS.md) · [Журнал](experiments/JOURNAL.md)

[Тимофей Манцуров · dosados](https://github.com/dosados)
