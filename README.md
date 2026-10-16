# Dojo · Machine Translation

**24-е место из 320 участников**

Моё решение соревнования по машинному переводу Data Dojo.

## Задача

| | EN → RU | RU → AB |
|---|---|---|
| Перевод | Английские абзацы на русский с сохранением смысла, стиля и терминологии | Короткие русские тексты на малоресурсный абхазский язык |
| Тест | 69 абзацев | 515 текстов |
| Лимит времени | 45 минут | 30 минут |

Качество оценивалось по среднему sentence BLEU. Решение запускалось без сети в Docker на NVIDIA L4 с 24 ГБ VRAM и 16 ГБ RAM. Размер образа — до 20 GB.

## Решение

**Для EN → RU я выбрал Gemma-4-E2B.** Сравнил её с TranslateGemma, проверил варианты промптов, few-shot, постобработку и QLoRA на OPUS. В итоговом коде использовал инструкцию на русском, beam search с тремя лучами и length penalty 0.8. Длина генерации зависит от длины абзаца.

**Для RU → AB я выбрал TranslateGemma-4B.** Сравнил её с Gemma и NLLB и дообучил в два этапа: CLM на русском и абхазском моноязычных корпусах, затем SFT на параллельных переводах. Обучение — QLoRA с 4-bit NF4; при SFT loss считается по токенам перевода. Для инференса объединил адаптер с базовой моделью в bf16 и использовал beam search с тремя лучами.

**Инференс.** Тексты группируются по длине и обрабатываются батчами. TranslateGemma использует SDPA. Веса включены в конкурсный контейнер, поэтому во время проверки загрузка из сети не требуется.

**Архитектура.** У каждой задачи свой модуль перевода и Dockerfile. Подготовка данных, обучение, оценка BLEU и сборка посылки вынесены в отдельные скрипты. Конфиги задают параметры QLoRA и последовательность CLM → SFT.

## Код и точки входа

В репозитории — код перевода, подготовка данных, обучение адаптеров, оценка BLEU и сборка конкурсного ZIP. CLI `dojo.py` запускает эти операции из корня проекта.

| Путь | Назначение |
|---|---|
| `dojo.py` | Общий CLI для обеих задач |
| `task-1/submission/solution.py` | Выбранное решение EN → RU на Gemma |
| `task-2/submission_translate/solution.py` | Выбранное решение RU → AB на TranslateGemma |
| `task-*/scripts/` | Подготовка данных, обучение, merge адаптеров и эксперименты |
| `task-*/train/`, `task-*/configs/` | Модули обучения и YAML-конфиги |
| `task-1/eval/` | Расчёт macro BLEU для обеих задач |

В `submission*` также лежат альтернативные модели и Dockerfile. Параметр `--task` выбирает задачу: `1` — EN → RU, `2` — RU → AB. Параметр `--model` выбирает `gemma`, `translate` или `nllb` (только для задачи 2).

## Быстрый старт

Python 3.10+, Linux, NVIDIA GPU с CUDA.

```bash
git clone https://github.com/dosados/dojo-solutions.git
cd dojo-solutions
conda create -n dojo python=3.10 -y
conda activate dojo
python -m pip install -r requirements/inference.txt

python dojo.py download --task 1 --model gemma
python -u dojo.py infer --task 1 --model gemma \
  --input examples/en-ru.json --output out/en-ru.json
```

`download` скачивает базовые веса. Свою дообученную модель можно передать в `infer` через `--weights PATH`. Для RU → AB используйте `--task 2 --model translate` и вход `examples/ru-ab.json`.

## Функционал CLI

| Команда | Действие | Основные параметры |
|---|---|---|
| `download` | Скачать базовую модель | `--task`, `--model` |
| `infer` | Перевести тексты | `--task`, `--model`, `--input`, `--output` |
| `evaluate` | Посчитать macro BLEU | `--hyp`, `--ref` |
| `train` | Обучить адаптер по конфигу | `--task`, `--model`, `--config` |
| `pack` | Собрать ZIP с кодом и весами | `--task`, `--model`, `--output` |

Для оценки установите зависимости из `requirements/evaluation.txt`, для обучения — из `requirements/training.txt` той же командой `pip install -r`. Конфиги обучения лежат в `task-*/configs/`; пути к корпусам заданы внутри YAML.

Вход `infer` — JSON/JSONL с полями `rid, src`, выход — JSON с `rid, translation`. Для `evaluate` нужны эталоны с `rid, ref`. Справка по параметрам: `python dojo.py <команда> --help`.

[Условие EN → RU](task-1/docs/SPECS.md) · [Условие RU → AB](task-2/docs/SPECS.md)
