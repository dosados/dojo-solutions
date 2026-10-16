# Dojo · Machine Translation

**24-е место среди 320 участников · top 7,5%**

Моё решение соревнования по машинному переводу в Yandex Contest. Я занял **24-е место из 320**.

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

`dojo.py` — общая точка входа для всех моделей. Она вызывает `solution.py` выбранной задачи. Скрипты отдельных экспериментов лежат в `task-*/scripts/`.

## Быстрый старт

Python 3.10+, Linux; для полноценного запуска рекомендуется NVIDIA GPU. Конкурсная конфигурация — L4 24 ГБ. Обучение и 4-bit режим требуют CUDA. CPU/offload доступны в некоторых ветках, но их скорость не соответствует конкурсным лимитам.

```bash
git clone https://github.com/dosados/dojo-solutions.git
cd dojo-solutions
conda create -n dojo python=3.10 -y
conda activate dojo
python -m pip install -r requirements/inference.txt

# Проверка формата и маршрута запуска без модели и GPU
python dojo.py infer --task 1 --model gemma \
  --input examples/en-ru.json --output out/en-ru.json --dry-run

# Загрузка базовой модели и перевод двух примеров
python dojo.py download --task 1 --model gemma
python -u dojo.py infer --task 1 --model gemma \
  --input examples/en-ru.json --output out/en-ru.json
```

Если Hugging Face запрашивает доступ, примите условия на странице модели и выполните `hf auth login`. Версию PyTorch выбирайте под свою CUDA. В конкурсных Dockerfile указаны PyTorch 2.8 и Transformers 5.5.0.

Для GPU с небольшим объёмом памяти используйте 4-bit квантование:

```bash
python -m pip install bitsandbytes
python -u dojo.py infer --task 1 --model gemma \
  --input examples/en-ru.json --output out/en-ru-4bit.json \
  -- --device cuda --quantize 4bit --batch-size 1
```

## Режимы моделей

| Задача | `--model` | Исходная точка входа | Что запускается |
|---|---|---|---|
| 1 | `gemma` | `task-1/submission/solution.py` | Базовый RU-промпт, без постобработки, beam=3, length penalty=0.8 |
| 1 | `translate` | `task-1/submission_translate/solution.py` | TranslateGemma EN→RU, greedy, динамический бюджет токенов |
| 2 | `gemma` | `task-2/submission/solution.py` | Исходный Gemma RU→AB baseline |
| 2 | `translate` | `task-2/submission_translate/solution.py` | TranslateGemma RU→AB, beam=3, length penalty=0.8 |
| 2 | `nllb` | `task-2/submission_nllb/solution.py` | NLLB RU→AB, beam=4 |

```bash
# Для каждой линии сначала загрузите соответствующие веса
python dojo.py download --task 2 --model translate
python -u dojo.py infer --task 2 --model translate \
  --input examples/ru-ab.json --output out/ru-ab.json

# Свою обученную/merged модель можно подключить без изменения исходного кода
python -u dojo.py infer --task 2 --model translate \
  --weights /path/to/merged-model \
  --input examples/ru-ab.json --output out/ru-ab-trained.json
```

`download` скачивает базовые модели. Для дообученного RU→AB варианта нужно пройти CLM → SFT и объединить адаптер с базой. Веса и корпуса в Git не включены. [Источники](docs/RESOURCES.md) · [Обучение и запуск](docs/REPRODUCING.md).

## Результаты экспериментов

Результаты моих экспериментов. Команды и подробные записи — в [журнале](experiments/JOURNAL.md).

| Измерение | Выборка / режим | Macro BLEU |
|---|---|---:|
| Task 1 · промпт организаторов | OPUS val, 300 примеров, 4-bit | 26.97 |
| Task 1 · выбранный English rules prompt | Та же выборка | **28.42** (+1.45) |
| Task 2 · TranslateGemma после CLM → SFT | Закрытый тест, посылка 31 мая | **8.7** |
| Task 2 · merged TranslateGemma, greedy | Локальный dev, 500 примеров | 25.154 |
| Task 2 · та же модель, beam=3, lp=0.8 | Та же выборка | **26.401** (+1.247) |

На Task 2 получился большой разрыв между dev и закрытым тестом: около 21 BLEU локально и 8.7 на конкурсе. Позже beam search улучшил локальный результат; балл этой версии на закрытом тесте не записан. Подробности разбиения данных и экспериментов — в [результатах](docs/RESULTS.md).

## Устройство репозитория

```text
dojo.py                      единый CLI: download / infer / evaluate / train / pack
requirements/                зависимости по режимам работы
examples/                    небольшие JSON-входы без скачивания корпусов
docs/                        архитектура, воспроизведение, ресурсы, результаты
task-1/
  submission*/               автономные EN→RU решения и Dockerfile
  scripts/                   подготовка, промпты, оценка, обучение
  eval/                      выравнивание по rid и macro BLEU
  train/ + configs/          QLoRA EN→RU
  docs/                      условие и исследовательские заметки
task-2/
  submission*/               Gemma / TranslateGemma / NLLB для RU→AB
  scripts/                   подготовка, CLM/SFT, merge и валидация
  train/ + configs/          общие модули обучения и конфиги трёх линий
  docs/                      условие и детали обучения
experiments/JOURNAL.md        хронология экспериментов
```

- [Архитектура и границы модулей](docs/ARCHITECTURE.md)
- [Воспроизведение: данные, обучение, оценка, Docker](docs/REPRODUCING.md)
- [Модели, корпуса и восстановление файлов](docs/RESOURCES.md)
- [Результаты экспериментов](docs/RESULTS.md)
- [Task 1](task-1/README.md) · [Task 2](task-2/README.md)

[Тимофей Манцуров · dosados](https://github.com/dosados)

Условия задач и стартовые шаблоны — от организаторов соревнования. Ссылки на модели и датасеты — в [ресурсах](docs/RESOURCES.md).
