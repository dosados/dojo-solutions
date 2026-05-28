# dojo-solutions

Репозиторий для двух задач Yandex Contest (машинный перевод, сдача Docker + zip). **Локально** — conda **ml-env** (только на вашем ПК). **В Docker** — без Anaconda, зависимости через `pip` в `Dockerfile` (как на контесте). Журнал: `experiments/JOURNAL.md`.

## Структура

| Путь | Назначение |
|------|------------|
| `task-1/` | Задача A: EN→RU, абзацы, gemma-4-E2B бейзлайн |
| `task-2/` | Задача B: RU→Abkhaz, малоресурсный язык |
| `task-*/docs/SPECS.md` | **Условие контеста (копипаста, не редактировать)** — ground truth |
| `scripts/requirements.txt` | Зависимости для локальной работы (если используете) |
| `experiments/JOURNAL.md` | Хронология экспериментов (append-only, текст) |
| `.cursor/rules/` | Правила для AI-агента в Cursor |

## Task 1 — Англо-русский переводчик

Перевод **английских абзацев** на русский с максимальным **BLEU** (sacrebleu, среднее по примерам). Подходы свободные: промпты, постобработка, finetune. Рекомендуемая модель: **google/gemma-4-E2B**.

- Тест: **69** примеров, длинные тексты (~846 символов в среднем).
- Сдача: контейнер читает `input.pickle`, пишет `out/output.json`.
- Бейзлайн: **`task-1/submission/`** (`solution.py`, `Dockerfile`, `weights/`) — это zip на контест; рядом `download_weights.py`, `examples/`, `docs/`, скрипты обучения.
- Ограничения: без сети в рантайме, **≤45 мин** на 69 запросов, образ **<20 GB**, GPU L4 24GB.

Подробности: [task-1/docs/SPECS.md](task-1/docs/SPECS.md). Метрика BLEU: [task-1/docs/BLEU.md](task-1/docs/BLEU.md).

## Task 2 — Русско-абхазский переводчик

Перевод **с русского на абхазский** (малоресурсный язык, спецсимволы алфавита). Цель — максимальный **BLEU**. Даны параллельный и моноязычные корпуса (ссылки в SPECS).

- Тест: **515** коротких примеров (~78 символов).
- Gemma-4 бейзлайн — нижняя планка; копирование русского ≈ **5.2 BLEU**, ориентир сильного решения ≈ **17.6 BLEU**.
- Ограничения: **≤30 мин** на 515 запросов, остальное как в task-1.

- Бейзлайн: **`task-2/submission/`** (`solution.py`, `Dockerfile`, `weights/`); архив организаторов: `ru-ab-gemma-4-baseline-submission/`.

Подробности: [task-2/docs/SPECS.md](task-2/docs/SPECS.md)

## Общий пайплайн сдачи

1. Локально: `conda activate ml-env` → `python download_weights.py` (из `task-1/`) → `submission/weights/` (не `conda run -n`, см. `.cursor/rules/conda-env.mdc`)
2. `docker build ./submission` && `docker run` с монтированием `input.pickle` и `out/`
3. Zip: содержимое **`task-1/submission/`** (`scripts/zip-submission.sh`) — без `__MACOSX` и без двойной вложенности папок
4. Загрузка на contest.yandex.ru (4 попытки / 24 ч)

## Для агента

- Всегда: [.cursor/rules/contest-specs.mdc](.cursor/rules/contest-specs.mdc) — **`task-*/docs/SPECS.md` не менять**, приоритет над остальной документацией
- Всегда: [.cursor/rules/conda-env.mdc](.cursor/rules/conda-env.mdc), [.cursor/rules/experiments-journal.mdc](.cursor/rules/experiments-journal.mdc)
- При работе в `task-1/` или `task-2/`: соответствующие `task-*.mdc`
- Перед новым экспериментом — прочитать `experiments/JOURNAL.md`; после прогона — дописать запись в конец
