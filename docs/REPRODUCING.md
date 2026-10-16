# Воспроизведение и исследование

Команды ниже выполняются из корня репозитория, если явно не указан `cd`. Для локальной работы создайте окружение по [README](../README.md); в моём локальном окружении используется `conda activate ml-env`. Скачивание моделей и данных требует сети, автономный инференс использует подготовленные локальные файлы.

## 1. Запуск исходного решения

```bash
python -m pip install -r requirements/inference.txt
python dojo.py download --task 1 --model gemma
python -u dojo.py infer --task 1 --model gemma \
  --input examples/en-ru.json --output out/en-ru.json
```

Поддерживаются JSON-массив, JSONL и доверенный pickle с полями `rid` и `src`. JSONL может также содержать `ref`: CLI передаст модели только `rid,src`. Идентификаторы должны быть уникальными целыми числами. Выход — JSON с `rid,translation`. `--dry-run` проверяет вход и выводит план без загрузки модели.

Модель можно хранить вне репозитория:

```bash
python dojo.py download --task 2 --model translate --weights /path/to/base-model
python -u dojo.py infer --task 2 --model translate \
  --weights /path/to/base-model --input examples/ru-ab.json --output out/ru-ab.json
```

Это базовая, не дообученная модель. `download` не пишет в непустую папку. Чтобы зафиксировать базовую ревизию для своего исследования, передайте `--revision` с commit SHA из Hub.

## 2. Данные EN→RU и оценка

```bash
python -m pip install -r requirements/evaluation.txt
cd task-1
python -u scripts/download_opus100.py
python -u scripts/convert_datasets.py --skip-self
cd ..

python -u dojo.py infer --task 1 --model gemma \
  --input task-1/data/processed/opus100/validation.jsonl \
  --output out/opus100-validation.json
python dojo.py evaluate \
  --hyp out/opus100-validation.json \
  --ref task-1/data/processed/opus100/validation.jsonl
```

Этот полный validation-прогон может занимать значительное время. Для первого запуска используйте двухстрочный `examples/en-ru.json` или свою небольшую JSONL-выборку. `evaluate` считает среднее sentence BLEU, выравнивая строки по `rid`; corpus BLEU — другая величина. Для полной диагностики:

```bash
python task-1/scripts/eval_bleu.py \
  --hyp out/opus100-validation.json \
  --ref task-1/data/processed/opus100/validation.jsonl --worst 10 --show-text
```

`convert_datasets.py --skip-self` использует официальные splits скачанного OPUS. Другой исследовательский инструмент `prepare_data.py` строит собственное разбиение и псевдоабзацы; не смешивайте его метрики с официальным validation. Описание — [Task 1 datasets](../task-1/docs/DATASETS.md).

## 3. Подготовка RU→AB

Нужны конкурсные файлы, которых нет в Git (см. [источники](RESOURCES.md)):

```text
task-2/data/ab-ru-parallel.csv    UTF-8 CSV: ru,ab
task-2/data/ab-mono.txt           абхазский текст по строкам
```

```bash
python -m pip install -r requirements/training.txt
cd task-2
python -u download_ru_wikipedia.py
python -u scripts/prepare_mono.py --ru-wikipedia --parallel-ru
python -u scripts/prepare_parallel.py --val-n 500
cd ..
```

Получатся `data/processed/mono/train.jsonl` (`text,lang`) и `data/processed/parallel/{train,val}.jsonl` (`src,ref`). Для оценки задайте отдельным парам в `val.jsonl` уникальные числовые `rid`, затем передайте этот файл в `infer` и `evaluate`.

Старый `prepare_dev.py` создаёт собственную случайную выборку из CSV с отдельными фильтрами. Она не обязана быть непересекающейся с train из `prepare_parallel.py`. Для новых честных сравнений используйте единственное зафиксированное разбиение с проверкой пересечений по тексту; исторические метрики не переименовывайте в независимый held-out результат.

## 4. QLoRA: CLM → SFT

Для TranslateGemma (основная обученная RU→AB линия):

```bash
python dojo.py download --task 2 --model translate
python -u dojo.py train --task 2 --model translate \
  --config task-2/configs/translate/clm_pilot.yaml
python -u dojo.py train --task 2 --model translate \
  --config task-2/configs/translate/sft_pilot.yaml
```

SFT-конфиг указывает на CLM-адаптер. Этапы идут последовательно: сначала должен появиться `runs/lora/translate/clm_pilot/final`. Это обучение на исходной базе; не подставляйте в `model_dir` уже merged-веса.

Аналогично доступны Gemma (`configs/gemma/`) и NLLB (`configs/nllb/`). Для NLLB выберите `--model nllb`: CLI вызовет `train_qlora_nllb.py`, а не causal-LM trainer. Исторические `*_smoke.yaml` рассчитаны на короткие проверки, `*_pilot.yaml` — на исследовательские прогоны; поздние `*_r24_*.yaml` содержат конкретные цепочки адаптеров и локальных путей. Это конфиги, не автоматически взаимозаменяемые пресеты.

Основные параметры задаются в YAML: модель и файлы данных, `adapter_init`, `max_seq_length`, `lora_r`, `lora_alpha`, batch size, gradient accumulation, learning rate, число эпох и сохранение checkpoints. Подробности: [Task 2 training](../task-2/docs/TRAINING.md).

## 5. Валидация и merge

```bash
cd task-2
python -u scripts/validate_lora.py \
  --model translate --adapter runs/lora/translate/sft_pilot/final --limit 50

# База остаётся исходной; merged результат — в отдельном каталоге.
python -u scripts/merge_lora.py --model translate \
  --base submission_translate/weights \
  --adapter runs/lora/translate/sft_pilot/final \
  --out submission_translate/weights_merged --device cpu
cd ..

python -u dojo.py infer --task 2 --model translate \
  --weights task-2/submission_translate/weights_merged \
  --input examples/ru-ab.json --output out/ru-ab-trained.json
```

CPU merge требует достаточно RAM для полной bf16-модели. Он может быть существенно тяжелее обучения самого адаптера. Пример `validate_lora.py` использует исторический dev-путь; подготовьте его через `prepare_dev.py` для сопоставления с прежними экспериментами или передайте собственные `--input` и `--ref` (см. `--help`). Измерение не становится независимым от train автоматически.

Task 1 также содержит исходную ветку QLoRA EN→RU:

```bash
python -u dojo.py train --task 1 --model gemma \
  --config task-1/configs/lora_pilot.yaml
```

Для неё сначала нужны базовые веса и OPUS из шагов 1–2. Ветка включает загрузку обучающих пар, QLoRA и merge адаптера.

## 6. Docker и ZIP

В каждой папке `submission*` сохранён исходный Dockerfile. Его базовый образ находится в реестре организаторов Yandex; доступность этого образа в произвольной внешней среде не гарантирована. Локальный CLI позволяет изучать решение без этого реестра.

```bash
docker build -t dojo-task1 task-1/submission
mkdir -p out
docker run --rm --gpus all \
  -v "$PWD/task-1/examples/example-input.pickle:/workspace/input.pickle:ro" \
  -v "$PWD/out:/workspace/out" \
  dojo-task1 --output /workspace/out/output.json
```

Последний `--output` поддерживается Task 1 / Gemma. У других исходных submissions жёстко задан `output.json` в рабочей папке. Для них заберите результат из остановленного контейнера:

```bash
# Для обученного варианта weights/ в Docker context должен содержать merged модель.
docker build -t dojo-task2 task-2/submission_translate
docker create --name dojo-task2-run --gpus all \
  -v "$PWD/task-2/examples/example-input.pickle:/workspace/input.pickle:ro" \
  dojo-task2
docker start -a dojo-task2-run
docker cp dojo-task2-run:/workspace/output.json out/task2-docker.json
docker rm dojo-task2-run
```

Сборка плоского архива, с выбранной моделью в `weights/`:

```bash
python dojo.py pack --task 2 --model translate \
  --weights task-2/submission_translate/weights_merged \
  --output out/task2-translate.zip
```

Упаковщик включает исходные `.py`, Dockerfile и выбранные веса, исключает HF cache, незавершённые загрузки, резервные копии и файлы локального inference. ZIP не загружается в Git. Время инференса и размер Docker-образа проверяйте на целевом оборудовании.

## 7. Исторические исследовательские режимы

`task-1/scripts/run_prompt_iteration.py` и `task-2/scripts/run_translate_prompt_iteration.py` воспроизводят варианты промптов; `compare_models.py` и `bench_inference.py` сравнивают линии. Запускайте из каталога соответствующей задачи и начните с `--help`.

Эти инструменты сохранены как исследовательские программы. Их промпты, greedy/beam режимы, batching и postprocess могут отличаться от текущего конкурсного `solution.py`. Для запуска именно сохранённой посылки используйте `dojo.py infer`. Зафиксированные результаты и неудачные эксперименты — в [журнале](../experiments/JOURNAL.md).

## Проверки интерфейса

```bash
python -m unittest discover -s tests -v
```

12 CPU-only тестов проверяют маршруты моделей, JSON/JSONL/pickle, ошибки контракта, вызов исходного модуля, восстановление рабочего каталога при сбое, отказ перезаписи весов, маршрутизацию обучения и состав ZIP. Для тестов нужен PyYAML, GPU и скачивание моделей не нужны. Тот же набор запускается через GitHub Actions.
