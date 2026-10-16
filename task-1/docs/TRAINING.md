# QLoRA EN→RU

Исследовательская ветка обучения Gemma на OPUS; её наличие не означает улучшение финального конкурсного балла. Текущий inference использует код из `submission/solution.py`.

Сначала установите `requirements/training.txt`, скачайте Task 1 / Gemma и подготовьте OPUS по [общему руководству](../../docs/REPRODUCING.md). Затем из корня:

```bash
python -u dojo.py train --task 1 --model gemma \
  --config task-1/configs/lora_pilot.yaml
```

`train/data.py` делает фильтрацию и детерминированный sampling JSONL по seed; `train/tokenize_sft.py` применяет training prompt и маскирует target для SFT. `gemma_peft.py`, `kbit.py`, `gemma_trainer.py` содержат исходные адаптации Gemma/PEFT для ограниченной VRAM. Конфиг задаёт LoRA rank, длину последовательности, число примеров и gradient accumulation.

Скрипт сохраняет адаптер в `task-1/runs/lora/pilot_10k/final`. Для проверки через текущую посылку сначала объедините адаптер с исходной базой. Историческая утилита Task 1 выполняет merge **в указанный base-dir**; для сохранения базы работайте на отдельной копии:

```bash
cp -a task-1/submission/weights task-1/submission/weights_merged
cd task-1
python -u scripts/merge_lora.py \
  --adapter runs/lora/pilot_10k/final \
  --base-dir submission/weights_merged --device cpu
cd ..
python -u dojo.py infer --task 1 --model gemma \
  --weights task-1/submission/weights_merged \
  --input task-1/data/processed/opus100/validation.jsonl \
  --output out/opus100-lora.json
python dojo.py evaluate --hyp out/opus100-lora.json \
  --ref task-1/data/processed/opus100/validation.jsonl
```

Merged модель требует существенно больше места и RAM, чем адаптер. Для небольшой GPU после параметров `infer` используйте `-- --device cuda --quantize 4bit --batch-size 1`. Исходные `train/data.py` и `merge_lora.py` восстановлены из истории `62e1c30`; алгоритм обучения не изменён. Старые обёртки с отсутствующим в текущем inference флагом `--adapter` в рабочий интерфейс не включены.
