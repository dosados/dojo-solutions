# Task 1 · Английский → русский

Перевод абзацев с сохранением смысла, стиля и терминологии. Конкурсная метрика — среднее sentence BLEU на 69 закрытых примерах; лимит 45 минут на L4 24 ГБ. [Оригинальное условие](docs/SPECS.md).

## Запуск из корня репозитория

```bash
python dojo.py download --task 1 --model gemma
python -u dojo.py infer --task 1 --model gemma \
  --input examples/en-ru.json --output out/en-ru.json
```

Вторая линия: замените `--model gemma` на `--model translate` и отдельно скачайте её веса. Квантование исходного Gemma inference: после аргументов CLI добавьте `-- --device cuda --quantize 4bit --batch-size 1` (нужен bitsandbytes).

## Код

| Каталог | Назначение |
|---|---|
| `submission/` | Исходная Gemma-посылка: baseline-промпт, beam=3, lp=0.8, без постобработки |
| `submission_translate/` | Альтернатива TranslateGemma EN→RU |
| `scripts/` | OPUS, преобразование форматов, локальная оценка, промпты, QLoRA, merge |
| `eval/` | Загрузка эталонов и переводов, выравнивание rid, macro BLEU |
| `train/`, `configs/` | Исследовательская QLoRA-ветка |
| `docs/` | Условие и исторические исследовательские материалы |

Экспериментальные `prompts.py`, `postprocess.py`, `local_inference.py` сохранены рядом, но общий CLI вызывает текущий `solution.py`. Настройки этих веток не обязаны совпадать. На OPUS val исследовательский English rules prompt улучшил 26.97 → 28.42 BLEU; это отдельный эксперимент, а не метрика текущего default inference.

[Полное воспроизведение](../docs/REPRODUCING.md) · [Архитектура](../docs/ARCHITECTURE.md) · [Ресурсы](../docs/RESOURCES.md) · [BLEU](docs/BLEU.md) · [Датасеты](docs/DATASETS.md) · [Обучение](docs/TRAINING.md) · [Журнал](../experiments/JOURNAL.md).
