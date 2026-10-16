# TranslateGemma-4B — сдача на контест (task-1)

Вариант с [google/translategemma-4b-it](https://huggingface.co/google/translategemma-4b-it) вместо Gemma-4-E2B-it. Формат zip/Docker тот же: `Dockerfile`, `solution.py`, `weights/`.

Перевод **EN→RU** через chat-template (`source_lang_code` / `target_lang_code`). Inference: батчинг по длине, `sdpa`, `torch.compile`, динамический `max_new_tokens` под абзацы.

## Подготовка весов

Из `task-1/` (лицензия Gemma на Hugging Face):

```bash
conda activate ml-env
python download_weights_translate.py
```

Веса только в `submission_translate/weights/` этой задачи (~8 GB), отдельно от `submission/weights/` (Gemma).

## Локальная оценка и сравнение с Gemma

Из `task-1/`:

```bash
conda activate ml-env
python -u scripts/compare_models.py --limit 20 --gpu-only --rerun
python -u scripts/bench_inference.py --limit 10 --gpu-only
```

## Упаковка в zip

```bash
cd task-1/submission_translate
zip -0 -r ../task-1-translategemma-submission.zip . -x "weights/.cache/*"
```
