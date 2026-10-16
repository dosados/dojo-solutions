# Task 2 · Русский → абхазский

Машинный перевод на малоресурсный язык. Конкурсная метрика — среднее sentence BLEU на 515 закрытых примерах; лимит 30 минут на L4 24 ГБ. [Оригинальное условие](docs/SPECS.md).

## Запуск из корня репозитория

```bash
python dojo.py download --task 2 --model translate
python -u dojo.py infer --task 2 --model translate \
  --input examples/ru-ab.json --output out/ru-ab.json
```

Это скачивание базовой модели. Обученные RU→AB веса не опубликованы; для повторения CLM/SFT нужны конкурсные корпуса и обучение. Скрипты подготовки и конфиги включены, ссылки и ограничения — в [ресурсах](../docs/RESOURCES.md).

## Три линии

| `--model` | Код | Подход |
|---|---|---|
| `gemma` | `submission/` | Gemma-4-E2B baseline, обычный RU-промпт |
| `translate` | `submission_translate/` | TranslateGemma-4B, structured translation template, CLM → SFT, beam=3 / lp=0.8 |
| `nllb` | `submission_nllb/` | NLLB-200-3.3B seq2seq, `rus_Cyrl` → `abk_Cyrl`, forced BOS, beam=4 |

В `configs/{gemma,translate,nllb}` находятся собственные конфиги; в `train/` — подготовка данных, токенизация, модели, PEFT и Trainer callbacks; в `scripts/` — точки входа подготовки, обучения, валидации и merge.

Успешная TranslateGemma CLM → SFT посылка 31 мая получила 8.7 BLEU на скрытом тесте. Позднее beam search дал локальный прирост 25.154 → 26.401 на dev; закрытый балл последней версии не зафиксирован. Подробные настройки и результаты — в [журнале](../experiments/JOURNAL.md).

[Воспроизведение](../docs/REPRODUCING.md) · [Архитектура](../docs/ARCHITECTURE.md) · [Обучение](docs/TRAINING.md) · [Результаты](../docs/RESULTS.md) · [Журнал](../experiments/JOURNAL.md).
