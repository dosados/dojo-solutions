# Agentic prompt tuning — Gemma-4-E2B (task-1)

**Модель:** `google/gemma-4-E2B-it`, 4-bit, `postprocess` on.
**Метрики:** macro BLEU (sacrebleu). **Главный ориентир — OPUS-100 val** (первые 300 предложений); self/dev (117 абзацев) — sanity, без подгонки под Пушкина.
**Прогон:** один промпт → правка `submission/prompt_tune_active.py` →
`python -u scripts/run_prompt_iteration.py --iter N --gpu-only --quantize 4bit`

## Сводка (16 итераций)

| # | Кратко | OPUS BLEU | self BLEU | Δ OPUS vs 01 |
|---|--------|-----------|-----------|----------------|
| 01 | RU бейзлайн организаторов | 26.97 | 54.57 | — |
| 02 | одно предложение, RU | **27.65** | 54.70 | +0.68 |
| 03 | дословность + согласование | 27.43 | 54.43 | +0.46 |
| 04 | антипаттерны ТЗ (кратко) | 26.93 | 55.23 | −0.04 |
| 05 | близость к формулировкам (RU) | 27.53 | 55.07 | +0.56 |
| **06** | **EN rules (лучший)** | **28.42** | 55.20 | **+1.45** |
| 07 | EN + generic few-shot | 27.16 | 55.24 | +0.19 |
| 08 | EN без меток EN/RU в конце | 25.27 | 53.85 | −1.70 |
| 09 | EN минимальный | 27.32 | 54.98 | +0.35 |
| 10 | EN rules + closeness wording | 28.33 | 55.13 | +1.36 |
| 11 | EN короткий (3 правила) | 28.23 | 54.94 | +1.26 |
| 12 | RU как v1, sentence | 27.63 | 54.45 | +0.66 |
| 13 | EN + короткие лейблы | 28.02 | 55.36 | +1.05 |
| 14 | EN + IT-термины | 27.58 | 55.05 | +0.61 |
| 15 | повтор #06 | **28.42** | 55.20 | +1.45 |
| 16 | EN + «professional translator» | 27.90 | 54.75 | +0.93 |

**Победитель:** **#06 / #15** — английская инструкция, 4 правила, метки `English:` / `Russian:`.
В коде: `prompts.build_opus_en_prompt` (предложения), `prompts.build_contest_en_prompt` (абзацы контеста).

**Выводы по направлениям:**
- **EN > RU** для Gemma на OPUS (+~1 BLEU к бейзлайну).
- **Few-shot из статей и длинные антипримеры** на OPUS не помогли (#07 хуже #06).
- **Фрейм «одно предложение»** полезен для OPUS; для self абзацы лучше `v1` (~57 vs ~55).
- Стабильные провалы: архаичный EN (rid 156), несовпадение пары ref/src (rid 73), подмена длинного контекста (rid 217) — промптом не лечится.

Артефакты: `out/prompt_tune/iterNN_*.json`, `iterNN_metrics.json`.

---

## Промпт 01 — бейзлайн организаторов

**Гипотеза:** Нижняя планка.

```
Ты профессиональный переводчик. Твоя задача перевести текст с английского на русский.
ВАЖНО чтобы в ответе был только перевод без каких-либо комментариев и уточнений.

Текст для перевода:
{src}
```

**Результаты:** OPUS **26.97**, self **54.57**.

**Вывод:** Близко к журнальному OPUS ~26.5. Частые подмены смысла (rid 22, 156, 217).

---

## Промпт 02 — одно предложение (RU)

**Гипотеза:** OPUS = предложения; убрать абзачный дрейф.

```
Переведи одно английское предложение на русский.
Правила:
- переводи только данный текст, не добавляй и не продолжай мысль;
- сохраняй имена, числа, кавычки;
- в ответе только перевод, без пояснений.

English:
{src}

Russian:
```

**Результаты:** OPUS **27.65**, self **54.70**.

**Вывод:** +0.7 OPUS; худшие rid те же. Идём к дословности и EN.

---

## Промпт 03 — дословность + согласование (RU)

**Гипотеза:** Снизить подмену смысла.

```
Перевод с английского на русский (одно предложение).
Дословно передавай смысл; не заменяй текст другим предложением.
Согласование местоимений по роду и числу подлежащего.
Имена, аббревиатуры и устоявшиеся термины сохраняй.
Ответ — только перевод.

English:
{src}

Russian:
```

**Результаты:** OPUS **27.43**, self **54.43**.

**Вывод:** Чуть хуже #02; избыточные RU-формулировки не дают выигрыша.

---

## Промпт 04 — антипаттерны ТЗ (кратко, RU)

**Гипотеза:** Антипримеры из SPECS без few-shot.

```
Переведи одно английское предложение на русский. Только перевод в ответе.
Не путай род местоимений (he → он). IT-термины: fork, merge — не «вилка».
Прямую речь и кавычки сохраняй, не перестраивай в придаточные.
Не добавляй фраз, которых нет в исходнике.

English:
{src}

Russian:
```

**Результаты:** OPUS **26.93**, self **55.23**.

**Вывод:** На OPUS регресс; self чуть выше — не ориентир для OPUS.

---

## Промпт 05 — близость к формулировкам (RU)

**Гипотеза:** Явный BLEU-стиль без few-shot из статей.

```
Переведи одно английское предложение на русский.
Перевод должен быть близок к исходнику по формулировкам (для машинной оценки):
сохраняй порядок частей, не перефразируй без нужды.
Только данный текст; не добавляй предложений.
Ответ — только русский перевод.

English:
{src}

Russian:
```

**Результаты:** OPUS **27.53**, self **55.07**.

**Вывод:** Средний результат; пробуем EN.

---

## Промпт 06 — EN rules (лучший)

**Гипотеза:** Gemma лучше следует EN-инструкции.

```
Translate the following English sentence into Russian.
Rules:
- output only the translation, no comments;
- translate exactly this sentence, do not add or omit content;
- keep names, numbers, and quotes;
- match pronoun gender/number to the subject.

English:
{src}

Russian:
```

**Результаты:** OPUS **28.42**, self **55.20**.

**Вывод:** **Лучший OPUS** (+1.45 к #01). Принят как финальный для OPUS.

---

## Промпт 07 — EN + generic few-shot

**Гипотеза:** Нейтральные пары (не из self/dev).

```
Translate the English sentence into Russian. Output only the translation.

Examples:
EN: The meeting will take place on Monday at 3 p.m.
RU: Встреча состоится в понедельник в 15:00.

EN: She said she would return next week.
RU: Она сказала, что вернётся на следующей неделе.

EN: {src}
RU:
```

**Результаты:** OPUS **27.16**, self **55.24**.

**Вывод:** Few-shot вредит OPUS; не использовать для обобщающего промпта.

---

## Промпт 08 — EN без меток в конце

**Гипотеза:** Короче шаблон → меньше служебного текста в ответе.

```
Translate the English sentence into Russian.
Stay close to the source wording and sentence structure (for machine translation evaluation).
Output only the Russian translation. Do not add or omit content.

{src}
```

**Результаты:** OPUS **25.27**, self **53.85**.

**Вывод:** Сильная просадка; метки `English:`/`Russian:` нужны.

---

## Промпт 09 — EN минимальный

```
Translate this English sentence into Russian.
Output only the translation. Do not add or omit anything.

English:
{src}

Russian:
```

**Результаты:** OPUS **27.32**, self **54.98**.

**Вывод:** Хуже #06; явные rules важны.

---

## Промпт 10 — EN rules + closeness wording

**Гипотеза:** Добавить к #06 правило близости формулировок.

```
Translate the following English sentence into Russian.
Rules:
- output only the translation, no comments;
- translate exactly this sentence, do not add or omit content;
- keep names, numbers, and quotes;
- match pronoun gender/number to the subject;
- stay close to source wording and clause order when natural.

English:
{src}

Russian:
```

**Результаты:** OPUS **28.33**, self **55.13**.

**Вывод:** Почти как #06 (−0.09 OPUS); лишнее правило не нужно.

---

## Промпт 11 — EN короткий (3 правила)

```
Translate the following English sentence into Russian.
Output only the translation. Translate exactly this sentence;
keep names, numbers, and quotes.

English:
{src}

Russian:
```

**Результаты:** OPUS **28.23**, self **54.94**.

**Вывод:** Чуть хуже #06; правило про местоимения полезно.

---

## Промпт 12 — RU «как v1», sentence

**Гипотеза:** Русская BLEU-инструкция без few-shot.

```
Ты переводчик EN→RU. Переведи одно английское предложение.
Сохраняй структуру и порядок частей; не перефразируй без нужды.
Имена, числа, кавычки и термины сохраняй. Согласуй местоимения с подлежащим.
Только русский перевод, без преамбул.

English:
{src}

Russian:
```

**Результаты:** OPUS **27.63**, self **54.45**.

**Вывод:** EN (#06) лучше RU (#12) на OPUS.

---

## Промпт 13 — EN + короткие лейблы

**Гипотеза:** Помочь rid с коротким src (CIOR, Description).

```
Translate the following English sentence into Russian.
Rules:
- output only the translation, no comments;
- translate exactly this sentence, do not add or omit content;
- if the source is a short label or title, give the closest natural Russian equivalent;
- keep names, numbers, and quotes;
- match pronoun gender/number to the subject.

English:
{src}

Russian:
```

**Результаты:** OPUS **28.02**, self **55.36**.

**Вывод:** Self чуть выше, OPUS ниже #06; правило про лейблы не даёт стабильного выигрыша.

---

## Промпт 14 — EN + IT-термины

**Результаты:** OPUS **27.58**, self **55.05**.

**Вывод:** Хуже #06; для OPUS лишнее.

---

## Промпт 15 — контрольный повтор #06

**Результаты:** OPUS **28.42**, self **55.20** (совпадает с #06).

**Вывод:** Результат воспроизводим.

---

## Промпт 16 — EN + «professional translator»

**Гипотеза:** Роль переводчика усилит дисциплину вывода.

```
You are a professional translator. Translate the following English sentence into Russian.
Rules:
- output only the translation, no comments or preambles;
- translate exactly this sentence, do not add or omit content;
- keep names, numbers, and quotes;
- match pronoun gender/number to the subject.

English:
{src}

Russian:
```

**Результаты:** OPUS **27.90**, self **54.75**.

**Вывод:** Хуже #06; роль не нужна.

---

## Рекомендации для сдачи

| Сценарий | Промпт | `--prompt` |
|----------|--------|------------|
| OPUS / предложения | #06 | `opus_en` |
| Контест (абзацы) | #06 с «text» | `contest_en` |
| self/dev IT-абзацы | v1 + few-shot | `v1` (по-прежнему ~57 self) |

Следующий шаг: прогнать `contest_en` на полном OPUS (500) и self/dev; сравнить с `v1` на абзацах перед zip.
