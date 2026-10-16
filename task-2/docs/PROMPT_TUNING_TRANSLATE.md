# Prompt tuning — task-2 `submission_translate`

Date: 2026-06-04

Model: `submission_translate/weights` (TranslateGemma-4B, merged adapter).
Eval: `data/processed/dev.input.pickle` / `dev.jsonl`, first 100 examples for prompt search.
Runner: `scripts/run_translate_prompt_iteration.py --iter N --limit 100 --gpu-only --quantize 4bit`.

Important context: the merged adapter was trained only on RU→AB pairs. The bundled TranslateGemma chat template is English, so iteration 02 tests a Russian translation of that template.

## Planned Iterations

| Iter | Name | Hypothesis | BLEU |
|---:|---|---|---:|
| 01 | baseline template EN | Current bundled English TranslateGemma template | **21.55** |
| 02 | Russian full template | Translate the bundled prompt into Russian | 21.47 |
| 03 | Russian short | Less prompt overhead, Russian instruction | 20.27 |
| 04 | Russian close/no-copy | Preserve content and discourage Russian copying | 19.99 |
| 05 | Russian native style | Ask for natural corpus-like Abkhaz | 18.94 |
| 06 | Minimal RU/AB labels | Minimal labels may match SFT behavior better | 20.73 |
| 07 | Russian + `аԥсуа` | Explicitly name Abkhaz in Abkhaz | 20.33 |
| 08 | Short English control | Separate language vs length effect | 20.09 |

## Prompt Texts and Results

### 01 — baseline template EN (winner)

Uses the bundled `weights/chat_template.jinja`:

```text
You are a professional Russian (ru) to Abkhazian (ab) translator...
Produce only the Abkhazian translation, without any additional explanations or commentary.
Please translate the following Russian text into Abkhazian:

{src}
```

Result: **21.55** BLEU (n=100).

### 02 — Russian full template

Russian translation of the bundled instruction:

```text
Ты профессиональный переводчик с русского языка на абхазский. Твоя задача — точно передать смысл и оттенки исходного русского текста, соблюдая грамматику, лексику и естественный стиль абхазского языка.
В ответе выведи только абхазский перевод, без пояснений, вариантов и комментариев. Переведи следующий русский текст на абхазский:

{src}
```

Result: **21.47** BLEU. Close to baseline, but not better.

### 03 — Russian short

```text
Переведи русский текст на абхазский язык. Ответ должен содержать только перевод на абхазский, без пояснений.

{src}
```

Result: **20.27** BLEU.

### 04 — Russian close/no-copy

```text
Переведи с русского на абхазский. Сохраняй смысл, имена, числа и порядок фраз.
Не копируй русский текст, если есть абхазский перевод. Только перевод.

{src}
```

Result: **19.99** BLEU.

### 05 — Russian native style

```text
Ты переводчик русско-абхазского корпуса. Переведи предложение так, как оно могло бы звучать в естественном абхазском тексте. Без комментариев.

{src}
```

Result: **18.94** BLEU.

### 06 — Minimal labels

```text
Русский:
{src}

Абхазский:
```

Result: **20.73** BLEU.

### 07 — Russian + `аԥсуа`

```text
Переведи русский текст на абхазский язык (аԥсуа). Выведи только абхазский перевод, без русского текста и без пояснений.

{src}
```

Result: **20.33** BLEU.

### 08 — Short English control

```text
Translate the Russian text into Abkhazian. Output only the Abkhazian translation, no explanations.

{src}
```

Result: **20.09** BLEU.

## Findings

- The Russian translation of the bundled prompt was the strongest alternative, but it still lost narrowly to the original template: **21.47 vs 21.55**.
- Short custom prompts underperform the bundled TranslateGemma template. The adapter was trained with the structured TranslateGemma message format, so changing the rendered chat prompt is more disruptive than expected.
- During iteration 01, a batched decoding bug was found: with left padding, outputs must be sliced at the padded input length, not each example's unpadded length. This is fixed in `submission_translate/solution.py` and in the runner.
- Fix verification: after the change, baseline `batch_size=40` vs `batch_size=1` on the first 40 dev examples gives close BLEU (**22.36** vs **22.23**) and zero prompt artifacts (`model`, `Abkhazian:`, `Русский:`) in both outputs. Before the fix, batched output included prompt fragments such as `Abkhazian:\n\n\n... model ...`.

Decision: keep the current bundled English template for submission, but include the decode fix.
