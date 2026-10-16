"""Экспериментальные промпты для agentic tuning (не в zip контеста по умолчанию)."""

from __future__ import annotations

from typing import Callable

from prompts import (
    _BAD_EXAMPLES,
    _BASELINE_INSTRUCTION,
    _BASELINE_NEG_HEADER,
    build_baseline_negative_prompt,
    build_baseline_prompt,
    build_translation_prompt,
)

PromptBuilder = Callable[[str], str]

# Нейтральные few-shot (не из self/dev): общий EN→RU, без домена статей.
_GENERIC_FEW_SHOT: list[tuple[str, str]] = [
    (
        "The meeting will take place on Monday at 3 p.m.",
        "Встреча состоится в понедельник в 15:00.",
    ),
    (
        "Please save the file before closing the application.",
        "Пожалуйста, сохраните файл перед закрытием приложения.",
    ),
    (
        "She said she would return next week.",
        "Она сказала, что вернётся на следующей неделе.",
    ),
]

_ONLY_TRANSLATION = """\
Переведи с английского на русский. В ответе — только перевод, без пояснений и заголовков.
"""

_LITERAL = """\
Ты переводчик EN→RU. Переводи точно и нейтрально:
- сохраняй порядок предложений и смысловых частей;
- не добавляй и не убирай информацию;
- имена, числа, кавычки и устоявшиеся термины (IT, бренды) оставляй узнаваемыми;
- в ответе только русский текст, без «Вот перевод» и markdown.
"""

_SENTENCE = """\
Переведи одно английское предложение на русский (не абзац, не список).
Сохраняй грамматику и согласование (род, число, время). Термины — устоявшиеся русские эквиваленты.
Ответ: только перевод, одно предложение.
"""

_BLEU_LITE = """\
Перевод с английского на русский для машинной оценки качества:
близко к эталону по формулировкам, без вольного пересказа; структура как в оригинале.
Только русский текст в ответе.
"""


def _append_src(header: str, src: str, *, label_en: str = "Английский", label_ru: str = "Русский") -> str:
    return f"{header}\n\n--- Перевод ---\n{label_en}:\n{src}\n\n{label_ru}:"


def _with_few_shot(
    instruction: str,
    examples: list[tuple[str, str]],
    src: str,
) -> str:
    parts = [instruction, "\nПримеры:\n"]
    for i, (en, ru) in enumerate(examples, start=1):
        parts.append(f"--- {i} ---\nEN: {en}\nRU: {ru}\n\n")
    parts.append(f"--- Перевод ---\nEN:\n{src}\n\nRU:")
    return "".join(parts)


def _with_negatives_only(src: str) -> str:
    parts = [_ONLY_TRANSLATION, "\nТипичные ошибки (так НЕ делай):\n"]
    for i, (en, bad_ru, why) in enumerate(_BAD_EXAMPLES, start=1):
        parts.append(f"{i}. EN: {en}\n   Плохо: {bad_ru}\n   Почему: {why}\n\n")
    parts.append(f"EN:\n{src}\n\nRU:")
    return "".join(parts)


def build_iter01_baseline(src: str) -> str:
    return build_baseline_prompt(src)


def build_iter02_baseline_neg(src: str) -> str:
    return build_baseline_negative_prompt(src)


def build_iter03_v1(src: str) -> str:
    return build_translation_prompt(src)


def build_iter04_minimal(src: str) -> str:
    return _append_src(_ONLY_TRANSLATION, src)


def build_iter05_literal(src: str) -> str:
    return _append_src(_LITERAL, src)


def build_iter06_sentence(src: str) -> str:
    return _append_src(_SENTENCE, src)


def build_iter07_neg_only(src: str) -> str:
    return _with_negatives_only(src)


def build_iter08_generic_fewshot(src: str) -> str:
    return _with_few_shot(_LITERAL, _GENERIC_FEW_SHOT, src)


def build_iter09_bleu_lite(src: str) -> str:
    return _append_src(_BLEU_LITE, src)


def build_iter10_short_neg(src: str) -> str:
    header = (
        "EN→RU перевод. Только русский текст.\n"
        "Не путай род местоимений; IT-термины — fork, merge, не «вилка».\n"
    )
    return _append_src(header, src, label_en="EN", label_ru="RU")


def build_iter11_hybrid_a(src: str) -> str:
    """Антипримеры ТЗ + нейтральный few-shot + краткая инструкция."""
    header = (
        "Профессиональный перевод EN→RU. Только перевод в ответе.\n"
        "Сохраняй структуру, согласование, устоявшиеся термины.\n\n"
        "Ошибки, которых избегать:\n"
    )
    for i, (_en, bad, why) in enumerate(_BAD_EXAMPLES[:2], start=1):
        header += f"  {i}) {why}\n"
    return _with_few_shot(header, _GENERIC_FEW_SHOT[:2], src)


def build_iter12_hybrid_b(src: str) -> str:
    """Sentence-mode + generic few-shot (для OPUS)."""
    return _with_few_shot(_SENTENCE, _GENERIC_FEW_SHOT[:2], src)


def build_iter13_ultra_short(src: str) -> str:
    return f"Translate to Russian. Output only the translation.\n\n{src}"


def build_iter14_ru_instruction(src: str) -> str:
    return (
        "Переведи на русский. Только перевод.\n\n"
        f"English:\n{src}\n\nRussian:"
    )


# iter15+ заполняются после первого прогона (см. refine_* ниже)


def build_iter15_refined(src: str) -> str:
    """После прогона 1–14: sentence + literal, без few-shot."""
    return _append_src(_SENTENCE + "\n" + _LITERAL.strip(), src)


def build_iter16_refined(src: str) -> str:
    """literal + 1 generic example."""
    return _with_few_shot(_LITERAL, [_GENERIC_FEW_SHOT[1]], src)


PROMPT_REGISTRY: dict[str, PromptBuilder] = {
    "iter01_baseline": build_iter01_baseline,
    "iter02_baseline_neg": build_iter02_baseline_neg,
    "iter03_v1": build_iter03_v1,
    "iter04_minimal": build_iter04_minimal,
    "iter05_literal": build_iter05_literal,
    "iter06_sentence": build_iter06_sentence,
    "iter07_neg_only": build_iter07_neg_only,
    "iter08_generic_fewshot": build_iter08_generic_fewshot,
    "iter09_bleu_lite": build_iter09_bleu_lite,
    "iter10_short_neg": build_iter10_short_neg,
    "iter11_hybrid_a": build_iter11_hybrid_a,
    "iter12_hybrid_b": build_iter12_hybrid_b,
    "iter13_ultra_short": build_iter13_ultra_short,
    "iter14_ru_instruction": build_iter14_ru_instruction,
    "iter15_refined": build_iter15_refined,
    "iter16_refined": build_iter16_refined,
}
