"""Промпты EN→RU: baseline (организаторы), baseline_neg (бейзлайн + антипримеры из ТЗ), v1 (BLEU + few-shot)."""

# Оригинальный бейзлайн из шаблона организаторов.
_BASELINE_INSTRUCTION = """\
Ты профессиональный переводчик. Твоя задача перевести текст с английского на русский.
ВАЖНО чтобы в ответе был только перевод без каких-либо комментариев и уточнений.

Текст для перевода:
"""

# Антипримеры из task-1/docs/SPECS.md (типичные ошибки абзачного перевода).
_BAD_EXAMPLES: list[tuple[str, str, str]] = [
    (
        "Danius said that ... when she",
        "Даниус заявил ... когда она",
        "теряется согласованность текста: у Danius мужской род, местоимение должно быть «он», не «она»",
    ),
    (
        "He said in the interview: 'I am made out of metal'",
        "Он, заявив что сделан из металла, рассказал об этом в интервью",
        "нарушается стилистика: не перестраивай цитату в придаточное; сохраняй прямую речь и порядок фраз",
    ),
    (
        'If you want to fork GitHub project... Button "Fork" will help',
        'Если вы хотите отвести GitHub проект... Кнопка "Вилка" поможет',
        "неверно переводятся термины: fork/Fork — устоявшиеся IT-термины, не «отвести» и не «Вилка»",
    ),
]

_BASELINE_NEG_HEADER = """\
Ты профессиональный переводчик. Твоя задача перевести текст с английского на русский.
ВАЖНО чтобы в ответе был только перевод без каких-либо комментариев и уточнений.

Ниже — примеры ПЛОХИХ переводов из типичных ошибок абзачного MT. Так делать НЕ нужно:

"""


def build_baseline_prompt(src: str) -> str:
    return _BASELINE_INSTRUCTION + src


def build_baseline_negative_prompt(src: str) -> str:
    parts = [_BASELINE_NEG_HEADER]
    for i, (en, bad_ru, why) in enumerate(_BAD_EXAMPLES, start=1):
        parts.append(
            f"--- Плохой пример {i} ---\n"
            f"Английский:\n{en}\n\n"
            f"Плохой перевод:\n{bad_ru}\n\n"
            f"Почему плохо: {why}\n\n"
        )
    parts.append("--- Перевод ---\n")
    parts.append("Текст для перевода:\n")
    parts.append(src)
    return "".join(parts)


# Пары из self/dev (article-1–2): близкий к ref стиль, без архаики Пушкина.
FEW_SHOT_EXAMPLES: list[tuple[str, str]] = [
    (
        "To train artificial intelligence, different methods can be used. "
        "They differ from one another in the degree of human involvement in the process.",
        "Чтобы обучить искусственный интеллект, можно использовать разные методы. "
        "Они отличаются друг от друга степенью вовлечённости человека в процесс.",
    ),
    (
        "Advantages: what is good about Python",
        "Преимущества: чем хорош Python",
    ),
    (
        "Specialists highlight a mass of advantages of Python—let us focus on the key ones.",
        "Специалисты выделяют массу преимуществ Python — остановимся на ключевых из них.",
    ),
]

_INSTRUCTION = """\
Ты профессиональный переводчик с английского на русский (абзац целиком).

Цель — перевод, который хорошо оценивается метрикой BLEU (совпадение формулировок с эталоном):
- сохраняй структуру и порядок предложений исходника, не переставляй смысловые блоки;
- не перефразируй без нужды: ближе к дословному переводу, чем к вольной пересказке;
- сохраняй имена, числа, кавычки, аббревиатуры (Python, ML, AI) и единую терминологию внутри абзаца;
- длина перевода сопоставима с английским текстом (не сокращай и не дополняй абзац);
- в ответе только русский перевод, без преамбул («Вот перевод…»), пояснений и markdown.
"""


def build_translation_prompt(src: str) -> str:
    parts = [_INSTRUCTION, "\nПримеры стиля:\n"]
    for i, (en, ru) in enumerate(FEW_SHOT_EXAMPLES, start=1):
        parts.append(f"--- Пример {i} ---\nАнглийский:\n{en}\n\nРусский:\n{ru}\n\n")
    parts.append("--- Перевод ---\nАнглийский:\n")
    parts.append(src)
    parts.append("\n\nРусский:")
    return "".join(parts)


# Лучший после agentic tuning на OPUS (см. docs/PROMPT_TUNING.md, итерации 06/15).
_OPUS_EN_RULES = """\
Translate the following English sentence into Russian.
Rules:
- output only the translation, no comments;
- translate exactly this sentence, do not add or omit content;
- keep names, numbers, and quotes;
- match pronoun gender/number to the subject.
"""


def build_opus_en_prompt(src: str) -> str:
    """OPUS / предложения: EN-инструкция, без few-shot из статей."""
    return f"{_OPUS_EN_RULES}\n\nEnglish:\n{src}\n\nRussian:"


def build_contest_en_prompt(src: str) -> str:
    """Тот же стиль, но «text» вместо sentence — для абзацев контеста."""
    header = _OPUS_EN_RULES.replace("sentence", "text", 1).replace(
        "this sentence", "this text", 1
    )
    return f"{header}\n\nEnglish:\n{src}\n\nRussian:"
