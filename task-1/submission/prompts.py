"""Промпт для EN→RU: ориентация на macro BLEU + few-shot из dev-корпуса."""

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
