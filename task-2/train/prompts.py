"""Промпты для SFT Gemma-4-E2B — совпадают с submission/solution.py."""

GEMMA_TRANSLATE_PROMPT = """
Ты профессиональный переводчик. Твоя задача перевести текст с русского на абхазский язык.
ВАЖНО чтобы в ответе был только перевод без каких-либо комментариев и уточнений.

Текст для перевода:
"""


def build_gemma_user_content(src: str) -> str:
    return GEMMA_TRANSLATE_PROMPT + src.strip()
