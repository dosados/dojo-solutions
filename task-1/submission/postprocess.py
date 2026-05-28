"""Очистка вывода модели перед записью в output.json."""

import re

# Преамбулы и служебные строки (многократно — модель иногда повторяет).
_PREAMBLE_RES = [
    re.compile(
        r"^(?:вот\s+)?(?:готовый\s+)?перевод\s*(?:на\s+русский(?:\s+язык)?)?\s*:?\s*",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(r"^русский(?:\s+перевод)?\s*:?\s*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^перевод\s*:?\s*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^translation\s*:?\s*", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^here is the translation\s*:?\s*", re.IGNORECASE | re.MULTILINE),
]

_THINKING_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.IGNORECASE | re.DOTALL)
_FENCE_RE = re.compile(r"^```(?:\w+)?\s*|\s*```$", re.MULTILINE)


def postprocess_translation(text: str) -> str:
    if not text:
        return text

    t = text.strip()
    t = _THINKING_RE.sub("", t).strip()
    t = _FENCE_RE.sub("", t).strip()

    # Если модель повторила «Русский:» в теле ответа — оставить последний блок.
    if "\nРусский:" in t or t.startswith("Русский:"):
        parts = re.split(r"(?:^|\n)Русский:\s*", t, flags=re.IGNORECASE)
        t = parts[-1].strip()

    changed = True
    while changed and t:
        changed = False
        for pat in _PREAMBLE_RES:
            new_t = pat.sub("", t, count=1).strip()
            if new_t != t:
                t = new_t
                changed = True

    # Обёртка в кавычки целиком
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'«»":
        t = t[1:-1].strip()

    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()
