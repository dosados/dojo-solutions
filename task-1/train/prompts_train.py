"""Короткий промпт для SFT — без few-shot (они только в inference)."""

TRAIN_USER_TEMPLATE = (
    "Переведи с английского на русский. В ответе только русский перевод, без пояснений.\n\n"
    "Английский:\n{src}\n\n"
    "Русский:"
)


def build_train_user_content(src: str) -> str:
    return TRAIN_USER_TEMPLATE.format(src=src.strip())
