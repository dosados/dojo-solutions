"""Лучший промпт после 16 итераций (=#06/#15). История: task-1/docs/PROMPT_TUNING.md"""


def build_prompt(src: str) -> str:
    return (
        "Translate the following English sentence into Russian.\n"
        "Rules:\n"
        "- output only the translation, no comments;\n"
        "- translate exactly this sentence, do not add or omit content;\n"
        "- keep names, numbers, and quotes;\n"
        "- match pronoun gender/number to the subject.\n\n"
        f"English:\n{src}\n\nRussian:"
    )
