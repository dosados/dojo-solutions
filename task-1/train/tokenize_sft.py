"""Токенизация train-примеров через chat template Gemma (как при инференсе)."""

from __future__ import annotations

from train.prompts_train import build_train_user_content


def _encode_text(processor, text: str) -> list[int]:
    batch = processor(text=text, return_tensors="pt")
    return batch["input_ids"][0].tolist()


def _prefix_len(input_ids: list[int], prompt_ids: list[int]) -> int:
    """Длина префикса prompt в full (с допуском сдвига спецтокенов)."""
    if input_ids[: len(prompt_ids)] == prompt_ids:
        return len(prompt_ids)
    max_shift = min(8, len(input_ids))
    for start in range(max_shift):
        end = start + len(prompt_ids)
        if end <= len(input_ids) and input_ids[start:end] == prompt_ids:
            return end
    n = 0
    for a, b in zip(input_ids, prompt_ids, strict=False):
        if a != b:
            break
        n += 1
    return n


def tokenize_translation_row(
    example: dict[str, str],
    processor,
    *,
    max_length: int,
) -> dict[str, list]:
    """Один проход chat template + маска loss только на ответ модели."""
    user = build_train_user_content(example["src"])
    ref = str(example["ref"]).strip()
    messages = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": ref},
    ]

    full_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    input_ids = _encode_text(processor, full_text)

    prompt_text = processor.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = _encode_text(processor, prompt_text)

    plen = _prefix_len(input_ids, prompt_ids)
    completion_mask = [0] * plen + [1] * (len(input_ids) - plen)

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        completion_mask = completion_mask[:max_length]

    return {
        "input_ids": input_ids,
        "completion_mask": completion_mask,
        "prefix_mismatch": int(plen != len(prompt_ids)),
    }
