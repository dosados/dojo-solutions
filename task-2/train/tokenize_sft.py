"""Stage 2: SFT RU→AB — формат совпадает с inference (gemma / translate)."""

from __future__ import annotations

from typing import Literal

from train.prompts import build_gemma_user_content

ModelType = Literal["gemma", "translate"]
SOURCE_LANG = "ru"
TARGET_LANG = "ab"


def _encode_gemma(processor, text: str) -> list[int]:
    batch = processor(text=text, return_tensors="pt")
    return batch["input_ids"][0].tolist()


def _encode_translate_prompt(
    processor,
    src: str,
) -> tuple[list[int], list[int]]:
    """User-only turn; tokenize=True работает только без assistant message."""
    messages = _build_translate_messages(src, None)
    batch = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return (
        batch["input_ids"][0].tolist(),
        batch["token_type_ids"][0].tolist(),
    )


def _encode_translate_sft(
    processor,
    src: str,
    ref: str,
) -> tuple[list[int], list[int], int]:
    """Полная пара user+assistant: prompt через template+tokenize, ответ — suffix encode.

    apply_chat_template(..., tokenize=True) падает, если assistant content — строка
    (Gemma3Processor ожидает list content для multimodal path).
    """
    prompt_messages = _build_translate_messages(src, None)
    full_messages = _build_translate_messages(src, ref)

    prompt_ids, prompt_tts = _encode_translate_prompt(processor, src)
    prompt_text = processor.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = processor.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    if not full_text.startswith(prompt_text):
        raise ValueError("TranslateGemma chat template: prompt is not a prefix of full")

    suffix = full_text[len(prompt_text) :]
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    comp_ids = tok.encode(suffix, add_special_tokens=False)

    input_ids = prompt_ids + comp_ids
    token_type_ids = prompt_tts + [0] * len(comp_ids)
    return input_ids, token_type_ids, len(prompt_ids)


def _prefix_len(input_ids: list[int], prompt_ids: list[int]) -> int:
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


def _build_translate_messages(src: str, ref: str | None = None) -> list[dict]:
    user = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "source_lang_code": SOURCE_LANG,
                "target_lang_code": TARGET_LANG,
                "text": src.strip(),
            }
        ],
    }
    if ref is None:
        return [user]
    return [user, {"role": "assistant", "content": ref.strip()}]


def _apply_template(processor, messages: list[dict], *, add_generation_prompt: bool) -> str:
    kwargs: dict = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    try:
        return processor.apply_chat_template(messages, **kwargs, enable_thinking=False)
    except TypeError:
        return processor.apply_chat_template(messages, **kwargs)


def tokenize_translation_row(
    example: dict[str, str],
    processor,
    *,
    model_type: ModelType,
    max_length: int,
) -> dict[str, list]:
    src = str(example["src"]).strip()
    ref = str(example["ref"]).strip()
    token_type_ids: list[int] | None = None

    if model_type == "translate":
        input_ids, token_type_ids, plen = _encode_translate_sft(processor, src, ref)
        prefix_mismatch = 0
    else:
        user = build_gemma_user_content(src)
        full_messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": ref},
        ]
        prompt_messages = [{"role": "user", "content": user}]
        full_text = _apply_template(processor, full_messages, add_generation_prompt=False)
        prompt_text = _apply_template(processor, prompt_messages, add_generation_prompt=True)
        input_ids = _encode_gemma(processor, full_text)
        prompt_ids = _encode_gemma(processor, prompt_text)
        plen = _prefix_len(input_ids, prompt_ids)
        prefix_mismatch = int(plen != len(prompt_ids))

    completion_mask = [0] * plen + [1] * (len(input_ids) - plen)

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        completion_mask = completion_mask[:max_length]
        if token_type_ids is not None:
            token_type_ids = token_type_ids[:max_length]

    out: dict[str, list] = {
        "input_ids": input_ids,
        "completion_mask": completion_mask,
        "prefix_mismatch": prefix_mismatch,
    }
    if token_type_ids is not None:
        out["token_type_ids"] = token_type_ids
    return out


def tokenize_translation_batch(
    batch: dict[str, list],
    processor,
    *,
    model_type: ModelType,
    max_length: int,
) -> dict[str, list]:
    """Батч-токенизация для datasets.map(batched=True).

    Для TranslateGemma SFT (chat-template + suffix) полноценная векторизация сложнее,
    поэтому оставляем поэлементно.

    Для Gemma SFT можно ускорить основную часть токенизации, но из-за prefix-логики
    всё равно есть цикл; выгода меньше, чем у CLM.
    """
    srcs = [str(x).strip() for x in batch["src"]]
    refs = [str(x).strip() for x in batch["ref"]]

    if model_type == "translate":
        outs = [
            tokenize_translation_row(
                {"src": s, "ref": r},
                processor,
                model_type=model_type,
                max_length=max_length,
            )
            for s, r in zip(srcs, refs, strict=True)
        ]
        # transpose list[dict] → dict[list]
        keys = outs[0].keys() if outs else []
        return {k: [o[k] for o in outs] for k in keys}

    # gemma: частичная батч-токенизация через processor(text=[...])
    users = [build_gemma_user_content(s) for s in srcs]
    full_messages = [
        [{"role": "user", "content": u}, {"role": "assistant", "content": r}]
        for u, r in zip(users, refs, strict=True)
    ]
    prompt_messages = [[{"role": "user", "content": u}] for u in users]

    full_texts = [
        _apply_template(processor, m, add_generation_prompt=False) for m in full_messages
    ]
    prompt_texts = [
        _apply_template(processor, m, add_generation_prompt=True) for m in prompt_messages
    ]

    full_enc = processor(text=full_texts, return_tensors=None)
    prompt_enc = processor(text=prompt_texts, return_tensors=None)
    full_ids = full_enc["input_ids"]
    prompt_ids = prompt_enc["input_ids"]

    out_input_ids: list[list[int]] = []
    out_completion_mask: list[list[int]] = []
    out_prefix_mismatch: list[int] = []

    for ids, pids in zip(full_ids, prompt_ids, strict=True):
        plen = _prefix_len(ids, pids)
        completion_mask = [0] * plen + [1] * (len(ids) - plen)
        prefix_mismatch = int(plen != len(pids))

        if len(ids) > max_length:
            ids = ids[:max_length]
            completion_mask = completion_mask[:max_length]

        out_input_ids.append(ids)
        out_completion_mask.append(completion_mask)
        out_prefix_mismatch.append(prefix_mismatch)

    return {
        "input_ids": out_input_ids,
        "completion_mask": out_completion_mask,
        "prefix_mismatch": out_prefix_mismatch,
    }
