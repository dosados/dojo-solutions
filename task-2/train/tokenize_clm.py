"""Stage 1: CLM — next-token prediction на смеси ru/ab моноязычного текста."""

from __future__ import annotations

from typing import Literal

ModelType = Literal["gemma", "translate"]


def _encode_gemma(processor, text: str) -> list[int]:
    batch = processor(text=text, return_tensors="pt", truncation=False)
    return batch["input_ids"][0].tolist()


def _encode_translate(processor, text: str) -> tuple[list[int], list[int]]:
    batch = processor(
        text=text,
        return_tensors="pt",
        truncation=False,
        return_mm_token_type_ids=True,
    )
    return (
        batch["input_ids"][0].tolist(),
        batch["token_type_ids"][0].tolist(),
    )


def tokenize_clm_row(
    example: dict[str, str],
    processor,
    *,
    model_type: ModelType,
    max_length: int,
) -> dict[str, list]:
    text = example["text"].strip()

    if model_type == "translate":
        input_ids, token_type_ids = _encode_translate(processor, text)
    else:
        input_ids = _encode_gemma(processor, text)
        token_type_ids = None

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        if token_type_ids is not None:
            token_type_ids = token_type_ids[:max_length]

    out: dict[str, list] = {
        "input_ids": input_ids,
        "completion_mask": [1] * len(input_ids),
    }
    if token_type_ids is not None:
        out["token_type_ids"] = token_type_ids
    return out


def tokenize_clm_batch(
    batch: dict[str, list],
    processor,
    *,
    model_type: ModelType,
    max_length: int,
) -> dict[str, list]:
    """Батч-токенизация для datasets.map(batched=True).

    В CLM нет chat-template — можно эффективно токенизировать списком строк.
    """
    texts = [str(t).strip() for t in batch["text"]]
    if model_type == "translate":
        enc = processor(
            text=texts,
            return_tensors=None,
            truncation=False,
            return_mm_token_type_ids=True,
        )
        input_ids = enc["input_ids"]
        token_type_ids = enc["token_type_ids"]
    else:
        enc = processor(text=texts, return_tensors=None, truncation=False)
        input_ids = enc["input_ids"]
        token_type_ids = None

    out_input_ids: list[list[int]] = []
    out_completion_mask: list[list[int]] = []
    out_tti: list[list[int]] | None = [] if token_type_ids is not None else None

    for i, ids in enumerate(input_ids):
        if len(ids) > max_length:
            ids = ids[:max_length]
            if token_type_ids is not None:
                token_type_ids[i] = token_type_ids[i][:max_length]
        out_input_ids.append(ids)
        out_completion_mask.append([1] * len(ids))
        if out_tti is not None:
            out_tti.append(token_type_ids[i])

    out: dict[str, list] = {
        "input_ids": out_input_ids,
        "completion_mask": out_completion_mask,
    }
    if out_tti is not None:
        out["token_type_ids"] = out_tti
    return out
