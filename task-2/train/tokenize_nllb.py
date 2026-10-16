"""Токенизация mono (stage clm) и parallel (stage sft) для NLLB-200."""

from __future__ import annotations

from transformers import PreTrainedTokenizerBase

from train.nllb_langs import MONO_LANG_TO_NLLB, SRC_LANG, TGT_LANG


def _set_langs(tokenizer: PreTrainedTokenizerBase, src: str, tgt: str) -> None:
    tokenizer.src_lang = src
    if hasattr(tokenizer, "set_src_lang_special_tokens"):
        tokenizer.set_src_lang_special_tokens(src)
    tokenizer.tgt_lang = tgt
    if hasattr(tokenizer, "set_tgt_lang_special_tokens"):
        tokenizer.set_tgt_lang_special_tokens(tgt)


def tokenize_nllb_mono_row(
    example: dict[str, str],
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int,
) -> dict[str, list]:
    """Same-language reconstruction (адаптация к ru/ab перед SFT)."""
    lang = str(example["lang"]).strip().lower()
    code = MONO_LANG_TO_NLLB.get(lang)
    if not code:
        raise ValueError(f"unknown mono lang: {lang!r}")

    text = str(example["text"]).strip()
    _set_langs(tokenizer, code, code)
    enc = tokenizer(
        text,
        text_target=text,
        max_length=max_length,
        truncation=True,
    )
    return {k: enc[k] for k in ("input_ids", "attention_mask", "labels")}


def tokenize_nllb_mono_batch(
    batch: dict[str, list],
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int,
) -> dict[str, list]:
    keys = ("input_ids", "attention_mask", "labels")
    out: dict[str, list] = {k: [] for k in keys}
    for text, lang in zip(batch["text"], batch["lang"], strict=True):
        row = tokenize_nllb_mono_row(
            {"text": str(text), "lang": str(lang)},
            tokenizer,
            max_length=max_length,
        )
        for k in keys:
            out[k].append(row[k])
    return out


def tokenize_nllb_sft_row(
    example: dict[str, str],
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int,
) -> dict[str, list]:
    src = str(example["src"]).strip()
    ref = str(example["ref"]).strip()
    _set_langs(tokenizer, SRC_LANG, TGT_LANG)
    enc = tokenizer(
        src,
        text_target=ref,
        max_length=max_length,
        truncation=True,
    )
    return {k: enc[k] for k in ("input_ids", "attention_mask", "labels")}


def tokenize_nllb_sft_batch(
    batch: dict[str, list],
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int,
) -> dict[str, list]:
    keys = ("input_ids", "attention_mask", "labels")
    out: dict[str, list] = {k: [] for k in keys}
    for src, ref in zip(batch["src"], batch["ref"], strict=True):
        row = tokenize_nllb_sft_row(
            {"src": str(src), "ref": str(ref)},
            tokenizer,
            max_length=max_length,
        )
        for k in keys:
            out[k].append(row[k])
    return out
