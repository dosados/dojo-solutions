"""Коды языков NLLB-200 (FLORES-200) для task-2 RU→Abkhaz."""

from __future__ import annotations

from transformers import PreTrainedTokenizerBase

# Монокорпус: lang в jsonl → NLLB code
MONO_LANG_TO_NLLB: dict[str, str] = {
    "ru": "rus_Cyrl",
    "ab": "abk_Cyrl",
}

SRC_LANG = "rus_Cyrl"
TGT_LANG = "abk_Cyrl"


def set_src_lang(tokenizer: PreTrainedTokenizerBase) -> None:
    tokenizer.src_lang = SRC_LANG
    if hasattr(tokenizer, "set_src_lang_special_tokens"):
        tokenizer.set_src_lang_special_tokens(SRC_LANG)


def set_tgt_lang(tokenizer: PreTrainedTokenizerBase) -> None:
    tokenizer.tgt_lang = TGT_LANG
    if hasattr(tokenizer, "set_tgt_lang_special_tokens"):
        tokenizer.set_tgt_lang_special_tokens(TGT_LANG)


def prepare_nllb_tokenizer(tokenizer: PreTrainedTokenizerBase) -> int:
    """src + tgt lang tokens; returns forced_bos_token_id для decoder."""
    set_src_lang(tokenizer)
    set_tgt_lang(tokenizer)
    return forced_bos_token_id(tokenizer)


def nllb_generate_kwargs(
    forced_bos: int,
    *,
    max_new_tokens: int = 128,
    num_beams: int = 4,
) -> dict:
    """Декодирование: beam + min_length снижают пустой вывод и «уход» в eng."""
    return {
        "forced_bos_token_id": forced_bos,
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "do_sample": False,
        "min_length": 2,
        "no_repeat_ngram_size": 3,
        "early_stopping": True,
    }


def forced_bos_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    """ID языка-мишени для decoder BOS (transformers 5.x: convert_tokens_to_ids)."""
    if hasattr(tokenizer, "lang_code_to_id"):
        lid = tokenizer.lang_code_to_id.get(TGT_LANG)  # type: ignore[attr-defined]
        if lid is not None:
            return int(lid)
    tid = tokenizer.convert_tokens_to_ids(TGT_LANG)
    # У NLLB id языка abk_Cyrl может совпадать с unk_token_id (3) — это нормально.
    if tid is None or (isinstance(tid, list) and not tid):
        raise RuntimeError(f"cannot resolve forced_bos for {TGT_LANG!r}")
    return int(tid[0] if isinstance(tid, list) else tid)
