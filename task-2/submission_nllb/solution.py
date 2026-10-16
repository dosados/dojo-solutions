"""NLLB-200-3.3B (RU→AB) — batched inference для контеста.

На входе:  /workspace/input.pickle
На выходе: /workspace/output.json
Веса:      /workspace/weights (скачать: ../download_weights_nllb.py)

Языки NLLB-200: rus_Cyrl → abk_Cyrl
"""
import json
import pickle

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_DIR = "./weights"
SRC_LANG = "rus_Cyrl"
TGT_LANG = "abk_Cyrl"
MAX_INPUT_LENGTH = 512
MAX_NEW_TOKENS = 128
BATCH_SIZE = 32
# beam=4: меньше пустых/английских ответов; beam=1 быстрее, но хуже на ab
NUM_BEAMS = 4
MIN_DECODE_LENGTH = 2


def _forced_bos_token_id(tokenizer) -> int:
    if hasattr(tokenizer, "lang_code_to_id"):
        lid = tokenizer.lang_code_to_id.get(TGT_LANG)
        if lid is not None:
            return int(lid)
    tid = tokenizer.convert_tokens_to_ids(TGT_LANG)
    if tid is None or (isinstance(tid, list) and not tid):
        raise RuntimeError(f"cannot resolve forced_bos for {TGT_LANG!r}")
    return int(tid[0] if isinstance(tid, list) else tid)


def _prepare_tokenizer(tokenizer) -> int:
    tokenizer.src_lang = SRC_LANG
    if hasattr(tokenizer, "set_src_lang_special_tokens"):
        tokenizer.set_src_lang_special_tokens(SRC_LANG)
    tokenizer.tgt_lang = TGT_LANG
    if hasattr(tokenizer, "set_tgt_lang_special_tokens"):
        tokenizer.set_tgt_lang_special_tokens(TGT_LANG)
    return _forced_bos_token_id(tokenizer)


def _input_device(model) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    with open("input.pickle", "rb") as f:
        rows = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    forced_bos = _prepare_tokenizer(tokenizer)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    device = _input_device(model)

    ordered = sorted(rows, key=lambda r: len(r["src"]))
    results: list[dict] = []

    for start in range(0, len(ordered), BATCH_SIZE):
        batch = ordered[start : start + BATCH_SIZE]
        texts = [r["src"] for r in batch]
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
        ).to(device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=NUM_BEAMS,
                do_sample=False,
                min_length=MIN_DECODE_LENGTH,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for row, translation in zip(batch, translations, strict=True):
            text = translation.strip()
            results.append({"rid": row["rid"], "translation": text})

    results.sort(key=lambda x: x["rid"])

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
