"""TranslateGemma-4B-it (RU→AB) — batched inference for contest.

На входе:  /workspace/input.pickle
На выходе: /workspace/output.json
Веса:      /workspace/weights (скачать: ../download_weights_translate.py)
"""
import json
import pickle

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_DIR = "./weights"
MAX_NEW_TOKENS = 128
BATCH_SIZE = 40
NUM_BEAMS = 3
LENGTH_PENALTY = 0.8
SOURCE_LANG = "ru"
TARGET_LANG = "ab"


def _load_processor(model_dir: str):
    try:
        return AutoProcessor.from_pretrained(model_dir, fix_mistral_regex=True)
    except (TypeError, AttributeError):
        return AutoProcessor.from_pretrained(model_dir)


def _translate_pad_token_id(processor, model) -> int:
    tok = getattr(processor, "tokenizer", None)
    if tok is not None and tok.pad_token_id is not None:
        return int(tok.pad_token_id)
    gen = model.generation_config
    pid = gen.pad_token_id
    if pid is not None:
        return int(pid[0] if isinstance(pid, list) else pid)
    eos = gen.eos_token_id
    if eos is None:
        raise RuntimeError("cannot determine pad_token_id")
    return int(eos[0] if isinstance(eos, list) else eos)


def build_messages(text: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": SOURCE_LANG,
                    "target_lang_code": TARGET_LANG,
                    "text": text,
                }
            ],
        }
    ]


def _input_device(model) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _prepare_batch(processor, batch: list[dict], device: torch.device):
    input_ids_list = []
    attention_mask_list = []

    for row in batch:
        inputs = processor.apply_chat_template(
            build_messages(row["src"]),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        ids = inputs["input_ids"][0]
        mask = inputs.get("attention_mask")
        if mask is None:
            mask = torch.ones_like(ids)
        else:
            mask = mask[0]
        input_ids_list.append(ids)
        attention_mask_list.append(mask)

    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"
    padded = tokenizer.pad(
        {"input_ids": input_ids_list, "attention_mask": attention_mask_list},
        padding=True,
        return_tensors="pt",
    )
    return padded.to(device)


def main() -> None:
    with open("input.pickle", "rb") as f:
        rows = pickle.load(f)

    processor = _load_processor(MODEL_DIR)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_DIR,
        dtype="auto",
        device_map="auto",
        attn_implementation="sdpa",
    )
    device = _input_device(model)
    pad_token_id = _translate_pad_token_id(processor, model)

    ordered = sorted(rows, key=lambda r: len(r["src"]))
    results: list[dict] = []

    for start in range(0, len(ordered), BATCH_SIZE):
        batch = ordered[start : start + BATCH_SIZE]
        inputs = _prepare_batch(processor, batch, device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=NUM_BEAMS,
                length_penalty=LENGTH_PENALTY,
                early_stopping=True,
                pad_token_id=pad_token_id,
            )

        for row, out_ids in zip(batch, outputs, strict=True):
            translation = processor.decode(
                out_ids[input_len:], skip_special_tokens=True
            ).strip()
            results.append({"rid": row["rid"], "translation": translation})

    results.sort(key=lambda x: x["rid"])

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
