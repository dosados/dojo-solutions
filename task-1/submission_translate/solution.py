"""TranslateGemma-4B-it (EN→RU) — batched inference for contest.

На входе:  /workspace/input.pickle
На выходе: /workspace/output.json
Веса:      /workspace/weights (скачать: ../download_weights_translate.py)
"""
import json
import pickle

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_DIR = "./weights"
BATCH_SIZE = 4
SOURCE_LANG = "en"
TARGET_LANG = "ru"
MAX_NEW_TOKENS_CAP = 1024


def estimate_max_new_tokens(src: str) -> int:
    """Бюджет генерации от длины абзаца (тест до ~2229 символов EN)."""
    n = len(src)
    if n < 80:
        return 256
    if n < 300:
        return 384
    budget = int(n * 0.45) + 256
    return max(384, min(MAX_NEW_TOKENS_CAP, budget))


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
    input_lens = []

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
        input_lens.append(int(ids.shape[0]))

    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"
    padded = tokenizer.pad(
        {"input_ids": input_ids_list, "attention_mask": attention_mask_list},
        padding=True,
        return_tensors="pt",
    )
    return padded.to(device), input_lens


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
    try:
        model.forward = torch.compile(model.forward, mode="default", fullgraph=False)
    except Exception:
        pass
    device = _input_device(model)
    pad_token_id = _translate_pad_token_id(processor, model)

    ordered = sorted(rows, key=lambda r: len(r["src"]))
    results: list[dict] = []

    for start in range(0, len(ordered), BATCH_SIZE):
        batch = ordered[start : start + BATCH_SIZE]
        max_new_tokens = max(estimate_max_new_tokens(r["src"]) for r in batch)
        inputs, input_lens = _prepare_batch(processor, batch, device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )

        for row, in_len, out_ids in zip(batch, input_lens, outputs, strict=True):
            translation = processor.decode(
                out_ids[in_len:], skip_special_tokens=True
            ).strip()
            results.append({"rid": row["rid"], "translation": translation})

    results.sort(key=lambda x: x["rid"])

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
