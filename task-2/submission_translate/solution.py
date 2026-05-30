"""Бейзлайн: TranslateGemma-4B-it (RU→AB)

На входе:  /workspace/input.pickle
На выходе: /workspace/output.json
Веса:      /workspace/weights (скачать: ../download_weights_translate.py)
"""
import json
import pickle

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


MODEL_DIR = "./weights"
MAX_NEW_TOKENS = 512
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


def main() -> None:
    with open("input.pickle", "rb") as f:
        rows = pickle.load(f)

    processor = _load_processor(MODEL_DIR)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_DIR,
        dtype="auto",
        device_map="auto",
    )
    pad_token_id = _translate_pad_token_id(processor, model)

    results = []
    for row in rows:
        inputs = processor.apply_chat_template(
            build_messages(row["src"]),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=pad_token_id,
            )

        translation = processor.decode(
            outputs[0][input_len:], skip_special_tokens=True
        ).strip()
        results.append({"rid": row["rid"], "translation": translation})

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
