"""Бейзлайн-решение: gemma-4-E2B-it

На входе:  /workspace/input.pickle
На выходе: /workspace/output.json
Веса:      /workspace/weights (предварительно скачиваются download_weights.py)
"""
import json
import os
import pickle

from transformers import AutoProcessor, AutoModelForCausalLM


MODEL_DIR = "./weights"
MAX_NEW_TOKENS = 1024
MAX_MODEL_LEN = 4096


GENERAL_PROMPT = """
Ты профессиональный переводчик. Твоя задача перевести текст с русского на абхазский язык.
ВАЖНО чтобы в ответе был только перевод без каких-либо комментариев и уточнений.

Текст для перевода:
"""


def main() -> None:
    with open("input.pickle", "rb") as f:
        rows = pickle.load(f)

    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype="auto",
        device_map="auto"
    )

    # batch-size 1 for simplicity
    results = []
    for row in rows:
        text = processor.apply_chat_template(
            [{"role": "user", "content": GENERAL_PROMPT + row["src"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        inputs = processor(text=text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[-1]

        # Generate output
        outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

        # Parse output
        results.append({
            'rid': row['rid'],
            'translation': processor.parse_response(response)['content'],
        })
        inputs.to('cpu')

    with open("output.json", "w") as f:
        json.dump(results, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
