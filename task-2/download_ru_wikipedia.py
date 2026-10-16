"""Скачивает misterkirill/ru-wikipedia в task-2/data/ru-wikipedia.jsonl.

Запуск из task-2/:

    conda activate ml-env
    python -u download_ru_wikipedia.py

Файл ~26 MB, ~115k строк — используется в scripts/prepare_mono.py для stage-1 CLM.
"""
from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset


REPO_ID = "misterkirill/ru-wikipedia"
OUT_PATH = Path(__file__).resolve().parent / "data" / "ru-wikipedia.jsonl"


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"loading {REPO_ID} …", flush=True)
    ds = load_dataset(REPO_ID, split="train")

    n = 0
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in ds:
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            f.write(json.dumps({"text": text, "lang": "ru"}, ensure_ascii=False) + "\n")
            n += 1
            if n % 20000 == 0:
                print(f"  written {n} …", flush=True)

    size_mb = OUT_PATH.stat().st_size / 2**20
    print(f"done: {OUT_PATH} ({n} rows, {size_mb:.1f} MB)", flush=True)
    print(
        "next:\n"
        "  python -u scripts/prepare_mono.py --ru-wikipedia --parallel-ru",
        flush=True,
    )


if __name__ == "__main__":
    main()
