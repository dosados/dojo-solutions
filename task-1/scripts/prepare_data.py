#!/usr/bin/env python3
"""Скачать EN–RU данные, разбить train/val, опционально склеить псевдо-абзацы.

Рекомендуемый старт (большой train + отдельный val):
  conda run -n ml-env python scripts/prepare_data.py --dataset opus100 --val-size 3000
  conda run -n ml-env python scripts/prepare_data.py --dataset flores --split devtest

Быстрый sentence-level бенчмарк (не абзацы):
  conda run -n ml-env python scripts/prepare_data.py --dataset flores --split dev

Псевдо-абзацы (~5 предложений) для val, ближе к контесту:
  conda run -n ml-env python scripts/prepare_data.py --dataset opus100 \\
    --val-size 500 --paragraph-sentences 5 --paragraph-val-only
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_TASK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = _TASK_ROOT / "data"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  {path}  ({len(rows)} rows)")


def _merge_paragraphs(
    rows: list[dict],
    *,
    sentences_per_para: int,
    separator_en: str = " ",
    separator_ru: str = " ",
) -> list[dict]:
    """Склеить последовательные предложения в псевдо-абзацы."""
    out: list[dict] = []
    buf_src: list[str] = []
    buf_ref: list[str] = []
    rid = 0

    def flush() -> None:
        nonlocal rid
        if not buf_src:
            return
        out.append(
            {
                "rid": rid,
                "src": separator_en.join(buf_src),
                "ref": separator_ru.join(buf_ref),
                "n_sentences": len(buf_src),
            }
        )
        rid += 1
        buf_src.clear()
        buf_ref.clear()

    for row in rows:
        buf_src.append(row["src"])
        buf_ref.append(row["ref"])
        if len(buf_src) >= sentences_per_para:
            flush()
    flush()
    return out


def _load_opus100_en_ru() -> list[dict]:
    from datasets import load_dataset

    # ~1M параллельных предложений, несколько доменов
    ds = load_dataset("Helsinki-NLP/opus-100", "en-ru", split="train")
    rows: list[dict] = []
    for i, ex in enumerate(ds):
        rows.append(
            {
                "rid": i,
                "src": ex["translation"]["en"].strip(),
                "ref": ex["translation"]["ru"].strip(),
            }
        )
    return rows


def _load_flores(split: str) -> list[dict]:
    from datasets import load_dataset

    # eng_Latn ↔ rus_Cyrl; dev ~997, devtest ~1012 предложений
    ds = load_dataset("facebook/flores", "all", split=split)
    rows: list[dict] = []
    for i, ex in enumerate(ds):
        rows.append(
            {
                "rid": i,
                "src": ex["sentence_eng_Latn"].strip(),
                "ref": ex["sentence_rus_Cyrl"].strip(),
            }
        )
    return rows


def _split_train_val(
    rows: list[dict],
    val_size: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    if val_size <= 0:
        return rows, []
    if val_size >= len(rows):
        raise ValueError(f"val_size={val_size} >= dataset size {len(rows)}")

    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    val_idx = set(indices[:val_size])
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    for i, row in enumerate(rows):
        if i in val_idx:
            val_rows.append(row)
        else:
            train_rows.append(row)

    for new_rid, row in enumerate(train_rows):
        row = dict(row)
        row["rid"] = new_rid
        train_rows[new_rid] = row
    for new_rid, row in enumerate(val_rows):
        row = dict(row)
        row["rid"] = new_rid
        val_rows[new_rid] = row

    return train_rows, val_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare EN–RU train/val JSONL")
    parser.add_argument(
        "--dataset",
        choices=("opus100", "flores"),
        required=True,
        help="opus100: большой train; flores: фиксированный sentence-level val",
    )
    parser.add_argument(
        "--split",
        default="devtest",
        help="Только flores: dev | devtest",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help="Каталог вывода (gitignored): data/",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=3000,
        help="Для opus100: сколько предложений в val (остальное — train)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--paragraph-sentences",
        type=int,
        default=0,
        help="Если >0: склеить N предложений в один пример (псевдо-абзац)",
    )
    parser.add_argument(
        "--paragraph-val-only",
        action="store_true",
        help="Склейка только для val; train остаётся sentence-level",
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=0,
        help="Обрезать train до N предложений (0 = без обрезки)",
    )
    args = parser.parse_args()

    print(f"Loading {args.dataset}...")
    if args.dataset == "opus100":
        rows = _load_opus100_en_ru()
        train_rows, val_rows = _split_train_val(rows, args.val_size, args.seed)
        if args.max_train > 0:
            train_rows = train_rows[: args.max_train]
        prefix = "opus100"
    else:
        val_rows = _load_flores(args.split)
        train_rows = []
        prefix = f"flores_{args.split}"

    if args.paragraph_sentences > 0:
        n = args.paragraph_sentences
        if args.paragraph_val_only or not train_rows:
            val_rows = _merge_paragraphs(val_rows, sentences_per_para=n)
        if train_rows and not args.paragraph_val_only:
            train_rows = _merge_paragraphs(train_rows, sentences_per_para=n)
        elif train_rows and args.paragraph_val_only:
            print(
                f"  train: sentence-level ({len(train_rows)}); "
                f"val: pseudo-paragraphs ({len(val_rows)})"
            )

    out = args.out_dir
    meta = {
        "dataset": args.dataset,
        "split": args.split if args.dataset == "flores" else "train",
        "val_size": args.val_size,
        "paragraph_sentences": args.paragraph_sentences,
        "seed": args.seed,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
    }
    (out / f"{prefix}_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"Writing to {out}/")
    if train_rows:
        _write_jsonl(out / f"{prefix}_train.jsonl", train_rows)
    _write_jsonl(out / f"{prefix}_val.jsonl", val_rows)

    print("\nОценка перевода:")
    print(
        f"  conda run -n ml-env python scripts/eval_bleu.py "
        f"--hyp submission/output.json --ref {out / f'{prefix}_val.jsonl'}"
    )
    print("\nПикль для solution.py (только src):")
    print(f"  см. scripts/jsonl_to_pickle.py или вручную из val.jsonl")


if __name__ == "__main__":
    main()
