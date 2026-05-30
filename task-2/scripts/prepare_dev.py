#!/usr/bin/env python3
"""Собрать dev-набор для локальной оценки task-2 из параллельного корпуса.

Выход (по умолчанию task-2/data/processed/):

  dev.jsonl           # rid, src (RU), ref (AB)
  dev.input.pickle    # rid, src — для solution.py

Запуск из task-2/:

    conda activate ml-env
    python -u scripts/prepare_dev.py
    python -u scripts/prepare_dev.py --n 100 --max-len 120
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import sys
from pathlib import Path

_TASK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = _TASK_ROOT / "data" / "ab-ru-parallel.csv"
DEFAULT_OUT = _TASK_ROOT / "data" / "processed"


def _sent_count(text: str) -> int:
    return max(1, text.count(".") + text.count("!") + text.count("?"))


def load_parallel_rows(
    csv_path: Path,
    *,
    min_len: int,
    max_len: int,
    max_sents: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            ru = row["ru"].strip()
            ab = row["ab"].strip()
            if not ru or not ab:
                continue
            if ru == ab:
                continue
            n = len(ru)
            if n < min_len or n > max_len:
                continue
            if _sent_count(ru) > max_sents:
                continue
            rows.append({"ru": ru, "ab": ab})
    return rows


def write_dev(rows: list[dict[str, str]], out_dir: Path) -> dict[str, int | Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "dev.jsonl"
    pickle_path = out_dir / "dev.input.pickle"

    pickle_rows: list[dict] = []
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rid, ex in enumerate(rows):
            rec = {"rid": rid, "src": ex["ru"], "ref": ex["ab"]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            pickle_rows.append({"rid": rid, "src": ex["ru"]})

    with pickle_path.open("wb") as f:
        pickle.dump(pickle_rows, f)

    return {
        "n": len(rows),
        "jsonl": jsonl_path,
        "pickle": pickle_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare task-2 dev set from parallel CSV")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n", type=int, default=500, help="Размер dev-выборки")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-len", type=int, default=5)
    parser.add_argument(
        "--max-len",
        type=int,
        default=120,
        help="Макс. длина RU (тест ~78 символов, <=120 — «как тест»)",
    )
    parser.add_argument("--max-sents", type=int, default=3)
    args = parser.parse_args()

    if not args.csv.is_file():
        print(f"error: missing {args.csv}", file=sys.stderr)
        sys.exit(1)

    pool = load_parallel_rows(
        args.csv,
        min_len=args.min_len,
        max_len=args.max_len,
        max_sents=args.max_sents,
    )
    if not pool:
        print("error: no rows matched filters", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)
    sample = pool if args.n >= len(pool) else rng.sample(pool, args.n)
    info = write_dev(sample, args.out_dir)

    print(f"pool after filters: {len(pool)}")
    print(f"dev sample: {info['n']} (seed={args.seed})")
    print(f"  {info['jsonl']}")
    print(f"  {info['pickle']}")


if __name__ == "__main__":
    main()
