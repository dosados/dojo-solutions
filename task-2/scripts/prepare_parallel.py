#!/usr/bin/env python3
"""Собрать train/val jsonl для stage-2 SFT из параллельного корпуса.

Выход: data/processed/parallel/train.jsonl, val.jsonl  — поля src (ru), ref (ab)

Запуск из task-2/:

    conda activate ml-env
    python -u scripts/prepare_parallel.py
    python -u scripts/prepare_parallel.py --val-n 500
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_TASK_ROOT = Path(__file__).resolve().parents[1]
if str(_TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(_TASK_ROOT))

from train.data import load_parallel_pool  # noqa: E402

DEFAULT_CSV = _TASK_ROOT / "data" / "ab-ru-parallel.csv"
DEFAULT_OUT = _TASK_ROOT / "data" / "processed" / "parallel"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare parallel SFT jsonl for task-2")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--val-n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-len", type=int, default=5)
    parser.add_argument(
        "--max-len",
        type=int,
        default=4096,
        help="Макс. длина RU в символах (0 = без верхнего лимита)",
    )
    parser.add_argument(
        "--max-sents",
        type=int,
        default=20,
        help="Макс. число предложений в RU (0 = без лимита)",
    )
    args = parser.parse_args()

    if not args.csv.is_file():
        print(f"error: missing {args.csv}", file=sys.stderr)
        sys.exit(1)

    pool = load_parallel_pool(
        args.csv,
        max_samples=None,
        seed=args.seed,
        min_chars=args.min_len,
        max_chars=args.max_len,
        max_sents=args.max_sents,
    )
    rng = random.Random(args.seed)
    rng.shuffle(pool)

    val_n = min(args.val_n, len(pool) // 10 or 1)
    val_rows = pool[:val_n]
    train_rows = pool[val_n:]

    train_path = args.out_dir / "train.jsonl"
    val_path = args.out_dir / "val.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)

    print(f"pool after filters: {len(pool)}", flush=True)
    print(f"  train: {len(train_rows)} → {train_path}", flush=True)
    print(f"  val:   {len(val_rows)} → {val_path}", flush=True)


if __name__ == "__main__":
    main()
