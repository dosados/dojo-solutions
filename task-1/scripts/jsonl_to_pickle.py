#!/usr/bin/env python3
"""Конвертировать JSONL (rid, src) в input.pickle для solution.py."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_TASK_ROOT = Path(__file__).resolve().parents[1]
if str(_TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(_TASK_ROOT))

from eval.formats import jsonl_to_pickle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path, help="JSONL с полями rid, src")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("input.pickle"),
    )
    args = parser.parse_args()

    n = jsonl_to_pickle(args.jsonl, args.output)
    print(f"Wrote {n} rows to {args.output}")


if __name__ == "__main__":
    main()
