#!/usr/bin/env python3
"""Собрать jsonl для stage-1 CLM: смесь абхазского и русского моноязычного текста.

Выход: data/processed/mono/train.jsonl  — поля text, lang (ru|ab)

Источники абхазского:
  --ab-file data/ab-mono.txt  (конкурсный корпус, по строкам)

Источники русского (один или несколько):
  --ru-wikipedia              — data/ru-wikipedia.jsonl (скачать: download_ru_wikipedia.py)
  --parallel-ru               — уникальные RU из ab-ru-parallel.csv (in-domain)
  --ru-file PATH              — свой .txt (по строкам/абзацам)

Запуск из task-2/:

    conda activate ml-env
    python -u download_ru_wikipedia.py
    python -u scripts/prepare_mono.py --ru-wikipedia --parallel-ru
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

_TASK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AB = _TASK_ROOT / "data" / "ab-mono.txt"
DEFAULT_PARALLEL = _TASK_ROOT / "data" / "ab-ru-parallel.csv"
DEFAULT_RU_WIKI = _TASK_ROOT / "data" / "ru-wikipedia.jsonl"
DEFAULT_OUT = _TASK_ROOT / "data" / "processed" / "mono" / "train.jsonl"


def _char_ok(n: int, *, min_chars: int, max_chars: int) -> bool:
    if n < min_chars:
        return False
    return max_chars <= 0 or n <= max_chars


def _iter_lines(path: Path, *, min_chars: int, max_chars: int):
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            if _char_ok(len(text), min_chars=min_chars, max_chars=max_chars):
                yield text


def _iter_paragraphs(path: Path, *, min_chars: int, max_chars: int):
    buf: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                if buf:
                    text = " ".join(buf).strip()
                    buf.clear()
                    if _char_ok(len(text), min_chars=min_chars, max_chars=max_chars):
                        yield text
                continue
            buf.append(s)
    if buf:
        text = " ".join(buf).strip()
        if _char_ok(len(text), min_chars=min_chars, max_chars=max_chars):
            yield text


def collect_ab(path: Path, args: argparse.Namespace) -> list[dict]:
    rows = [{"text": t, "lang": "ab"} for t in _iter_lines(path, min_chars=args.min_chars, max_chars=args.max_chars)]
    print(f"  ab from {path.name}: {len(rows)} lines", flush=True)
    return rows


def collect_parallel_ru(path: Path, args: argparse.Namespace) -> list[dict]:
    seen: set[str] = set()
    rows: list[dict] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            ru = row["ru"].strip()
            if not ru or ru in seen:
                continue
            if not _char_ok(len(ru), min_chars=args.min_chars, max_chars=args.max_chars):
                continue
            seen.add(ru)
            rows.append({"text": ru, "lang": "ru"})
    print(f"  ru from parallel csv: {len(rows)} unique sentences", flush=True)
    return rows


def collect_ru_file(path: Path, args: argparse.Namespace) -> list[dict]:
    if path.suffix == ".txt":
        texts = list(_iter_lines(path, min_chars=args.min_chars, max_chars=args.max_chars))
    else:
        texts = list(_iter_paragraphs(path, min_chars=args.min_chars, max_chars=args.max_chars))
    rows = [{"text": t, "lang": "ru"} for t in texts]
    print(f"  ru from {path.name}: {len(rows)} segments", flush=True)
    return rows


def collect_ru_wikipedia(path: Path, args: argparse.Namespace) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            if not _char_ok(len(text), min_chars=args.min_chars, max_chars=args.max_chars):
                continue
            rows.append({"text": text, "lang": "ru"})
            if args.wiki_samples is not None and len(rows) >= args.wiki_samples:
                break
    print(f"  ru from {path.name}: {len(rows)} chunks", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare mono CLM jsonl for task-2")
    parser.add_argument("--ab-file", type=Path, default=DEFAULT_AB)
    parser.add_argument("--parallel-csv", type=Path, default=DEFAULT_PARALLEL)
    parser.add_argument("--ru-wiki-file", type=Path, default=DEFAULT_RU_WIKI)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-chars", type=int, default=20)
    parser.add_argument(
        "--max-chars",
        type=int,
        default=4096,
        help="Макс. длина сегмента в символах (0 = без верхнего лимита)",
    )
    parser.add_argument("--max-ab", type=int, default=None, help="Лимит абхазских строк")
    parser.add_argument("--max-ru", type=int, default=None, help="Лимит русских строк (суммарно)")
    parser.add_argument(
        "--ru-wikipedia",
        action="store_true",
        help="RU из data/ru-wikipedia.jsonl (misterkirill/ru-wikipedia)",
    )
    parser.add_argument(
        "--wiki-samples",
        type=int,
        default=None,
        help="Макс. строк из ru-wikipedia (по умолчанию — все после фильтра)",
    )
    parser.add_argument("--parallel-ru", action="store_true", help="RU из параллельного csv")
    parser.add_argument("--ru-file", type=Path, default=None, help="Свой русский монокорпус")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.ru_wikipedia and not args.parallel_ru and args.ru_file is None:
        print(
            "error: укажите хотя бы один источник RU: "
            "--ru-wikipedia, --parallel-ru и/или --ru-file",
            file=sys.stderr,
        )
        sys.exit(1)

    all_rows: list[dict] = []

    if args.ab_file.is_file():
        ab_rows = collect_ab(args.ab_file, args)
        if args.max_ab is not None and len(ab_rows) > args.max_ab:
            rng = random.Random(args.seed)
            ab_rows = rng.sample(ab_rows, args.max_ab)
        all_rows.extend(ab_rows)
    else:
        print(f"warning: missing {args.ab_file}", flush=True)

    ru_rows: list[dict] = []
    if args.ru_wikipedia:
        if not args.ru_wiki_file.is_file():
            print(
                f"error: missing {args.ru_wiki_file}\n"
                "  Run: python -u download_ru_wikipedia.py",
                file=sys.stderr,
            )
            sys.exit(1)
        ru_rows.extend(collect_ru_wikipedia(args.ru_wiki_file, args))
    if args.parallel_ru and args.parallel_csv.is_file():
        ru_rows.extend(collect_parallel_ru(args.parallel_csv, args))
    elif args.parallel_ru:
        print(f"warning: missing {args.parallel_csv}", flush=True)
    if args.ru_file is not None:
        if not args.ru_file.is_file():
            print(f"error: missing {args.ru_file}", file=sys.stderr)
            sys.exit(1)
        ru_rows.extend(collect_ru_file(args.ru_file, args))

    if args.max_ru is not None and len(ru_rows) > args.max_ru:
        rng = random.Random(args.seed + 1)
        ru_rows = rng.sample(ru_rows, args.max_ru)

    all_rows.extend(ru_rows)
    if not all_rows:
        print("error: no rows collected", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)
    rng.shuffle(all_rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_ab = sum(1 for r in all_rows if r["lang"] == "ab")
    n_ru = sum(1 for r in all_rows if r["lang"] == "ru")
    print(f"wrote {args.out}: {len(all_rows)} total (ab={n_ab}, ru={n_ru})", flush=True)


if __name__ == "__main__":
    main()
