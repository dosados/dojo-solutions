#!/usr/bin/env python3
"""Привести self_data и локальный opus-100 к форматам solution.py и eval_bleu.

Выход: task-1/data/processed/

  self/
    all.jsonl              # rid, src, ref, article, para_idx
    all.input.pickle         # для solution.py
    dev.jsonl                # статьи 1–4 (без heldout)
    dev.input.pickle
    dev_it.jsonl             # dev без article-4 (Пушкин)
    dev_it.input.pickle
    heldout.jsonl            # одна статья (по умолчанию article-5)
    heldout.input.pickle

  opus100/
    train.jsonl              # HF train, ~1M предложений
    validation.jsonl         # HF validation, 2000
    test.jsonl               # HF test, 2000
    validation.input.pickle
    test.input.pickle

Запуск из task-1/:

    conda run -n ml-env python scripts/convert_datasets.py
    conda run -n ml-env python scripts/convert_datasets.py --skip-opus-train
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_TASK_ROOT = Path(__file__).resolve().parents[1]
if str(_TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(_TASK_ROOT))

from eval.articles import DEV_IT_EXCLUDE  # noqa: E402
from eval.formats import jsonl_to_pickle, parse_paragraph_blocks, write_jsonl  # noqa: E402

DEFAULT_SELF = _TASK_ROOT / "data" / "self_data"
DEFAULT_OPUS = _TASK_ROOT / "data" / "opus-100-en-ru"
DEFAULT_OUT = _TASK_ROOT / "data" / "processed"

ARTICLE_RE = re.compile(r"article-(\d+)")


def _article_id(path: Path) -> str:
    m = ARTICLE_RE.search(path.stem)
    if not m:
        raise ValueError(f"cannot parse article id from {path.name}")
    return f"article-{m.group(1)}"


def convert_self_data(
    self_root: Path,
    out_dir: Path,
    *,
    heldout_article: str = "article-5",
) -> dict:
    en_dir = self_root / "en"
    if not en_dir.is_dir():
        raise FileNotFoundError(f"missing {en_dir}")

    all_rows: list[dict] = []
    rid = 0

    for en_path in sorted(en_dir.glob("article-*-en.txt")):
        article = _article_id(en_path)
        ru_path = self_root / "ru" / en_path.name.replace("-en.txt", "-ru.txt")
        if not ru_path.is_file():
            raise FileNotFoundError(f"missing pair for {en_path.name}: {ru_path}")

        en_paras = parse_paragraph_blocks(en_path)
        ru_paras = parse_paragraph_blocks(ru_path)
        if len(en_paras) != len(ru_paras):
            raise ValueError(
                f"{article}: {len(en_paras)} EN vs {len(ru_paras)} RU paragraphs"
            )

        for para_idx, (src, ref) in enumerate(zip(en_paras, ru_paras)):
            all_rows.append(
                {
                    "rid": rid,
                    "src": src,
                    "ref": ref,
                    "article": article,
                    "para_idx": para_idx,
                }
            )
            rid += 1

    self_out = out_dir / "self"
    n_all = write_jsonl(self_out / "all.jsonl", all_rows)
    jsonl_to_pickle(self_out / "all.jsonl", self_out / "all.input.pickle")

    dev_rows = [r for r in all_rows if r["article"] != heldout_article]
    heldout_rows = [r for r in all_rows if r["article"] == heldout_article]

    dev_it_rows = [r for r in dev_rows if r["article"] not in DEV_IT_EXCLUDE]

    for rows, name in (
        (dev_rows, "dev"),
        (dev_it_rows, "dev_it"),
        (heldout_rows, "heldout"),
    ):
        renumbered = []
        for new_rid, row in enumerate(rows):
            renumbered.append({**row, "rid": new_rid})
        write_jsonl(self_out / f"{name}.jsonl", renumbered)
        jsonl_to_pickle(
            self_out / f"{name}.jsonl",
            self_out / f"{name}.input.pickle",
        )

    return {
        "n_all": n_all,
        "n_dev": len(dev_rows),
        "n_dev_it": len(dev_it_rows),
        "dev_it_excluded": list(DEV_IT_EXCLUDE),
        "n_heldout": len(heldout_rows),
        "heldout_article": heldout_article,
        "articles": sorted({r["article"] for r in all_rows}),
    }


def _opus_rows_from_split(ds_split, *, start_rid: int = 0) -> list[dict]:
    rows: list[dict] = []
    for i, ex in enumerate(ds_split):
        rows.append(
            {
                "rid": start_rid + i,
                "src": ex["translation"]["en"].strip(),
                "ref": ex["translation"]["ru"].strip(),
            }
        )
    return rows


def convert_opus100(
    opus_disk: Path,
    out_dir: Path,
    *,
    skip_train: bool = False,
) -> dict:
    from datasets import load_from_disk

    if not opus_disk.is_dir():
        raise FileNotFoundError(
            f"missing {opus_disk}; run: python scripts/download_opus100.py"
        )

    ds = load_from_disk(str(opus_disk))
    opus_out = out_dir / "opus100"
    stats: dict[str, int | str] = {}

    for split_name in ("validation", "test", "train"):
        if split_name == "train" and skip_train:
            stats["train_skipped"] = True
            continue
        if split_name not in ds:
            continue

        print(f"  opus100/{split_name} …")
        rows = _opus_rows_from_split(ds[split_name])
        path = opus_out / f"{split_name}.jsonl"
        n = write_jsonl(path, rows)
        stats[f"n_{split_name}"] = n

        if split_name in ("validation", "test"):
            jsonl_to_pickle(path, opus_out / f"{split_name}.input.pickle")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert self_data + opus-100 to program formats")
    parser.add_argument("--self-dir", type=Path, default=DEFAULT_SELF)
    parser.add_argument("--opus-dir", type=Path, default=DEFAULT_OPUS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--heldout-article",
        default="article-5",
        help="Статья только для финального val (остальные → dev)",
    )
    parser.add_argument("--skip-self", action="store_true")
    parser.add_argument("--skip-opus", action="store_true")
    parser.add_argument(
        "--skip-opus-train",
        action="store_true",
        help="Не писать train.jsonl (~1M строк, долго)",
    )
    args = parser.parse_args()

    manifest: dict = {"out_dir": str(args.out_dir.relative_to(_TASK_ROOT))}

    if not args.skip_self:
        print("Converting self_data …")
        manifest["self"] = convert_self_data(
            args.self_dir,
            args.out_dir,
            heldout_article=args.heldout_article,
        )

    if not args.skip_opus:
        print("Converting opus-100 …")
        manifest["opus100"] = convert_opus100(
            args.opus_dir,
            args.out_dir,
            skip_train=args.skip_opus_train,
        )

    manifest_path = args.out_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nDone. Manifest: {manifest_path}")
    print("\nВалидация (dev без Пушкина, article-4):")
    print("  cd submission && python -u solution.py \\")
    print("    --input ../data/processed/self/dev_it.input.pickle \\")
    print("    --output ../out/dev_it_output.json --progress")
    print("  python scripts/eval_bleu.py \\")
    print("    --hyp out/dev_it_output.json \\")
    print("    --ref data/processed/self/dev_it.jsonl --worst 10 --show-text")
    print("\nФинальный held-out (статья 5):")
    print("  … --ref data/processed/self/heldout.jsonl")


if __name__ == "__main__":
    main()
