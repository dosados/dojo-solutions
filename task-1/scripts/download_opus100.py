#!/usr/bin/env python3
"""Скачать Helsinki-NLP/opus-100 (en-ru) в task-1/data/.

Запуск из корня task-1:

    conda run -n ml-env python scripts/download_opus100.py

Повторный запуск пропускает загрузку, если каталог уже есть (--force — перекачать).
Дальше: scripts/prepare_data.py читает с Hub напрямую; для офлайн-работы:

    from datasets import load_from_disk
    ds = load_from_disk("data/opus-100-en-ru")
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

_TASK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = _TASK_ROOT / "data" / "opus-100-en-ru"

REPO_ID = "Helsinki-NLP/opus-100"
CONFIG = "en-ru"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download OPUS-100 en-ru to data/opus-100-en-ru/"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help="Куда сохранить DatasetDict (save_to_disk)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Удалить существующий каталог и скачать заново",
    )
    args = parser.parse_args()
    out: Path = args.out_dir

    if out.exists():
        if not args.force:
            print(f"Already exists: {out}  (use --force to re-download)")
            return
        shutil.rmtree(out)

    from datasets import load_dataset

    print(f"Loading {REPO_ID} [{CONFIG}] from Hugging Face Hub...")
    ds = load_dataset(REPO_ID, CONFIG)

    out.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out))

    meta = {
        "repo_id": REPO_ID,
        "config": CONFIG,
        "path": str(out.relative_to(_TASK_ROOT)),
        "splits": {name: ds[name].num_rows for name in ds},
        "columns": ds["train"].column_names,
    }
    meta_path = out.parent / "opus-100-en-ru_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved to: {out}")
    for name, n in meta["splits"].items():
        print(f"  {name}: {n:,} rows")
    print(f"Meta: {meta_path}")


if __name__ == "__main__":
    main()
