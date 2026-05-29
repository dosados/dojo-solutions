"""Загрузка и сэмплирование JSONL для LoRA."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def _length_ok(src: str, ref: str, *, min_chars: int, max_chars: int) -> bool:
    n = len(src)
    if n < min_chars or n > max_chars:
        return False
    if not src.strip() or not ref.strip():
        return False
    return True


def read_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def reservoir_sample_jsonl(
    path: Path,
    k: int,
    *,
    seed: int,
    min_chars: int = 10,
    max_chars: int = 800,
) -> list[dict[str, str]]:
    """Случайные k строк из большого JSONL без загрузки всего файла в RAM."""
    rng = random.Random(seed)
    reservoir: list[dict[str, str]] = []
    seen = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            src = str(row["src"]).strip()
            ref = str(row.get("ref") or row.get("tgt") or "").strip()
            if not _length_ok(src, ref, min_chars=min_chars, max_chars=max_chars):
                continue
            item = {"src": src, "ref": ref}
            seen += 1
            if len(reservoir) < k:
                reservoir.append(item)
            else:
                j = rng.randint(0, seen - 1)
                if j < k:
                    reservoir[j] = item
    if len(reservoir) < k:
        print(
            f"  warning: only {len(reservoir)} rows passed filters (asked {k})",
            flush=True,
        )
    return reservoir


def load_train_pool(
    train_path: Path,
    *,
    max_samples: int | None,
    seed: int,
    min_chars: int,
    max_chars: int,
    self_mix_path: Path | None = None,
    self_mix_max: int = 0,
) -> list[dict[str, str]]:
    if max_samples is None:
        rows = read_jsonl_rows(train_path)
        pool = []
        for row in rows:
            src = str(row["src"]).strip()
            ref = str(row.get("ref") or "").strip()
            if _length_ok(src, ref, min_chars=min_chars, max_chars=max_chars):
                pool.append({"src": src, "ref": ref})
        rng = random.Random(seed)
        rng.shuffle(pool)
        print(f"  train: {len(pool)} rows (full file, filtered)", flush=True)
    else:
        pool = reservoir_sample_jsonl(
            train_path,
            max_samples,
            seed=seed,
            min_chars=min_chars,
            max_chars=max_chars,
        )
        print(f"  train: {len(pool)} rows (sampled {max_samples})", flush=True)

    if self_mix_path and self_mix_max > 0 and self_mix_path.is_file():
        self_rows = read_jsonl_rows(self_mix_path)
        extra: list[dict[str, str]] = []
        for row in self_rows:
            src = str(row["src"]).strip()
            ref = str(row.get("ref") or "").strip()
            article = str(row.get("article") or "")
            if article.startswith("article-3"):
                continue
            if _length_ok(src, ref, min_chars=min_chars, max_chars=max_chars):
                extra.append({"src": src, "ref": ref})
        rng = random.Random(seed + 1)
        rng.shuffle(extra)
        extra = extra[:self_mix_max]
        pool.extend(extra)
        print(f"  self mix: +{len(extra)} from {self_mix_path.name}", flush=True)

    rng = random.Random(seed + 2)
    rng.shuffle(pool)
    return pool


def load_val_pool(
    val_path: Path,
    *,
    max_samples: int,
    seed: int,
    min_chars: int,
    max_chars: int,
) -> list[dict[str, str]]:
    if max_samples <= 0:
        return []
    pool = reservoir_sample_jsonl(
        val_path,
        max_samples,
        seed=seed,
        min_chars=min_chars,
        max_chars=max_chars,
    )
    print(f"  val: {len(pool)} rows", flush=True)
    return pool
