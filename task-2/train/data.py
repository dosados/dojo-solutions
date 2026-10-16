"""Загрузка данных для CLM (mono) и SFT (parallel)."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path


def _sent_count(text: str) -> int:
    return max(1, text.count(".") + text.count("!") + text.count("?"))


def load_jsonl_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_mono_pool(
    path: Path,
    *,
    max_samples: int | None,
    seed: int,
    lang_weights: dict[str, float] | None = None,
    min_chars: int = 20,
    max_chars: int = 4096,
) -> list[dict[str, str]]:
    """jsonl: {"text": str, "lang": "ru"|"ab"}."""
    pool = load_jsonl_rows(path)
    filtered: list[dict[str, str]] = []
    for row in pool:
        text = str(row.get("text", "")).strip()
        lang = str(row.get("lang", "")).strip().lower()
        if lang not in ("ru", "ab") or not text:
            continue
        n = len(text)
        if n < min_chars:
            continue
        if max_chars > 0 and n > max_chars:
            continue
        filtered.append({"text": text, "lang": lang})

    if not filtered:
        raise ValueError(f"no mono rows after filter: {path}")

    if lang_weights:
        by_lang: dict[str, list[dict]] = {"ru": [], "ab": []}
        for row in filtered:
            by_lang[row["lang"]].append(row)
        rng = random.Random(seed)
        total_w = sum(lang_weights.get(k, 0) for k in ("ru", "ab"))
        if total_w <= 0:
            raise ValueError("lang_weights must sum to > 0")
        if max_samples is None:
            max_samples = len(filtered)
        n_ab = int(max_samples * lang_weights.get("ab", 0.5) / total_w)
        n_ru = max_samples - n_ab
        ab_sample = (
            by_lang["ab"]
            if n_ab >= len(by_lang["ab"])
            else rng.sample(by_lang["ab"], n_ab)
        )
        ru_sample = (
            by_lang["ru"]
            if n_ru >= len(by_lang["ru"])
            else rng.sample(by_lang["ru"], n_ru)
        )
        mixed = ab_sample + ru_sample
        rng.shuffle(mixed)
        return mixed

    rng = random.Random(seed)
    if max_samples is not None and max_samples < len(filtered):
        return rng.sample(filtered, max_samples)
    rng.shuffle(filtered)
    return filtered


def load_parallel_pool(
    path: Path,
    *,
    max_samples: int | None,
    seed: int,
    min_chars: int = 5,
    max_chars: int = 4096,
    max_sents: int = 20,
) -> list[dict[str, str]]:
    """jsonl: {"src": ru, "ref": ab} или csv ru,ab."""
    if path.suffix == ".csv":
        rows = _load_parallel_csv(path, min_chars, max_chars, max_sents)
    else:
        rows = []
        for row in load_jsonl_rows(path):
            ru = str(row.get("src", row.get("ru", ""))).strip()
            ab = str(row.get("ref", row.get("ab", ""))).strip()
            if not ru or not ab or ru == ab:
                continue
            if len(ru) < min_chars:
                continue
            if max_chars > 0 and len(ru) > max_chars:
                continue
            if max_sents > 0 and _sent_count(ru) > max_sents:
                continue
            rows.append({"src": ru, "ref": ab})

    if not rows:
        raise ValueError(f"no parallel rows after filter: {path}")

    rng = random.Random(seed)
    if max_samples is not None and max_samples < len(rows):
        return rng.sample(rows, max_samples)
    return rows


def _load_parallel_csv(
    path: Path,
    min_chars: int,
    max_chars: int,
    max_sents: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            ru = row["ru"].strip()
            ab = row["ab"].strip()
            if not ru or not ab or ru == ab:
                continue
            if len(ru) < min_chars:
                continue
            if max_chars > 0 and len(ru) > max_chars:
                continue
            if max_sents > 0 and _sent_count(ru) > max_sents:
                continue
            rows.append({"src": ru, "ref": ab})
    return rows
