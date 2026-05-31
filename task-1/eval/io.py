"""Загрузка гипотез и эталонов для eval."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any


def _read_text_lines(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line for line in lines if line.strip() or line == ""]


def load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected JSON list, got {type(data).__name__}")
    return data


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_rid_text_map(path: Path, text_key: str) -> dict[int, str]:
    """JSON / JSONL / pickle со списком {rid, <text_key>}."""
    suffix = path.suffix.lower()
    if suffix == ".pickle":
        with path.open("rb") as f:
            rows = pickle.load(f)
    elif suffix == ".jsonl":
        rows = load_jsonl(path)
    else:
        rows = load_json_list(path)

    out: dict[int, str] = {}
    for row in rows:
        rid = int(row["rid"])
        if rid in out:
            raise ValueError(f"{path}: duplicate rid={rid}")
        if text_key not in row:
            raise KeyError(f"{path}: rid={rid} missing key {text_key!r}")
        out[rid] = str(row[text_key]).strip()
    return out


def load_hypotheses(path: Path) -> dict[int, str]:
    """output.json или dev-файл с полем translation."""
    suffix = path.suffix.lower()
    if suffix == ".txt":
        lines = _read_text_lines(path)
        return {i: line for i, line in enumerate(lines)}
    return load_rid_text_map(path, "translation")


def _resolve_ref_key(rows: list[dict[str, Any]], ref_key: str | None) -> str:
    if ref_key is not None:
        return ref_key
    sample = rows[0]
    if "ref" in sample:
        return "ref"
    if "translation" in sample:
        return "translation"
    if "tgt" in sample:
        return "tgt"
    raise KeyError(
        f"specify --ref-key; found keys: {list(sample.keys())}"
    )


def _load_ref_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return load_jsonl(path)
    if suffix == ".json":
        return load_json_list(path)
    with path.open("rb") as f:
        rows = pickle.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected list rows, got {type(rows).__name__}")
    return rows


def load_references(
    path: Path,
    ref_key: str | None = None,
    *,
    exclude_articles: tuple[str, ...] | list[str] | None = None,
) -> dict[int, str]:
    """Эталоны: ref, translation или plain .txt по порядку rid."""
    suffix = path.suffix.lower()
    if suffix == ".txt":
        lines = _read_text_lines(path)
        return {i: line for i, line in enumerate(lines)}

    rows = _load_ref_rows(path)
    if exclude_articles:
        skip = set(exclude_articles)
        before = len(rows)
        rows = [r for r in rows if r.get("article") not in skip]
        dropped = before - len(rows)
        if dropped:
            print(
                f"excluded {dropped} example(s) from articles: "
                f"{', '.join(sorted(skip))}",
                flush=True,
            )

    key = _resolve_ref_key(rows, ref_key)

    out: dict[int, str] = {}
    for row in rows:
        rid = int(row["rid"])
        out[rid] = str(row[key]).strip()
    return out


def load_parallel_jsonl(path: Path) -> tuple[dict[int, str], dict[int, str]]:
    """Один файл: {rid, src, ref} — для проверки без output.json."""
    rows = load_jsonl(path)
    src: dict[int, str] = {}
    ref: dict[int, str] = {}
    for row in rows:
        rid = int(row["rid"])
        src[rid] = str(row["src"]).strip()
        ref[rid] = str(row["ref"]).strip()
    return src, ref
