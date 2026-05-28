"""Общие форматы данных task-1: JSONL, input.pickle."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Iterable


def parse_paragraph_blocks(path: Path) -> list[str]:
    """Абзацы из .txt: блоки непустых строк, внутри абзаца — через пробел."""
    blocks: list[str] = []
    current: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            current.append(line.strip())
        elif current:
            blocks.append(" ".join(current))
            current = []
    if current:
        blocks.append(" ".join(current))
    return blocks


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def jsonl_to_pickle(
    jsonl_path: Path,
    pickle_path: Path,
    *,
    src_key: str = "src",
) -> int:
    rows: list[dict[str, int | str]] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            rows.append({"rid": int(ex["rid"]), "src": str(ex[src_key])})
    pickle_path.parent.mkdir(parents=True, exist_ok=True)
    with pickle_path.open("wb") as f:
        pickle.dump(rows, f)
    return len(rows)
