"""Пути к весам task-2 (обязательны локально перед обучением/merge)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

ModelType = Literal["gemma", "translate", "nllb"]

_TASK2_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_WEIGHTS: dict[ModelType, Path] = {
    "gemma": _TASK2_ROOT / "submission" / "weights",
    "translate": _TASK2_ROOT / "submission_translate" / "weights",
    "nllb": _TASK2_ROOT / "submission_nllb" / "weights",
}

_DOWNLOAD_HINT: dict[ModelType, str] = {
    "gemma": "python download_weights.py",
    "translate": "python download_weights_translate.py",
    "nllb": "python download_weights_nllb.py",
}


def resolve_weights_dir(model_type: ModelType, path: str | Path | None = None) -> Path:
    """Каталог весов из конфига или default для model_type."""
    if path is not None:
        p = Path(path)
        if not p.is_absolute():
            p = (_TASK2_ROOT / p).resolve()
    else:
        p = DEFAULT_WEIGHTS[model_type]

    if not p.is_dir():
        raise FileNotFoundError(
            f"weights not found: {p}\n  run: {_DOWNLOAD_HINT[model_type]}"
        )
    return p
