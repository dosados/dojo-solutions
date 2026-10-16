"""Чекпоинты LoRA-адаптера без дублирования tokenizer и без пика VRAM."""

from __future__ import annotations

import gc
import json
import re
import shutil
from pathlib import Path
from typing import Any

import torch
from transformers import TrainerCallback

# ~32 MB tokenizer.json на каждый checkpoint — не нужен для resume адаптера
_CHECKPOINT_STRIP_NAMES = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "processor_config.json",
    "preprocessor_config.json",
    "chat_template.jinja",
    "added_tokens.json",
)


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    """Последний checkpoint-N в output_dir (по номеру шага)."""
    if not output_dir.is_dir():
        return None
    best: tuple[int, Path] | None = None
    pat = re.compile(r"^checkpoint-(\d+)$")
    for child in output_dir.iterdir():
        m = pat.match(child.name)
        if not m or not child.is_dir():
            continue
        step = int(m.group(1))
        adapter = child / "adapter_model.safetensors"
        if not adapter.is_file():
            continue
        if best is None or step > best[0]:
            best = (step, child)
    return best[1] if best else None


def _strip_checkpoint_extras(ckpt_dir: Path) -> None:
    removed = 0
    for name in _CHECKPOINT_STRIP_NAMES:
        path = ckpt_dir / name
        if path.is_file():
            path.unlink()
            removed += 1
    if removed:
        print(f"  checkpoint trim: removed {removed} redundant files from {ckpt_dir.name}", flush=True)


def _checkpoint_size_mb(ckpt_dir: Path) -> float:
    total = sum(f.stat().st_size for f in ckpt_dir.rglob("*") if f.is_file())
    return total / 2**20


class LowVRAMCheckpointCallback(TrainerCallback):
    """Перед/после save: gc + cuda cache; после save — только adapter + trainer state."""

    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not ckpt_dir.is_dir():
            return
        _strip_checkpoint_extras(ckpt_dir)
        mb = _checkpoint_size_mb(ckpt_dir)
        print(f"  checkpoint saved: {ckpt_dir.name} ({mb:.1f} MiB, adapter-only)", flush=True)


# Файлы LoRA, достаточные для merge_lora / validate_lora
_BEST_EVAL_ADAPTER_FILES = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "adapter_config.json",
)


class BestEvalCheckpointCallback(TrainerCallback):
    """Сохраняет лучший адаптер по eval-метрике в отдельную папку (не для resume).

    Resume обучения по-прежнему идёт с checkpoint-N / final, best_eval/ только для merge.
    """

    def __init__(
        self,
        run_dir: Path,
        *,
        metric: str = "eval_loss",
        best_dir_name: str = "best_eval",
        greater_is_better: bool = False,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.metric = metric
        self.best_dir = self.run_dir / best_dir_name
        self.greater_is_better = greater_is_better
        self._best_value: float | None = self._read_stored_best()

    def _meta_path(self) -> Path:
        return self.best_dir / "best_metric.json"

    def _has_adapter(self) -> bool:
        d = self.best_dir
        return (d / "adapter_model.safetensors").is_file() or (d / "adapter_model.bin").is_file()

    def _read_stored_best(self) -> float | None:
        """Порог только если в best_eval/ реально лежит адаптер, не один json."""
        if not self._has_adapter():
            return None
        path = self._meta_path()
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            val = data.get(self.metric)
            return float(val) if val is not None else None
        except (json.JSONDecodeError, TypeError, ValueError):
            return None

    def _is_improvement(self, value: float) -> bool:
        if self._best_value is None:
            return True
        if self.greater_is_better:
            return value > self._best_value
        return value < self._best_value

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        del args, control, kwargs
        if not metrics or self.metric not in metrics:
            return
        value = float(metrics[self.metric])
        if not self._is_improvement(value):
            return

        step = int(state.global_step)
        epoch = float(state.epoch) if state.epoch is not None else None
        src_ckpt = self.run_dir / f"checkpoint-{step}"

        ok = False
        if model is not None:
            ok = self._promote_from_model(model, step=step, epoch=epoch, value=value)
        elif src_ckpt.is_dir() and (src_ckpt / "adapter_model.safetensors").is_file():
            ok = self._promote_from_checkpoint(src_ckpt, step=step, epoch=epoch, value=value)
        else:
            print(
                f"  best eval: skip step {step} (no checkpoint-{step} and no model in callback)",
                flush=True,
            )
            return

        if not ok:
            print(f"  best eval: save failed at step {step}, not updating {self.best_dir.name}/", flush=True)
            return

        self._best_value = value
        print(
            f"  best eval saved → {self.best_dir.name}/  "
            f"{self.metric}={value:.4f}  step={step}",
            flush=True,
        )

    def _promote_from_checkpoint(
        self,
        src_ckpt: Path,
        *,
        step: int,
        epoch: float | None,
        value: float,
    ) -> bool:
        prepare_for_checkpoint_save()
        if self.best_dir.is_dir():
            shutil.rmtree(self.best_dir)
        self.best_dir.mkdir(parents=True, exist_ok=True)
        for name in _BEST_EVAL_ADAPTER_FILES:
            src = src_ckpt / name
            if src.is_file():
                shutil.copy2(src, self.best_dir / name)
        if not self._has_adapter():
            shutil.rmtree(self.best_dir, ignore_errors=True)
            return False
        self._write_meta(step=step, epoch=epoch, value=value, source=str(src_ckpt.name))
        return True

    def _promote_from_model(
        self,
        model: Any,
        *,
        step: int,
        epoch: float | None,
        value: float,
    ) -> bool:
        prepare_for_checkpoint_save()
        if self.best_dir.is_dir():
            shutil.rmtree(self.best_dir)
        self.best_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(self.best_dir)
        _strip_checkpoint_extras(self.best_dir)
        if not self._has_adapter():
            shutil.rmtree(self.best_dir, ignore_errors=True)
            return False
        self._write_meta(step=step, epoch=epoch, value=value, source="model.save_pretrained")
        return True

    def _write_meta(
        self,
        *,
        step: int,
        epoch: float | None,
        value: float,
        source: str,
    ) -> None:
        meta = {
            self.metric: value,
            "global_step": step,
            "epoch": epoch,
            "source": source,
        }
        self._meta_path().write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def prepare_for_checkpoint_save() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resolve_resume_checkpoint(
    output_dir: Path,
    *,
    explicit: Path | None,
    auto_resume: bool,
) -> Path | None:
    if explicit is not None:
        path = explicit.resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"resume checkpoint not found: {path}")
        print(f"  resume: explicit {path}", flush=True)
        return path
    if not auto_resume:
        return None
    final_adapter = output_dir / "final" / "adapter_model.safetensors"
    if final_adapter.is_file():
        print(
            "  resume: skip auto (final/ already exists — удалите final/ или checkpoint-* для нового прогона)",
            flush=True,
        )
        return None
    latest = find_latest_checkpoint(output_dir)
    if latest is not None:
        print(f"  resume: auto from {latest}", flush=True)
    return latest
