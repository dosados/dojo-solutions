#!/usr/bin/env python3
"""Merge QLoRA adapter в базовые веса bf16 (для zip контеста).

Обучение — 4-bit на GPU; merge — полные веса bf16 (~8 GB для 4B).
На GPU ≤8 GiB VRAM используйте --device cpu (нужно ~16 GiB RAM).

Запуск из task-2/:

    conda activate ml-env
    python -u scripts/merge_lora.py --model translate --adapter runs/lora/translate/sft_pilot/final
    python -u scripts/merge_lora.py --model translate --adapter runs/lora/translate/sft_pilot/final \\
        --out submission_translate/weights --backup --device cpu

    python -u scripts/merge_lora.py --model gemma --adapter runs/lora/gemma/sft_pilot/final \\
        --out submission/weights --backup --device cpu

    python -u scripts/merge_lora.py --model nllb --adapter runs/lora/nllb/sft_pilot/final \\
        --out submission_nllb/weights --backup --device cpu
"""
from __future__ import annotations

import argparse
import gc
import shutil
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSeq2SeqLM,
)

_TASK_ROOT = Path(__file__).resolve().parents[1]
if str(_TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(_TASK_ROOT))

from train.gemma_peft import patch_peft_gemma4_clippable_linear  # noqa: E402
from train.load_model import load_processor  # noqa: E402
from train.paths import resolve_weights_dir  # noqa: E402

# bf16 4B ~8 GiB; merge на GPU нужен запас
_GPU_MERGE_MIN_GIB = 10.0


def _gpu_total_gib() -> float | None:
    if not torch.cuda.is_available():
        return None
    _, total = torch.cuda.mem_get_info(0)
    return total / 2**30


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    total = _gpu_total_gib()
    if total is None:
        print("  auto: CUDA недоступна — merge на CPU", flush=True)
        return "cpu"
    if total < _GPU_MERGE_MIN_GIB:
        print(
            f"  auto: GPU {total:.1f} GiB — bf16 merge не влезет; используем CPU "
            f"(нужно ~16 GiB RAM)",
            flush=True,
        )
        return "cpu"
    return "cuda"


def _load_base_bf16(model_type: str, base: Path, *, device: str):
    kwargs: dict = {
        "dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
    }
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda: CUDA недоступна")
        kwargs["device_map"] = {"": 0}
    elif device == "cpu":
        kwargs["device_map"] = "cpu"
    else:
        raise ValueError(f"unknown device: {device}")

    if model_type == "translate":
        return AutoModelForImageTextToText.from_pretrained(str(base), **kwargs)
    if model_type == "nllb":
        return AutoModelForSeq2SeqLM.from_pretrained(str(base), **kwargs)
    return AutoModelForCausalLM.from_pretrained(str(base), **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base weights")
    parser.add_argument("--model", choices=("gemma", "translate", "nllb"), required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="auto: CPU если VRAM < 10 GiB (default: auto)",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="If --out equals base, copy base to weights_backup first",
    )
    args = parser.parse_args()

    base = resolve_weights_dir(args.model, args.base)
    adapter = args.adapter.resolve()
    out = (args.out or base.parent / f"{base.name}_merged").resolve()
    device = _resolve_device(args.device)

    if not base.is_dir():
        raise FileNotFoundError(f"base weights not found: {base}")
    if not adapter.is_dir():
        raise FileNotFoundError(f"adapter not found: {adapter}")

    if args.backup and out == base:
        backup = base.parent / f"{base.name}_backup"
        if backup.exists():
            shutil.rmtree(backup)
        print(f"backup {base} → {backup}", flush=True)
        shutil.copytree(base, backup)

    if args.model == "gemma":
        patch_peft_gemma4_clippable_linear()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"loading bf16 base from {base} on {device} …", flush=True)
    model = _load_base_bf16(args.model, base, device=device)
    print(f"loading adapter from {adapter} …", flush=True)
    model = PeftModel.from_pretrained(model, str(adapter))

    print("merge_and_unload …", flush=True)
    merged = model.merge_and_unload()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    in_place = out == base
    save_dir = out
    staging: Path | None = None
    if in_place:
        staging = out.parent / f".{out.name}_merge_staging"
        if staging.exists():
            shutil.rmtree(staging)
        save_dir = staging

    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"saving merged bf16 weights to {save_dir} …", flush=True)
    merged.save_pretrained(str(save_dir), safe_serialization=True)
    del merged
    gc.collect()

    processor = load_processor(args.model, base)
    processor.save_pretrained(str(save_dir))

    if staging is not None:
        print(f"replacing {out} …", flush=True)
        if out.exists():
            shutil.rmtree(out)
        staging.rename(out)

    print(f"done → {out}", flush=True)


if __name__ == "__main__":
    main()
