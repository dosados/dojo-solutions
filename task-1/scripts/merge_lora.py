#!/usr/bin/env python3
"""Слить QLoRA-адаптер в полные веса для посылки (bf16, без peft в рантайме).

Пример:
  cd task-1
  python -u scripts/merge_lora.py --adapter runs/lora/pilot_10k/final
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor

_TASK_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into submission weights")
    parser.add_argument(
        "--adapter",
        type=Path,
        required=True,
        help="Директория адаптера (…/final с adapter_config.json)",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=_TASK_ROOT / "submission" / "weights",
        help="Базовые веса Gemma",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="cpu",
        help="cpu — для 8GB GPU; cuda — если хватает VRAM на merge",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Сохранить копию model.safetensors перед заменой",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Только показать пути, без записи",
    )
    args = parser.parse_args()

    adapter_dir = args.adapter.expanduser().resolve()
    base_dir = args.base_dir.expanduser().resolve()
    if not (adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(f"no adapter at {adapter_dir}")

    if args.device == "auto":
        device_map = "auto"
    elif args.device == "cuda":
        device_map = {"": 0}
    else:
        device_map = "cpu"

    print(f"base:    {base_dir}", flush=True)
    print(f"adapter: {adapter_dir}", flush=True)
    print(f"device:  {args.device}", flush=True)

    if args.dry_run:
        return

    model_path = base_dir / "model.safetensors"
    if args.backup and model_path.is_file():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = base_dir / f"model.safetensors.bak.{ts}"
        print(f"backup → {backup}", flush=True)
        shutil.copy2(model_path, backup)

    print("loading base (bf16) …", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_dir,
        dtype=torch.bfloat16,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    print("loading adapter …", flush=True)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    print("merge_and_unload …", flush=True)
    model = model.merge_and_unload()

    merged_dir = base_dir / "_merged_tmp"
    if merged_dir.exists():
        shutil.rmtree(merged_dir)
    merged_dir.mkdir(parents=True)

    print(f"saving merged weights to {merged_dir} …", flush=True)
    model.save_pretrained(merged_dir, safe_serialization=True)

    new_weights = merged_dir / "model.safetensors"
    if not new_weights.is_file():
        raise FileNotFoundError(f"expected {new_weights}")

    print(f"replace {model_path}", flush=True)
    shutil.copy2(new_weights, model_path)
    shutil.rmtree(merged_dir)

    # Сохранить копию адаптера в submission для отладки (опционально)
    lora_dest = base_dir / "lora"
    if lora_dest.exists():
        print(f"note: {lora_dest} exists — not overwriting (use merged weights)", flush=True)
    else:
        print(f"optional: copy adapter to {lora_dest} for --lora-dir inference", flush=True)

    print("done. submission/weights/model.safetensors is merged.", flush=True)
    print(
        "smoke: cd submission && docker build -t task-1-solution .",
        flush=True,
    )


if __name__ == "__main__":
    main()
