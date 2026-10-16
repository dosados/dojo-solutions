#!/usr/bin/env python3
"""Быстрый бенчмарк Gemma vs TranslateGemma (inference only, n≤10)."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_TASK_ROOT = Path(__file__).resolve().parents[1]


def _run(
    *,
    model: str,
    limit: int,
    gpu_only: bool,
    quantize: str,
    batch_size: int,
    compile: bool,
    warmup: bool,
) -> float:
    cmd = [
        sys.executable,
        "-u",
        str(_TASK_ROOT / "scripts" / "run_local_eval.py"),
        "--model",
        model,
        "--limit",
        str(limit),
        "--skip-bleu",
        "--batch-size",
        str(batch_size),
    ]
    if gpu_only:
        cmd.append("--gpu-only")
    if quantize != "none":
        cmd.extend(["--quantize", quantize])
    if compile:
        cmd.append("--compile")
    if warmup:
        cmd.append("--warmup")
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if "s/example" in line and "wrote" in line:
            part = line.split("(")[-1].rstrip(")")
            return float(part.replace("s/example", ""))
    raise RuntimeError(f"could not parse timing:\n{proc.stdout}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--gpu-only", action="store_true")
    parser.add_argument("--quantize", choices=("none", "4bit"), default="4bit")
    args = parser.parse_args()

    print(f"=== baseline (batch=1) n={args.limit} ===", flush=True)
    gemma = _run(
        model="gemma",
        limit=args.limit,
        gpu_only=args.gpu_only,
        quantize=args.quantize,
        batch_size=1,
        compile=False,
        warmup=False,
    )
    trans_seq = _run(
        model="translate",
        limit=args.limit,
        gpu_only=args.gpu_only,
        quantize=args.quantize,
        batch_size=1,
        compile=False,
        warmup=False,
    )
    print(f"Gemma:              {gemma:.2f} s/example")
    print(f"Translate (seq):    {trans_seq:.2f} s/example  ({trans_seq / gemma:.2f}x)")

    print(f"\n=== translate optimized (batch=8, max_tok=128) n={args.limit} ===", flush=True)
    trans_batched = _run(
        model="translate",
        limit=args.limit,
        gpu_only=args.gpu_only,
        quantize=args.quantize,
        batch_size=8,
        compile=False,
        warmup=False,
    )
    print(
        f"Translate (batched): {trans_batched:.2f} s/example  "
        f"({trans_batched / gemma:.2f}x vs gemma)"
    )


if __name__ == "__main__":
    main()
