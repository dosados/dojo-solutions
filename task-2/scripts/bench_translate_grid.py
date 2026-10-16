#!/usr/bin/env python3
"""Benchmark TranslateGemma generation with one model load."""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from local_inference import (  # noqa: E402
    _input_device,
    _load_translate_processor,
    _maybe_compile,
    _model_load_kwargs,
    _prepare_translate_batch,
    _translate_pad_token_id,
    resolve_load_mode,
)

_TASK_ROOT = Path(__file__).resolve().parents[1]


def _load_rows(path: Path, limit: int | None) -> list[dict]:
    with path.open("rb") as f:
        rows = pickle.load(f)
    return rows if limit is None else rows[:limit]


def _iter_batches(rows: list[dict], batch_size: int):
    ordered = sorted(rows, key=lambda r: len(r["src"]))
    for start in range(0, len(ordered), batch_size):
        yield ordered[start : start + batch_size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=_TASK_ROOT / "data" / "processed" / "dev.input.pickle",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=_TASK_ROOT / "submission_translate" / "weights",
    )
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 12, 16, 20, 24, 28, 32])
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--quantize", choices=("none", "4bit"), default="4bit")
    parser.add_argument("--gpu-only", action="store_true", default=True)
    args = parser.parse_args()

    rows = _load_rows(args.input, args.limit)
    gpu_only, quantize = resolve_load_mode(gpu_only=args.gpu_only, quantize=args.quantize)
    print(
        f"rows={len(rows)} weights={args.weights} quantize={quantize} "
        f"compile={args.compile} warmup={args.warmup}",
        flush=True,
    )
    processor = _load_translate_processor(args.weights)
    model = AutoModelForImageTextToText.from_pretrained(
        args.weights,
        **_model_load_kwargs(gpu_only=gpu_only, quantize=quantize),
    )
    model = _maybe_compile(model, enabled=args.compile)
    device = _input_device(model)
    pad_token_id = _translate_pad_token_id(processor, model)

    def generate_all(batch_size: int) -> None:
        for batch in _iter_batches(rows, batch_size):
            inputs, input_lens = _prepare_translate_batch(
                processor,
                batch,
                device=device,
                pad_token_id=pad_token_id,
            )
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=pad_token_id,
                )
            for in_len, out_ids in zip(input_lens, outputs, strict=True):
                processor.decode(out_ids[in_len:], skip_special_tokens=True).strip()

    if args.warmup:
        generate_all(min(args.batch_sizes))
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    for batch_size in args.batch_sizes:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        generate_all(batch_size)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_gib = torch.cuda.max_memory_allocated() / 2**30
        else:
            peak_gib = 0.0
        elapsed = time.perf_counter() - t0
        print(
            f"batch={batch_size:2d} elapsed={elapsed:6.1f}s "
            f"per_example={elapsed / max(len(rows), 1):.3f}s peak={peak_gib:.2f}GiB",
            flush=True,
        )


if __name__ == "__main__":
    main()
