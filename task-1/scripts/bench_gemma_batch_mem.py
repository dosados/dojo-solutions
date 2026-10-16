#!/usr/bin/env python3
"""Подбор batch_size для task-1 Gemma: пик VRAM на данных, близких к контесту.

Контест (SPECS): 69 абзацев, avg ~846 символов, max 2229.
Локально на 8GB GPU грузим 4bit (--quantize 4bit), прирост VRAM от batch
экстраполируем на bf16 L4 24GB: contest_peak ≈ MODEL_GIB + (peak_B - peak_1).
"""
from __future__ import annotations

import argparse
import pickle
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

_SCRIPTS = Path(__file__).resolve().parent
_SUBMISSION = _SCRIPTS.parent / "submission"
if str(_SUBMISSION) not in sys.path:
    sys.path.insert(0, str(_SUBMISSION))

from prompts import build_opus_en_prompt  # noqa: E402
from solution import estimate_max_new_tokens  # noqa: E402

_TASK_ROOT = _SCRIPTS.parent
_SELF_EN = _TASK_ROOT / "data" / "self_data" / "en"
_CONTEST_MODEL_GIB = 7.3
_CONTEST_VRAM_GIB = 24.0
_MARGIN_GIB = 2.0


def _load_corpus_text() -> str:
    parts = []
    for p in sorted(_SELF_EN.glob("*.txt")):
        parts.append(p.read_text(encoding="utf-8").strip())
    return "\n\n".join(parts)


def _slice_text(corpus: str, length: int) -> str:
    if length <= len(corpus):
        return corpus[:length]
    reps = (length // max(len(corpus), 1)) + 1
    return (corpus * reps)[:length]


def build_contest_proxy_rows(n: int = 69, *, seed: int = 42) -> list[dict]:
    """Длины по распределению из SPECS (avg 846, max 2229)."""
    rng = random.Random(seed)
    corpus = _load_corpus_text()
    lengths: list[int] = []
    for i in range(n):
        if i < 8:
            lengths.append(rng.randint(60, 120))
        elif i < 20:
            lengths.append(rng.randint(200, 450))
        elif i < 55:
            lengths.append(int(rng.gauss(846, 280)))
        else:
            lengths.append(rng.randint(1600, 2229))
    lengths = [max(40, min(2229, L)) for L in lengths]
    rows = [{"rid": i, "src": _slice_text(corpus, L)} for i, L in enumerate(lengths)]
    return rows


def _iter_batches(rows: list[dict], batch_size: int) -> list[list[dict]]:
    ordered = sorted(rows, key=lambda r: len(r["src"]))
    return [
        ordered[start : start + batch_size]
        for start in range(0, len(ordered), batch_size)
    ]


def _bnb_4bit_config():
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def _mem_gib() -> tuple[float, float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 2**30
    reserved = torch.cuda.memory_reserved() / 2**30
    free, total = torch.cuda.mem_get_info(0)
    return alloc, reserved, total / 2**30


def _generate_batch(
    *,
    processor,
    tokenizer,
    model,
    device: torch.device,
    pad_token_id: int,
    build_prompt,
    batch: list[dict],
) -> None:
    batch_max = max(estimate_max_new_tokens(r["src"]) for r in batch)
    prompts = [
        processor.apply_chat_template(
            [{"role": "user", "content": build_prompt(row["src"])}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for row in batch
    ]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    with torch.inference_mode():
        model.generate(
            **inputs,
            max_new_tokens=batch_max,
            do_sample=False,
            pad_token_id=pad_token_id,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=Path,
        default=_SUBMISSION / "weights",
    )
    parser.add_argument("--limit", type=int, default=69, help="Сколько абзацев из proxy")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 6, 8, 10, 12, 16],
    )
    parser.add_argument(
        "--quantize",
        choices=("4bit", "none"),
        default="4bit",
        help="Локально 4bit; none только если влезает bf16 (~7.3 GiB модель)",
    )
    parser.add_argument(
        "--worst-only",
        action="store_true",
        help="Только худший батч: batch_size самых длинных абзацев (пик VRAM)",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Дополнительно: полный проход с лучшим batch для оценки времени",
    )
    parser.add_argument("--out-pickle", type=Path, default=None)
    args = parser.parse_args()

    rows = build_contest_proxy_rows(args.limit)
    lens = [len(r["src"]) for r in rows]
    print(
        f"proxy rows={len(rows)} chars min={min(lens)} max={max(lens)} "
        f"avg={sum(lens)/len(lens):.0f}",
        flush=True,
    )
    if args.out_pickle:
        args.out_pickle.parent.mkdir(parents=True, exist_ok=True)
        with args.out_pickle.open("wb") as f:
            pickle.dump(rows, f)
        print(f"wrote {args.out_pickle}", flush=True)

    if not torch.cuda.is_available():
        print("CUDA недоступна — замер невозможен", flush=True)
        return

    _, _, gpu_total = _mem_gib()
    print(f"GPU total={gpu_total:.2f} GiB quantize={args.quantize}", flush=True)

    load_kw: dict = {
        "device_map": {"": 0},
        "attn_implementation": "sdpa",
    }
    if args.quantize == "4bit":
        load_kw["quantization_config"] = _bnb_4bit_config()
        load_kw["dtype"] = torch.bfloat16
    else:
        load_kw["dtype"] = torch.bfloat16

    print("loading model …", flush=True)
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.weights)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = int(tokenizer.pad_token_id)

    model = AutoModelForCausalLM.from_pretrained(args.weights, **load_kw)
    device = torch.device("cuda:0")
    build_prompt = build_opus_en_prompt
    print(f"model ready in {time.perf_counter() - t0:.1f}s", flush=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _generate_batch(
        processor=processor,
        tokenizer=tokenizer,
        model=model,
        device=device,
        pad_token_id=pad_token_id,
        build_prompt=build_prompt,
        batch=[rows[0]],
    )
    torch.cuda.synchronize()
    warmup_peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"warmup batch=1 peak={warmup_peak:.2f} GiB", flush=True)

    results: list[dict] = []
    for bs in sorted(set(args.batch_sizes)):
        try:
            if args.worst_only:
                ordered = sorted(rows, key=lambda r: len(r["src"]))
                batch = ordered[-bs:]
                batches = [batch]
            else:
                batches = _iter_batches(rows, bs)

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            t_start = time.perf_counter()
            for batch in batches:
                _generate_batch(
                    processor=processor,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    pad_token_id=pad_token_id,
                    build_prompt=build_prompt,
                    batch=batch,
                )
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 2**30
            elapsed = time.perf_counter() - t_start
            oom = False
        except torch.cuda.OutOfMemoryError:
            peak = float("inf")
            elapsed = 0.0
            oom = True

        results.append({"batch_size": bs, "peak_gib": peak, "elapsed": elapsed, "oom": oom})
        status = "OOM" if oom else f"peak={peak:.2f}GiB"
        print(
            f"batch={bs:2d}  {status}  elapsed={elapsed:.1f}s  "
            f"n_batches={len(batches) if not oom else '-'}",
            flush=True,
        )

    ok = [r for r in results if not r["oom"]]
    if not ok:
        print("\nВсе batch sizes OOM", flush=True)
        return

    peak1 = next(r["peak_gib"] for r in ok if r["batch_size"] == 1)
    # baseline = короткий warmup (модель + минимальный контекст)
    kv_baseline = peak1 - warmup_peak
    print(f"\n--- Экстраполяция на контест L4 {_CONTEST_VRAM_GIB:.0f} GiB (bf16 ~{_CONTEST_MODEL_GIB} GiB модель) ---", flush=True)
    print(
        f"local warmup_peak={warmup_peak:.2f} GiB (короткий абзац); "
        f"worst batch=1 peak={peak1:.2f} GiB",
        flush=True,
    )
    print(f"margin={_MARGIN_GIB} GiB → лимит пика ~{_CONTEST_VRAM_GIB - _MARGIN_GIB:.1f} GiB", flush=True)
    for r in ok:
        kv_extra = r["peak_gib"] - peak1
        contest_est = _CONTEST_MODEL_GIB + kv_baseline + kv_extra
        headroom = _CONTEST_VRAM_GIB - _MARGIN_GIB - contest_est
        mark = "OK" if headroom >= 0 else "OVER"
        print(
            f"  batch={r['batch_size']:2d}  local_peak={r['peak_gib']:.2f}  "
            f"kv_extra={kv_extra:+.2f}  contest_est≈{contest_est:.2f}  headroom={headroom:+.2f}  [{mark}]",
            flush=True,
        )

    limit_gib = _CONTEST_VRAM_GIB - _MARGIN_GIB
    best = max(
        (
            r
            for r in ok
            if _CONTEST_MODEL_GIB + kv_baseline + (r["peak_gib"] - peak1) <= limit_gib
        ),
        key=lambda r: r["batch_size"],
        default=None,
    )
    if best:
        est = _CONTEST_MODEL_GIB + kv_baseline + (best["peak_gib"] - peak1)
        print(
            f"\nРекомендация для submission: BATCH_SIZE={best['batch_size']} "
            f"(contest_est≈{est:.1f} GiB, запас до лимита {limit_gib:.0f} GiB)",
            flush=True,
        )
    worst_n = min(4, len(rows))
    worst = sorted(rows, key=lambda r: len(r["src"]))[-worst_n:]
    w_max_tok = max(estimate_max_new_tokens(r["src"]) for r in worst)
    print(
        f"\nХудший батч ({worst_n} самых длинных): chars={[len(r['src']) for r in worst]} "
        f"max_new_tokens={w_max_tok}",
        flush=True,
    )

    if args.timing and best and not args.worst_only:
        bs = best["batch_size"]
        torch.cuda.empty_cache()
        t_start = time.perf_counter()
        for batch in _iter_batches(rows, bs):
            _generate_batch(
                processor=processor,
                tokenizer=tokenizer,
                model=model,
                device=device,
                pad_token_id=pad_token_id,
                build_prompt=build_prompt,
                batch=batch,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t_start
        print(
            f"\nTiming full proxy n={len(rows)} batch={bs}: {elapsed:.1f}s "
            f"({elapsed / len(rows):.1f}s/para)",
            flush=True,
        )
        print(
            f"Double pass estimate: {2 * elapsed:.0f}s ({2 * elapsed / 60:.1f} min) "
            f"vs contest limit 45 min",
            flush=True,
        )


if __name__ == "__main__":
    main()
