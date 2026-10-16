#!/usr/bin/env python3
"""Локальный прогон Gemma-4-E2B или TranslateGemma + macro BLEU на self/dev.

Запуск из task-1/:

    conda activate ml-env
    python -u scripts/run_local_eval.py --model gemma --limit 20 --quantize 4bit
    python -u scripts/compare_models.py --limit 20 --quantize 4bit --rerun
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import torch

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from local_inference import run_gemma, run_translate  # noqa: E402

_TASK_ROOT = Path(__file__).resolve().parents[1]

MODEL_PRESETS = {
    "translate": {
        "weights": _TASK_ROOT / "submission_translate" / "weights",
        "output": _TASK_ROOT / "out" / "translategemma_dev_output.json",
    },
    "gemma": {
        "weights": _TASK_ROOT / "submission" / "weights",
        "output": _TASK_ROOT / "out" / "gemma_dev_output.json",
    },
}

_DEFAULT_INPUT = _TASK_ROOT / "data" / "processed" / "self" / "dev.input.pickle"
_DEFAULT_REF = _TASK_ROOT / "data" / "processed" / "self" / "dev.jsonl"


def _load_rows(path: Path, *, limit: int | None) -> list[dict]:
    with path.open("rb") as f:
        rows = pickle.load(f)
    if limit is not None:
        rows = rows[:limit]
    return rows


def _run_inference(args: argparse.Namespace) -> None:
    if not args.weights.is_dir():
        script = (
            "download_weights.py"
            if args.model == "gemma"
            else "download_weights_translate.py"
        )
        raise FileNotFoundError(
            f"weights not found: {args.weights}\n  run: python {script}"
        )
    if not args.input.is_file():
        raise FileNotFoundError(f"input not found: {args.input}")

    import logging
    import warnings

    logging.getLogger("transformers").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*generation flags are not valid.*")

    rows = _load_rows(args.input, limit=args.limit)
    run_fn = run_translate if args.model == "translate" else run_gemma
    infer_kw = dict(
        model_dir=args.weights,
        gpu_only=args.gpu_only,
        quantize=args.quantize,
        show_progress=True,
        batch_size=args.batch_size,
        use_compile=args.compile,
        warmup=args.warmup,
        max_new_tokens=args.max_new_tokens,
    )
    print(
        f"[{args.model}] inference: {len(rows)} paragraph(s)  "
        f"batch={args.batch_size} compile={args.compile} "
        f"max_new_tokens={args.max_new_tokens or 'dynamic'}",
        flush=True,
    )
    t0 = time.perf_counter()
    results = run_fn(rows, **infer_kw)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"wrote {args.output} in {elapsed:.1f}s "
        f"({elapsed / max(len(rows), 1):.2f}s/paragraph)"
    )


def _run_bleu(args: argparse.Namespace) -> None:
    if str(_TASK_ROOT) not in sys.path:
        sys.path.insert(0, str(_TASK_ROOT))

    from eval.bleu import evaluate_pairs  # noqa: E402
    from eval.io import load_hypotheses, load_references  # noqa: E402

    if not args.ref.is_file():
        raise FileNotFoundError(f"ref not found: {args.ref}")
    if not args.output.is_file():
        raise FileNotFoundError(f"hyp not found: {args.output}")

    refs = load_references(args.ref)
    hyps = load_hypotheses(args.output)

    if args.limit is not None:
        keep = set(range(args.limit))
        refs = {rid: text for rid, text in refs.items() if rid in keep}
        hyps = {rid: text for rid, text in hyps.items() if rid in keep}

    report = evaluate_pairs(hyps, refs, require_all_refs=True)

    print(f"sacrebleu {report.sacrebleu_version}")
    print(f"examples: {report.n_examples}")
    print(f"macro BLEU: {report.macro_bleu:.2f}")
    print(f"min / max:  {report.min_score:.2f} / {report.max_score:.2f}")

    if args.worst > 0:
        ranked = sorted(report.per_example, key=lambda x: x.score)
        print(f"\n--- worst {min(args.worst, len(ranked))} ---")
        for ex in ranked[: args.worst]:
            print(f"rid={ex.rid}  BLEU={ex.score:.2f}")
            if args.show_text:
                ref_prev = ex.ref[:120] + ("..." if len(ex.ref) > 120 else "")
                hyp_prev = ex.hyp[:120] + ("..." if len(ex.hyp) > 120 else "")
                print(f"  ref: {ref_prev}")
                print(f"  hyp: {hyp_prev}")

    if args.json_out:
        payload = {
            "macro_bleu": round(report.macro_bleu, 4),
            "n_examples": report.n_examples,
            "sacrebleu_version": report.sacrebleu_version,
            "per_example": [
                {"rid": ex.rid, "bleu": round(ex.score, 4)}
                for ex in report.per_example
            ],
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nWrote {args.json_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local task-1 model eval")
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_PRESETS),
        default="translate",
        help="gemma = Gemma-4-E2B (prompt v1), translate = TranslateGemma-4B",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Каталог весов (по умолчанию — из --model)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=_DEFAULT_INPUT,
        help="dev.input.pickle с полями rid, src",
    )
    parser.add_argument(
        "--ref",
        type=Path,
        default=_DEFAULT_REF,
        help="dev.jsonl с полями rid, ref",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Куда писать переводы",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Только первые N абзацев",
    )
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--skip-bleu", action="store_true")
    parser.add_argument(
        "--gpu-only",
        action="store_true",
        help="Полные веса на GPU; на ≤8.5 GiB VRAM автоматически 4bit",
    )
    parser.add_argument(
        "--quantize",
        choices=("none", "4bit"),
        default="none",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Размер батча (default: translate=4, gemma=1)",
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Фиксированный лимит (default: dynamic по длине src)",
    )
    parser.add_argument("--worst", type=int, default=10)
    parser.add_argument("--show-text", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    preset = MODEL_PRESETS[args.model]
    if args.weights is None:
        args.weights = preset["weights"]
    if args.output is None:
        args.output = preset["output"]
    if args.batch_size is None:
        args.batch_size = 4 if args.model == "translate" else 1
    if args.max_new_tokens is None and args.model == "gemma":
        pass  # dynamic per paragraph in run_gemma
    elif args.max_new_tokens is None and args.model == "translate":
        pass  # dynamic in run_translate

    if not args.skip_inference:
        _run_inference(args)

    if not args.skip_bleu:
        _run_bleu(args)


if __name__ == "__main__":
    main()
