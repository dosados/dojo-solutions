#!/usr/bin/env python3
"""DEPRECATED: используйте docs/PROMPT_TUNING.md + run_prompt_iteration.py (по одному промпту).

Пакетная оценка промптов Gemma: одна загрузка модели, несколько вариантов.

    cd task-1
    conda activate ml-env
    python -u scripts/prompt_tune_eval.py --opus-limit 300 --gpu-only --quantize 4bit
    python -u scripts/prompt_tune_eval.py --names iter06_sentence iter15_refined
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import torch

_TASK_ROOT = Path(__file__).resolve().parents[1]
_SUBMISSION = _TASK_ROOT / "submission"
_SCRIPTS = Path(__file__).resolve().parent
for p in (_TASK_ROOT, _SCRIPTS, _SUBMISSION):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eval.bleu import evaluate_pairs  # noqa: E402
from eval.io import load_hypotheses, load_references  # noqa: E402
from local_inference import load_gemma, run_gemma_loaded  # noqa: E402
from prompts_experiments import PROMPT_REGISTRY, PromptBuilder  # noqa: E402

_SELF_INPUT = _TASK_ROOT / "data" / "processed" / "self" / "dev.input.pickle"
_SELF_REF = _TASK_ROOT / "data" / "processed" / "self" / "dev.jsonl"
_OPUS_INPUT = _TASK_ROOT / "data" / "processed" / "opus100" / "validation.input.pickle"
_OPUS_REF = _TASK_ROOT / "data" / "processed" / "opus100" / "validation.jsonl"
_WEIGHTS = _SUBMISSION / "weights"
_OUT_DIR = _TASK_ROOT / "out" / "prompt_tune"


def _load_rows(path: Path, limit: int | None) -> list[dict]:
    with path.open("rb") as f:
        rows = pickle.load(f)
    if limit is not None:
        rows = rows[:limit]
    return rows


def _bleu_on(rows: list[dict], hyps: list[dict], ref_path: Path) -> dict:
    refs = load_references(ref_path)
    hyp_map = {h["rid"]: h["translation"] for h in hyps}
    if rows:
        keep = {r["rid"] for r in rows}
        refs = {rid: t for rid, t in refs.items() if rid in keep}
        hyp_map = {rid: t for rid, t in hyp_map.items() if rid in keep}
    report = evaluate_pairs(hyp_map, refs, require_all_refs=True)
    return {
        "macro_bleu": round(report.macro_bleu, 2),
        "n": report.n_examples,
        "min": round(report.min_score, 2),
        "max": round(report.max_score, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt tuning batch eval (Gemma)")
    parser.add_argument(
        "--names",
        nargs="*",
        default=None,
        help=f"Подмножество промптов (default: все {len(PROMPT_REGISTRY)})",
    )
    parser.add_argument("--opus-limit", type=int, default=300)
    parser.add_argument("--self-limit", type=int, default=None)
    parser.add_argument("--gpu-only", action="store_true")
    parser.add_argument("--quantize", choices=("none", "4bit"), default="4bit")
    parser.add_argument("--no-postprocess", action="store_true")
    parser.add_argument(
        "--summary",
        type=Path,
        default=_OUT_DIR / "summary.json",
    )
    args = parser.parse_args()

    names = args.names or sorted(PROMPT_REGISTRY)
    for n in names:
        if n not in PROMPT_REGISTRY:
            raise SystemExit(f"unknown prompt {n!r}; known: {', '.join(sorted(PROMPT_REGISTRY))}")

    if not _WEIGHTS.is_dir():
        raise SystemExit(f"weights missing: {_WEIGHTS}")

    self_rows = _load_rows(_SELF_INPUT, args.self_limit)
    opus_rows = _load_rows(_OPUS_INPUT, args.opus_limit)

    print(
        f"prompt_tune: {len(names)} variants, "
        f"opus n={len(opus_rows)}, self n={len(self_rows)}, "
        f"postprocess={not args.no_postprocess}",
        flush=True,
    )

    t_load = time.perf_counter()
    runtime = load_gemma(
        _WEIGHTS,
        gpu_only=args.gpu_only,
        quantize=args.quantize,  # type: ignore[arg-type]
    )
    print(f"model loaded in {time.perf_counter() - t_load:.1f}s", flush=True)

    results: list[dict] = []
    for name in names:
        builder: PromptBuilder = PROMPT_REGISTRY[name]
        t0 = time.perf_counter()
        print(f"\n{'=' * 60}\n>>> {name}\n{'=' * 60}", flush=True)

        opus_hyps = run_gemma_loaded(
            runtime,
            opus_rows,
            prompt_builder=builder,
            show_progress=True,
            use_postprocess=not args.no_postprocess,
        )
        opus_bleu = _bleu_on(opus_rows, opus_hyps, _OPUS_REF)

        self_hyps = run_gemma_loaded(
            runtime,
            self_rows,
            prompt_builder=builder,
            show_progress=True,
            use_postprocess=not args.no_postprocess,
        )
        self_bleu = _bleu_on(self_rows, self_hyps, _SELF_REF)

        elapsed = time.perf_counter() - t0
        _OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_opus = _OUT_DIR / f"{name}_opus.json"
        out_self = _OUT_DIR / f"{name}_self.json"
        out_opus.write_text(json.dumps(opus_hyps, ensure_ascii=False), encoding="utf-8")
        out_self.write_text(json.dumps(self_hyps, ensure_ascii=False), encoding="utf-8")

        row = {
            "name": name,
            "opus_bleu": opus_bleu["macro_bleu"],
            "opus_n": opus_bleu["n"],
            "self_bleu": self_bleu["macro_bleu"],
            "self_n": self_bleu["n"],
            "elapsed_s": round(elapsed, 1),
        }
        results.append(row)
        print(
            f"  OPUS BLEU={opus_bleu['macro_bleu']:.2f}  "
            f"self BLEU={self_bleu['macro_bleu']:.2f}  time={elapsed:.0f}s",
            flush=True,
        )

    results.sort(key=lambda r: r["opus_bleu"], reverse=True)
    print("\n=== ranking by OPUS BLEU ===", flush=True)
    for r in results:
        print(
            f"  {r['name']:28s}  opus={r['opus_bleu']:5.2f}  self={r['self_bleu']:5.2f}  "
            f"{r['elapsed_s']:5.0f}s",
            flush=True,
        )

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "opus_limit": args.opus_limit,
        "self_limit": args.self_limit,
        "quantize": args.quantize,
        "postprocess": not args.no_postprocess,
        "results": results,
    }
    args.summary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {args.summary}", flush=True)


if __name__ == "__main__":
    main()
