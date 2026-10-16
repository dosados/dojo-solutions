#!/usr/bin/env python3
"""Один прогон активного промпта (submission/prompt_tune_active.py).

    cd task-1
    conda activate ml-env
    python -u scripts/run_prompt_iteration.py --iter 1 --gpu-only --quantize 4bit
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

_TASK_ROOT = Path(__file__).resolve().parents[1]
_SUBMISSION = _TASK_ROOT / "submission"
_SCRIPTS = Path(__file__).resolve().parent
for p in (_TASK_ROOT, _SCRIPTS, _SUBMISSION):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eval.bleu import evaluate_pairs  # noqa: E402
from eval.io import load_references  # noqa: E402
from local_inference import load_gemma, run_gemma_loaded  # noqa: E402
from prompt_tune_active import build_prompt  # noqa: E402

_OPUS_IN = _TASK_ROOT / "data" / "processed" / "opus100" / "validation.input.pickle"
_OPUS_REF = _TASK_ROOT / "data" / "processed" / "opus100" / "validation.jsonl"
_SELF_IN = _TASK_ROOT / "data" / "processed" / "self" / "dev.input.pickle"
_SELF_REF = _TASK_ROOT / "data" / "processed" / "self" / "dev.jsonl"
_WEIGHTS = _SUBMISSION / "weights"
_OUT = _TASK_ROOT / "out" / "prompt_tune"


def _load_rows(path: Path, limit: int | None) -> list[dict]:
    with path.open("rb") as f:
        rows = pickle.load(f)
    return rows[:limit] if limit else rows


def _eval(rows: list[dict], hyps: list[dict], ref_path: Path) -> tuple[dict, list]:
    refs = load_references(ref_path)
    keep = {r["rid"] for r in rows}
    refs = {rid: t for rid, t in refs.items() if rid in keep}
    hyp_map = {h["rid"]: h["translation"] for h in hyps if h["rid"] in keep}
    report = evaluate_pairs(hyp_map, refs, require_all_refs=True)
    worst = sorted(report.per_example, key=lambda x: x.score)[:8]
    return {
        "macro_bleu": round(report.macro_bleu, 2),
        "n": report.n_examples,
        "min": round(report.min_score, 2),
        "max": round(report.max_score, 2),
    }, worst


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iter", type=int, required=True)
    p.add_argument("--opus-limit", type=int, default=300)
    p.add_argument("--gpu-only", action="store_true")
    p.add_argument("--quantize", choices=("none", "4bit"), default="4bit")
    p.add_argument("--no-postprocess", action="store_true")
    args = p.parse_args()

    tag = f"iter{args.iter:02d}"
    opus_rows = _load_rows(_OPUS_IN, args.opus_limit)
    self_rows = _load_rows(_SELF_IN, None)

    print(f"=== {tag} ===", flush=True)
    print(f"opus n={len(opus_rows)}, self n={len(self_rows)}", flush=True)

    t0 = time.perf_counter()
    runtime = load_gemma(_WEIGHTS, gpu_only=args.gpu_only, quantize=args.quantize)
    print(f"model loaded {time.perf_counter() - t0:.1f}s", flush=True)

    pp = not args.no_postprocess
    opus_hyps = run_gemma_loaded(
        runtime, opus_rows, prompt_builder=build_prompt, show_progress=True, use_postprocess=pp
    )
    self_hyps = run_gemma_loaded(
        runtime, self_rows, prompt_builder=build_prompt, show_progress=True, use_postprocess=pp
    )

    opus_m, opus_worst = _eval(opus_rows, opus_hyps, _OPUS_REF)
    self_m, self_worst = _eval(self_rows, self_hyps, _SELF_REF)

    _OUT.mkdir(parents=True, exist_ok=True)
    metrics = {
        "iter": args.iter,
        "tag": tag,
        "opus": opus_m,
        "self": self_m,
        "opus_limit": args.opus_limit,
        "postprocess": pp,
    }
    (_OUT / f"{tag}_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (_OUT / f"{tag}_opus.json").write_text(
        json.dumps(opus_hyps, ensure_ascii=False), encoding="utf-8"
    )
    (_OUT / f"{tag}_self.json").write_text(
        json.dumps(self_hyps, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nOPUS  macro BLEU = {opus_m['macro_bleu']:.2f}  (n={opus_m['n']})")
    print(f"self  macro BLEU = {self_m['macro_bleu']:.2f}  (n={self_m['n']})")
    print(f"\n--- worst OPUS ({tag}) ---")
    for ex in opus_worst:
        print(f"rid={ex.rid} BLEU={ex.score:.2f}")
        print(f"  ref: {ex.ref[:140]}{'...' if len(ex.ref) > 140 else ''}")
        print(f"  hyp: {ex.hyp[:140]}{'...' if len(ex.hyp) > 140 else ''}")


if __name__ == "__main__":
    main()
