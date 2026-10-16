#!/usr/bin/env python3
"""Сравнить Gemma-4-E2B (prompt v1) и TranslateGemma-4B на self/dev task-1.

Запуск из task-1/:

    conda activate ml-env
    python -u scripts/compare_models.py --limit 20 --gpu-only --rerun
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_TASK_ROOT = Path(__file__).resolve().parents[1]
if str(_TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(_TASK_ROOT))

from eval.bleu import evaluate_pairs  # noqa: E402
from eval.io import load_hypotheses, load_references  # noqa: E402

MODELS = {
    "gemma": {
        "label": "Gemma-4-E2B (v1 prompt)",
        "weights": _TASK_ROOT / "submission" / "weights",
        "output": _TASK_ROOT / "out" / "gemma_dev_output.json",
    },
    "translate": {
        "label": "TranslateGemma-4B",
        "weights": _TASK_ROOT / "submission_translate" / "weights",
        "output": _TASK_ROOT / "out" / "translategemma_dev_output.json",
    },
}


def _bleu(output: Path, ref: Path, limit: int | None) -> dict:
    refs = load_references(ref)
    hyps = load_hypotheses(output)
    if limit is not None:
        keep = set(range(limit))
        refs = {rid: text for rid, text in refs.items() if rid in keep}
        hyps = {rid: text for rid, text in hyps.items() if rid in keep}
    report = evaluate_pairs(hyps, refs, require_all_refs=True)
    return {
        "macro_bleu": report.macro_bleu,
        "n_examples": report.n_examples,
        "min_score": report.min_score,
        "max_score": report.max_score,
    }


def _run_model(model_key: str, args: argparse.Namespace) -> Path:
    info = MODELS[model_key]
    output = info["output"]

    if args.reuse_existing and output.is_file() and not args.rerun:
        print(f"[{info['label']}] reuse {output}", flush=True)
        return output

    cmd = [
        sys.executable,
        "-u",
        str(_TASK_ROOT / "scripts" / "run_local_eval.py"),
        "--model",
        model_key,
        "--weights",
        str(info["weights"]),
        "--output",
        str(output),
        "--input",
        str(args.input),
        "--ref",
        str(args.ref),
        "--skip-bleu",
    ]
    if args.gpu_only:
        cmd.append("--gpu-only")
    if args.quantize != "none":
        cmd.extend(["--quantize", args.quantize])
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.batch_size is not None:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if args.compile:
        cmd.append("--compile")
    print(f"[{info['label']}] inference …", flush=True)
    subprocess.run(cmd, check=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Gemma vs TranslateGemma on task-1 self/dev"
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--input",
        type=Path,
        default=_TASK_ROOT / "data" / "processed" / "self" / "dev.input.pickle",
    )
    parser.add_argument(
        "--ref",
        type=Path,
        default=_TASK_ROOT / "data" / "processed" / "self" / "dev.jsonl",
    )
    parser.add_argument("--reuse-existing", action="store_true", default=True)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--gpu-only", action="store_true")
    parser.add_argument("--quantize", choices=("none", "4bit"), default="none")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Переопределить batch для обеих моделей (по умолчанию gemma=1, translate=4)",
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=_TASK_ROOT / "out" / "compare_models.json",
    )
    args = parser.parse_args()

    if not args.input.is_file() or not args.ref.is_file():
        raise FileNotFoundError(
            f"dev not found — need {args.input} and {args.ref}\n"
            "  run: python scripts/convert_datasets.py"
        )

    results = {}
    for key in ("gemma", "translate"):
        out = _run_model(key, args)
        metrics = _bleu(out, args.ref, args.limit)
        results[key] = {"label": MODELS[key]["label"], "output": str(out), **metrics}

    print("\n=== comparison (macro BLEU, self/dev) ===")
    for key in ("gemma", "translate"):
        r = results[key]
        print(
            f"{r['label']:28s}  BLEU={r['macro_bleu']:6.2f}  "
            f"n={r['n_examples']}  min/max={r['min_score']:.2f}/{r['max_score']:.2f}"
        )

    gemma_bleu = results["gemma"]["macro_bleu"]
    trans_bleu = results["translate"]["macro_bleu"]
    delta = trans_bleu - gemma_bleu
    winner = results["translate"]["label"] if delta >= 0 else results["gemma"]["label"]
    print(f"\nΔ (translate - gemma): {delta:+.2f}  →  {winner}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "limit": args.limit,
            "ref": str(args.ref),
            "models": results,
            "delta_translate_minus_gemma": round(delta, 4),
        }
        args.json_out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
