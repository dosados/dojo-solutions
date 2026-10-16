#!/usr/bin/env python3
"""Сравнить Gemma-4-E2B и TranslateGemma-4B на одном dev-наборе task-2.

Запуск из task-2/:

    conda activate ml-env
    python -u scripts/compare_models.py --limit 50 --quantize 4bit --rerun
    # на ~8GB GPU: --gpu-only автоматически переключится на 4bit
    python -u scripts/compare_models.py --prepare-dev
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
_TASK1_ROOT = _TASK_ROOT.parent / "task-1"

from train.paths import resolve_weights_dir  # noqa: E402

MODELS = {
    "gemma": {
        "label": "Gemma-4-E2B",
        "submission": _TASK_ROOT / "submission",
        "output": _TASK_ROOT / "out" / "gemma_dev_output.json",
    },
    "translate": {
        "label": "TranslateGemma-4B",
        "submission": _TASK_ROOT / "submission_translate",
        "output": _TASK_ROOT / "out" / "translategemma_dev_output.json",
    },
    "nllb": {
        "label": "NLLB-3.3B",
        "submission": _TASK_ROOT / "submission_nllb",
        "output": _TASK_ROOT / "out" / "nllb_dev_output.json",
    },
}


def _resolve_weights(model_key: str) -> Path:
    return resolve_weights_dir(model_key)


def _bleu(output: Path, ref: Path, limit: int | None) -> dict:
    if str(_TASK1_ROOT) not in sys.path:
        sys.path.insert(0, str(_TASK1_ROOT))
    from eval.bleu import evaluate_pairs
    from eval.io import load_hypotheses, load_references

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
    weights = _resolve_weights(model_key)
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
        str(weights),
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
    print(f"[{info['label']}] inference …", flush=True)
    subprocess.run(cmd, check=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Gemma vs TranslateGemma on task-2 dev")
    parser.add_argument("--prepare-dev", action="store_true")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--input",
        type=Path,
        default=_TASK_ROOT / "data" / "processed" / "dev.input.pickle",
    )
    parser.add_argument(
        "--ref",
        type=Path,
        default=_TASK_ROOT / "data" / "processed" / "dev.jsonl",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        default=True,
        help="Не пересчитывать, если output уже есть (default: true)",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Пересчитать обе модели даже если output уже есть",
    )
    parser.add_argument(
        "--gpu-only",
        action="store_true",
        help="Полные веса на GPU; на ≤8.5 GiB VRAM автоматически 4bit",
    )
    parser.add_argument(
        "--quantize",
        choices=("none", "4bit"),
        default="none",
        help="4bit NF4 (bitsandbytes), рекомендуется на ~8GB GPU",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=_TASK_ROOT / "out" / "compare_models.json",
    )
    args = parser.parse_args()

    if args.prepare_dev:
        subprocess.run(
            [sys.executable, "-u", str(_TASK_ROOT / "scripts" / "prepare_dev.py")],
            check=True,
        )

    if not args.input.is_file() or not args.ref.is_file():
        raise FileNotFoundError("dev not found — run with --prepare-dev first")

    results = {}
    model_keys = tuple(MODELS)
    for key in model_keys:
        out = _run_model(key, args)
        metrics = _bleu(out, args.ref, args.limit)
        results[key] = {
            "label": MODELS[key]["label"],
            "output": str(out),
            **metrics,
        }

    print("\n=== comparison (macro BLEU) ===")
    for key in model_keys:
        r = results[key]
        print(
            f"{r['label']:20s}  BLEU={r['macro_bleu']:6.2f}  "
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
