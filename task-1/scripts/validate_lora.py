#!/usr/bin/env python3
"""Инференс + macro BLEU: base vs LoRA (adapter + 4-bit).

Пример:
  cd task-1
  python -u scripts/validate_lora.py --adapter runs/lora/pilot_10k/final
  python -u scripts/validate_lora.py --merged   # после merge_lora (полные веса, 4-bit)
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


def _run_inference(
    *,
    input_pickle: Path,
    output_json: Path,
    adapter: Path | None,
    merged: bool,
    device: str,
    progress: bool,
) -> None:
    cmd = [
        sys.executable,
        "-u",
        str(_TASK_ROOT / "submission" / "solution.py"),
        "--input",
        str(input_pickle),
        "--output",
        str(output_json),
        "--device",
        device,
        "--quantize",
        "4bit",
    ]
    if progress:
        cmd.append("--progress")
    if adapter is not None:
        cmd.extend(["--lora-dir", str(adapter)])
    if merged:
        cmd.append("--no-lora")

    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(_TASK_ROOT / "submission"), check=True)


def _bleu(hyp_path: Path, ref_path: Path, *, worst: int, show_text: bool) -> float:
    refs = load_references(ref_path)
    hyps = load_hypotheses(hyp_path)
    report = evaluate_pairs(hyps, refs)
    print(f"\n=== {hyp_path.name} ===", flush=True)
    print(f"macro BLEU: {report.macro_bleu:.2f} (n={report.n_examples})", flush=True)
    if worst > 0:
        ranked = sorted(report.per_example, key=lambda x: x.score)
        print(f"--- worst {min(worst, len(ranked))} ---", flush=True)
        for ex in ranked[:worst]:
            print(f"rid={ex.rid}  BLEU={ex.score:.2f}", flush=True)
            if show_text:
                hyp_prev = ex.hyp[:100] + ("..." if len(ex.hyp) > 100 else "")
                ref_prev = ex.ref[:100] + ("..." if len(ex.ref) > 100 else "")
                print(f"  hyp: {hyp_prev}", flush=True)
                print(f"  ref: {ref_prev}", flush=True)
    return report.macro_bleu


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate LoRA vs baseline BLEU")
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="Путь к …/final (adapter). Если не задан — только baseline.",
    )
    parser.add_argument(
        "--merged",
        action="store_true",
        help="Прогон без adapter (уже merged в submission/weights)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=_TASK_ROOT / "data/processed/opus100/validation.input.pickle",
        help="Pickle для solution.py (по умолчанию OPUS val 2k)",
    )
    parser.add_argument(
        "--ref",
        type=Path,
        default=_TASK_ROOT / "data/processed/opus100/validation.jsonl",
    )
    parser.add_argument(
        "--self-dev",
        action="store_true",
        help="Дополнительно self/dev (117 абзацев)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=300,
        help="Урезать OPUS val для быстрой проверки (0 = все)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_TASK_ROOT / "out" / "lora_val",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--worst", type=int, default=5)
    parser.add_argument("--show-text", action="store_true")
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Не гонять baseline inference; если есть out/lora_val/baseline.json — только BLEU",
    )
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    input_pickle = args.input.resolve()
    ref_path = args.ref.resolve()

    if args.max_samples > 0 and input_pickle.suffix == ".pickle":
        import pickle

        with input_pickle.open("rb") as f:
            rows = pickle.load(f)
        rows = rows[: args.max_samples]
        subset_pickle = out_dir / "val_subset.input.pickle"
        with subset_pickle.open("wb") as f:
            pickle.dump(rows, f)
        input_pickle = subset_pickle
        refs = load_references(ref_path)
        subset_ref = out_dir / "val_subset.jsonl"
        with subset_ref.open("w", encoding="utf-8") as f:
            for row in rows:
                rid = int(row["rid"])
                if rid in refs:
                    f.write(
                        json.dumps({"rid": rid, "ref": refs[rid]}, ensure_ascii=False)
                        + "\n"
                    )
        ref_path = subset_ref
        print(f"subset: {len(rows)} examples", flush=True)

    scores: dict[str, float] = {}

    hyp_base = out_dir / "baseline.json"
    if not args.merged and (not args.skip_baseline or hyp_base.is_file()):
        if args.skip_baseline and hyp_base.is_file():
            print("\n>>> baseline BLEU (cached baseline.json)", flush=True)
        elif not args.skip_baseline:
            print("\n>>> baseline (no LoRA)", flush=True)
            _run_inference(
                input_pickle=input_pickle,
                output_json=hyp_base,
                adapter=None,
                merged=False,
                device=args.device,
                progress=args.progress,
            )
        if hyp_base.is_file():
            scores["baseline"] = _bleu(
                hyp_base, ref_path, worst=args.worst, show_text=args.show_text
            )
        elif args.skip_baseline:
            print(f"warning: --skip-baseline but missing {hyp_base}", flush=True)

    if args.adapter or args.merged:
        tag = "merged" if args.merged else args.adapter.name
        hyp_lora = out_dir / f"lora_{tag}.json"
        print(f"\n>>> LoRA ({tag})", flush=True)
        _run_inference(
            input_pickle=input_pickle,
            output_json=hyp_lora,
            adapter=args.adapter.resolve() if args.adapter else None,
            merged=args.merged,
            device=args.device,
            progress=args.progress,
        )
        scores["lora"] = _bleu(hyp_lora, ref_path, worst=args.worst, show_text=args.show_text)

    if args.self_dev:
        self_in = _TASK_ROOT / "data/processed/self/dev.input.pickle"
        self_ref = _TASK_ROOT / "data/processed/self/dev.jsonl"
        if self_in.is_file():
            hyp_self = out_dir / "self_dev_lora.json"
            print("\n>>> self/dev", flush=True)
            _run_inference(
                input_pickle=self_in,
                output_json=hyp_self,
                adapter=args.adapter.resolve() if args.adapter and not args.merged else None,
                merged=args.merged,
                device=args.device,
                progress=args.progress,
            )
            scores["self_dev"] = _bleu(
                hyp_self, self_ref, worst=args.worst, show_text=args.show_text
            )

    summary_path = out_dir / "scores.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)
    print(f"\nsummary: {summary_path}", flush=True)
    for k, v in scores.items():
        print(f"  {k}: {v:.2f}", flush=True)


if __name__ == "__main__":
    main()
