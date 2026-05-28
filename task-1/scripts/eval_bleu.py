#!/usr/bin/env python3
"""Локальный macro BLEU (sacrebleu sentence_bleu), как на контесте.

Примеры:
  # output.json vs эталоны
  conda run -n ml-env python scripts/eval_bleu.py \\
    --hyp submission/output.json \\
    --ref data/val.jsonl

  # два текстовых файла (строка = один пример, порядок важен)
  conda run -n ml-env python scripts/eval_bleu.py \\
    --hyp runs/hyp.txt --ref data/val.ref.txt

  # только эталоны (проверка dev-набора: ref vs ref → 100)
  conda run -n ml-env python scripts/eval_bleu.py --ref data/val.jsonl --sanity-ref
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# task-1/ в PYTHONPATH при запуске из task-1/
_TASK_ROOT = Path(__file__).resolve().parents[1]
if str(_TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(_TASK_ROOT))

from eval.bleu import evaluate_pairs  # noqa: E402
from eval.io import load_hypotheses, load_references  # noqa: E402


def _print_report(report, *, worst_k: int, show_text: bool) -> None:
    print(f"sacrebleu {report.sacrebleu_version}")
    print(f"examples: {report.n_examples}")
    print(f"macro BLEU: {report.macro_bleu:.2f}")
    print(f"min / max:  {report.min_score:.2f} / {report.max_score:.2f}")

    if worst_k > 0:
        ranked = sorted(report.per_example, key=lambda x: x.score)
        print(f"\n--- worst {min(worst_k, len(ranked))} ---")
        for ex in ranked[:worst_k]:
            print(f"rid={ex.rid}  BLEU={ex.score:.2f}")
            if show_text:
                hyp_prev = ex.hyp[:120] + ("..." if len(ex.hyp) > 120 else "")
                ref_prev = ex.ref[:120] + ("..." if len(ex.ref) > 120 else "")
                print(f"  hyp: {hyp_prev}")
                print(f"  ref: {ref_prev}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Macro-average BLEU (contest-style)")
    parser.add_argument(
        "--hyp",
        type=Path,
        help="Переводы: output.json, .jsonl или .txt (по строке)",
    )
    parser.add_argument(
        "--ref",
        type=Path,
        required=True,
        help="Эталоны: .jsonl / .json ({rid, ref}) или .txt",
    )
    parser.add_argument(
        "--ref-key",
        default=None,
        help="Поле эталона в JSON (по умолчанию: ref, иначе translation, tgt)",
    )
    parser.add_argument(
        "--worst",
        type=int,
        default=5,
        help="Сколько худших примеров показать (0 = не показывать)",
    )
    parser.add_argument(
        "--show-text",
        action="store_true",
        help="Печатать фрагменты hyp/ref у худших примеров",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Сохранить per-rid scores в JSON",
    )
    parser.add_argument(
        "--allow-missing-hyp",
        action="store_true",
        help="Не требовать перевод для каждого rid из ref",
    )
    parser.add_argument(
        "--sanity-ref",
        action="store_true",
        help="Сравнить ref с самим собой (ожидается BLEU=100)",
    )
    args = parser.parse_args()

    refs = load_references(args.ref, ref_key=args.ref_key)

    if args.sanity_ref:
        hyps = refs
    elif args.hyp is None:
        parser.error("--hyp is required unless --sanity-ref")
    else:
        hyps = load_hypotheses(args.hyp)

    report = evaluate_pairs(
        hyps,
        refs,
        require_all_refs=not args.allow_missing_hyp,
    )
    _print_report(report, worst_k=args.worst, show_text=args.show_text)

    if args.json_out:
        payload = [
            {"rid": ex.rid, "bleu": round(ex.score, 4)}
            for ex in report.per_example
        ]
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(
                {
                    "macro_bleu": round(report.macro_bleu, 4),
                    "n_examples": report.n_examples,
                    "sacrebleu_version": report.sacrebleu_version,
                    "per_example": payload,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
