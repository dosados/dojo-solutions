#!/usr/bin/env python3
"""Локальный BLEU: свои абзацы (self/) и OPUS-100 (предложения).

Только метрика (переводы уже есть):

    conda activate ml-env
    cd task-1
    python -u scripts/eval_local.py bleu --preset self-dev --hyp out/dev_output.json
    python -u scripts/eval_local.py bleu --preset opus-val --hyp out/opus_val.json --limit 500

Inference + BLEU (Gemma или TranslateGemma):

    python -u scripts/eval_local.py run --preset self-dev --model gemma --gpu-only --quantize 4bit
    python -u scripts/eval_local.py run --preset opus-val --model translate --limit 300

Несколько наборов подряд:

    python -u scripts/eval_local.py run --suite self --model gemma --gpu-only
    python -u scripts/eval_local.py bleu --suite all --hyp-template 'out/{preset}.json'

Перед первым запуском:

    python scripts/convert_datasets.py          # self + opus → data/processed/
    python scripts/convert_datasets.py --skip-opus-train   # без 1M train
    # OPUS с HF: scripts/download_opus100.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_TASK_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
if str(_TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(_TASK_ROOT))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from eval.bleu import evaluate_pairs  # noqa: E402
from eval.io import load_hypotheses, load_references  # noqa: E402

_PROCESSED = _TASK_ROOT / "data" / "processed"


@dataclass(frozen=True)
class Benchmark:
    name: str
    label: str
    ref: Path
    input: Path | None
    exclude_articles: tuple[str, ...] = ()
    default_limit: int | None = None  # для opus — ускорить прогон по умолчанию


PRESETS: dict[str, Benchmark] = {
    "self-dev": Benchmark(
        name="self-dev",
        label="self/dev — абзацы, статьи 1–4 (117)",
        ref=_PROCESSED / "self" / "dev.jsonl",
        input=_PROCESSED / "self" / "dev.input.pickle",
    ),
    "self-dev-it": Benchmark(
        name="self-dev-it",
        label="self/dev_it — без Пушкина (article-4)",
        ref=_PROCESSED / "self" / "dev_it.jsonl",
        input=_PROCESSED / "self" / "dev_it.input.pickle",
    ),
    "self-heldout": Benchmark(
        name="self-heldout",
        label="self/heldout — одна статья (финальный val)",
        ref=_PROCESSED / "self" / "heldout.jsonl",
        input=_PROCESSED / "self" / "heldout.input.pickle",
    ),
    "opus-val": Benchmark(
        name="opus-val",
        label="OPUS-100 validation — предложения (2000)",
        ref=_PROCESSED / "opus100" / "validation.jsonl",
        input=_PROCESSED / "opus100" / "validation.input.pickle",
        default_limit=500,
    ),
    "opus-test": Benchmark(
        name="opus-test",
        label="OPUS-100 test — предложения (2000)",
        ref=_PROCESSED / "opus100" / "test.jsonl",
        input=_PROCESSED / "opus100" / "test.input.pickle",
        default_limit=500,
    ),
}

SUITES: dict[str, list[str]] = {
    "self": ["self-dev", "self-dev-it", "self-heldout"],
    "opus": ["opus-val", "opus-test"],
    "all": ["self-dev", "self-dev-it", "opus-val"],
}


def _resolve_presets(names: list[str]) -> list[Benchmark]:
    out: list[Benchmark] = []
    for name in names:
        if name not in PRESETS:
            raise KeyError(f"unknown preset {name!r}; choose from: {', '.join(PRESETS)}")
        out.append(PRESETS[name])
    return out


def _expand_suite(suite: str) -> list[str]:
    if suite not in SUITES:
        raise KeyError(f"unknown suite {suite!r}; choose from: {', '.join(SUITES)}")
    return SUITES[suite]


def _default_hyp_path(preset: str, model: str | None) -> Path:
    tag = model or "hyp"
    return _TASK_ROOT / "out" / f"{tag}_{preset.replace('-', '_')}.json"


def _limit_for(preset: Benchmark, limit: int | None) -> int | None:
    if limit is not None:
        return limit
    return preset.default_limit


def _run_bleu(
    preset: Benchmark,
    hyp_path: Path,
    *,
    limit: int | None,
    worst: int,
    show_text: bool,
    json_out: Path | None,
) -> dict:
    if not preset.ref.is_file():
        raise FileNotFoundError(
            f"ref not found: {preset.ref}\n  run: python scripts/convert_datasets.py"
        )
    if not hyp_path.is_file():
        raise FileNotFoundError(f"hyp not found: {hyp_path}")

    refs = load_references(preset.ref, exclude_articles=preset.exclude_articles or None)
    hyps = load_hypotheses(hyp_path)

    eff_limit = _limit_for(preset, limit)
    if eff_limit is not None:
        keep = set(range(eff_limit))
        refs = {rid: t for rid, t in refs.items() if rid in keep}
        hyps = {rid: t for rid, t in hyps.items() if rid in keep}

    report = evaluate_pairs(hyps, refs, require_all_refs=True)

    print(f"\n=== {preset.label} ===")
    print(f"  hyp: {hyp_path}")
    print(f"  ref: {preset.ref}")
    if eff_limit is not None:
        print(f"  limit: first {eff_limit} rids")
    print(f"  sacrebleu {report.sacrebleu_version}")
    print(f"  macro BLEU: {report.macro_bleu:.2f}  (n={report.n_examples})")
    print(f"  min / max:  {report.min_score:.2f} / {report.max_score:.2f}")

    if worst > 0:
        ranked = sorted(report.per_example, key=lambda x: x.score)
        print(f"\n  --- worst {min(worst, len(ranked))} ---")
        for ex in ranked[:worst]:
            print(f"  rid={ex.rid}  BLEU={ex.score:.2f}")
            if show_text:
                hyp_prev = ex.hyp[:100] + ("..." if len(ex.hyp) > 100 else "")
                ref_prev = ex.ref[:100] + ("..." if len(ex.ref) > 100 else "")
                print(f"    hyp: {hyp_prev}")
                print(f"    ref: {ref_prev}")

    result = {
        "preset": preset.name,
        "label": preset.label,
        "hyp": str(hyp_path),
        "ref": str(preset.ref),
        "limit": eff_limit,
        "macro_bleu": round(report.macro_bleu, 4),
        "n_examples": report.n_examples,
        "min_score": round(report.min_score, 4),
        "max_score": round(report.max_score, 4),
    }

    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        detail = {
            **result,
            "per_example": [
                {"rid": ex.rid, "bleu": round(ex.score, 4)}
                for ex in report.per_example
            ],
        }
        json_out.write_text(
            json.dumps(detail, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  wrote {json_out}")

    return result


def _run_inference(
    preset: Benchmark,
    *,
    model: str,
    output: Path,
    limit: int | None,
    gpu_only: bool,
    quantize: str,
    batch_size: int | None,
) -> None:
    if preset.input is None or not preset.input.is_file():
        raise FileNotFoundError(
            f"input not found: {preset.input}\n  run: python scripts/convert_datasets.py"
        )

    cmd = [
        sys.executable,
        "-u",
        str(_SCRIPTS / "run_local_eval.py"),
        "--model",
        model,
        "--input",
        str(preset.input),
        "--ref",
        str(preset.ref),
        "--output",
        str(output),
        "--skip-bleu",
    ]
    eff_limit = _limit_for(preset, limit)
    if eff_limit is not None:
        cmd.extend(["--limit", str(eff_limit)])
    if gpu_only:
        cmd.append("--gpu-only")
    if quantize != "none":
        cmd.extend(["--quantize", quantize])
    if batch_size is not None:
        cmd.extend(["--batch-size", str(batch_size)])

    print(f"\n>>> inference [{preset.name}] → {output}", flush=True)
    subprocess.run(cmd, check=True)


def cmd_bleu(args: argparse.Namespace) -> None:
    presets = _resolve_presets(args.preset)
    results = []
    for preset in presets:
        if args.hyp:
            hyp = Path(args.hyp)
        elif args.hyp_template:
            hyp = Path(args.hyp_template.format(preset=preset.name, model=args.model or ""))
        else:
            hyp = _default_hyp_path(preset.name, args.model)
        jout = None
        if args.json_out:
            jout = args.json_out.parent / f"{args.json_out.stem}_{preset.name}{args.json_out.suffix}"
        results.append(
            _run_bleu(
                preset,
                hyp,
                limit=args.limit,
                worst=args.worst,
                show_text=args.show_text,
                json_out=jout,
            )
        )

    if len(results) > 1:
        print("\n=== summary ===")
        for r in results:
            lim = f" (n≤{r['limit']})" if r["limit"] else ""
            print(f"  {r['preset']:16s}  BLEU={r['macro_bleu']:6.2f}{lim}")


def cmd_run(args: argparse.Namespace) -> None:
    if args.preset:
        names = args.preset
    elif args.suite:
        names = _expand_suite(args.suite)
    else:
        raise SystemExit("укажите --preset или --suite")

    presets = _resolve_presets(names)
    summary = []

    for preset in presets:
        output = (
            Path(args.output)
            if args.output and len(presets) == 1
            else _default_hyp_path(preset.name, args.model)
        )
        _run_inference(
            preset,
            model=args.model,
            output=output,
            limit=args.limit,
            gpu_only=args.gpu_only,
            quantize=args.quantize,
            batch_size=args.batch_size,
        )
        summary.append(
            _run_bleu(
                preset,
                output,
                limit=args.limit,
                worst=args.worst,
                show_text=args.show_text,
                json_out=(
                    args.json_out.parent / f"{args.json_out.stem}_{preset.name}{args.json_out.suffix}"
                    if args.json_out
                    else None
                ),
            )
        )

    if len(summary) > 1:
        print("\n=== summary ===")
        for r in summary:
            print(f"  {r['preset']:16s}  BLEU={r['macro_bleu']:6.2f}  →  {r['hyp']}")

    if args.json_summary:
        args.json_summary.parent.mkdir(parents=True, exist_ok=True)
        args.json_summary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nWrote {args.json_summary}")


def cmd_list(_args: argparse.Namespace) -> None:
    print("Presets:")
    for name, p in PRESETS.items():
        lim = f", default --limit {p.default_limit}" if p.default_limit else ""
        print(f"  {name:16s}  {p.label}{lim}")
        print(f"                  ref: {p.ref.relative_to(_TASK_ROOT)}")
    print("\nSuites:")
    for name, presets in SUITES.items():
        print(f"  {name:16s}  {', '.join(presets)}")


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--preset",
        nargs="+",
        choices=sorted(PRESETS),
        metavar="PRESET",
        help=f"Один или несколько: {', '.join(PRESETS)}",
    )
    p.add_argument(
        "--suite",
        choices=sorted(SUITES),
        help="Группа пресетов: self | opus | all",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Только первые N rid (для opus по умолчанию 500, если не задано)",
    )
    p.add_argument("--worst", type=int, default=5)
    p.add_argument("--show-text", action="store_true")
    p.add_argument("--json-out", type=Path, default=None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BLEU на self/dev и OPUS-100 (task-1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_bleu = sub.add_parser("bleu", help="Только BLEU по готовому output.json")
    _add_common(p_bleu)
    p_bleu.add_argument(
        "--hyp",
        type=Path,
        help="Один файл переводов для всех --preset (иначе out/{model}_{preset}.json)",
    )
    p_bleu.add_argument(
        "--hyp-template",
        type=str,
        help="Шаблон пути, напр. 'out/{model}_{preset}.json'",
    )
    p_bleu.add_argument(
        "--model",
        choices=("gemma", "translate"),
        help="Для выбора пути out/{model}_*.json без --hyp",
    )
    p_bleu.set_defaults(func=cmd_bleu)

    p_run = sub.add_parser("run", help="Inference + BLEU")
    _add_common(p_run)
    p_run.add_argument(
        "--model",
        choices=("gemma", "translate"),
        default="gemma",
    )
    p_run.add_argument("--output", type=Path, default=None, help="Только при одном --preset")
    p_run.add_argument("--gpu-only", action="store_true")
    p_run.add_argument("--quantize", choices=("none", "4bit"), default="none")
    p_run.add_argument("--batch-size", type=int, default=None)
    p_run.add_argument(
        "--json-summary",
        type=Path,
        default=None,
        help="Сводка по всем пресетам в один JSON",
    )
    p_run.set_defaults(func=cmd_run)

    p_list = sub.add_parser("list", help="Список пресетов и путей")
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
