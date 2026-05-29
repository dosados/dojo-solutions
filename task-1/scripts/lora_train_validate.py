#!/usr/bin/env python3
"""Этап 1: обучение QLoRA + локальная валидация BLEU (без merge и без zip).

  cd task-1
  python -u scripts/lora_train_validate.py --config configs/lora_pilot.yaml

  # Только валидация уже обученного адаптера:
  python -u scripts/lora_train_validate.py --config configs/lora_pilot.yaml --skip-train
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

_TASK_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=_TASK_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QLoRA: train + BLEU validation (no merge, no zip)",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-validate", action="store_true")
    parser.add_argument("--val-samples", type=int, default=300)
    parser.add_argument("--self-dev", action="store_true")
    parser.add_argument("--worst", type=int, default=5)
    args = parser.parse_args()

    config_path = args.config.resolve()
    with config_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run_name = cfg["run_name"]
    adapter_dir = (_TASK_ROOT / cfg["output_dir"] / run_name / "final").resolve()

    py = sys.executable

    if not args.skip_train:
        print("=== этап: train ===", flush=True)
        _run([py, "-u", str(_TASK_ROOT / "scripts" / "train_lora.py"), "--config", str(config_path)])

    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"adapter not found: {adapter_dir}")

    if not args.skip_validate:
        print("=== этап: validate (adapter, 4-bit) ===", flush=True)
        _run(
            [
                py,
                "-u",
                str(_TASK_ROOT / "scripts" / "validate_lora.py"),
                "--adapter",
                str(adapter_dir),
                "--max-samples",
                str(args.val_samples),
                "--progress",
                "--worst",
                str(args.worst),
            ]
            + (["--self-dev"] if args.self_dev else [])
        )

    print("\n=== train + validate done ===", flush=True)
    print(f"adapter: {adapter_dir}", flush=True)
    print("scores:  out/lora_val/scores.json", flush=True)
    print(
        "\nДальше (отдельные этапы):\n"
        f"  merge:  python -u scripts/merge_lora.py --adapter {adapter_dir} --device cpu --backup\n"
        "  check:  python -u scripts/validate_lora.py --merged --max-samples 300 --skip-baseline --progress\n"
        "  zip:    ./scripts/zip-submission.sh",
        flush=True,
    )


if __name__ == "__main__":
    main()
