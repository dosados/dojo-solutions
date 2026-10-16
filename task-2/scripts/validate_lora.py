#!/usr/bin/env python3
"""Inference с LoRA-адаптером + macro BLEU на dev (task-2).

Запуск из task-2/:

    conda activate ml-env
    python -u scripts/prepare_parallel.py   # если ещё нет dev
    python -u scripts/validate_lora.py --model translate --adapter runs/lora/translate/sft_pilot/final
    python -u scripts/validate_lora.py --model gemma --adapter runs/lora/gemma/sft_pilot/final --limit 50
    python -u scripts/validate_lora.py --model nllb --adapter runs/lora/nllb/sft_pilot/final --limit 50
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

_TASK_ROOT = Path(__file__).resolve().parents[1]
if str(_TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(_TASK_ROOT))
_TASK1_ROOT = _TASK_ROOT.parent / "task-1"

from train.nllb_langs import nllb_generate_kwargs, prepare_nllb_tokenizer  # noqa: E402
from train.paths import resolve_weights_dir  # noqa: E402

DEFAULT_INPUT = _TASK_ROOT / "data" / "processed" / "dev.input.pickle"
DEFAULT_REF = _TASK_ROOT / "data" / "processed" / "dev.jsonl"


def _load_rows(path: Path, limit: int | None) -> list[dict]:
    with path.open("rb") as f:
        rows = pickle.load(f)
    return rows[:limit] if limit else rows


def _run_with_adapter(
    model_key: str,
    rows: list[dict],
    *,
    base_dir: Path,
    adapter_dir: Path,
    quantize: str,
) -> list[dict]:
    """Load base+adapter once, run greedy decode (same logic as local_inference)."""
    if str(_TASK_ROOT) not in sys.path:
        sys.path.insert(0, str(_TASK_ROOT))

    from train.gemma_peft import patch_peft_gemma4_clippable_linear  # noqa: E402
    from train.load_model import load_processor, load_quantized_base  # noqa: E402

    if model_key == "gemma":
        patch_peft_gemma4_clippable_linear()

    model = load_quantized_base(model_key, base_dir)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    processor = load_processor(model_key, base_dir)

    device = next(p.device for p in model.parameters() if p.device.type != "meta")
    pad_token_id = None
    if model_key == "translate":
        from local_inference import _translate_pad_token_id  # noqa: E402

        pad_token_id = _translate_pad_token_id(processor, model)

    if model_key == "nllb":
        forced_bos = prepare_nllb_tokenizer(processor)
        gen_kw = nllb_generate_kwargs(forced_bos, max_new_tokens=128, num_beams=4)
        results = []
        for row in rows:
            inputs = processor(
                row["src"],
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(device)
            with torch.inference_mode():
                outputs = model.generate(**inputs, **gen_kw)
            translation = processor.decode(outputs[0], skip_special_tokens=True).strip()
            results.append({"rid": row["rid"], "translation": translation})
        return results

    from train.prompts import build_gemma_user_content  # noqa: E402
    from train.tokenize_sft import _build_translate_messages  # noqa: E402

    results = []
    for row in rows:
        src = row["src"]
        if model_key == "translate":
            messages = _build_translate_messages(src, None)
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(device)
            input_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=pad_token_id,
                )
            translation = processor.decode(
                outputs[0][input_len:], skip_special_tokens=True
            ).strip()
        else:
            user = build_gemma_user_content(src)
            prompt = processor.apply_chat_template(
                [{"role": "user", "content": user}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = processor(text=prompt, return_tensors="pt").to(device)
            input_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                outputs = model.generate(**inputs, max_new_tokens=1024)
            response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
            translation = processor.parse_response(response)["content"]

        results.append({"rid": row["rid"], "translation": translation})
    return results


def _bleu(output: Path, ref: Path, limit: int | None) -> None:
    if str(_TASK1_ROOT) not in sys.path:
        sys.path.insert(0, str(_TASK1_ROOT))
    from eval.bleu import evaluate_pairs  # noqa: E402
    from eval.io import load_hypotheses, load_references  # noqa: E402

    refs = load_references(ref)
    hyps = load_hypotheses(output)
    if limit is not None:
        keep = set(range(limit))
        refs = {k: v for k, v in refs.items() if k in keep}
        hyps = {k: v for k, v in hyps.items() if k in keep}
    report = evaluate_pairs(hyps, refs, require_all_refs=True)
    print(f"macro BLEU: {report.macro_bleu:.2f}  n={report.n_examples}", flush=True)
    print(f"min/max: {report.min_score:.2f} / {report.max_score:.2f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate LoRA adapter on task-2 dev")
    parser.add_argument("--model", choices=("gemma", "translate", "nllb"), required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=None)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--ref", type=Path, default=DEFAULT_REF)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-bleu", action="store_true")
    args = parser.parse_args()

    base = resolve_weights_dir(args.model, args.base).resolve()
    adapter = args.adapter.resolve()

    if not args.input.is_file():
        raise FileNotFoundError(
            f"missing {args.input} — run: python -u scripts/prepare_dev.py"
        )

    rows = _load_rows(args.input, args.limit)
    out = args.output or (
        _TASK_ROOT / "out" / f"lora_{args.model}_dev_output.json"
    )

    t0 = time.perf_counter()
    print(f"[{args.model}] adapter inference: {len(rows)} examples", flush=True)
    results = _run_with_adapter(
        args.model, rows, base_dir=base, adapter_dir=adapter, quantize="4bit"
    )
    elapsed = time.perf_counter() - t0

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out} in {elapsed:.1f}s", flush=True)

    if not args.skip_bleu:
        if not args.ref.is_file():
            print(f"warning: no ref {args.ref}, skip BLEU", flush=True)
        else:
            _bleu(out, args.ref, args.limit)


if __name__ == "__main__":
    main()
