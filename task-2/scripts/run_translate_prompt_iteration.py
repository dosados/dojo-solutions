#!/usr/bin/env python3
"""Один прогон prompt-варианта для task-2 TranslateGemma submission.

Запуск из task-2/:

    conda activate ml-env
    python -u scripts/run_translate_prompt_iteration.py --iter 2 --limit 100 --gpu-only --quantize 4bit

Лог промптов: docs/PROMPT_TUNING_TRANSLATE.md
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

_TASK_ROOT = Path(__file__).resolve().parents[1]
_TASK1_ROOT = _TASK_ROOT.parent / "task-1"
if str(_TASK1_ROOT) not in sys.path:
    sys.path.insert(0, str(_TASK1_ROOT))

from eval.bleu import evaluate_pairs  # noqa: E402
from eval.io import load_references  # noqa: E402

WEIGHTS = _TASK_ROOT / "submission_translate" / "weights"
DEFAULT_INPUT = _TASK_ROOT / "data" / "processed" / "dev.input.pickle"
DEFAULT_REF = _TASK_ROOT / "data" / "processed" / "dev.jsonl"
OUT_DIR = _TASK_ROOT / "out" / "prompt_tune_translate"
MAX_NEW_TOKENS = 128
BATCH_SIZE = 40


@dataclass(frozen=True)
class PromptVariant:
    name: str
    hypothesis: str
    builder: Callable[[object, str], str | None]


def _bnb_4bit_config():
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def _model_load_kwargs(*, gpu_only: bool, quantize: str) -> dict:
    if quantize == "4bit":
        return {
            "device_map": {"": 0},
            "dtype": torch.bfloat16,
            "quantization_config": _bnb_4bit_config(),
            "attn_implementation": "sdpa",
        }
    if gpu_only:
        return {"device_map": {"": 0}, "dtype": "auto", "attn_implementation": "sdpa"}
    return {"device_map": "auto", "dtype": "auto", "attn_implementation": "sdpa"}


def _load_rows(path: Path, limit: int | None) -> list[dict]:
    with path.open("rb") as f:
        rows = pickle.load(f)
    return rows[:limit] if limit else rows


def _input_device(model) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pad_token_id(processor, model) -> int:
    tok = getattr(processor, "tokenizer", None)
    if tok is not None and tok.pad_token_id is not None:
        return int(tok.pad_token_id)
    pid = model.generation_config.pad_token_id
    if pid is not None:
        return int(pid[0] if isinstance(pid, list) else pid)
    eos = model.generation_config.eos_token_id
    if eos is None:
        raise RuntimeError("cannot determine pad/eos token")
    return int(eos[0] if isinstance(eos, list) else eos)


def _baseline_messages(src: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": "ru",
                    "target_lang_code": "ab",
                    "text": src.strip(),
                }
            ],
        }
    ]


def _baseline_prompt(processor, src: str) -> str | None:
    # None means: use the model's bundled TranslateGemma chat template.
    return None


def _turn(processor, instruction: str, src: str) -> str:
    bos = processor.tokenizer.bos_token or "<bos>"
    return (
        f"{bos}<start_of_turn>user\n"
        f"{instruction.strip()}\n\n\n"
        f"{src.strip()}"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def _ru_full(processor, src: str) -> str:
    return _turn(
        processor,
        (
            "Ты профессиональный переводчик с русского языка на абхазский. "
            "Твоя задача — точно передать смысл и оттенки исходного русского текста, "
            "соблюдая грамматику, лексику и естественный стиль абхазского языка.\n"
            "В ответе выведи только абхазский перевод, без пояснений, вариантов и комментариев. "
            "Переведи следующий русский текст на абхазский:"
        ),
        src,
    )


def _ru_short(processor, src: str) -> str:
    return _turn(
        processor,
        (
            "Переведи русский текст на абхазский язык. "
            "Ответ должен содержать только перевод на абхазский, без пояснений."
        ),
        src,
    )


def _ru_close(processor, src: str) -> str:
    return _turn(
        processor,
        (
            "Переведи с русского на абхазский. Сохраняй смысл, имена, числа и порядок фраз. "
            "Не копируй русский текст, если есть абхазский перевод. Только перевод."
        ),
        src,
    )


def _ru_native(processor, src: str) -> str:
    return _turn(
        processor,
        (
            "Ты переводчик русско-абхазского корпуса. Переведи предложение так, как оно "
            "могло бы звучать в естественном абхазском тексте. Без комментариев."
        ),
        src,
    )


def _ru_minimal_labels(processor, src: str) -> str:
    bos = processor.tokenizer.bos_token or "<bos>"
    return (
        f"{bos}<start_of_turn>user\n"
        "Русский:\n"
        f"{src.strip()}\n\n"
        "Абхазский:"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def _ab_word(processor, src: str) -> str:
    return _turn(
        processor,
        (
            "Переведи русский текст на абхазский язык (аԥсуа). "
            "Выведи только абхазский перевод, без русского текста и без пояснений."
        ),
        src,
    )


def _en_short(processor, src: str) -> str:
    return _turn(
        processor,
        (
            "Translate the Russian text into Abkhazian. "
            "Output only the Abkhazian translation, no explanations."
        ),
        src,
    )


VARIANTS: dict[int, PromptVariant] = {
    1: PromptVariant("baseline_template_en", "Current bundled English TranslateGemma template.", _baseline_prompt),
    2: PromptVariant("ru_full_template", "Russian translation of the bundled system prompt.", _ru_full),
    3: PromptVariant("ru_short", "Short Russian instruction, less prompt overhead.", _ru_short),
    4: PromptVariant("ru_close_no_copy", "Russian instruction with no-copy and preservation rules.", _ru_close),
    5: PromptVariant("ru_native_style", "Ask for natural Abkhaz corpus-like wording.", _ru_native),
    6: PromptVariant("ru_minimal_labels", "Minimal Russian/Abkhazian labels.", _ru_minimal_labels),
    7: PromptVariant("ru_abkhaz_word", "Russian instruction explicitly naming Abkhaz as аԥсуа.", _ab_word),
    8: PromptVariant("en_short_control", "Short English control without long bundled wording.", _en_short),
}


def _prepare_batch(processor, rows: list[dict], variant: PromptVariant, device: torch.device):
    prompts = []

    for row in rows:
        rendered = variant.builder(processor, row["src"])
        if rendered is None:
            encoded = processor.apply_chat_template(
                _baseline_messages(row["src"]),
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            ids = encoded["input_ids"][0]
            mask = encoded.get("attention_mask")
            if mask is None:
                mask = torch.ones_like(ids)
            else:
                mask = mask[0]
            prompts.append((ids, mask))
        else:
            tok = processor.tokenizer
            encoded = tok(rendered, return_tensors="pt", add_special_tokens=False)
            ids = encoded["input_ids"][0]
            mask = encoded["attention_mask"][0]
            prompts.append((ids, mask))

    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"
    padded = tokenizer.pad(
        {
            "input_ids": [p[0] for p in prompts],
            "attention_mask": [p[1] for p in prompts],
        },
        padding=True,
        return_tensors="pt",
    )
    return padded.to(device)


def _decode_batch(processor, rows: list[dict], input_len: int, outputs) -> list[dict]:
    results = []
    for row, out_ids in zip(rows, outputs, strict=True):
        results.append(
            {
                "rid": row["rid"],
                "translation": processor.decode(out_ids[input_len:], skip_special_tokens=True).strip(),
            }
        )
    return results


def _bleu(rows: list[dict], results: list[dict], ref_path: Path) -> tuple[dict, list]:
    refs = load_references(ref_path)
    keep = {r["rid"] for r in rows}
    refs = {rid: text for rid, text in refs.items() if rid in keep}
    hyps = {r["rid"]: r["translation"] for r in results}
    report = evaluate_pairs(hyps, refs, require_all_refs=True)
    worst = sorted(report.per_example, key=lambda x: x.score)[:8]
    return (
        {
            "macro_bleu": round(report.macro_bleu, 2),
            "n": report.n_examples,
            "min": round(report.min_score, 2),
            "max": round(report.max_score, 2),
        },
        worst,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one TranslateGemma prompt iteration")
    parser.add_argument("--iter", type=int, required=True, choices=sorted(VARIANTS))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--ref", type=Path, default=DEFAULT_REF)
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--gpu-only", action="store_true")
    parser.add_argument("--quantize", choices=("none", "4bit"), default="4bit")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    variant = VARIANTS[args.iter]
    rows = _load_rows(args.input, args.limit)

    print(f"=== iter{args.iter:02d}: {variant.name} ===", flush=True)
    print(f"hypothesis: {variant.hypothesis}", flush=True)
    print(f"n={len(rows)} weights={args.weights}", flush=True)

    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.weights, fix_mistral_regex=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.weights,
        **_model_load_kwargs(gpu_only=args.gpu_only, quantize=args.quantize),
    )
    model.eval()
    device = _input_device(model)
    pad_token_id = _pad_token_id(processor, model)
    print(f"loaded in {time.perf_counter() - t0:.1f}s device={device}", flush=True)

    ordered = sorted(rows, key=lambda r: len(r["src"]))
    results: list[dict] = []
    for start in range(0, len(ordered), args.batch_size):
        batch = ordered[start : start + args.batch_size]
        inputs = _prepare_batch(processor, batch, variant, device)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        results.extend(_decode_batch(processor, batch, input_len, outputs))

    results.sort(key=lambda x: x["rid"])
    metrics, worst = _bleu(rows, results, args.ref)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"iter{args.iter:02d}_{variant.name}"
    (OUT_DIR / f"{tag}.json").write_text(
        json.dumps(results, ensure_ascii=False),
        encoding="utf-8",
    )
    payload = {
        "iter": args.iter,
        "name": variant.name,
        "hypothesis": variant.hypothesis,
        "limit": args.limit,
        "metrics": metrics,
    }
    (OUT_DIR / f"{tag}_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"BLEU={metrics['macro_bleu']:.2f} n={metrics['n']} min={metrics['min']} max={metrics['max']}", flush=True)
    print("--- worst ---", flush=True)
    for ex in worst:
        print(f"rid={ex.rid} BLEU={ex.score:.2f}", flush=True)
        print(f"  hyp: {ex.hyp[:120]}", flush=True)
        print(f"  ref: {ex.ref[:120]}", flush=True)


if __name__ == "__main__":
    main()
