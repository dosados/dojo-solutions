"""Бейзлайн-решение: gemma-4-E2B-it

На входе:  /workspace/input.pickle (или --input)
На выходе: /workspace/output.json (или --output)
Веса:      /workspace/weights (скачать: task-1/download_weights.py → submission/weights/)
"""
import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Iterator

import torch
from transformers import AutoProcessor, AutoModelForCausalLM

from postprocess import postprocess_translation
from prompts import build_translation_prompt


MODEL_DIR = "./weights"
MAX_NEW_TOKENS_DEFAULT = 1536
MAX_NEW_TOKENS_CAP = 2048
MIN_NEW_TOKENS = 384


def estimate_max_new_tokens(src: str) -> int:
    """Бюджет генерации от длины абзаца (тест до ~2229 символов EN)."""
    n = len(src)
    if n < 80:
        return 512
    if n < 300:
        return 768
    budget = int(n * 0.55) + 384
    return max(MIN_NEW_TOKENS, min(MAX_NEW_TOKENS_CAP, budget))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EN→RU translation for contest input.pickle")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("input.pickle"),
        help="Путь к input.pickle (по умолчанию: ./input.pickle в cwd)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output.json"),
        help="Путь к output.json (по умолчанию: ./output.json в cwd)",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Прогресс-бар и ETA по абзацам (локальная отладка)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help=(
            "auto: device_map=auto (на 8GB VRAM часть весов на CPU); "
            "cuda: вся модель на GPU (нужно ~8+ GiB свободно или --quantize 4bit)"
        ),
    )
    parser.add_argument(
        "--quantize",
        choices=("none", "4bit", "8bit"),
        default="none",
        help="Локально на 8GB: --device cuda --quantize 4bit (нужен pip install bitsandbytes)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help=(
            f"Фиксированный лимит генерации для всех абзацев "
            f"(по умолчанию: динамически, потолок {MAX_NEW_TOKENS_CAP})"
        ),
    )
    parser.add_argument(
        "--no-dynamic-tokens",
        action="store_true",
        help=f"Не масштабировать лимит по длине src; использовать {MAX_NEW_TOKENS_DEFAULT}",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=1,
        metavar="N",
        help="Beam search: N>1 — beam, N=1 — greedy (по умолчанию: 1)",
    )
    parser.add_argument(
        "--length-penalty",
        type=float,
        default=1.0,
        help="Length penalty для beam search (только при --num-beams > 1)",
    )
    return parser.parse_args()


def _build_generate_kwargs(
    max_tokens: int,
    *,
    num_beams: int,
    length_penalty: float,
) -> dict:
    kw: dict = {"max_new_tokens": max_tokens}
    if num_beams > 1:
        kw["num_beams"] = num_beams
        kw["length_penalty"] = length_penalty
        kw["early_stopping"] = True
    return kw


def _bnb_config(quantize: str):
    from transformers import BitsAndBytesConfig

    if quantize == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    if quantize == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError(quantize)


def _gpu_free_gib() -> float | None:
    if not torch.cuda.is_available():
        return None
    free, _total = torch.cuda.mem_get_info(0)
    return free / 2**30


def _load_model(
    device_mode: str,
    *,
    quantize: str,
) -> tuple[AutoModelForCausalLM, torch.device]:
    if quantize != "none" and device_mode == "cpu":
        raise RuntimeError("--quantize с --device cpu не поддерживается")

    if device_mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda: CUDA недоступна")
        device = torch.device("cuda:0")
        free_gib = _gpu_free_gib()
        if quantize == "none" and free_gib is not None and free_gib < 8.0:
            print(
                f"  warning: свободно VRAM ~{free_gib:.1f} GiB — bf16 целиком (~7.3 GiB) "
                f"может не влезть; используйте --quantize 4bit",
                flush=True,
            )
        load_kw: dict = {"device_map": {"": 0}}
        if quantize != "none":
            load_kw["quantization_config"] = _bnb_config(quantize)
            load_kw["dtype"] = torch.bfloat16
        else:
            load_kw["dtype"] = torch.bfloat16
        try:
            model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, **load_kw)
        except torch.OutOfMemoryError as exc:
            raise RuntimeError(
                "CUDA OOM при загрузке полной модели. "
                "На ~8GB GPU: --device cuda --quantize 4bit "
                "(pip install bitsandbytes) или --device auto --max-new-tokens 512"
            ) from exc
    elif device_mode == "cpu":
        device = torch.device("cpu")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            dtype="auto",
            device_map="auto",
        )
        device = _infer_model_device(model)

    _report_device_placement(model, device_mode, quantize=quantize)
    return model, device


def _infer_model_device(model: AutoModelForCausalLM) -> torch.device:
    devices = {p.device for p in model.parameters()}
    if len(devices) == 1:
        return next(iter(devices))
    if any(d.type == "cuda" for d in devices):
        return torch.device("cuda:0")
    return torch.device("cpu")


def _report_device_placement(
    model: AutoModelForCausalLM,
    device_mode: str,
    *,
    quantize: str,
) -> None:
    on_cuda = sum(p.numel() for p in model.parameters() if p.device.type == "cuda")
    on_cpu = sum(p.numel() for p in model.parameters() if p.device.type == "cpu")
    on_meta = sum(p.numel() for p in model.parameters() if p.device.type == "meta")
    total = on_cuda + on_cpu + on_meta
    if total == 0:
        return
    print(
        f"  params: cuda={100 * on_cuda / total:.1f}% "
        f"cpu={100 * on_cpu / total:.1f}% meta={100 * on_meta / total:.1f}%",
        flush=True,
    )
    if quantize != "none":
        print(f"  quantization: {quantize}", flush=True)
    if device_mode == "auto" and on_cpu + on_meta > 0:
        print(
            "  hint: offload на CPU → медленно на длинных абзацах; "
            "на 8GB: --device cuda --quantize 4bit",
            flush=True,
        )


def _tokens_for_paragraph(
    src: str,
    *,
    fixed: int | None,
    dynamic: bool,
) -> int:
    if fixed is not None:
        return fixed
    if dynamic:
        return estimate_max_new_tokens(src)
    return MAX_NEW_TOKENS_DEFAULT


def _iter_rows(rows: list, *, progress: bool) -> Iterator[dict]:
    if not progress:
        yield from rows
        return
    try:
        from tqdm import tqdm
    except ImportError:
        print("warning: tqdm not installed, --progress ignored", flush=True)
        yield from rows
        return
    yield from tqdm(rows, desc="translate", unit="para", dynamic_ncols=True)


def main() -> None:
    args = parse_args()
    if args.num_beams < 1:
        raise SystemExit("--num-beams must be >= 1")
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    use_dynamic = not args.no_dynamic_tokens

    with input_path.open("rb") as f:
        rows = pickle.load(f)

    n = len(rows)
    print(f"input: {input_path} ({n} paragraphs)", flush=True)
    if args.max_new_tokens is not None:
        print(f"max_new_tokens: fixed {args.max_new_tokens}", flush=True)
    elif use_dynamic:
        print(
            f"max_new_tokens: dynamic (cap {MAX_NEW_TOKENS_CAP}, "
            f"fallback {MAX_NEW_TOKENS_DEFAULT})",
            flush=True,
        )
    else:
        print(f"max_new_tokens: {MAX_NEW_TOKENS_DEFAULT}", flush=True)
    if args.num_beams > 1:
        print(
            f"decoding: beam search (num_beams={args.num_beams}, "
            f"length_penalty={args.length_penalty})",
            flush=True,
        )
    else:
        print("decoding: greedy (num_beams=1)", flush=True)

    print(
        "loading weights (~10 GB from ./weights) — progress bar starts after this",
        flush=True,
    )
    t_load = time.perf_counter()
    print("  processor …", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    print(f"  model (device={args.device}) …", flush=True)
    model, device = _load_model(args.device, quantize=args.quantize)
    if device.type == "cuda":
        alloc = torch.cuda.memory_allocated(device) / 2**30
        reserved = torch.cuda.memory_reserved(device) / 2**30
        print(
            f"model ready in {time.perf_counter() - t_load:.1f}s "
            f"(device={device}, VRAM alloc={alloc:.2f} GiB, reserved={reserved:.2f} GiB)",
            flush=True,
        )
    else:
        print(
            f"model ready in {time.perf_counter() - t_load:.1f}s (device={device})",
            flush=True,
        )
    if args.progress:
        print(f"translating {n} paragraphs …", flush=True)

    results = []
    t0 = time.perf_counter()
    for row in _iter_rows(rows, progress=args.progress):
        src = row["src"]
        max_tokens = _tokens_for_paragraph(
            src,
            fixed=args.max_new_tokens,
            dynamic=use_dynamic,
        )
        text = processor.apply_chat_template(
            [{"role": "user", "content": build_translation_prompt(src)}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = processor(text=text, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[-1]

        gen_kw = _build_generate_kwargs(
            max_tokens,
            num_beams=args.num_beams,
            length_penalty=args.length_penalty,
        )
        outputs = model.generate(**inputs, **gen_kw)
        response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
        raw = processor.parse_response(response)["content"]
        translation = postprocess_translation(raw)

        results.append({
            "rid": row["rid"],
            "translation": translation,
        })
        inputs.to("cpu")

    elapsed = time.perf_counter() - t0
    if n:
        print(
            f"done: {n} paragraphs in {elapsed:.1f}s "
            f"({elapsed / n:.1f}s/para)",
            flush=True,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"output: {output_path}", flush=True)


if __name__ == "__main__":
    main()
