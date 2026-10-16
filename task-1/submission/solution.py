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


MODEL_DIR = "./weights"
# scripts/bench_gemma_batch_mem.py — L4 24GB, худшие батчи.
BATCH_SIZE = 4
NUM_BEAMS = 3
LENGTH_PENALTY = 0.8
MAX_NEW_TOKENS_DEFAULT = 1536
MAX_NEW_TOKENS_CAP = 2048
MIN_NEW_TOKENS = 384


def build_baseline_prompt(src: str) -> str:
    """Оригинальный промпт организаторов из baseline submission."""
    return (
        "Ты профессиональный переводчик. Твоя задача перевести текст с английского "
        "на русский.\n"
        "ВАЖНО чтобы в ответе был только перевод без каких-либо комментариев "
        "и уточнений.\n\n"
        "Текст для перевода:\n"
        f"{src}"
    )


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
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Размер батча inference (по умолчанию {BATCH_SIZE}; 1 = последовательно)",
    )
    return parser.parse_args()


def _bnb_config(quantize: str):
    from transformers import BitsAndBytesConfig

    if quantize == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
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
        load_kw: dict = {
            "device_map": {"": 0},
            "attn_implementation": "sdpa",
        }
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
            attn_implementation="sdpa",
        )
        device = _input_device(model)

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


def _input_device(model: AutoModelForCausalLM) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _iter_batches(rows: list[dict], batch_size: int) -> Iterator[list[dict]]:
    if batch_size <= 1:
        for row in rows:
            yield [row]
        return
    ordered = sorted(rows, key=lambda r: len(r["src"]))
    for start in range(0, len(ordered), batch_size):
        yield ordered[start : start + batch_size]


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
    print("prompt: baseline", flush=True)
    print("postprocess: off", flush=True)
    print(f"batch_size: {args.batch_size}", flush=True)
    print(f"num_beams: {NUM_BEAMS}", flush=True)
    print(f"length_penalty: {LENGTH_PENALTY}", flush=True)

    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = int(tokenizer.pad_token_id)

    def _tokens_for(src: str) -> int:
        return _tokens_for_paragraph(
            src,
            fixed=args.max_new_tokens,
            dynamic=use_dynamic,
        )

    def _generate_batch(batch: list[dict], results: list[dict]) -> None:
        batch_max = max(_tokens_for(r["src"]) for r in batch)
        prompts = [
            processor.apply_chat_template(
                [{"role": "user", "content": build_baseline_prompt(row["src"])}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for row in batch
        ]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=batch_max,
                do_sample=False,
                num_beams=NUM_BEAMS,
                length_penalty=LENGTH_PENALTY,
                early_stopping=True,
                pad_token_id=pad_token_id,
            )

        for row, out_ids in zip(batch, outputs, strict=True):
            response = processor.decode(out_ids[prompt_len:], skip_special_tokens=False)
            raw = processor.parse_response(response)["content"]
            translation = raw.strip()
            results.append({"rid": row["rid"], "translation": translation})

    results: list[dict] = []
    t0 = time.perf_counter()
    batch_size = max(1, args.batch_size)

    if args.progress and batch_size > 1:
        try:
            from tqdm import tqdm

            batches = list(_iter_batches(rows, batch_size))
            batch_iter = tqdm(batches, desc="translate", unit="batch", dynamic_ncols=True)
        except ImportError:
            batch_iter = _iter_batches(rows, batch_size)
    elif args.progress:
        batch_iter = ([row] for row in _iter_rows(rows, progress=True))
    else:
        batch_iter = _iter_batches(rows, batch_size)

    for batch in batch_iter:
        _generate_batch(batch, results)

    results.sort(key=lambda x: x["rid"])

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
