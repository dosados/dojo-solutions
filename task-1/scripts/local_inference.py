"""Локальный inference для task-1 (не входит в zip контеста)."""
from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Iterable, Literal

import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

_SUBMISSION = Path(__file__).resolve().parents[1] / "submission"
if str(_SUBMISSION) not in sys.path:
    sys.path.insert(0, str(_SUBMISSION))

from postprocess import postprocess_translation  # noqa: E402
from prompts import build_translation_prompt  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]

QuantizeMode = Literal["none", "4bit"]

_GPU_4BIT_TOTAL_GIB = 8.5

GEMMA_MAX_NEW_TOKENS = 1536
TRANSLATE_MAX_NEW_TOKENS_CAP = 1024


def estimate_translate_max_new_tokens(src: str) -> int:
    n = len(src)
    if n < 80:
        return 256
    if n < 300:
        return 384
    budget = int(n * 0.45) + 256
    return max(384, min(TRANSLATE_MAX_NEW_TOKENS_CAP, budget))


def estimate_gemma_max_new_tokens(src: str) -> int:
    n = len(src)
    if n < 80:
        return 512
    if n < 300:
        return 768
    budget = int(n * 0.55) + 384
    return max(384, min(2048, budget))


def _bnb_4bit_config():
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def _gpu_mem_gib() -> tuple[float | None, float | None]:
    if not torch.cuda.is_available():
        return None, None
    free, total = torch.cuda.mem_get_info(0)
    return free / 2**30, total / 2**30


def resolve_load_mode(
    *,
    gpu_only: bool,
    quantize: QuantizeMode,
) -> tuple[bool, QuantizeMode]:
    if quantize != "none" or not gpu_only:
        return gpu_only, quantize
    _free, total = _gpu_mem_gib()
    if total is not None and total < _GPU_4BIT_TOTAL_GIB:
        print(
            f"  auto: GPU {total:.1f} GiB — bf16 целиком не влезет; "
            f"включаем 4bit (или уберите --gpu-only для CPU offload)",
            flush=True,
        )
        return True, "4bit"
    return gpu_only, quantize


def _model_load_kwargs(*, gpu_only: bool, quantize: QuantizeMode) -> dict:
    if quantize == "4bit":
        if not torch.cuda.is_available():
            raise RuntimeError("--quantize 4bit: нужна CUDA и pip install bitsandbytes")
        return {
            "device_map": {"": 0},
            "dtype": torch.bfloat16,
            "quantization_config": _bnb_4bit_config(),
            "attn_implementation": "sdpa",
        }
    if gpu_only:
        if not torch.cuda.is_available():
            raise RuntimeError("--gpu-only: CUDA недоступна")
        return {
            "dtype": "auto",
            "device_map": {"": 0},
            "attn_implementation": "sdpa",
        }
    return {"dtype": "auto", "device_map": "auto", "attn_implementation": "sdpa"}


def _input_device(model) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _maybe_compile(model, *, enabled: bool):
    if not enabled:
        return model
    if not torch.cuda.is_available():
        print("  compile: skipped (no CUDA)", flush=True)
        return model
    try:
        print("  compile: torch.compile(forward, mode=default)", flush=True)
        model.forward = torch.compile(model.forward, mode="default", fullgraph=False)
    except Exception as exc:  # noqa: BLE001
        print(f"  compile: skipped ({exc})", flush=True)
    return model


def _iter_batches(rows: list[dict], batch_size: int) -> Iterable[list[dict]]:
    if batch_size <= 1:
        for row in rows:
            yield [row]
        return
    ordered = sorted(rows, key=lambda r: len(r["src"]))
    for start in range(0, len(ordered), batch_size):
        yield ordered[start : start + batch_size]


def _batch_progress(total_rows: int, batch_size: int, *, show_progress: bool):
    n_batches = (total_rows + batch_size - 1) // max(batch_size, 1)
    if show_progress and tqdm is not None:
        return tqdm(total=n_batches, unit="batch", dynamic_ncols=True)
    return None


def _log_load_mode(*, quantize: QuantizeMode, gpu_only: bool) -> None:
    if quantize == "4bit":
        print("  load: 4-bit NF4 on GPU (bitsandbytes)", flush=True)
    elif gpu_only:
        print("  load: full weights on GPU", flush=True)
    else:
        print("  load: device_map=auto (может offload на CPU)", flush=True)


@dataclass
class GemmaRuntime:
    processor: AutoProcessor
    model: AutoModelForCausalLM
    device: torch.device
    pad_token_id: int
    max_new_tokens: int | None


def load_gemma(
    model_dir: str | Path,
    *,
    gpu_only: bool = False,
    quantize: QuantizeMode = "none",
    use_compile: bool = False,
    max_new_tokens: int | None = None,
) -> GemmaRuntime:
    gpu_only, quantize = resolve_load_mode(gpu_only=gpu_only, quantize=quantize)
    _log_load_mode(quantize=quantize, gpu_only=gpu_only)
    processor = AutoProcessor.from_pretrained(model_dir)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        **_model_load_kwargs(gpu_only=gpu_only, quantize=quantize),
    )
    model = _maybe_compile(model, enabled=use_compile)
    return GemmaRuntime(
        processor=processor,
        model=model,
        device=_input_device(model),
        pad_token_id=int(tokenizer.pad_token_id),
        max_new_tokens=max_new_tokens,
    )


def run_gemma_loaded(
    runtime: GemmaRuntime,
    rows: list[dict],
    *,
    prompt_builder: Callable[[str], str] | None = None,
    show_progress: bool = False,
    batch_size: int = 1,
    use_postprocess: bool = True,
    warmup: bool = False,
    use_compile: bool = False,
) -> list[dict]:
    processor = runtime.processor
    tokenizer = processor.tokenizer
    model = runtime.model
    device = runtime.device
    pad_token_id = runtime.pad_token_id

    def _tokens_for(src: str) -> int:
        if runtime.max_new_tokens is not None:
            return runtime.max_new_tokens
        return estimate_gemma_max_new_tokens(src)

    build_prompt = prompt_builder or build_translation_prompt

    def _generate_batch(batch: list[dict], results: list[dict]) -> None:
        batch_max = max(_tokens_for(r["src"]) for r in batch)
        prompts = [
            processor.apply_chat_template(
                [{"role": "user", "content": build_prompt(row["src"])}],
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
        input_lens = inputs["attention_mask"].sum(dim=1).tolist()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=batch_max,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        for row, in_len, out_ids in zip(batch, input_lens, outputs, strict=True):
            response = processor.decode(out_ids[int(in_len) :], skip_special_tokens=False)
            raw = processor.parse_response(response)["content"]
            text = raw.strip() if not use_postprocess else postprocess_translation(raw)
            results.append({"rid": row["rid"], "translation": text})

    results: list[dict] = []
    if warmup and rows and use_compile:
        _generate_batch(rows[: min(batch_size, len(rows))], results)
        results.clear()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    progress = _batch_progress(len(rows), batch_size, show_progress=show_progress)
    for batch in _iter_batches(rows, batch_size):
        _generate_batch(batch, results)
        if progress is not None:
            progress.update(1)
    if progress is not None:
        progress.close()
    return results


def run_gemma(
    rows: list[dict],
    *,
    model_dir: str | Path,
    gpu_only: bool = False,
    quantize: QuantizeMode = "none",
    show_progress: bool = False,
    max_new_tokens: int | None = None,
    batch_size: int = 1,
    use_compile: bool = False,
    warmup: bool = False,
    use_postprocess: bool = True,
    prompt_builder: Callable[[str], str] | None = None,
) -> list[dict]:
    runtime = load_gemma(
        model_dir,
        gpu_only=gpu_only,
        quantize=quantize,
        use_compile=use_compile,
        max_new_tokens=max_new_tokens,
    )
    return run_gemma_loaded(
        runtime,
        rows,
        prompt_builder=prompt_builder,
        show_progress=show_progress,
        batch_size=batch_size,
        use_postprocess=use_postprocess,
        warmup=warmup,
        use_compile=use_compile,
    )


def _load_translate_processor(model_dir: str | Path):
    try:
        return AutoProcessor.from_pretrained(model_dir, fix_mistral_regex=True)
    except (TypeError, AttributeError):
        return AutoProcessor.from_pretrained(model_dir)


def _translate_pad_token_id(processor, model) -> int:
    tok = getattr(processor, "tokenizer", None)
    if tok is not None and tok.pad_token_id is not None:
        return int(tok.pad_token_id)
    gen = model.generation_config
    pid = gen.pad_token_id
    if pid is not None:
        return int(pid[0] if isinstance(pid, list) else pid)
    eos = gen.eos_token_id
    if eos is None:
        raise RuntimeError("cannot determine pad_token_id for TranslateGemma")
    return int(eos[0] if isinstance(eos, list) else eos)


def _translate_messages(src: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": "en",
                    "target_lang_code": "ru",
                    "text": src,
                }
            ],
        }
    ]


def _prepare_translate_batch(
    processor,
    batch: list[dict],
    *,
    device: torch.device,
):
    input_ids_list: list[torch.Tensor] = []
    attention_mask_list: list[torch.Tensor] = []
    input_lens: list[int] = []

    for row in batch:
        inputs = processor.apply_chat_template(
            _translate_messages(row["src"]),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        ids = inputs["input_ids"][0]
        mask = inputs.get("attention_mask")
        if mask is None:
            mask = torch.ones_like(ids)
        else:
            mask = mask[0]
        input_ids_list.append(ids)
        attention_mask_list.append(mask)
        input_lens.append(int(ids.shape[0]))

    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"
    padded = tokenizer.pad(
        {"input_ids": input_ids_list, "attention_mask": attention_mask_list},
        padding=True,
        return_tensors="pt",
    )
    return padded.to(device), input_lens


def run_translate(
    rows: list[dict],
    *,
    model_dir: str | Path,
    gpu_only: bool = False,
    quantize: QuantizeMode = "none",
    show_progress: bool = False,
    max_new_tokens: int | None = None,
    batch_size: int = 4,
    use_compile: bool = False,
    warmup: bool = False,
) -> list[dict]:
    gpu_only, quantize = resolve_load_mode(gpu_only=gpu_only, quantize=quantize)
    _log_load_mode(quantize=quantize, gpu_only=gpu_only)
    processor = _load_translate_processor(model_dir)

    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        **_model_load_kwargs(gpu_only=gpu_only, quantize=quantize),
    )
    model = _maybe_compile(model, enabled=use_compile)
    device = _input_device(model)
    pad_token_id = _translate_pad_token_id(processor, model)

    def _tokens_for(src: str) -> int:
        if max_new_tokens is not None:
            return max_new_tokens
        return estimate_translate_max_new_tokens(src)

    def _generate_batch(batch: list[dict]) -> None:
        batch_max = max(_tokens_for(r["src"]) for r in batch)
        inputs, input_lens = _prepare_translate_batch(
            processor,
            batch,
            device=device,
        )
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=batch_max,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        for row, in_len, out_ids in zip(batch, input_lens, outputs, strict=True):
            results.append({
                "rid": row["rid"],
                "translation": processor.decode(
                    out_ids[in_len:], skip_special_tokens=True
                ).strip(),
            })

    results: list[dict] = []
    if warmup and rows and use_compile:
        _generate_batch(rows[: min(batch_size, len(rows))])
        results.clear()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    progress = _batch_progress(len(rows), batch_size, show_progress=show_progress)
    for batch in _iter_batches(rows, batch_size):
        _generate_batch(batch)
        if progress is not None:
            progress.update(1)
    if progress is not None:
        progress.close()
    return results
