"""Локальный inference для task-2 (не входит в zip контеста)."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]

QuantizeMode = Literal["none", "4bit"]

# bf16 Gemma-4-E2B ~7.3 GiB; на GPU ≤ этого порога включаем 4bit при --gpu-only
_GPU_4BIT_TOTAL_GIB = 8.5

GEMMA_MAX_NEW_TOKENS = 1024
TRANSLATE_MAX_NEW_TOKENS = 512

GEMMA_PROMPT = """
Ты профессиональный переводчик. Твоя задача перевести текст с русского на абхазский язык.
ВАЖНО чтобы в ответе был только перевод без каких-либо комментариев и уточнений.

Текст для перевода:
"""


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
    """На ~8GB GPU bf16 целиком не влезает — автоматически 4bit при --gpu-only."""
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
        }
    if gpu_only:
        if not torch.cuda.is_available():
            raise RuntimeError("--gpu-only: CUDA недоступна")
        return {"dtype": "auto", "device_map": {"": 0}}
    return {"dtype": "auto", "device_map": "auto"}


def _input_device(model) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _iter_rows(rows: list[dict], *, show_progress: bool) -> Iterable[dict]:
    if show_progress and tqdm is not None:
        return tqdm(rows, unit="ex", dynamic_ncols=True)
    return rows


def _log_load_mode(*, quantize: QuantizeMode, gpu_only: bool) -> None:
    if quantize == "4bit":
        print("  load: 4-bit NF4 on GPU (bitsandbytes)", flush=True)
    elif gpu_only:
        print("  load: full weights on GPU", flush=True)
    else:
        print("  load: device_map=auto (может offload на CPU)", flush=True)


def run_gemma(
    rows: list[dict],
    *,
    model_dir: str | Path,
    gpu_only: bool = False,
    quantize: QuantizeMode = "none",
    show_progress: bool = False,
    max_new_tokens: int = GEMMA_MAX_NEW_TOKENS,
) -> list[dict]:
    gpu_only, quantize = resolve_load_mode(gpu_only=gpu_only, quantize=quantize)
    _log_load_mode(quantize=quantize, gpu_only=gpu_only)
    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        **_model_load_kwargs(gpu_only=gpu_only, quantize=quantize),
    )
    device = _input_device(model)

    results = []
    for row in _iter_rows(rows, show_progress=show_progress):
        prompt = processor.apply_chat_template(
            [{"role": "user", "content": GEMMA_PROMPT + row["src"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = processor(text=prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
        results.append({
            "rid": row["rid"],
            "translation": processor.parse_response(response)["content"],
        })
    return results


def _load_translate_processor(model_dir: str | Path):
    try:
        return AutoProcessor.from_pretrained(model_dir, fix_mistral_regex=True)
    except (TypeError, AttributeError):
        # fix_mistral_regex не поддерживается некоторыми версиями tokenizers
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


def run_translate(
    rows: list[dict],
    *,
    model_dir: str | Path,
    gpu_only: bool = False,
    quantize: QuantizeMode = "none",
    show_progress: bool = False,
    max_new_tokens: int = TRANSLATE_MAX_NEW_TOKENS,
) -> list[dict]:
    gpu_only, quantize = resolve_load_mode(gpu_only=gpu_only, quantize=quantize)
    _log_load_mode(quantize=quantize, gpu_only=gpu_only)
    processor = _load_translate_processor(model_dir)

    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        **_model_load_kwargs(gpu_only=gpu_only, quantize=quantize),
    )
    device = _input_device(model)
    pad_token_id = _translate_pad_token_id(processor, model)

    results = []
    for row in _iter_rows(rows, show_progress=show_progress):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": "ru",
                        "target_lang_code": "ab",
                        "text": row["src"],
                    }
                ],
            }
        ]
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
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )

        results.append({
            "rid": row["rid"],
            "translation": processor.decode(
                outputs[0][input_len:], skip_special_tokens=True
            ).strip(),
        })
    return results
