"""Загрузка Gemma-4-E2B, TranslateGemma и NLLB в 4-bit (GPU, опционально CPU offload)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSeq2SeqLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)

ModelType = Literal["gemma", "translate", "nllb"]


def bnb_4bit_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def _build_device_kwargs(cfg: dict[str, Any] | None) -> dict[str, Any]:
    """device_map / max_memory из yaml (cpu offload при нехватке VRAM)."""
    cfg = cfg or {}
    gpu_gib = cfg.get("gpu_max_memory_gib")
    cpu_gib = cfg.get("cpu_max_memory_gib", 32)
    cpu_offload = bool(cfg.get("cpu_offload", False))

    if gpu_gib is not None or cpu_offload:
        g = float(gpu_gib if gpu_gib is not None else 6.5)
        c = float(cpu_gib)
        print(
            f"  device_map=auto  max_memory: GPU {g:.1f} GiB, CPU {c:.0f} GiB "
            f"(часть слоёв на CPU — медленнее, но меньше OOM)",
            flush=True,
        )
        return {
            "device_map": "auto",
            "max_memory": {0: f"{g}GiB", "cpu": f"{int(c)}GiB"},
        }

    return {"device_map": {"": 0}}


def _build_attn_kwargs(cfg: dict[str, Any] | None) -> dict[str, Any]:
    cfg = cfg or {}
    impl = cfg.get("attn_implementation")
    if not impl:
        return {}
    # transformers ожидает строку: "sdpa" | "flash_attention_2" | ...
    print(f"  attn_implementation={impl!r}", flush=True)
    return {"attn_implementation": str(impl)}


def load_tokenizer(model_type: ModelType, model_dir: str | Path):
    """Tokenizer для nllb; для gemma/translate — processor (совместимость с merge/validate)."""
    model_dir = str(model_dir)
    if model_type == "nllb":
        return AutoTokenizer.from_pretrained(model_dir)
    return load_processor(model_type, model_dir)


def load_processor(model_type: ModelType, model_dir: str | Path):
    model_dir = str(model_dir)
    if model_type == "nllb":
        return load_tokenizer(model_type, model_dir)
    if model_type == "translate":
        try:
            return AutoProcessor.from_pretrained(model_dir, fix_mistral_regex=True)
        except (TypeError, AttributeError):
            return AutoProcessor.from_pretrained(model_dir)
    return AutoProcessor.from_pretrained(model_dir)


def load_quantized_base(
    model_type: ModelType,
    model_dir: str | Path,
    *,
    load_cfg: dict[str, Any] | None = None,
):
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA training needs CUDA")

    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(0)
    print(f"  GPU VRAM: {total / 2**30:.1f} GiB total, {free / 2**30:.1f} GiB free", flush=True)

    kwargs: dict[str, Any] = {
        "quantization_config": bnb_4bit_config(),
        "dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        **_build_device_kwargs(load_cfg),
        **_build_attn_kwargs(load_cfg),
    }
    model_dir = str(model_dir)
    dm = kwargs.get("device_map")
    print(f"  loading {model_type} base (4-bit, device_map={dm!r}) …", flush=True)

    if model_type == "translate":
        return AutoModelForImageTextToText.from_pretrained(model_dir, **kwargs)
    if model_type == "nllb":
        return AutoModelForSeq2SeqLM.from_pretrained(model_dir, **kwargs)
    return AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
