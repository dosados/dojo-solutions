"""Подготовка 4-bit модели к QLoRA (модель целиком на GPU)."""

from __future__ import annotations

import torch


def prepare_model_for_kbit_training_low_vram(
    model,
    *,
    use_gradient_checkpointing: bool = True,
    cast_norms_to_fp32: bool = True,
) -> torch.nn.Module:
    """Аналог peft.prepare_model_for_kbit_training без пика VRAM на cast всех norm."""
    model.config.use_cache = False

    for param in model.parameters():
        param.requires_grad = False

    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:

        def make_inputs_require_grad(module, _input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if cast_norms_to_fp32:
        _cast_norm_params_incremental(model)

    return model


def _cast_norm_params_incremental(model) -> None:
    n_cast = 0
    for _name, param in model.named_parameters():
        if param.ndim != 1:
            continue
        if param.dtype not in (torch.float16, torch.bfloat16):
            continue
        param.data = param.data.to(torch.float32)
        n_cast += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f"  kbit prepare: cast {n_cast} norm/bias params to fp32", flush=True)
