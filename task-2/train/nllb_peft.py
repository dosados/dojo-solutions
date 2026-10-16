"""LoRA defaults для M2M100 / NLLB (encoder-decoder)."""

from __future__ import annotations


def default_lora_target_modules() -> list[str]:
    """Linear projections в encoder и decoder M2M100."""
    return ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
