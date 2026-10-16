"""PEFT patch для Gemma-4-E2B (ClippableLinear → внутренний .linear)."""

from __future__ import annotations

_PATCHED = False


def patch_peft_gemma4_clippable_linear() -> None:
    global _PATCHED
    if _PATCHED:
        return

    from peft.tuners.lora import model as lora_model

    orig = lora_model.LoraModel._create_and_replace

    def _create_and_replace(
        self,
        lora_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
        **kwargs,
    ):
        if type(target).__name__ == "Gemma4ClippableLinear":
            inner = getattr(target, "linear", None)
            if inner is not None:
                return orig(
                    self,
                    lora_config,
                    adapter_name,
                    inner,
                    target_name,
                    parent,
                    current_key,
                    **kwargs,
                )
        return orig(
            self,
            lora_config,
            adapter_name,
            target,
            target_name,
            parent,
            current_key,
            **kwargs,
        )

    lora_model.LoraModel._create_and_replace = _create_and_replace
    _PATCHED = True
    print("  peft: patch Gemma4ClippableLinear → .linear", flush=True)


def default_lora_target_modules() -> list[str]:
    return [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
