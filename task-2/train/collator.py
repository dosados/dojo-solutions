"""Data collator для TranslateGemma (Gemma3): пробрасывает token_type_ids в batch."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
from trl.trainer.utils import pad


@dataclass
class DataCollatorForGemma3TextLM(DataCollatorForLanguageModeling):
    """DataCollatorForLanguageModeling + token_type_ids (обязательны при training Gemma3)."""

    def torch_call(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        output = super().torch_call(examples)
        if examples and "token_type_ids" in examples[0]:
            token_type_ids = [
                torch.tensor(example["token_type_ids"]) for example in examples
            ]
            output["token_type_ids"] = pad(
                token_type_ids,
                padding_value=0,
                padding_side="right",
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
        return output


def build_gemma3_collator(
    processor,
    *,
    completion_only_loss: bool,
    pad_to_multiple_of: int | None = None,
) -> DataCollatorForGemma3TextLM:
    tok = getattr(processor, "tokenizer", processor)
    pad_token_id = tok.pad_token_id
    if pad_token_id is None:
        pad_token_id = tok.eos_token_id
    if pad_token_id is None:
        raise ValueError("processor has no pad_token_id or eos_token_id")
    return DataCollatorForGemma3TextLM(
        pad_token_id=int(pad_token_id),
        completion_only_loss=completion_only_loss,
        padding_free=False,
        pad_to_multiple_of=pad_to_multiple_of,
    )
