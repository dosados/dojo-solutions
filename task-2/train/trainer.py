"""SFTTrainer для QLoRA на ~8 GB VRAM."""

from __future__ import annotations

import os

import torch
from trl import SFTTrainer

from train.checkpoint import prepare_for_checkpoint_save


class LowVRAMSFTTrainer(SFTTrainer):
    """Loss без entropy/chunked CE; save checkpoint без пика VRAM."""

    max_logits_to_keep: int | None = None

    def _save_checkpoint(self, model, trial, metrics=None):
        prepare_for_checkpoint_save()
        try:
            super()._save_checkpoint(model, trial)
        finally:
            prepare_for_checkpoint_save()

    def save_model(self, output_dir: str | os.PathLike | None = None, _internal_call: bool = False):
        prepare_for_checkpoint_save()
        try:
            super().save_model(output_dir, _internal_call=_internal_call)
        finally:
            prepare_for_checkpoint_save()

    def compute_loss(
        self,
        model,
        inputs: dict,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        inputs = dict(inputs)
        # Safety: Gemma3 training требует token_type_ids; PeftModel может не пробросить из batch
        if "token_type_ids" not in inputs and "input_ids" in inputs:
            inputs["token_type_ids"] = inputs["input_ids"].new_zeros(
                inputs["input_ids"].shape
            )
        labels = inputs.get("labels")
        if labels is not None:
            supervised = int((labels != -100).sum(dim=-1).max().item())
            if supervised > 0:
                k = supervised + 1
                if self.max_logits_to_keep is not None:
                    k = min(k, self.max_logits_to_keep)
                inputs["logits_to_keep"] = k
                inputs["labels"] = labels[..., -k:]

        outputs = model(**inputs)
        loss = outputs.loss
        if loss is None:
            raise RuntimeError("model forward did not return loss; check labels in batch")
        if return_outputs:
            return loss, outputs
        return loss
