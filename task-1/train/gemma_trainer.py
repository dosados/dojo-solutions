"""SFTTrainer для ~8GB: loss без entropy/chunked CE; lm_head только на completion-токены."""

from __future__ import annotations

from trl import SFTTrainer


class LowVRAMSFTTrainer(SFTTrainer):
    """Обходит SFTTrainer.compute_loss (entropy и chunked_nll жрут VRAM)."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = dict(inputs)
        labels = inputs.get("labels")
        if labels is not None:
            supervised = int((labels != -100).sum(dim=-1).max().item())
            if supervised > 0:
                k = supervised + 1
                inputs["logits_to_keep"] = k
                # lm_head только на хвосте; labels обрезаем так же, иначе CE по неверной длине
                inputs["labels"] = labels[:, -k:]
        outputs = model(**inputs)
        loss = outputs.loss
        if loss is None:
            raise RuntimeError("model forward did not return loss; check labels in batch")
        return (loss, outputs) if return_outputs else loss
