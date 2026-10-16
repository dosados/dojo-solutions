"""Seq2SeqTrainer с низким пиком VRAM при save checkpoint."""

from __future__ import annotations

import os

from transformers import Seq2SeqTrainer

from train.checkpoint import prepare_for_checkpoint_save


class LowVRAMSeq2SeqTrainer(Seq2SeqTrainer):
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
