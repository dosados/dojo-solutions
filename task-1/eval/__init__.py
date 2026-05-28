"""Локальная оценка BLEU (macro-average, sacrebleu) как на контесте."""

from eval.bleu import (
    BleuReport,
    ExampleScore,
    align_by_rid,
    evaluate_pairs,
    macro_bleu,
    per_example_bleu,
    sentence_bleu,
)

__all__ = [
    "BleuReport",
    "ExampleScore",
    "align_by_rid",
    "evaluate_pairs",
    "macro_bleu",
    "per_example_bleu",
    "sentence_bleu",
]
