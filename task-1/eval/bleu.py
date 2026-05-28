"""Macro-average BLEU через sacrebleu (как в SPECS / BLEU.md)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import sacrebleu


@dataclass(frozen=True)
class ExampleScore:
    rid: int
    score: float
    hyp: str
    ref: str


@dataclass(frozen=True)
class BleuReport:
    macro_bleu: float
    n_examples: int
    per_example: tuple[ExampleScore, ...]
    sacrebleu_version: str

    @property
    def min_score(self) -> float:
        return min(x.score for x in self.per_example)

    @property
    def max_score(self) -> float:
        return max(x.score for x in self.per_example)


def sentence_bleu(hypothesis: str, reference: str) -> float:
    """Один абзац: hypothesis vs один reference (0–100)."""
    return sacrebleu.sentence_bleu(hypothesis, [reference]).score


def per_example_bleu(
    hypotheses: Sequence[str],
    references: Sequence[str],
    rids: Sequence[int] | None = None,
) -> list[ExampleScore]:
    if len(hypotheses) != len(references):
        raise ValueError(
            f"length mismatch: {len(hypotheses)} hypotheses vs {len(references)} references"
        )
    if rids is None:
        rids = list(range(len(hypotheses)))
    if len(rids) != len(hypotheses):
        raise ValueError("rids length must match hypotheses")

    return [
        ExampleScore(
            rid=int(rid),
            score=sentence_bleu(hyp, ref),
            hyp=hyp,
            ref=ref,
        )
        for rid, hyp, ref in zip(rids, hypotheses, references)
    ]


def macro_bleu(hypotheses: Sequence[str], references: Sequence[str]) -> float:
    scores = per_example_bleu(hypotheses, references)
    if not scores:
        raise ValueError("empty input")
    return sum(s.score for s in scores) / len(scores)


def align_by_rid(
    hypotheses: dict[int, str],
    references: dict[int, str],
    *,
    require_all_refs: bool = True,
) -> tuple[list[int], list[str], list[str]]:
    """Сопоставить по rid; порядок — сортировка rid."""
    hyp_rids = set(hypotheses)
    ref_rids = set(references)

    if require_all_refs:
        missing = ref_rids - hyp_rids
        if missing:
            raise ValueError(
                f"hypotheses missing {len(missing)} rid(s), e.g. {sorted(missing)[:5]}"
            )

    common = sorted(hyp_rids & ref_rids)
    if not common:
        raise ValueError("no common rid between hypotheses and references")

    extra_hyp = hyp_rids - ref_rids
    if extra_hyp:
        print(
            f"warning: {len(extra_hyp)} hypothesis rid(s) without reference "
            f"(ignored), e.g. {sorted(extra_hyp)[:5]}"
        )

    missing_ref = ref_rids - hyp_rids
    if missing_ref and not require_all_refs:
        print(
            f"warning: {len(missing_ref)} reference rid(s) without hypothesis "
            f"(ignored), e.g. {sorted(missing_ref)[:5]}"
        )

    hyps = [hypotheses[i] for i in common]
    refs = [references[i] for i in common]
    return common, hyps, refs


def evaluate_pairs(
    hypotheses: dict[int, str],
    references: dict[int, str],
    *,
    require_all_refs: bool = True,
) -> BleuReport:
    rids, hyps, refs = align_by_rid(
        hypotheses, references, require_all_refs=require_all_refs
    )
    per_ex = per_example_bleu(hyps, refs, rids=rids)
    macro = sum(x.score for x in per_ex) / len(per_ex)
    return BleuReport(
        macro_bleu=macro,
        n_examples=len(per_ex),
        per_example=tuple(per_ex),
        sacrebleu_version=sacrebleu.__version__,
    )
