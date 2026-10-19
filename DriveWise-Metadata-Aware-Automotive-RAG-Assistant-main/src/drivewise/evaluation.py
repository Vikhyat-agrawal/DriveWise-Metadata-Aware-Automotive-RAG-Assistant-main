from __future__ import annotations

"""Lightweight quality signals for the displayed answer."""

from .models import EvaluationScore, RetrievedChunk
from .text_utils import overlap_ratio, tokenize


def evaluate_answer(
    question: str,
    answer: str,
    context: tuple[RetrievedChunk, ...],
    status: str,
) -> EvaluationScore:
    """Score whether the answer overlaps with the retrieved brochure context."""
    if status != "answered" or not context:
        return EvaluationScore(
            faithfulness=0.0,
            answer_coverage=0.0,
            context_relevance=0.0,
            status="needs_more_context",
        )

    context_text = " ".join(item.chunk.text for item in context)
    answer_terms = set(tokenize(answer))
    context_terms = set(tokenize(context_text))
    faithful_terms = answer_terms.intersection(context_terms)
    faithfulness = len(faithful_terms) / max(len(answer_terms), 1)

    answer_coverage = max(
        overlap_ratio(question, answer),
        overlap_ratio(question, context_text),
    )
    context_relevance = sum(item.rerank_score for item in context) / len(context)

    label = "good"
    if faithfulness < 0.35 or context_relevance < 0.12:
        label = "review"

    return EvaluationScore(
        faithfulness=_clamp(faithfulness),
        answer_coverage=_clamp(answer_coverage),
        context_relevance=_clamp(context_relevance),
        status=label,
    )


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
