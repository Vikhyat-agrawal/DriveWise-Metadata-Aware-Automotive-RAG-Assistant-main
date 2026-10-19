from __future__ import annotations

"""Local retrieval for brochure chunks.

This project stays offline and lightweight, so retrieval uses a TF-IDF style
score plus a small re-ranker instead of a hosted vector database.
"""

import math
from collections import Counter

from .chunking import section_for_question
from .library import BrochureLibrary
from .models import BrochureChunk, RetrievedChunk
from .text_utils import cosine, overlap_ratio, term_counter, tokenize


class DriveWiseRetriever:
    """Find the best chunks for the selected Hyundai model and question."""

    def __init__(self, library: BrochureLibrary):
        self.library = library

    def retrieve(
        self,
        brand: str,
        model: str,
        question: str,
        top_k: int = 5,
        candidate_k: int = 10,
        max_context_chars: int = 2400,
    ) -> tuple[RetrievedChunk, ...]:
        scoped_chunks = self.library.filter_chunks(brand, model)
        if not scoped_chunks:
            return tuple()

        idf = _idf(scoped_chunks)
        query_vector = _weighted_vector(question, idf)
        raw_candidates: list[RetrievedChunk] = []

        for chunk in scoped_chunks:
            chunk_vector = _weighted_vector(chunk.text, idf)
            vector_score = cosine(query_vector, chunk_vector)
            if vector_score <= 0:
                continue

            matched_terms = tuple(sorted(set(tokenize(question)).intersection(tokenize(chunk.text))))
            raw_candidates.append(
                RetrievedChunk(
                    chunk=chunk,
                    vector_score=vector_score,
                    rerank_score=0.0,
                    matched_terms=matched_terms,
                )
            )

        raw_candidates.sort(key=lambda item: item.vector_score, reverse=True)
        reranked = self._rerank(question, raw_candidates[:candidate_k])
        return _control_context(reranked, top_k=top_k, max_chars=max_context_chars)

    def _rerank(self, question: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        intent_section = section_for_question(question)
        reranked: list[RetrievedChunk] = []

        for item in candidates:
            section_boost = 1.0 if intent_section and item.chunk.section == intent_section else 0.0
            coverage = overlap_ratio(question, item.chunk.text)
            score = (0.72 * item.vector_score) + (0.18 * section_boost) + (0.10 * coverage)
            reranked.append(
                RetrievedChunk(
                    chunk=item.chunk,
                    vector_score=item.vector_score,
                    rerank_score=score,
                    matched_terms=item.matched_terms,
                )
            )

        reranked.sort(key=lambda item: item.rerank_score, reverse=True)
        return reranked


def _idf(chunks: list[BrochureChunk]) -> dict[str, float]:
    document_count = len(chunks)
    document_frequency: Counter[str] = Counter()
    for chunk in chunks:
        document_frequency.update(set(tokenize(chunk.text)))

    return {
        term: math.log((1 + document_count) / (1 + frequency)) + 1
        for term, frequency in document_frequency.items()
    }


def _weighted_vector(text: str, idf: dict[str, float]) -> Counter[str]:
    raw = term_counter(text)
    weighted: Counter[str] = Counter()
    for term, count in raw.items():
        weighted[term] = count * idf.get(term, 1.0)
    return weighted


def _control_context(
    candidates: list[RetrievedChunk],
    top_k: int,
    max_chars: int,
) -> tuple[RetrievedChunk, ...]:
    """Keep the prompt small enough that noisy brochure pages do not dominate."""
    selected: list[RetrievedChunk] = []
    used_chars = 0
    seen_sections: set[str] = set()

    for item in candidates:
        projected = used_chars + len(item.chunk.text)
        if selected and projected > max_chars:
            continue
        selected.append(item)
        used_chars += len(item.chunk.text)
        seen_sections.add(item.chunk.section)
        if len(selected) >= top_k:
            break

    if not selected:
        return tuple()

    selected.sort(key=lambda item: (item.chunk.section not in seen_sections, -item.rerank_score))
    return tuple(selected)
