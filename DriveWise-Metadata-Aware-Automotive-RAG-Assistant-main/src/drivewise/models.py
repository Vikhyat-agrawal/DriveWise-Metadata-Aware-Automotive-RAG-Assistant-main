from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BrochureSection:
    title: str
    page: int
    text: str


@dataclass(frozen=True)
class BrochureRecord:
    brand: str
    model: str
    brochure_name: str
    version: str
    source_path: str
    sections: tuple[BrochureSection, ...]

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> "BrochureRecord":
        sections = tuple(
            BrochureSection(
                title=str(section.get("title", "General")),
                page=int(section.get("page", 1) or 1),
                text=str(section.get("text", "")),
            )
            for section in item.get("sections", [])
            if str(section.get("text", "")).strip()
        )
        return cls(
            brand=str(item.get("brand", "")).strip(),
            model=str(item.get("model", "")).strip(),
            brochure_name=str(item.get("brochure_name", "Brochure")).strip(),
            version=str(item.get("version", "unknown")).strip(),
            source_path=str(item.get("source_path", "")).strip(),
            sections=sections,
        )


@dataclass(frozen=True)
class BrochureChunk:
    id: str
    brand: str
    model: str
    brochure_name: str
    version: str
    source_path: str
    section: str
    page: int
    text: str

    def metadata(self) -> dict[str, Any]:
        return {
            "brand": self.brand,
            "model": self.model,
            "brochure_name": self.brochure_name,
            "version": self.version,
            "source_path": self.source_path,
            "section": self.section,
            "page": self.page,
            "chunk_id": self.id,
        }


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: BrochureChunk
    vector_score: float
    rerank_score: float
    matched_terms: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SourceView:
    brand: str
    model: str
    section: str
    page: int
    brochure_name: str
    version: str
    source_path: str
    chunk_id: str
    score: float
    preview: str

    @classmethod
    def from_retrieved(cls, item: RetrievedChunk, preview_chars: int = 260) -> "SourceView":
        text = item.chunk.text.strip()
        preview = text[:preview_chars].rstrip()
        if len(text) > preview_chars:
            preview = f"{preview}..."
        return cls(
            brand=item.chunk.brand,
            model=item.chunk.model,
            section=item.chunk.section,
            page=item.chunk.page,
            brochure_name=item.chunk.brochure_name,
            version=item.chunk.version,
            source_path=item.chunk.source_path,
            chunk_id=item.chunk.id,
            score=item.rerank_score,
            preview=preview,
        )


@dataclass(frozen=True)
class EvaluationScore:
    faithfulness: float
    answer_coverage: float
    context_relevance: float
    status: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "faithfulness": round(self.faithfulness, 3),
            "answer_coverage": round(self.answer_coverage, 3),
            "context_relevance": round(self.context_relevance, 3),
            "status": self.status,
        }


@dataclass(frozen=True)
class AssistantResult:
    brand: str
    model: str
    question: str
    answer: str
    sources: tuple[SourceView, ...]
    evaluation: EvaluationScore
    response_ms: int
    status: str

    def to_log_record(self) -> dict[str, Any]:
        return {
            "brand": self.brand,
            "model": self.model,
            "question": self.question,
            "answer": self.answer,
            "response_ms": self.response_ms,
            "status": self.status,
            "sources": [
                {
                    "chunk_id": source.chunk_id,
                    "section": source.section,
                    "page": source.page,
                    "score": round(source.score, 4),
                    "brochure_name": source.brochure_name,
                }
                for source in self.sources
            ],
            "evaluation": self.evaluation.as_dict(),
        }


@dataclass(frozen=True)
class ClaimVerificationResult:
    brand: str
    model: str
    claim: str
    verdict: str
    explanation: str
    evidence: tuple[str, ...]
    sources: tuple[SourceView, ...]
    confidence: float
    evaluation: EvaluationScore
    response_ms: int
    status: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "brand": self.brand,
            "model": self.model,
            "claim": self.claim,
            "verdict": self.verdict,
            "explanation": self.explanation,
            "evidence": list(self.evidence),
            "confidence": round(self.confidence, 3),
            "response_ms": self.response_ms,
            "status": self.status,
            "evaluation": self.evaluation.as_dict(),
        }


@dataclass(frozen=True)
class ComparisonRow:
    parameter: str
    left_summary: str
    right_summary: str


@dataclass(frozen=True)
class ComparisonResult:
    brand: str
    left_model: str
    right_model: str
    question: str
    rows: tuple[ComparisonRow, ...]
    left_sources: tuple[SourceView, ...]
    right_sources: tuple[SourceView, ...]
    evaluation: EvaluationScore
    response_ms: int
    status: str

    def table_records(self) -> list[dict[str, str]]:
        return [
            {
                "parameter": row.parameter,
                self.left_model: row.left_summary,
                self.right_model: row.right_summary,
            }
            for row in self.rows
        ]


@dataclass(frozen=True)
class RecommendationCandidate:
    model: str
    fit_score: float
    evidence_strength: float
    matched_sections: tuple[str, ...]
    explanation: str
    evidence: tuple[str, ...]
    sources: tuple[SourceView, ...]


@dataclass(frozen=True)
class RecommendationResult:
    brand: str
    requirements: str
    candidates: tuple[RecommendationCandidate, ...]
    response_ms: int
    status: str

    def table_records(self) -> list[dict[str, str | int | float]]:
        return [
            {
                "rank": index,
                "model": candidate.model,
                "fit_score": round(candidate.fit_score, 3),
                "evidence_strength": round(candidate.evidence_strength, 3),
                "matched_sections": ", ".join(candidate.matched_sections),
                "explanation": candidate.explanation,
            }
            for index, candidate in enumerate(self.candidates, start=1)
        ]
