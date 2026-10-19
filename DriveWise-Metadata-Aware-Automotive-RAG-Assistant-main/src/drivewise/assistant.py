from __future__ import annotations

"""Mode orchestration for DriveWise Verify.

The assistant keeps the retrieval core local and evidence-first, then applies
small deterministic mode logic for asking, claim verification, comparison, and
requirement-based recommendation.
"""

import re
from collections import Counter, defaultdict
from time import perf_counter
from typing import Iterable

from .evaluation import evaluate_answer
from .generation import generate_grounded_answer, select_evidence_sentences
from .library import BrochureLibrary
from .models import (
    AssistantResult,
    ClaimVerificationResult,
    ComparisonResult,
    ComparisonRow,
    RecommendationCandidate,
    RecommendationResult,
    RetrievedChunk,
    SourceView,
)
from .retrieval import DriveWiseRetriever
from .text_utils import tokenize


MIN_EVIDENCE_SCORE = 0.10

SECTION_ORDER = (
    "Safety",
    "Interior and Comfort",
    "Dimensions",
    "Engine and Performance",
    "Mileage and Fuel Efficiency",
    "Battery and Range",
    "Charging",
    "Infotainment and Connectivity",
    "General",
)

NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}

GENERIC_CLAIM_TERMS = {
    "claim",
    "claims",
    "dealer",
    "feature",
    "features",
    "hyundai",
    "provides",
    "provide",
    "available",
    "comes",
    "offer",
    "offers",
    "supports",
    "support",
}

FEATURE_TERMS = {
    "airbag",
    "battery",
    "boot",
    "brake",
    "camera",
    "charger",
    "charging",
    "dimension",
    "engine",
    "mileage",
    "range",
    "screen",
    "seat",
    "speaker",
    "sunroof",
    "torque",
    "transmission",
    "wheelbase",
}

QUALIFIER_GROUPS = {
    "sunroof": {"panoramic", "electric", "voice", "smart"},
    "transmission": {"manual", "automatic", "amt", "dct", "ivt"},
    "engine": {"petrol", "diesel", "turbo", "cng", "electric"},
    "seat": {"ventilated", "powered", "leather", "reclining"},
}

REQUIREMENT_SECTION_WEIGHTS: dict[str, dict[str, float]] = {
    "family": {
        "Safety": 0.32,
        "Interior and Comfort": 0.26,
        "Dimensions": 0.24,
        "Infotainment and Connectivity": 0.08,
        "Engine and Performance": 0.10,
    },
    "highway": {
        "Safety": 0.30,
        "Engine and Performance": 0.24,
        "Interior and Comfort": 0.18,
        "Mileage and Fuel Efficiency": 0.16,
        "Dimensions": 0.12,
    },
    "city": {
        "Mileage and Fuel Efficiency": 0.30,
        "Dimensions": 0.22,
        "Safety": 0.18,
        "Interior and Comfort": 0.16,
        "Infotainment and Connectivity": 0.14,
    },
    "safety": {"Safety": 1.0},
    "comfort": {"Interior and Comfort": 0.74, "Dimensions": 0.16, "Safety": 0.10},
    "technology": {"Infotainment and Connectivity": 0.72, "Safety": 0.18, "Charging": 0.10},
    "tech": {"Infotainment and Connectivity": 0.72, "Safety": 0.18, "Charging": 0.10},
    "space": {"Dimensions": 0.58, "Interior and Comfort": 0.32, "Safety": 0.10},
    "performance": {"Engine and Performance": 0.78, "Mileage and Fuel Efficiency": 0.12, "Safety": 0.10},
    "ev": {"Battery and Range": 0.46, "Charging": 0.34, "Safety": 0.10, "Infotainment and Connectivity": 0.10},
    "electric": {"Battery and Range": 0.46, "Charging": 0.34, "Safety": 0.10, "Infotainment and Connectivity": 0.10},
}

DEFAULT_REQUIREMENT_WEIGHTS = {
    "Safety": 0.24,
    "Interior and Comfort": 0.20,
    "Dimensions": 0.16,
    "Engine and Performance": 0.14,
    "Mileage and Fuel Efficiency": 0.12,
    "Infotainment and Connectivity": 0.10,
    "Battery and Range": 0.04,
}

SECTION_QUERY_TERMS = {
    "Safety": "safety airbags brakes stability camera adas isofix",
    "Interior and Comfort": "comfort interior seats climate cabin sunroof ventilated",
    "Dimensions": "dimensions length width height wheelbase boot space",
    "Engine and Performance": "engine performance power torque transmission turbo",
    "Mileage and Fuel Efficiency": "mileage fuel efficiency kmpl range",
    "Battery and Range": "battery kwh electric ev driving range",
    "Charging": "charging charger dc fast ac wall charger",
    "Infotainment and Connectivity": "infotainment connectivity screen speaker app wireless navigation",
}


class DriveWiseAssistant:
    """Coordinates retrieval, answer generation, source display, and scoring."""

    def __init__(self, library: BrochureLibrary):
        self.library = library
        self.retriever = DriveWiseRetriever(library)

    def answer(self, brand: str, model: str, question: str) -> AssistantResult:
        started = perf_counter()
        context = self.retriever.retrieve(brand=brand, model=model, question=question)
        answer, status = generate_grounded_answer(
            brand=brand,
            model=model,
            question=question,
            context=context,
        )
        evaluation = evaluate_answer(
            question=question,
            answer=answer,
            context=context,
            status=status,
        )
        response_ms = int((perf_counter() - started) * 1000)
        sources = tuple(SourceView.from_retrieved(item) for item in context)

        return AssistantResult(
            brand=brand,
            model=model,
            question=question,
            answer=answer,
            sources=sources,
            evaluation=evaluation,
            response_ms=response_ms,
            status=status,
        )

    def verify_claim(self, brand: str, model: str, claim: str) -> ClaimVerificationResult:
        started = perf_counter()
        clean_claim = claim.strip()
        context = self.retriever.retrieve(
            brand=brand,
            model=model,
            question=clean_claim,
            top_k=5,
            candidate_k=12,
            max_context_chars=2600,
        )
        evidence = tuple(select_evidence_sentences(clean_claim, context, limit=4))
        verdict, explanation, confidence, status = self._classify_claim(
            brand=brand,
            model=model,
            claim=clean_claim,
            evidence=evidence,
            context=context,
        )
        answer_text = f"{verdict}. {explanation} {' '.join(evidence)}"
        evaluation = evaluate_answer(
            question=clean_claim,
            answer=answer_text,
            context=context,
            status="answered" if status == "answered" else "failed",
        )
        response_ms = int((perf_counter() - started) * 1000)

        return ClaimVerificationResult(
            brand=brand,
            model=model,
            claim=clean_claim,
            verdict=verdict,
            explanation=explanation,
            evidence=evidence,
            sources=tuple(SourceView.from_retrieved(item) for item in context),
            confidence=confidence,
            evaluation=evaluation,
            response_ms=response_ms,
            status=status,
        )

    def compare_models(
        self,
        brand: str,
        left_model: str,
        right_model: str,
        question: str,
    ) -> ComparisonResult:
        started = perf_counter()
        clean_question = question.strip() or (
            "Compare brochure evidence for safety, comfort, dimensions, performance, "
            "mileage, battery, charging, and technology."
        )

        if left_model.casefold() == right_model.casefold():
            evaluation = evaluate_answer(clean_question, "", tuple(), "failed")
            return ComparisonResult(
                brand=brand,
                left_model=left_model,
                right_model=right_model,
                question=clean_question,
                rows=(
                    ComparisonRow(
                        parameter="Selection",
                        left_summary="Choose two different models.",
                        right_summary="Choose two different models.",
                    ),
                ),
                left_sources=tuple(),
                right_sources=tuple(),
                evaluation=evaluation,
                response_ms=int((perf_counter() - started) * 1000),
                status="failed",
            )

        left_context = self.retriever.retrieve(
            brand=brand,
            model=left_model,
            question=clean_question,
            top_k=6,
            candidate_k=14,
            max_context_chars=3000,
        )
        right_context = self.retriever.retrieve(
            brand=brand,
            model=right_model,
            question=clean_question,
            top_k=6,
            candidate_k=14,
            max_context_chars=3000,
        )
        rows = self._build_comparison_rows(clean_question, left_context, right_context)
        combined_context = left_context + right_context
        answer_text = " ".join(
            f"{row.parameter}: {row.left_summary} {row.right_summary}" for row in rows
        )
        status = "answered" if rows and combined_context else "failed"
        evaluation = evaluate_answer(
            question=clean_question,
            answer=answer_text,
            context=combined_context,
            status=status,
        )

        return ComparisonResult(
            brand=brand,
            left_model=left_model,
            right_model=right_model,
            question=clean_question,
            rows=rows,
            left_sources=tuple(SourceView.from_retrieved(item) for item in left_context),
            right_sources=tuple(SourceView.from_retrieved(item) for item in right_context),
            evaluation=evaluation,
            response_ms=int((perf_counter() - started) * 1000),
            status=status,
        )

    def recommend(
        self,
        brand: str,
        requirements: str,
        top_n: int = 3,
    ) -> RecommendationResult:
        started = perf_counter()
        clean_requirements = requirements.strip()
        if not clean_requirements:
            return RecommendationResult(
                brand=brand,
                requirements=clean_requirements,
                candidates=tuple(),
                response_ms=int((perf_counter() - started) * 1000),
                status="failed",
            )

        weights = _section_weights_for_requirements(clean_requirements)
        candidates: list[RecommendationCandidate] = []

        for model in self.library.models_for_brand(brand):
            context, section_scores = self._retrieve_requirement_context(
                brand=brand,
                model=model,
                requirements=clean_requirements,
                weights=weights,
            )
            if not context:
                continue

            matched_sections = _ordered_unique(item.chunk.section for item in context)
            covered_weight = min(
                1.0,
                sum(weight for section, weight in weights.items() if section_scores.get(section, 0.0) >= MIN_EVIDENCE_SCORE),
            )
            weighted_score = sum(
                weights.get(section, 0.0) * min(section_scores.get(section, 0.0) / 0.42, 1.0)
                for section in weights
            )
            suitability_signal = _requirement_signal_score(clean_requirements, context)
            evidence_strength = weighted_score
            fit_score = (
                (0.42 * weighted_score)
                + (0.28 * covered_weight)
                + (0.30 * suitability_signal)
                + _requirement_bonus(clean_requirements, context)
            )
            retrieval_query = _expanded_requirement_query(clean_requirements, weights)
            evidence = tuple(select_evidence_sentences(retrieval_query, context, limit=3))
            explanation = _recommendation_explanation(model, matched_sections)

            candidates.append(
                RecommendationCandidate(
                    model=model,
                    fit_score=_clamp(fit_score),
                    evidence_strength=_clamp(evidence_strength),
                    matched_sections=matched_sections,
                    explanation=explanation,
                    evidence=evidence,
                    sources=tuple(SourceView.from_retrieved(item) for item in context),
                )
            )

        candidates.sort(key=lambda item: item.fit_score, reverse=True)
        selected = tuple(candidates[: max(1, top_n)])
        return RecommendationResult(
            brand=brand,
            requirements=clean_requirements,
            candidates=selected,
            response_ms=int((perf_counter() - started) * 1000),
            status="answered" if selected else "failed",
        )

    def _retrieve_requirement_context(
        self,
        brand: str,
        model: str,
        requirements: str,
        weights: dict[str, float],
    ) -> tuple[tuple[RetrievedChunk, ...], dict[str, float]]:
        selected: list[RetrievedChunk] = []
        seen_chunk_ids: set[str] = set()
        section_scores: dict[str, float] = {}

        for section, _weight in sorted(weights.items(), key=lambda item: item[1], reverse=True):
            section_query = f"{requirements} {SECTION_QUERY_TERMS.get(section, section)}"
            section_context = self.retriever.retrieve(
                brand=brand,
                model=model,
                question=section_query,
                top_k=2,
                candidate_k=10,
                max_context_chars=1300,
            )
            direct_scores = [
                item.rerank_score for item in section_context if item.chunk.section == section
            ]
            fallback_scores = [item.rerank_score * 0.55 for item in section_context]
            section_scores[section] = max(direct_scores or fallback_scores or [0.0])

            for item in section_context:
                if item.chunk.id in seen_chunk_ids:
                    continue
                selected.append(item)
                seen_chunk_ids.add(item.chunk.id)

        selected.sort(key=lambda item: item.rerank_score, reverse=True)
        return tuple(selected[:8]), section_scores

    def _classify_claim(
        self,
        brand: str,
        model: str,
        claim: str,
        evidence: tuple[str, ...],
        context: tuple[RetrievedChunk, ...],
    ) -> tuple[str, str, float, str]:
        if not claim:
            return "Not Found", "Enter an automotive claim to verify.", 0.0, "failed"

        if not context or context[0].rerank_score < MIN_EVIDENCE_SCORE:
            return (
                "Not Found",
                "No sufficiently strong official brochure evidence was retrieved. "
                "Because absence of evidence is not proof that the claim is false, the claim is not contradicted.",
                0.18,
                "failed",
            )

        evidence_text = " ".join(evidence) or " ".join(item.chunk.text for item in context)
        if not evidence_text.strip():
            return (
                "Not Found",
                "Related brochure chunks were retrieved, but no clear sentence-level evidence was available.",
                0.22,
                "failed",
            )

        claim_norm = _normalize_number_words(claim)
        evidence_norm = _normalize_number_words(evidence_text)
        model_terms = set(tokenize(f"{brand} {model}"))
        claim_terms = _important_terms(claim_norm, model_terms)
        evidence_terms = set(tokenize(evidence_norm))
        supported_terms = claim_terms.intersection(evidence_terms)
        support_ratio = len(supported_terms) / max(len(claim_terms), 1)
        feature_overlap = _feature_terms(claim_norm).intersection(_feature_terms(evidence_norm))
        top_score = context[0].rerank_score

        conflict = _conflict_reason(claim_norm, evidence_norm, feature_overlap)
        if conflict:
            confidence = _confidence(top_score, max(support_ratio, 0.5))
            return (
                "Contradicted",
                f"The retrieved brochure evidence conflicts with the claim: {conflict}",
                confidence,
                "answered",
            )

        if support_ratio >= 0.70 or _normalized_phrase(claim_norm) in _normalized_phrase(evidence_norm):
            confidence = _confidence(top_score, support_ratio)
            return (
                "Supported",
                "The retrieved official brochure evidence contains the key details needed to support the claim.",
                confidence,
                "answered",
            )

        if feature_overlap and support_ratio >= 0.34:
            missing = sorted(claim_terms.difference(evidence_terms))
            missing_text = ", ".join(missing[:4]) if missing else "one or more claim details"
            confidence = _confidence(top_score, support_ratio) * 0.82
            return (
                "Partially Supported",
                "The brochure evidence supports the general topic of the claim, but it does not clearly support "
                f"these detail(s): {missing_text}.",
                _clamp(confidence),
                "answered",
            )

        return (
            "Not Found",
            "The official brochure evidence retrieved for this model does not contain enough information to verify "
            "the claim. The claim is therefore not treated as false.",
            _clamp(0.20 + (top_score * 0.25)),
            "failed",
        )

    def _build_comparison_rows(
        self,
        question: str,
        left_context: tuple[RetrievedChunk, ...],
        right_context: tuple[RetrievedChunk, ...],
    ) -> tuple[ComparisonRow, ...]:
        left_by_section = _summaries_by_section(question, left_context)
        right_by_section = _summaries_by_section(question, right_context)
        sections = [
            section
            for section in SECTION_ORDER
            if section in left_by_section or section in right_by_section
        ]

        if not sections:
            return (
                ComparisonRow(
                    parameter="Evidence",
                    left_summary="No strong brochure-backed evidence was retrieved.",
                    right_summary="No strong brochure-backed evidence was retrieved.",
                ),
            )

        rows: list[ComparisonRow] = []
        for section in sections[:6]:
            rows.append(
                ComparisonRow(
                    parameter=section,
                    left_summary=left_by_section.get(
                        section,
                        f"No strong {section.lower()} evidence was retrieved for this model.",
                    ),
                    right_summary=right_by_section.get(
                        section,
                        f"No strong {section.lower()} evidence was retrieved for this model.",
                    ),
                )
            )
        return tuple(rows)


def _summaries_by_section(
    question: str,
    context: tuple[RetrievedChunk, ...],
) -> dict[str, str]:
    grouped: dict[str, list[RetrievedChunk]] = defaultdict(list)
    for item in context:
        grouped[item.chunk.section].append(item)

    summaries: dict[str, str] = {}
    for section in SECTION_ORDER:
        items = tuple(grouped.get(section, []))
        if not items:
            continue
        sentences = select_evidence_sentences(f"{question} {section}", items, limit=2)
        if sentences:
            summaries[section] = " ".join(sentences)
    return summaries


def _section_weights_for_requirements(requirements: str) -> dict[str, float]:
    lowered = requirements.lower()
    weights: Counter[str] = Counter()
    for trigger, mapping in REQUIREMENT_SECTION_WEIGHTS.items():
        if trigger in lowered:
            weights.update(mapping)

    if not weights:
        weights.update(DEFAULT_REQUIREMENT_WEIGHTS)

    total = sum(weights.values()) or 1.0
    return {section: value / total for section, value in weights.items()}


def _expanded_requirement_query(requirements: str, weights: dict[str, float]) -> str:
    parts = [requirements]
    for section, _weight in sorted(weights.items(), key=lambda item: item[1], reverse=True)[:5]:
        terms = SECTION_QUERY_TERMS.get(section)
        if terms:
            parts.append(terms)
    return " ".join(parts)


def _requirement_signal_score(
    requirements: str,
    context: tuple[RetrievedChunk, ...],
) -> float:
    lowered = requirements.lower()
    evidence_text = " ".join(item.chunk.text for item in context).lower()
    signals: list[float] = []

    if "family" in lowered:
        signals.append(_space_and_seating_signal(evidence_text))
    if "space" in lowered:
        signals.append(_space_and_seating_signal(evidence_text))
    if "highway" in lowered:
        signals.append(_keyword_signal(evidence_text, ("engine", "power", "torque", "transmission", "cruise", "drive mode")))
    if "city" in lowered:
        compact_signal = 1.0 - min(_max_dimension_value(evidence_text, ("overall length", "length")) / 4700, 1.0)
        signals.append(max(0.35, compact_signal))
    if "safety" in lowered:
        signals.append(_keyword_signal(evidence_text, ("safety", "airbag", "brake", "stability", "camera", "adas", "isofix")))
    if "comfort" in lowered:
        signals.append(_keyword_signal(evidence_text, ("comfort", "seat", "climate", "ventilated", "cabin", "sunroof")))
    if "technology" in lowered or "tech" in lowered:
        signals.append(_keyword_signal(evidence_text, ("infotainment", "screen", "speaker", "wireless", "navigation", "bluelink")))
    if "performance" in lowered:
        signals.append(_keyword_signal(evidence_text, ("engine", "power", "torque", "turbo", "transmission")))
    if "ev" in lowered or "electric" in lowered:
        signals.append(_keyword_signal(evidence_text, ("battery", "kwh", "charging", "charger", "electric", "range")))

    if not signals:
        return 0.50
    return _clamp(sum(signals) / len(signals))


def _requirement_bonus(requirements: str, context: tuple[RetrievedChunk, ...]) -> float:
    lowered = requirements.lower()
    evidence_text = " ".join(item.chunk.text for item in context).lower()
    bonus = 0.0

    if ("family" in lowered or "space" in lowered) and _has_large_family_seating(evidence_text):
        bonus += 0.07
    if "family" in lowered and _max_dimension_value(evidence_text, ("wheelbase",)) >= 2700:
        bonus += 0.03

    return min(bonus, 0.10)


def _space_and_seating_signal(text: str) -> float:
    if _has_large_family_seating(text):
        return 1.0

    length = _max_dimension_value(text, ("overall length", "length"))
    wheelbase = _max_dimension_value(text, ("wheelbase",))
    if length >= 4500 or wheelbase >= 2700:
        return 0.82
    if length >= 4300 or wheelbase >= 2600:
        return 0.66
    if "boot" in text or "space" in text or "dimension" in text:
        return 0.45
    return 0.25


def _has_large_family_seating(text: str) -> bool:
    return bool(re.search(r"\b(?:6|7)[-\s]?seater\b|\b(?:6|7)\s*seat\b|3rd row|third row", text))


def _max_dimension_value(text: str, labels: tuple[str, ...]) -> int:
    values: list[int] = []
    for label in labels:
        pattern = rf"{re.escape(label)}\s*(?:\(?mm\)?|\(mm\))?\s*((?:\d\s*){{3,5}})"
        for match in re.finditer(pattern, text):
            digits = re.sub(r"\D", "", match.group(1))
            if digits:
                values.append(int(digits))
    return max(values or [0])


def _keyword_signal(text: str, keywords: tuple[str, ...]) -> float:
    hits = sum(1 for keyword in keywords if keyword in text)
    if hits == 0:
        return 0.20
    return _clamp(0.30 + (hits / len(keywords)))


def _recommendation_explanation(model: str, matched_sections: tuple[str, ...]) -> str:
    if matched_sections:
        sections = ", ".join(matched_sections[:4])
        return (
            f"Based on official brochure evidence, {model} aligns with the requirement through "
            f"{sections} content."
        )
    return (
        f"{model} has some retrieved brochure evidence for the requirement, but the matching sections "
        "are limited."
    )


def _important_terms(text: str, extra_stop: set[str]) -> set[str]:
    return {
        term
        for term in tokenize(text)
        if term not in GENERIC_CLAIM_TERMS and term not in extra_stop and len(term) > 1
    }


def _feature_terms(text: str) -> set[str]:
    features: set[str] = set()
    for term in tokenize(text):
        root = _feature_root(term)
        if root in FEATURE_TERMS:
            features.add(root)
    return features


def _conflict_reason(
    claim: str,
    evidence: str,
    feature_overlap: set[str],
) -> str | None:
    claim_numbers = _numbers_near_features(claim, feature_overlap)
    evidence_numbers = _numbers_near_features(evidence, feature_overlap)
    if claim_numbers and evidence_numbers and feature_overlap and not claim_numbers.intersection(evidence_numbers):
        return (
            f"the claim mentions {', '.join(sorted(claim_numbers))}, while the retrieved evidence mentions "
            f"{', '.join(sorted(evidence_numbers)[:3])} for the related feature."
        )

    claim_negative = _has_negation(claim)
    evidence_negative = _has_negation(evidence)
    if feature_overlap and claim_negative != evidence_negative:
        if claim_negative:
            return "the claim says the feature is unavailable, while the brochure evidence describes it."
        return "the claim says the feature is available, while the brochure evidence marks it as unavailable."

    claim_terms = set(tokenize(claim))
    evidence_terms = set(tokenize(evidence))
    for feature, qualifiers in QUALIFIER_GROUPS.items():
        if feature not in feature_overlap:
            continue
        claim_qualifiers = qualifiers.intersection(claim_terms)
        evidence_qualifiers = qualifiers.intersection(evidence_terms)
        if claim_qualifiers and evidence_qualifiers and claim_qualifiers.isdisjoint(evidence_qualifiers):
            return (
                f"the claim describes the {feature} as {', '.join(sorted(claim_qualifiers))}, "
                f"but the brochure evidence describes it as {', '.join(sorted(evidence_qualifiers))}."
            )

    return None


def _numbers_near_features(text: str, features: set[str]) -> set[str]:
    if not features:
        return set()

    tokens = re.findall(r"[a-zA-Z0-9.]+", text.lower())
    numbers: set[str] = set()
    for index, token in enumerate(tokens):
        if not re.fullmatch(r"\d+(?:\.\d+)?", token):
            continue
        window = tokens[max(0, index - 5) : index + 6]
        window_features = {_feature_root(item) for item in window}
        if features.intersection(window_features):
            numbers.add(token)
    return numbers


def _feature_root(term: str) -> str:
    root = term[:-1] if term.endswith("s") else term
    return root if root in FEATURE_TERMS else term


def _has_negation(text: str) -> bool:
    return bool(
        re.search(
            r"\b(no|not|without|does not|doesn't|is not|isn't|not available|unavailable)\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def _normalize_number_words(text: str) -> str:
    normalized = text.lower()
    for word, number in NUMBER_WORDS.items():
        normalized = re.sub(rf"\b{word}\b", number, normalized)
    return normalized


def _normalized_phrase(text: str) -> str:
    return " ".join(tokenize(_normalize_number_words(text), keep_stop_words=True))


def _ordered_unique(sections: Iterable[str]) -> tuple[str, ...]:
    seen = set(sections)
    ordered = [section for section in SECTION_ORDER if section in seen]
    extras = sorted(section for section in seen if section not in SECTION_ORDER)
    return tuple(ordered + extras)


def _confidence(top_score: float, support_ratio: float) -> float:
    score_component = min(top_score / 0.42, 1.0)
    return _clamp((0.55 * score_component) + (0.45 * support_ratio))


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
