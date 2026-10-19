from __future__ import annotations

"""Turn brochure pages into searchable chunks with readable metadata."""

import hashlib

from .models import BrochureChunk, BrochureRecord, BrochureSection
from .text_utils import normalize_space, slugify, word_windows

SECTION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Engine and Performance": (
        "engine",
        "performance",
        "power",
        "torque",
        "transmission",
        "turbo",
        "petrol",
        "diesel",
    ),
    "Mileage and Fuel Efficiency": ("mileage", "fuel", "efficiency", "kmpl", "economy", "range"),
    "Battery and Range": ("battery", "electric", "ev", "kilowatt", "kwh"),
    "Charging": ("charging", "charger", "dc fast", "wall charger", "charge port"),
    "Safety": (
        "safety",
        "airbag",
        "brake",
        "stability",
        "isofix",
        "camera",
        "adas",
        "abs",
        "esc",
        "parking",
    ),
    "Dimensions": ("dimension", "length", "width", "height", "wheelbase", "boot", "space"),
    "Interior and Comfort": (
        "interior",
        "comfort",
        "seat",
        "seating",
        "climate",
        "ventilated",
        "cabin",
        "sunroof",
    ),
    "Infotainment and Connectivity": (
        "infotainment",
        "connectivity",
        "screen",
        "speaker",
        "app",
        "wireless",
        "navigation",
        "bluelink",
    ),
}


def infer_section(text: str) -> str:
    """Guess the brochure section from text that came out of a PDF page."""
    lowered = text.lower()
    dimension_hits = sum(
        1
        for keyword in ("dimension", "overall length", "overall width", "overall height", "wheelbase", "boot space")
        if keyword in lowered
    )
    if dimension_hits >= 3:
        return "Dimensions"

    if _looks_like_ev_range(lowered) and any(
        phrase in lowered
        for phrase in ("driving range", "capacity (kwh)", "high voltage battery")
    ):
        return "Battery and Range"
    if _looks_like_vehicle_charging(lowered) and any(
        phrase in lowered
        for phrase in ("fast charging", "ac fast charging", "dc fast charging")
    ):
        return "Charging"

    best_section = "General"
    best_hits = 0
    for section, keywords in SECTION_KEYWORDS.items():
        if section == "Battery and Range" and not _looks_like_ev_range(lowered):
            continue
        if section == "Charging" and not _looks_like_vehicle_charging(lowered):
            continue
        hits = sum(1 for keyword in keywords if keyword in lowered)
        if hits > best_hits:
            best_section = section
            best_hits = hits
    return best_section


def _looks_like_ev_range(lowered: str) -> bool:
    if "battery" in lowered or "kwh" in lowered:
        return True
    electric_terms = ("electric", "ev")
    return "range" in lowered and any(term in lowered for term in electric_terms)


def _looks_like_vehicle_charging(lowered: str) -> bool:
    vehicle_terms = ("battery", "ev", "electric", "range", "dc fast", "wall charger")
    charge_terms = ("charging", "charger", "charge port")
    return any(term in lowered for term in charge_terms) and any(
        term in lowered for term in vehicle_terms
    )


def section_for_question(question: str) -> str | None:
    section = infer_section(question)
    return None if section == "General" else section


def chunk_record(record: BrochureRecord, max_words: int = 115, overlap: int = 24) -> list[BrochureChunk]:
    """Split one brochure record while keeping brand, model, page, and section."""
    chunks: list[BrochureChunk] = []
    for section in record.sections:
        chunks.extend(_chunk_section(record, section, max_words=max_words, overlap=overlap))
    return chunks


def _chunk_section(
    record: BrochureRecord,
    section: BrochureSection,
    max_words: int,
    overlap: int,
) -> list[BrochureChunk]:
    clean_text = normalize_space(section.text)
    if not clean_text:
        return []

    words = clean_text.split()
    windows = word_windows(words, size=max_words, overlap=overlap)
    chunks: list[BrochureChunk] = []
    base_section = section.title.strip() or infer_section(clean_text)

    for index, window in enumerate(windows, start=1):
        text = " ".join(window)
        inferred_section = infer_section(text)
        section_name = inferred_section if inferred_section != "General" else base_section
        # Stable IDs make source references readable without storing whole paths in IDs.
        stable = hashlib.sha1(
            "|".join(
                [
                    record.brand,
                    record.model,
                    record.brochure_name,
                    record.version,
                    section_name,
                    str(section.page),
                    str(index),
                    text[:80],
                ]
            ).encode("utf-8")
        ).hexdigest()[:10]
        chunk_id = "-".join(
            [
                slugify(record.brand),
                slugify(record.model),
                slugify(section_name),
                str(section.page),
                stable,
            ]
        )
        chunks.append(
            BrochureChunk(
                id=chunk_id,
                brand=record.brand,
                model=record.model,
                brochure_name=record.brochure_name,
                version=record.version,
                source_path=record.source_path,
                section=section_name,
                page=section.page,
                text=text,
            )
        )
    return chunks
