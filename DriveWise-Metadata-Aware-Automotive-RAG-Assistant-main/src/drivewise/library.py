from __future__ import annotations

"""Load the generated brochure JSON and expose it as searchable records."""

import json
from io import BytesIO
from pathlib import Path
from typing import Iterable, Sequence

from .chunking import chunk_record, infer_section
from .models import BrochureRecord, BrochureSection, BrochureChunk


class BrochureLibrary:
    """In-memory brochure library used by the Streamlit app."""

    def __init__(self, records: Sequence[BrochureRecord]):
        self.records = tuple(records)
        self.chunks = tuple(chunk for record in self.records for chunk in chunk_record(record))

    def brands(self) -> list[str]:
        return sorted({chunk.brand for chunk in self.chunks})

    def models_for_brand(self, brand: str) -> list[str]:
        return sorted({chunk.model for chunk in self.chunks if chunk.brand == brand})

    def filter_chunks(self, brand: str, model: str) -> list[BrochureChunk]:
        return [
            chunk
            for chunk in self.chunks
            if chunk.brand.casefold() == brand.casefold()
            and chunk.model.casefold() == model.casefold()
        ]


def load_libraries_from_data_dir(data_dir: Path) -> BrochureLibrary:
    records: list[BrochureRecord] = []
    for path in sorted(data_dir.glob("*.json")):
        records.extend(load_records_from_json(path))
    return BrochureLibrary(records)


def load_records_from_json(path: Path) -> list[BrochureRecord]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw if isinstance(raw, list) else [raw]
    records = [BrochureRecord.from_dict(item) for item in items]
    return [record for record in records if record.brand and record.model and record.sections]


def extract_text_by_page(filename: str, raw: bytes) -> list[tuple[int, str]]:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("Install pypdf to read PDF brochures.") from exc

        reader = PdfReader(BytesIO(raw))
        pages: list[tuple[int, str]] = []
        for index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((index, text))
        return pages

    text = raw.decode("utf-8", errors="ignore")
    return [(1, text)]


def sections_from_pages(pages: Iterable[tuple[int, str]]) -> list[BrochureSection]:
    sections: list[BrochureSection] = []
    for page, text in pages:
        blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
        if not blocks:
            blocks = [text]
        for block in blocks:
            sections.append(
                BrochureSection(
                    title=infer_section(block),
                    page=page,
                    text=block,
                )
            )
    return sections
