from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from drivewise.library import extract_text_by_page, sections_from_pages

BROCHURES = {
    "ioniq-5.pdf": "IONIQ 5",
    "creta-ev.pdf": "CRETA Electric",
    "alcazar.pdf": "ALCAZAR",
    "exter.pdf": "EXTER",
    "creta-n-line.pdf": "CRETA N Line",
    "creta.pdf": "CRETA",
    "venue-n-line.pdf": "VENUE N Line",
    "venue.pdf": "VENUE",
    "verna.pdf": "VERNA",
    "aura.pdf": "AURA",
    "i20-n-line.pdf": "i20 N Line",
    "i20.pdf": "i20",
    "grand-i10-nios.pdf": "Grand i10 NIOS",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build DriveWise JSON from the Hyundai PDFs in a brochure folder."
    )
    parser.add_argument(
        "--folder",
        default=str(ROOT / "brochures"),
        help="Folder containing the Hyundai brochure PDFs",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "data" / "hyundai_brochures.json"),
        help="JSON file to write",
    )
    args = parser.parse_args()

    brochure_folder = Path(args.folder).resolve()
    output_path = Path(args.output).resolve()

    records = []
    missing = []
    for filename, model in BROCHURES.items():
        path = brochure_folder / filename
        if not path.exists():
            missing.append(str(path))
            continue

        pages = extract_text_by_page(path.name, path.read_bytes())
        sections = sections_from_pages(pages)
        try:
            source_path = str(path.relative_to(ROOT))
        except ValueError:
            source_path = str(path)

        records.append(
            {
                "brand": "Hyundai",
                "model": model,
                "brochure_name": f"Hyundai {model} brochure",
                "version": "user provided brochure",
                "source_path": source_path,
                "sections": [
                    {
                        "title": section.title,
                        "page": section.page,
                        "text": section.text,
                    }
                    for section in sections
                    if section.text.strip()
                ],
            }
        )

    output_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    print(f"Wrote {len(records)} brochure records to {output_path}")
    print(f"Source folder: {brochure_folder}")
    for record in records:
        print(f"- {record['model']}: {len(record['sections'])} sections")

    if missing:
        print("Missing files:")
        for path in missing:
            print(f"- {path}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
