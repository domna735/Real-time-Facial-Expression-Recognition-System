"""Extract AffectNet paper Table 9 into a machine-readable CSV.

This avoids relying on lossy pdf->txt extraction, which can reorder columns.

Outputs:
  outputs/paper_extract/affectnet__table9__raw_pages.tsv
  outputs/paper_extract/affectnet__table9__table.tsv

Note: Table extraction quality depends on the PDF structure.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable

import pdfplumber


REPO_ROOT = Path(__file__).resolve().parents[1]
PDF_PATH = REPO_ROOT / "research" / "paper compared" / (
    "AffectNet A Database for Facial Expression, Valence, and Arousal Computing in the Wild.pdf"
)
OUT_DIR = REPO_ROOT / "outputs" / "paper_extract"


def _iter_pages_with_table9(pdf: pdfplumber.PDF) -> Iterable[tuple[int, str]]:
    for i, page in enumerate(pdf.pages, start=1):
        text = page.extract_text() or ""
        if "TABLE 9" in text or "Table 9" in text:
            yield i, text


def _normalize_cell(cell: str | None) -> str:
    if cell is None:
        return ""
    return re.sub(r"\s+", " ", cell).strip()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not PDF_PATH.exists():
        raise FileNotFoundError(str(PDF_PATH))

    raw_pages_path = OUT_DIR / "affectnet__table9__raw_pages.tsv"
    table_path = OUT_DIR / "affectnet__table9__table.tsv"

    with pdfplumber.open(str(PDF_PATH)) as pdf:
        pages = list(_iter_pages_with_table9(pdf))
        if not pages:
            raise RuntimeError("Could not find 'TABLE 9' in extracted page text")

        # Save the page text around Table 9 for manual verification.
        with raw_pages_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["page", "text"])
            for page_num, text in pages:
                w.writerow([page_num, text])

        # Attempt table extraction on those pages.
        extracted_any = False
        with table_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["page", "table_index", "row_index", "col_index", "cell"])

            for page_num, _ in pages:
                page = pdf.pages[page_num - 1]

                # Try multiple settings; PDFs often need tweaks.
                table_settings_variants = [
                    {"vertical_strategy": "lines", "horizontal_strategy": "lines"},
                    {"vertical_strategy": "text", "horizontal_strategy": "text"},
                    {"vertical_strategy": "lines", "horizontal_strategy": "text"},
                    {"vertical_strategy": "text", "horizontal_strategy": "lines"},
                ]

                for variant_i, settings in enumerate(table_settings_variants):
                    try:
                        tables = page.extract_tables(table_settings=settings) or []
                    except Exception:
                        continue

                    for table_i, table in enumerate(tables):
                        # Heuristic: look for the metric labels row.
                        flat = " ".join(_normalize_cell(c) for r in table for c in r)
                        if not re.search(r"Accuracy\s+F1", flat, flags=re.IGNORECASE):
                            continue

                        extracted_any = True
                        for r_i, row in enumerate(table):
                            for c_i, cell in enumerate(row):
                                w.writerow(
                                    [
                                        page_num,
                                        f"{variant_i}:{table_i}",
                                        r_i,
                                        c_i,
                                        _normalize_cell(cell),
                                    ]
                                )

        if not extracted_any:
            # Still useful: raw page text is saved.
            raise RuntimeError(
                "Found TABLE 9 page(s) but could not extract a structured table. "
                "See outputs/paper_extract/affectnet__table9__raw_pages.tsv for the page text."
            )

    print(f"Wrote: {raw_pages_path}")
    print(f"Wrote: {table_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
