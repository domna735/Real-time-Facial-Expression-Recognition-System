"""Extract rough text + candidate benchmark snippets from FER paper PDFs.

Goal: create an evidence-backed comparison table by pulling out *reported* dataset names
(e.g., FER2013, FERPlus, AffectNet, RAF-DB/RAF-AU, ExpW) and metrics (accuracy, F1).

This is intentionally heuristic: PDFs vary a lot. It produces:
- outputs/paper_extract/<pdf_stem>.txt (full extracted text, for Ctrl+F)
- outputs/paper_extract/<pdf_stem>__snippets.md (best-effort snippets)

Usage:
  python scripts/extract_paper_metrics_from_pdfs.py --pdf-dir "research/paper compared"

Dependencies: pdfminer.six
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from pdfminer.high_level import extract_text


DATASET_KEYWORDS = [
    "FER2013",
    "FER-2013",
    "FERPlus",
    "FER+",
    "AffectNet",
    "RAF",
    "RAF-DB",
    "RAFDB",
    "RAF-AU",
    "ExpW",
    "Expression in-the-Wild",
]

METRIC_PATTERNS = [
    # accuracy-like
    re.compile(r"\bacc(?:uracy)?\b\s*[:=]?\s*(\d{1,3}(?:\.\d+)?)\s*%?", re.IGNORECASE),
    re.compile(r"\btop-?1\b\s*(\d{1,3}(?:\.\d+)?)\s*%", re.IGNORECASE),
    # f1-like
    re.compile(r"\bmacro\s*-?\s*f1\b\s*[:=]?\s*(\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"\bf1\b\s*[:=]?\s*(\d+(?:\.\d+)?)", re.IGNORECASE),
]


@dataclass(frozen=True)
class Snippet:
    pdf: str
    dataset_hits: list[str]
    metric_hits: list[str]
    text: str


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _find_snippets(text: str, *, window: int = 180) -> list[Snippet]:
    snippets: list[Snippet] = []

    # Index dataset keyword occurrences and take windows around them.
    lower = text.lower()
    for kw in DATASET_KEYWORDS:
        idx = 0
        kw_lower = kw.lower()
        while True:
            pos = lower.find(kw_lower, idx)
            if pos == -1:
                break
            start = max(0, pos - window)
            end = min(len(text), pos + len(kw) + window)
            chunk = _normalize_ws(text[start:end])

            dataset_hits = sorted({k for k in DATASET_KEYWORDS if k.lower() in chunk.lower()})
            metric_hits: list[str] = []
            for pat in METRIC_PATTERNS:
                for m in pat.finditer(chunk):
                    metric_hits.append(m.group(0))

            # Only keep chunks that look like they may contain results.
            if metric_hits:
                snippets.append(Snippet(pdf="", dataset_hits=dataset_hits, metric_hits=metric_hits[:10], text=chunk))

            idx = pos + len(kw_lower)

    # Deduplicate by text.
    seen = set()
    uniq: list[Snippet] = []
    for s in snippets:
        if s.text in seen:
            continue
        seen.add(s.text)
        uniq.append(s)

    return uniq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", required=True, help="Directory containing PDFs")
    ap.add_argument("--out-dir", default="outputs/paper_extract", help="Output directory")
    ap.add_argument("--max-snippets", type=int, default=40, help="Max snippets per PDF")
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in: {pdf_dir}")

    for pdf in pdfs:
        print(f"[extract] {pdf.name}")
        text = extract_text(str(pdf))
        txt_path = out_dir / f"{pdf.stem}.txt"
        txt_path.write_text(text, encoding="utf-8", errors="ignore")

        snippets = _find_snippets(text)
        # assign pdf name now
        snippets = [Snippet(pdf=pdf.name, dataset_hits=s.dataset_hits, metric_hits=s.metric_hits, text=s.text) for s in snippets]

        md_path = out_dir / f"{pdf.stem}__snippets.md"
        lines: list[str] = []
        lines.append(f"# Snippets: {pdf.name}\n")
        lines.append("Heuristic extraction; verify by opening the PDF and searching around these phrases.\n")

        for i, s in enumerate(snippets[: max(0, args.max_snippets)], start=1):
            lines.append(f"## #{i}\n")
            if s.dataset_hits:
                lines.append(f"- Datasets: {', '.join(s.dataset_hits)}\n")
            if s.metric_hits:
                lines.append(f"- Metrics: {', '.join(s.metric_hits)}\n")
            lines.append("\n")
            lines.append(s.text)
            lines.append("\n\n")

        md_path.write_text("".join(lines), encoding="utf-8", errors="ignore")

    print({"out_dir": str(out_dir), "pdfs": [p.name for p in pdfs]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
