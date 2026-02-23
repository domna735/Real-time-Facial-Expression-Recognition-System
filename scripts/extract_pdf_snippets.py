from __future__ import annotations

import argparse
import re
from pathlib import Path

from pypdf import PdfReader


def _clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text(pdf_path: Path, max_pages: int) -> str:
    reader = PdfReader(str(pdf_path))
    page_count = len(reader.pages)
    max_pages = min(max_pages, page_count)

    parts: list[str] = []
    for i in range(max_pages):
        page = reader.pages[i]
        try:
            parts.append(page.extract_text() or "")
        except Exception as e:  # pragma: no cover
            parts.append(f"[extract_error page={i}: {type(e).__name__}]\n")

    return _clean("\n".join(parts))


def find_snippets(text: str, keyword: str, context: int, max_hits: int) -> list[str]:
    hits: list[str] = []
    if not text:
        return hits

    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    for m in pattern.finditer(text):
        start = max(0, m.start() - context)
        end = min(len(text), m.end() + context)
        snippet = text[start:end].replace("\n", " ")
        snippet = re.sub(r"\s+", " ", snippet).strip()
        hits.append(snippet)
        if len(hits) >= max_hits:
            break

    return hits


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract short keyword-centered snippets from a PDF (no full-text dump).")
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--max-pages", type=int, default=6)
    ap.add_argument("--head-chars", type=int, default=1200)
    ap.add_argument("--keyword", action="append", default=[])
    ap.add_argument("--context", type=int, default=180)
    ap.add_argument("--max-hits", type=int, default=3)
    args = ap.parse_args()

    if not args.pdf.exists():
        raise SystemExit(f"PDF not found: {args.pdf}")

    text = extract_text(args.pdf, max_pages=args.max_pages)
    print(f"PDF: {args.pdf}")
    print(f"Extracted pages: {args.max_pages}")
    print(f"Extracted chars: {len(text)}")
    print("\n--- HEAD ---\n")
    print(text[: args.head_chars])

    for kw in args.keyword:
        print(f"\n--- KEYWORD: {kw} ---\n")
        snippets = find_snippets(text, kw, context=args.context, max_hits=args.max_hits)
        if not snippets:
            print("(no hits in extracted pages)")
        else:
            for i, s in enumerate(snippets, 1):
                print(f"[{i}] {s}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
