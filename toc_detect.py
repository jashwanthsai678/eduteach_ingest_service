"""
Automated chapter-boundary detection -- the piece that was done by hand
(reading the TOC, computing the page offset) while testing on Class 4 EVS
and Class 5 EVS. Needed before "upload any PDF" can actually work
unattended.

Three steps, same idea as manual testing, now as real code:
  1. Find TOC-candidate pages with a cheap keyword scan (no LLM).
  2. Parse the candidate page(s)' text into {title, printed_start_page}
     with one real LLM call (robust across different TOC row formats --
     a hand-written regex parser would break on the next book's slightly
     different layout).
  3. Detect the printed-vs-actual page offset by searching for each
     parsed chapter's real occurrence AFTER the TOC page, and taking the
     offset every sample agrees on -- fully deterministic, no guessing.
"""

import json
import os
import time
from collections import Counter

import requests

MODEL = "google/gemini-2.5-flash"
_API_URL = "https://openrouter.ai/api/v1/chat/completions"

TOC_PARSE_PROMPT = (
    "Below is the extracted text of one or more pages from a school textbook "
    "that may contain its table of contents / index. Find the real chapter "
    "list -- each chapter's title, and the PAGE NUMBER it starts on.\n\n"
    "IMPORTANT: each row often has MORE THAN ONE number (e.g. a number of "
    "teaching periods, a month name, and the page number, in some order that "
    "varies by book). Do not just grab the first or last number blindly -- "
    "identify the actual page number specifically: it is the one that "
    "INCREASES monotonically from one chapter to the next down the list (a "
    "periods count does NOT increase monotonically -- it's usually a small, "
    "non-increasing number like 8-16). If a column header row is present "
    "(e.g. 'S.No / Name / Month / Page No. / Periods'), use it to confirm "
    "which number is genuinely the page number, but the monotonic-increase "
    "check is the reliable signal, since header/column order can be printed "
    "in a different order than the data columns actually appear.\n\n"
    "List entries in the order they appear. Ignore anything that isn't a "
    "real chapter entry (a 'Revision' row with no page number, front "
    "matter, headers). If no real chapter list is present, return an empty "
    "array.\n\n"
    "Return ONLY a JSON array of {title, printed_start_page}, in order."
)

TOC_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "printed_start_page": {"type": "integer"},
        },
        "required": ["title", "printed_start_page"],
    },
}


def find_toc_candidate_pages(doc, scan_pages: int = 15) -> list[int]:
    """Cheap keyword scan, no LLM -- narrows 200 pages down to a handful."""
    candidates = []
    for i in range(min(scan_pages, doc.page_count)):
        text = doc[i].get_text("text").lower()
        if "content" in text or "index" in text or "syllabus" in text:
            candidates.append(i)
    return candidates


def parse_toc_with_llm(doc, candidate_pages: list[int], api_key: str, max_attempts: int = 3) -> list[dict]:
    if not candidate_pages:
        return []
    text = "\n\n".join(f"<!-- page {i+1} -->\n{doc[i].get_text('text')}" for i in candidate_pages)
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": TOC_PARSE_PROMPT + "\n\n" + text}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "toc", "schema": TOC_SCHEMA}},
    }
    last_exc = RuntimeError("failed on every attempt")
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.post(_API_URL, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json=payload, timeout=90)
            resp.raise_for_status()
            return json.loads(resp.json()["choices"][0]["message"]["content"])
        except Exception as exc:
            last_exc = exc
            print(f"    TOC parse attempt {attempt}/{max_attempts} failed: {exc!r}")
            if attempt < max_attempts:
                time.sleep(5 * attempt)
    raise last_exc


def _normalize_title(title: str) -> str:
    """Handles real, observed mismatches between a TOC's abbreviated/
    punctuated title and the actual chapter heading's text (e.g. TOC says
    "INDIAN HISTORY & CULTURE", the real heading spells out "...AND...")."""
    t = title.strip().upper()
    t = t.replace("&", "AND")
    t = " ".join(t.split())  # collapse whitespace/newlines
    return t


def detect_offset(doc, parsed_chapters: list[dict], toc_page_indices: list[int], min_agree_fraction: float = 0.5) -> int | None:
    """Searches for EVERY parsed chapter's real occurrence AFTER the TOC
    page (not just a few samples), takes the offset the largest group
    agrees on. A majority vote across many samples tolerates the real
    failure modes observed -- an incidental early mention of a later
    chapter's title (a false EARLY match, giving a too-small offset for
    that one sample) or a title with no exact match at all (skipped, not
    counted) -- without needing every single sample to agree. Returns None
    if even the best-agreeing offset doesn't cover at least
    min_agree_fraction of the chapters that found any match at all --
    caller should treat that as "needs manual review", not silently guess."""
    search_start = max(toc_page_indices) + 1 if toc_page_indices else 0
    # Extracted+normalized once, not once per chapter -- this search range
    # can be 150+ pages and there are usually a dozen or more chapters.
    page_texts_norm = [_normalize_title(doc[i].get_text("text")) for i in range(search_start, doc.page_count)]

    offsets = []
    for ch in parsed_chapters:
        title_norm = _normalize_title(ch["title"])
        if not title_norm:
            continue
        for offset_i, text_norm in enumerate(page_texts_norm):
            if title_norm in text_norm:
                page_idx = search_start + offset_i
                offsets.append((page_idx + 1) - ch["printed_start_page"])
                break

    if not offsets:
        return None
    counts = Counter(offsets)
    best_offset, n_agree = counts.most_common(1)[0]
    if n_agree / len(offsets) < min_agree_fraction:
        return None
    return best_offset


def build_chapter_ranges(parsed_chapters: list[dict], offset: int, last_page: int) -> list[dict]:
    cleaned = sorted(
        ({"title": c["title"].strip(), "printed_start_page": c["printed_start_page"]} for c in parsed_chapters if c.get("title", "").strip()),
        key=lambda c: c["printed_start_page"],
    )
    chapters = []
    for i, c in enumerate(cleaned):
        real_start = c["printed_start_page"] + offset
        real_end = (cleaned[i + 1]["printed_start_page"] + offset - 1) if i + 1 < len(cleaned) else last_page
        chapters.append({"index": i + 1, "title": c["title"], "start_page": real_start, "end_page": max(real_start, real_end)})
    return chapters


def detect_chapters(doc, api_key: str) -> dict:
    """Returns {"chapters": [...], "offset": int, "status": "ok"} on
    success, or {"chapters": [], "status": "needs_review", "reason": ...}
    when automatic detection can't be trusted -- never silently guesses."""
    candidates = find_toc_candidate_pages(doc)
    if not candidates:
        return {"chapters": [], "status": "needs_review", "reason": "no TOC/index page found in the first pages scanned"}

    parsed = parse_toc_with_llm(doc, candidates, api_key)
    if not parsed:
        return {"chapters": [], "status": "needs_review", "reason": "TOC page(s) found but no chapter list could be parsed from them"}

    offset = detect_offset(doc, parsed, candidates)
    if offset is None:
        return {"chapters": [], "status": "needs_review", "reason": "could not confirm a consistent printed-vs-actual page offset"}

    chapters = build_chapter_ranges(parsed, offset, doc.page_count)
    return {"chapters": chapters, "offset": offset, "status": "ok"}
