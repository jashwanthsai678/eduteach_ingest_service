"""
The full Phase 2 pipeline as an importable function -- validated logic
from paddle_ocr_vl/phase2_pymupdf_pipeline, made fully automatic (no
hardcoded per-book chapter list) via toc_detect.py.
"""

import re
from pathlib import Path

import pymupdf

import chapter_select as sel
import toc_detect


def stage5_build_blocks(doc, start_page: int, end_page: int) -> list[dict]:
    all_blocks = []
    for page_idx in range(start_page - 1, end_page):
        page = doc[page_idx]
        raw = list(page.get_text("blocks"))
        raw.sort(key=lambda b: (round(b[1] / 10), b[0]))
        images = page.get_image_info(xrefs=True)
        block_no = 0
        for b in raw:
            x0, y0, x1, y1, text, bno, btype = b
            text = text.strip()
            if btype == 0 and text:
                block_no += 1
                all_blocks.append({
                    "block_id": f"p{page_idx+1}_b{block_no:02d}", "page": page_idx + 1, "type": "text",
                    "bbox": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)], "text": text,
                })
        for img in images:
            block_no += 1
            bbox = img["bbox"]
            all_blocks.append({
                "block_id": f"p{page_idx+1}_b{block_no:02d}", "page": page_idx + 1, "type": "image",
                "bbox": [round(bbox[0], 1), round(bbox[1], 1), round(bbox[2], 1), round(bbox[3], 1)], "xref": img["xref"],
            })
    return all_blocks


def process_chapter(doc, ch: dict, book_id: str, images_dir: Path, api_key: str, image_cache: list | None = None) -> dict:
    blocks = stage5_build_blocks(doc, ch["start_page"], ch["end_page"])
    text_blocks = [b for b in blocks if b["type"] == "text"]
    image_blocks = [b for b in blocks if b["type"] == "image"]
    by_page = {}
    for b in blocks:
        by_page.setdefault(b["page"], []).append(b)

    text_input = [{"block_id": b["block_id"], "page": b["page"], "bbox": b["bbox"], "text": b["text"]} for b in text_blocks]
    text_decisions = sel.select_text_blocks(text_input, api_key) if text_input else {}

    def crop_fn(b):
        page = doc[b["page"] - 1]
        pix = page.get_pixmap(clip=pymupdf.Rect(*b["bbox"]), dpi=150)
        return pix.tobytes("png")

    def context_fn(b):
        siblings = by_page[b["page"]]
        idx = siblings.index(b)
        texts = [s["text"] for s in siblings[max(0, idx - 2):idx] + siblings[idx + 1:idx + 3] if s["type"] == "text"]
        return "\n".join(texts)

    image_decisions = sel.select_image_blocks(image_blocks, crop_fn, context_fn, api_key, image_cache) if image_blocks else {}

    kept = []
    for b in blocks:
        if b["type"] == "text":
            d = text_decisions.get(b["block_id"])
            if not d or not d.get("keep"):
                continue
            kept.append({"block_id": b["block_id"], "page": b["page"], "content_type": d["type"], "bbox": b["bbox"], "text": b["text"]})
        else:
            d = image_decisions.get(b["block_id"])
            if not d or not d.get("keep"):
                continue
            if d.get("save_image", True):
                img_path = images_dir / f"{b['block_id']}.png"
                images_dir.mkdir(parents=True, exist_ok=True)
                page = doc[b["page"] - 1]
                pix = page.get_pixmap(clip=pymupdf.Rect(*b["bbox"]), dpi=150)
                pix.save(img_path)
                kept.append({
                    "block_id": b["block_id"], "page": b["page"], "content_type": "image", "bbox": b["bbox"],
                    "image_path": str(img_path), "description": d.get("reason", ""),
                })
            else:
                # Content is real but simple/generic enough that the description alone
                # substitutes for the picture -- no crop, no file, no upload needed at all.
                kept.append({
                    "block_id": b["block_id"], "page": b["page"], "content_type": "image_description",
                    "bbox": b["bbox"], "description": d.get("reason", ""),
                })

    kept.sort(key=lambda e: (e["page"], round(e["bbox"][1] / 10), e["bbox"][0]))
    for i, e in enumerate(kept, start=1):
        e["sequence"] = i

    return {
        "book_id": book_id, "chapter_number": ch["index"], "title": ch["title"],
        "start_page": ch["start_page"], "end_page": ch["end_page"], "content": kept,
        "stats": {"total_blocks": len(blocks), "text_blocks": len(text_blocks), "image_blocks": len(image_blocks), "kept": len(kept)},
    }


def process_book(pdf_path: Path, book_id: str, images_dir: Path, api_key: str, on_progress=None) -> dict:
    def report(stage, detail=""):
        if on_progress:
            on_progress(stage, detail)

    doc = pymupdf.open(pdf_path)
    report("detecting_chapters", "Scanning for the table of contents")
    detection = toc_detect.detect_chapters(doc, api_key)
    if detection["status"] != "ok":
        report("failed", detection.get("reason", "chapter detection failed"))
        return {"status": "failed", "reason": detection.get("reason"), "chapters": []}

    chapters_meta = detection["chapters"]
    report("processing_chapters", f"{len(chapters_meta)} chapter(s) detected, offset={detection['offset']}")

    image_cache = []  # shared across every chapter below -- a recurring image (banner,
    # footer icon) is judged by Gemini once and reused for every later occurrence,
    # matched via tolerant perceptual-hash comparison (see chapter_select.py's _find_cached).
    processed = []
    for i, ch in enumerate(chapters_meta, start=1):
        report("processing_chapters", f"chapter {i}/{len(chapters_meta)}: {ch['title']}")
        try:
            canonical = process_chapter(doc, ch, book_id, images_dir, api_key, image_cache)
            processed.append(canonical)
        except Exception as exc:
            report("chapter_failed", f"chapter {i} '{ch['title']}' failed: {exc!r}")

    report("done", f"{len(processed)}/{len(chapters_meta)} chapter(s) processed")
    return {"status": "ok", "offset": detection["offset"], "chapters": processed}
