"""
Publishes Phase 2 pipeline output into the SAME Supabase project, SAME
three tables, SAME image-compression settings as paddle_ocr_vl's
publish_book.py -- copied here (not imported across repos, since this
service deploys standalone) rather than reimplemented, so behavior stays
identical to what Phase 1 already publishes.
"""

import hashlib
import io
import os
from pathlib import Path

import requests
from PIL import Image

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

HEADERS = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}", "Content-Type": "application/json"}

_STORAGE_BUCKET = "textbook-images"
_MAX_IMAGE_DIMENSION = 1600
_JPEG_QUALITY = 70  # same value paddle_ocr_vl/publish_book.py settled on after measuring real compression results

_TAG_MAP = {"heading": "HEADING", "concept": "CONCEPT", "activity": "ACTIVITY"}


def _sb_insert(table: str, rows: list[dict]) -> list[dict]:
    headers = {**HEADERS, "Prefer": "return=representation"}
    resp = requests.post(f"{SUPABASE_URL}/rest/v1/{table}", headers=headers, json=rows, timeout=60)
    if not resp.ok:
        raise RuntimeError(f"insert into {table} failed ({resp.status_code}): {resp.text}")
    return resp.json()


def compress_and_upload_image(image_path: Path, storage_key: str) -> tuple[str, int, int, int]:
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        if max(im.size) > _MAX_IMAGE_DIMENSION:
            im.thumbnail((_MAX_IMAGE_DIMENSION, _MAX_IMAGE_DIMENSION), Image.LANCZOS)
        width, height = im.size
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
        data = buf.getvalue()

    resp = requests.post(
        f"{SUPABASE_URL}/storage/v1/object/{_STORAGE_BUCKET}/{storage_key}",
        headers={"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}", "Content-Type": "image/jpeg"},
        data=data, timeout=60,
    )
    if not resp.ok:
        raise RuntimeError(f"Supabase Storage upload failed ({resp.status_code}): {resp.text}")
    return f"{_STORAGE_BUCKET}/{storage_key}", width, height, len(data)


def validate_chapter(shaped: dict, content: str, image_rows: list[dict]) -> list[str]:
    import re as _re
    problems = []
    if not (shaped.get("chapter_title") or "").strip():
        problems.append("empty chapter title")
    if not content.strip():
        problems.append("empty chapter content")
    start, end = shaped.get("start_page"), shaped.get("end_page")
    if start is None or end is None:
        problems.append("missing page range")
    elif start > end:
        problems.append(f"start_page {start} > end_page {end}")
    referenced_ids = set(_re.findall(r'<img id="([^"]+)"', content))
    uploaded_ids = {row["image_id"] for row in image_rows}
    orphaned = referenced_ids - uploaded_ids
    if orphaned:
        problems.append(f"{len(orphaned)} image reference(s) with no uploaded row: {sorted(orphaned)}")
    return problems


def adapt_chapter(canonical: dict) -> dict:
    lines = []
    images = []
    fig_n = 0
    for item in canonical["content"]:
        ctype = item["content_type"]
        if ctype == "image":
            fig_n += 1
            images.append({"path": item["image_path"], "caption": item.get("description", ""), "page": item["page"]})
            lines.append(f"[FIGURE {fig_n}]")
        elif ctype == "image_description":
            # Simple/generic content judged redrawable from text alone (Stage 8's
            # keep_description_only tier) -- no file was ever cropped or saved for
            # this one, so it has no [FIGURE n] slot and nothing to upload.
            lines.append(f"[FIGURE DESCRIPTION] {item.get('description', '')}")
        elif ctype in _TAG_MAP:
            lines.append(f"[{_TAG_MAP[ctype]}] {item['text']}")
    return {
        "chapter_title": canonical["title"], "start_page": canonical["start_page"],
        "end_page": canonical["end_page"], "content": "\n\n".join(lines), "images": images,
    }


def publish_phase2_book(
    pdf_path: Path, book_result: dict, board: str, grade: str, subject: str,
    language: str, school_id: str, auto_publish: bool = True,
) -> str:
    sha256 = hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest()
    book_id = f"{board.lower().replace(' ', '_')}_class{grade}_{subject.lower().replace(' ', '_')}_{language}"

    chapters = book_result["chapters"]
    book_row = _sb_insert("textbook_books", [{
        "school_id": school_id, "book_id": book_id, "board": board, "grade": str(grade),
        "subject": subject, "language": language, "source_pdf": Path(pdf_path).name,
        "source_pdf_sha256": sha256, "chapter_count": len(chapters),
    }])[0]
    book_uuid = book_row["id"]

    # Keyed by the LOCAL file path pipeline.py already dedup'd a recurring image
    # down to -- multiple chapters can still reference that same local file (the
    # cache in pipeline.py is book-wide), so without this, each chapter would
    # upload it to Supabase Storage again as a separate object. This makes sure
    # a given local file is only ever actually uploaded once per book.
    uploaded_cache: dict[str, tuple] = {}

    for canonical in chapters:
        shaped = adapt_chapter(canonical)
        index = canonical["chapter_number"]
        content = shaped["content"]
        image_rows = []
        for i, img in enumerate(shaped["images"], start=1):
            image_id = f"img_{book_id}_ch{index:02d}_{i:02d}"
            local_path = img["path"]
            if local_path in uploaded_cache:
                storage_path, width, height, n_bytes = uploaded_cache[local_path]
            else:
                storage_key = f"{book_id}/ch{index:02d}/{image_id}.jpg"
                storage_path, width, height, n_bytes = compress_and_upload_image(Path(local_path), storage_key)
                uploaded_cache[local_path] = (storage_path, width, height, n_bytes)
            content = content.replace(f"[FIGURE {i}]", f'<img id="{image_id}" />')
            image_rows.append({
                "school_id": school_id, "image_id": image_id, "caption": img["caption"],
                "storage_path": storage_path, "source_page": img["page"], "width": width,
                "height": height, "bytes": n_bytes, "order_index": i - 1,
            })

        problems = validate_chapter(shaped, content, image_rows)
        is_published = auto_publish and not problems

        chapter_row = _sb_insert("textbook_chapters", [{
            "book_uuid": book_uuid, "school_id": school_id, "chapter_number": index,
            "chapter_title": shaped["chapter_title"], "page_start": shaped["start_page"],
            "page_end": shaped["end_page"], "content_markdown": content,
            "content_sha256": hashlib.sha256(content.encode()).hexdigest(), "published": is_published,
        }])[0]
        chapter_uuid = chapter_row["id"]

        for row in image_rows:
            row["chapter_uuid"] = chapter_uuid
        if image_rows:
            _sb_insert("textbook_images", image_rows)

    return book_uuid
