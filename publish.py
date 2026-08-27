"""
Publishes Phase 2 pipeline output into the SAME Supabase project, SAME
three tables, SAME image-compression settings as paddle_ocr_vl's
publish_book.py -- copied here (not imported across repos, since this
service deploys standalone) rather than reimplemented, so behavior stays
identical to what Phase 1 already publishes.
"""

import gc
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

_TAG_MAP = {
    "heading": "HEADING", "concept": "CONCEPT", "activity": "ACTIVITY",
    "key_words": "KEY WORDS", "summary": "WHAT HAVE WE LEARNT", "textbook_question": "TEXTBOOK QUESTION",
}


def _sb_insert(table: str, rows: list[dict]) -> list[dict]:
    headers = {**HEADERS, "Prefer": "return=representation"}
    resp = requests.post(f"{SUPABASE_URL}/rest/v1/{table}", headers=headers, json=rows, timeout=60)
    if not resp.ok:
        raise RuntimeError(f"insert into {table} failed ({resp.status_code}): {resp.text}")
    return resp.json()


def _sb_get(table: str, **params) -> list[dict]:
    resp = requests.get(f"{SUPABASE_URL}/rest/v1/{table}", headers=HEADERS, params=params, timeout=30)
    if not resp.ok:
        raise RuntimeError(f"query on {table} failed ({resp.status_code}): {resp.text}")
    return resp.json()


def _sb_delete(table: str, **params) -> None:
    resp = requests.delete(f"{SUPABASE_URL}/rest/v1/{table}", headers=HEADERS, params=params, timeout=30)
    if not resp.ok:
        raise RuntimeError(f"delete from {table} failed ({resp.status_code}): {resp.text}")


def _delete_storage_prefix(prefix: str) -> int:
    """Deletes every file under a Storage prefix (e.g. "book_id/ch01/"). Supabase's list
    endpoint only returns the direct children of a prefix (files AND subfolders, not
    recursively) -- book_id/ is one level (chapter folders), so this recurses one level
    deep to reach the actual image files inside each ch0N/ folder. Returns how many files
    were actually removed, purely for the caller to report a real count instead of
    silently doing nothing if the prefix was already empty/wrong."""
    resp = requests.post(
        f"{SUPABASE_URL}/storage/v1/object/list/{_STORAGE_BUCKET}",
        headers={**HEADERS, "Content-Type": "application/json"},
        json={"prefix": prefix}, timeout=30,
    )
    if not resp.ok:
        return 0
    entries = resp.json()
    file_paths = []
    for entry in entries:
        child = f"{prefix}/{entry['name']}" if prefix else entry["name"]
        if entry.get("id") is None:
            # No file id -- this entry is a folder, not a file. Recurse one level.
            file_paths.extend(_list_storage_files(child))
        else:
            file_paths.append(child)
    if not file_paths:
        return 0
    del_resp = requests.delete(
        f"{SUPABASE_URL}/storage/v1/object/{_STORAGE_BUCKET}",
        headers={**HEADERS, "Content-Type": "application/json"},
        json={"prefixes": file_paths}, timeout=60,
    )
    if not del_resp.ok:
        raise RuntimeError(f"storage delete failed ({del_resp.status_code}): {del_resp.text}")
    return len(file_paths)


def _list_storage_files(prefix: str) -> list[str]:
    resp = requests.post(
        f"{SUPABASE_URL}/storage/v1/object/list/{_STORAGE_BUCKET}",
        headers={**HEADERS, "Content-Type": "application/json"},
        json={"prefix": prefix}, timeout=30,
    )
    if not resp.ok:
        return []
    return [f"{prefix}/{entry['name']}" for entry in resp.json() if entry.get("id") is not None]


def delete_book(book_id: str) -> dict | None:
    """Cascading delete: images (rows + their uploaded Storage files) -> chapters ->
    the book row itself, in that dependency order rather than relying on FK CASCADE
    (not confirmed to be configured on these tables). Returns a summary dict, or None
    if no book with this book_id exists at all -- lets the caller 404 correctly instead
    of silently reporting success for something that was never there."""
    books = _sb_get("textbook_books", book_id=f"eq.{book_id}", select="id")
    if not books:
        return None
    book_uuid = books[0]["id"]

    chapters = _sb_get("textbook_chapters", book_uuid=f"eq.{book_uuid}", select="id")
    chapter_uuids = [c["id"] for c in chapters]

    image_count = 0
    for chapter_uuid in chapter_uuids:
        images = _sb_get("textbook_images", chapter_uuid=f"eq.{chapter_uuid}", select="id")
        image_count += len(images)
        _sb_delete("textbook_images", chapter_uuid=f"eq.{chapter_uuid}")

    if chapter_uuids:
        _sb_delete("textbook_chapters", book_uuid=f"eq.{book_uuid}")

    _sb_delete("textbook_books", id=f"eq.{book_uuid}")

    deleted_files = _delete_storage_prefix(book_id)

    return {
        "book_id": book_id, "book_uuid": book_uuid, "chapters_deleted": len(chapter_uuids),
        "image_rows_deleted": image_count, "storage_files_deleted": deleted_files,
    }


def find_existing_book(book_id: str) -> tuple[str, set[int]] | None:
    """Checks whether this exact book_id already has a book row in Supabase
    -- confirmed necessary after a stopped-partway-through book was
    re-uploaded and would otherwise have created a second book row and
    re-paid to reprocess chapters that had already succeeded. Returns
    (book_uuid, {already-published chapter numbers}) if the book row
    exists, or None if this is genuinely a new book."""
    books = _sb_get("textbook_books", book_id=f"eq.{book_id}", select="id")
    if not books:
        return None
    book_uuid = books[0]["id"]
    chapters = _sb_get("textbook_chapters", book_uuid=f"eq.{book_uuid}", select="chapter_number")
    done = {row["chapter_number"] for row in chapters}
    return book_uuid, done


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
            images.append({
                "path": item["image_path"], "caption": item.get("description", ""),
                "page": item["page"], "usage": item.get("usage", "direct"),
            })
            lines.append(f"[FIGURE {fig_n}]")
        elif ctype == "image_description":
            # Simple/generic content judged redrawable/regeneratable from text alone
            # (Stage 8's keep_description_only tier) -- no file was ever cropped or
            # saved for this one, but it still gets a real placeholder + row (path
            # None, no upload) so a downstream consumer has ONE consistent lookup for
            # every figure instead of real images being structured and these being
            # buried as inline text.
            fig_n += 1
            images.append({
                "path": None, "caption": item.get("description", ""),
                "page": item["page"], "usage": item.get("usage", "draw"),
            })
            lines.append(f"[FIGURE {fig_n}]")
        elif ctype == "image_description_ref":
            # The SAME recurring image's description already appeared earlier in this
            # chapter (pipeline.py's within-chapter dedup) -- no new row, no repeated
            # paragraph, just a short pointer back to it.
            lines.append("[FIGURE: SAME AS ABOVE]")
        elif ctype in _TAG_MAP:
            lines.append(f"[{_TAG_MAP[ctype]}] {item['text']}")
    return {
        "chapter_title": canonical["title"], "start_page": canonical["start_page"],
        "end_page": canonical["end_page"], "content": "\n\n".join(lines), "images": images,
    }


def create_book_row(pdf_path: Path, book_id: str, board: str, grade: str, subject: str, language: str, school_id: str, chapter_count: int) -> str:
    """Creates the book row immediately, before any chapter is processed --
    lets publish_one_chapter() attach each chapter to a real book_uuid as
    soon as it finishes, instead of waiting for the whole book to be done."""
    sha256 = hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest()
    book_row = _sb_insert("textbook_books", [{
        "school_id": school_id, "book_id": book_id, "board": board, "grade": str(grade),
        "subject": subject, "language": language, "source_pdf": Path(pdf_path).name,
        "source_pdf_sha256": sha256, "chapter_count": chapter_count,
    }])[0]
    return book_row["id"]


def publish_one_chapter(book_uuid: str, book_id: str, school_id: str, canonical: dict, uploaded_cache: dict, auto_publish: bool = True) -> dict:
    """Publishes exactly one chapter, immediately -- main.py calls this right
    after pipeline.process_chapter() returns, chapter by chapter, instead of
    collecting a whole book's chapters first. That way a crash on chapter 15
    doesn't erase chapters 1-14, which had already made it into Supabase by
    the time it happened -- confirmed necessary after a real 18-chapter run
    hit Render's 512MB OOM ceiling with nothing published at all, since the
    old publish_phase2_book() only ever ran after every chapter succeeded.

    uploaded_cache: caller-owned dict of {local_image_path: (storage_path,
    width, height, n_bytes)}, shared across every call for one book -- a
    recurring image (already deduped to one local file by pipeline.py) is
    only ever actually uploaded to Supabase Storage once, reused by every
    later chapter that also references it."""
    shaped = adapt_chapter(canonical)
    index = canonical["chapter_number"]
    content = shaped["content"]
    image_rows = []
    for i, img in enumerate(shaped["images"], start=1):
        image_id = f"img_{book_id}_ch{index:02d}_{i:02d}"
        local_path = img["path"]
        if local_path is None:
            # keep_description_only content -- no file was ever cropped or saved for it
            # (see pipeline.py's process_chapter), so there's nothing to upload. Still gets
            # a real row + placeholder, just with no storage_path: the API's "usage" field
            # is what tells a downstream consumer this one has no url and must be
            # AI-generated or hand-drawn instead of fetched.
            storage_path, width, height, n_bytes = None, None, None, None
        elif local_path in uploaded_cache:
            storage_path, width, height, n_bytes = uploaded_cache[local_path]
        else:
            storage_key = f"{book_id}/ch{index:02d}/{image_id}.jpg"
            storage_path, width, height, n_bytes = compress_and_upload_image(Path(local_path), storage_key)
            uploaded_cache[local_path] = (storage_path, width, height, n_bytes)
            gc.collect()  # release the just-compressed image bytes before the next upload
        content = content.replace(f"[FIGURE {i}]", f'<img id="{image_id}" />')
        image_rows.append({
            "school_id": school_id, "image_id": image_id, "caption": img["caption"],
            "storage_path": storage_path, "source_page": img["page"], "width": width,
            "height": height, "bytes": n_bytes, "order_index": i - 1, "usage": img["usage"],
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

    return {"chapter_number": index, "chapter_title": shaped["chapter_title"], "published": is_published, "problems": problems, "image_count": len(image_rows)}


def publish_phase2_book(
    pdf_path: Path, book_result: dict, board: str, grade: str, subject: str,
    language: str, school_id: str, auto_publish: bool = True,
) -> str:
    """Kept for local one-off scripts (e.g. run_maths_book.py) that already
    have a whole book's worth of chapters in memory. Implemented on top of
    create_book_row()/publish_one_chapter() so both paths stay in sync."""
    book_id = f"{board.lower().replace(' ', '_')}_class{grade}_{subject.lower().replace(' ', '_')}_{language}"
    chapters = book_result["chapters"]
    book_uuid = create_book_row(pdf_path, book_id, board, grade, subject, language, school_id, len(chapters))

    uploaded_cache: dict[str, tuple] = {}
    for canonical in chapters:
        publish_one_chapter(book_uuid, book_id, school_id, canonical, uploaded_cache, auto_publish=auto_publish)

    return book_uuid
