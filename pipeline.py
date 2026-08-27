"""
The full Phase 2 pipeline as an importable function -- validated logic
from paddle_ocr_vl/phase2_pymupdf_pipeline, made fully automatic (no
hardcoded per-book chapter list) via toc_detect.py.
"""

import gc
import re
from pathlib import Path

import pymupdf

import chapter_select as sel
import toc_detect


_FRAGMENT_MAX_AREA = 10000  # pt^2 (100x100pt) -- comfortably above a real standalone photo/diagram's
# usual size; a small image block only counts as a possible fragment if it's under this.
_FRAGMENT_CLUSTER_GAP = 20  # pt -- how close two small image bboxes must be to count as one cluster
_FRAGMENT_MIN_CLUSTER_SIZE = 3  # only merge once several pieces cluster tightly -- same threshold
# Phase 1's merge_fragmented_images validated (build_chapters.py) -- avoids merging two genuinely
# separate small icons/bullets that just happen to sit near each other on the page.


def _bbox_area(bbox: list) -> float:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


_PUA_RE = re.compile(r"[-\U000F0000-\U000FFFFD\U00100000-\U0010FFFD]")
# Private Use Area codepoints -- confirmed on real data: some PDFs render bullet/number
# glyphs ("7." "8." "9.") through a custom symbol font mapped into PUA codepoints, which
# extract as meaningless characters like '' with no real text meaning outside
# that font. Kept as-is, they're pure garbage tokens a downstream reader can't interpret;
# stripped, the block either loses just a cosmetic numeral (its real content is elsewhere)
# or goes empty and is dropped entirely by stage5_build_blocks -- both strictly better than
# shipping unreadable codepoints.


def _clean_text(text: str) -> str:
    """PyMuPDF preserves the PDF's physical line-wrap as literal '\\n' inside
    a block's text -- e.g. "Nithya and her 5 friends are\\nplaying a game" --
    which is just where the original page happened to wrap the line, not a
    real paragraph break (every block here is already one classified unit;
    real breaks between units are separate blocks). Left in, every downstream
    reader (including whichever LLM a school's app uses to build materials
    from this) has to silently reflow it just to read it as a sentence, on
    every single use. Collapsing all whitespace to single spaces fixes that
    once, for free, at ingestion. Also shortens long fill-in-the-blank
    underscore runs ("____________") to a fixed-width placeholder -- the
    exact count of underscores on the original worksheet line carries no
    real information, just extra tokens."""
    text = _PUA_RE.sub("", text)
    text = " ".join(text.split())
    text = re.sub(r"_{4,}", "___", text)
    return text


def _normalize_heading(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


_MERGEABLE_TEXT_TYPES = {"concept", "activity", "textbook_question", "key_words", "summary"}
_SENTENCE_END_RE = re.compile(r'[.?!:;"\')\]]$')


def _join_text(prev_text: str, next_text: str) -> str:
    """Two blocks being merged are sometimes genuinely separate lines (two dialogue
    turns, two distinct activity items) and sometimes just ONE sentence the PDF split
    into two raw text objects purely from page layout -- confirmed real case: "who is
    this girl in half" / "sari?" as two separate blocks, which a blanket newline-join
    turns into "in half\\nsari?", breaking the sentence mid-word. Rejoin as a
    continuation (space, or no space for a hyphenated line-break) only when prev_text
    doesn't already end a sentence AND next_text starts lowercase -- both signals
    needed together, since either alone is too weak (many real activity fragments
    start lowercase after a real line break, e.g. list continuations)."""
    prev_text = prev_text.rstrip()
    if not next_text:
        return prev_text
    looks_like_continuation = (
        not _SENTENCE_END_RE.search(prev_text) and next_text[:1].islower()
    )
    if prev_text.endswith("-") and looks_like_continuation:
        return prev_text[:-1] + next_text  # dehyphenate a word split across a line break
    if looks_like_continuation:
        return prev_text + " " + next_text
    return prev_text + "\n" + next_text


def _merge_adjacent_text(kept: list[dict]) -> list[dict]:
    """PyMuPDF splits a page into text blocks by its own layout heuristics, not by
    logical unit -- a single family-tree diagram's box labels, or a fill-in-the-blank
    table's cells, routinely come back as many separate tiny blocks (confirmed on real
    data: "Name of" / "father" / "mother" / "uncle" / "wives" each as their own block).
    Each one was getting its own repeated "[ACTIVITY]"/"[CONCEPT]" tag -- tag overhead
    bigger than the actual content for these fragments, and a downstream reader (or an
    LLM building materials from this) has to mentally stitch dozens of one-word
    "blocks" back into the one table/diagram they actually are.

    Run AFTER final reading-order sort: collapses a RUN of consecutive kept entries
    that are the SAME type into one entry (see _join_text for how the two texts get
    joined). Never merges across a heading, image, or a type change (concept next to
    activity stays two tags) -- only true fragments of one contiguous same-type run
    collapse; genuinely distinct content keeps its own tag."""
    merged: list[dict] = []
    for item in kept:
        if (merged and item["content_type"] in _MERGEABLE_TEXT_TYPES
                and merged[-1]["content_type"] == item["content_type"]):
            prev = merged[-1]
            prev["text"] = _join_text(prev["text"], item["text"])
            x0 = min(prev["bbox"][0], item["bbox"][0])
            y0 = min(prev["bbox"][1], item["bbox"][1])
            x1 = max(prev["bbox"][2], item["bbox"][2])
            y1 = max(prev["bbox"][3], item["bbox"][3])
            prev["bbox"] = [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)]
        else:
            merged.append(dict(item))
    return merged


def _merge_fragmented_images(blocks: list[dict]) -> list[dict]:
    """A real, observed PDF-encoding artifact: one embedded picture sometimes gets split
    into hundreds or thousands of small tiled image XObjects instead of one (confirmed live
    -- a Render free-tier OOM crash traced to one chapter with ~3,638 image blocks that
    should have been a single photo). Every one of those fragments becomes its own block,
    its own bbox, its own render -- the actual driver of that memory blowup, independent of
    which Gemini tier each fragment would have hit.

    Purely geometric fix, no vision call needed (unlike Phase 1's merge_fragmented_images,
    which merges visually-separate figures like individual portraits in a family tree and so
    needs a model to confirm they're meant to be read as one image) -- here the pieces are
    already known to be slices of one image, a PDF structural artifact, so clustering small
    image blocks by proximity and collapsing each qualifying cluster to one block spanning
    their union bbox is enough. Runs before any tiny/QR/LLM tier ever sees the individual
    pieces, so both the block count AND every downstream render/API call collapse with it.
    Leaves isolated small blocks (clusters under _FRAGMENT_MIN_CLUSTER_SIZE) untouched -- a
    single small icon still goes through the normal per-image path unaffected."""
    by_page: dict[int, list[int]] = {}
    for i, b in enumerate(blocks):
        by_page.setdefault(b["page"], []).append(i)

    to_remove = set()
    replacements = {}

    for page, idxs in by_page.items():
        img_idxs = [i for i in idxs if blocks[i]["type"] == "image" and _bbox_area(blocks[i]["bbox"]) <= _FRAGMENT_MAX_AREA]
        if len(img_idxs) < _FRAGMENT_MIN_CLUSTER_SIZE:
            continue

        parent = {i: i for i in img_idxs}

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        def expanded(bbox, g):
            return [bbox[0] - g, bbox[1] - g, bbox[2] + g, bbox[3] + g]

        def overlaps(a, b):
            return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]

        for a in range(len(img_idxs)):
            for c in range(a + 1, len(img_idxs)):
                i, j = img_idxs[a], img_idxs[c]
                if overlaps(expanded(blocks[i]["bbox"], _FRAGMENT_CLUSTER_GAP), blocks[j]["bbox"]):
                    union(i, j)

        clusters: dict[int, list[int]] = {}
        for i in img_idxs:
            clusters.setdefault(find(i), []).append(i)

        for members in clusters.values():
            if len(members) < _FRAGMENT_MIN_CLUSTER_SIZE:
                continue
            keep_idx = min(members)
            member_blocks = [blocks[i] for i in members]
            x0 = min(b["bbox"][0] for b in member_blocks)
            y0 = min(b["bbox"][1] for b in member_blocks)
            x1 = max(b["bbox"][2] for b in member_blocks)
            y1 = max(b["bbox"][3] for b in member_blocks)
            replacements[keep_idx] = {
                **blocks[keep_idx],
                "bbox": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
                "merged_from": len(members),
            }
            to_remove.update(i for i in members if i != keep_idx)

    return [replacements.get(i, b) for i, b in enumerate(blocks) if i not in to_remove]


_COLLAGE_LABEL_MAX_CHARS = 20  # a text block this short (or shorter) sitting between candidate
# images -- a caption word, a diagonal watermark fragment, a page-footer fragment -- counts as a
# label and does NOT break a collage group; anything longer is real separating content and does.
_COLLAGE_MAX_GAP = 60  # pt -- looser than the fragment-merge's 20pt, since these are genuinely
# distinct drawings that may have real (if modest) spacing between them, not tight PDF-encoding
# fragments of one picture.
_COLLAGE_MIN_GROUP_SIZE = 3  # only treat as "meant to be seen as one group" once there are
# several -- avoids merging two incidental nearby images that aren't really a collage.
_COLLAGE_MAX_MEMBER_AREA = 120000  # pt^2 -- a real, standalone full-scene illustration is
# never a collage-grid member. Calibrated against two real observed cases: a validated true
# collage's largest member (a background board behind ~16 small objects) is 95,351pt^2; a
# false-positive merge (a QR code + an unrelated full-scene illustration + a footer strip,
# with no real text between them) had a 174,952pt^2 illustration as its dominant member. This
# cap sits between the two -- comfortably above any legitimate collage piece, comfortably
# below a real single illustration -- so an image this large always breaks/excludes itself
# from a run, same as real separating text.
_COLLAGE_MIN_FILL_RATIO = 0.15  # sum of member areas / the run's merged bbox area. Confirmed
# real OOM cause: a page-wide watermark/texture pattern (thousands of ~1-2pt tiles, already
# consolidated by _merge_fragmented_images into a handful of moderate blocks up to ~47,000pt^2
# each) sits with no real text between pieces, so this function chained several of THOSE into
# one "collage" spanning most of the page -- one observed run produced a 395,182pt^2 merged
# block from members totaling only 5,290pt^2 (a 1.3% fill ratio), which then needed a
# ~3300x3400px crop (tens of MB) sent to a real judgment call -- the actual driver of a real
# Render OOM crash. A genuine collage's members visually fill the space they're grouped in (the
# validated true case: 49.6% fill ratio); scattered watermark noise does not. Runs below this
# ratio are left unmerged -- their members go through the normal per-image path individually,
# where each (already confirmed small) piece correctly hits the free tiny-icon tier instead.


def _merge_composite_images(blocks: list[dict]) -> list[dict]:
    """A different problem from _merge_fragmented_images: not one image accidentally sliced
    into meaningless pieces by a PDF-encoding artifact, but SEVERAL genuinely distinct small
    drawings meant to be seen together as one group -- e.g. a "categorize these objects by
    shape" page showing ~15 different everyday-object illustrations side by side, where only
    a short one-word label (or nothing) separates them. Judged independently, each drawing
    gets its own isolated decision with no awareness it's part of a set meant to be read
    together -- confirmed on a real chapter: one such collage came out as 1 real kept image
    (only 3 of the ~15 objects) plus 12 separate description-only text blocks for the rest,
    losing the "look at all of them and sort them" point of the exercise.

    Runs on the now-correctly-interleaved block sequence (see stage5_build_blocks): walks each
    page in true reading order and groups a RUN of image blocks that (a) have no real text
    (longer than _COLLAGE_LABEL_MAX_CHARS) breaking the run, and (b) aren't spaced more than
    _COLLAGE_MAX_GAP apart. A qualifying run collapses into one block spanning the union of
    its members' boxes, the same mechanical merge _merge_fragmented_images already uses --
    only the detection signal differs (real-text-in-between, not just tight tiny clustering).
    Runs of fewer than _COLLAGE_MIN_GROUP_SIZE are left untouched: an isolated small image (or
    two unrelated ones that happen to be near each other) still goes through the normal
    per-image path on its own.

    Confirmed regression (before the size cap below existed): a QR code sitting just above
    an unrelated full-scene illustration, with no real text between them, got swept into the
    same composite group -- once merged, the crop is no longer "just a QR code" so
    select_image_blocks's deterministic _is_qr_code check on the merged crop would likely
    miss it. Gating on _is_qr_code itself here was tried and reverted: it's unreliable in
    BOTH directions on real data -- it missed the actual QR code in that exact case (a known
    limitation, see _is_qr_code's docstring: real QR crops around 170x185px slip through even
    upscaled), and separately flagged a real small object drawing as a QR code in a genuine
    collage, incorrectly splitting a group that should have stayed merged. _COLLAGE_MAX_MEMBER_AREA
    below is what actually fixes the QR case (the illustration next to the QR code is far larger
    than any real collage member), without this false-positive/false-negative risk."""
    by_page: dict[int, list[int]] = {}
    for i, b in enumerate(blocks):
        by_page.setdefault(b["page"], []).append(i)

    to_remove = set()
    replacements = {}

    for _, idxs in by_page.items():
        run: list[int] = []
        last_bbox = None

        def flush():
            if len(run) >= _COLLAGE_MIN_GROUP_SIZE:
                keep_idx = run[0]
                member_blocks = [blocks[i] for i in run]
                x0 = min(b["bbox"][0] for b in member_blocks)
                y0 = min(b["bbox"][1] for b in member_blocks)
                x1 = max(b["bbox"][2] for b in member_blocks)
                y1 = max(b["bbox"][3] for b in member_blocks)
                bbox_area = max(0, x1 - x0) * max(0, y1 - y0)
                sum_area = sum(_bbox_area(b["bbox"]) for b in member_blocks)
                fill_ratio = (sum_area / bbox_area) if bbox_area else 1.0
                if fill_ratio >= _COLLAGE_MIN_FILL_RATIO:
                    replacements[keep_idx] = {
                        **blocks[keep_idx],
                        "bbox": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
                        "merged_from": len(run),
                    }
                    to_remove.update(i for i in run if i != keep_idx)
                # else: scattered, not a real collage -- leave every member as its own block
            run.clear()

        for i in idxs:
            b = blocks[i]
            if b["type"] == "image":
                if _bbox_area(b["bbox"]) > _COLLAGE_MAX_MEMBER_AREA:
                    flush()
                    last_bbox = None
                    continue
                if run and last_bbox is not None and (b["bbox"][1] - last_bbox[3]) > _COLLAGE_MAX_GAP:
                    flush()
                run.append(i)
                last_bbox = b["bbox"]
            else:
                if len(b.get("text", "")) <= _COLLAGE_LABEL_MAX_CHARS:
                    continue  # short label/watermark/footer fragment -- doesn't break the run
                flush()
                last_bbox = None
        flush()

    return [replacements.get(i, b) for i, b in enumerate(blocks) if i not in to_remove]


_FULL_WIDTH_FRACTION = 0.92  # an image at least this fraction of the page's own width is
# almost certainly a decorative full-bleed background graphic (a paper-texture strip, a
# gradient panel) -- real illustrations/photos in these textbooks are always inset with
# visible margins, never edge-to-edge. Confirmed real bug: one page's background was split
# into ~9 stacked full-width strips that, having no real text between them, chained
# together with the chapter's title banner, its QR code, AND the page's real illustration
# into one meaningless composite block -- which then got dropped outright as a "title
# banner" (since the title dominated what the model saw), losing the real illustration
# along with it. Dropped here structurally, before either merge pass ever sees them --
# same free, no-API-call treatment as the tiny-icon and QR tiers, so they can never again
# drag real content into a bad merge or cost a wasted judgment call apiece.


def stage5_build_blocks(doc, start_page: int, end_page: int) -> list[dict]:
    all_blocks = []
    for page_idx in range(start_page - 1, end_page):
        page = doc[page_idx]
        page_width = page.rect.width
        raw_text = list(page.get_text("blocks"))
        images = page.get_image_info(xrefs=True)

        # Sort text AND images together into ONE true reading-order sequence.
        # Previously text blocks were sorted among themselves and images were
        # appended after ALL of them (in raw PDF xref order, not position) --
        # meaning every image on a page landed after every text block regardless
        # of where it actually sat, which broke both "nearby text" context
        # lookups (pipeline.py's context_fn) for any image past the first couple,
        # and any attempt to detect real text separating two images (needed for
        # _merge_composite_images below).
        combined = []
        for b in raw_text:
            x0, y0, x1, y1, text, bno, btype = b
            text = _clean_text(text)
            if btype == 0 and text:
                combined.append((y0, x0, "text", [x0, y0, x1, y1], text, None))
        for img in images:
            bbox = img["bbox"]
            if (bbox[2] - bbox[0]) >= page_width * _FULL_WIDTH_FRACTION:
                continue  # decorative full-bleed background strip -- see _FULL_WIDTH_FRACTION
            combined.append((bbox[1], bbox[0], "image", bbox, None, img["xref"]))
        combined.sort(key=lambda item: (round(item[0] / 10), item[1]))

        block_no = 0
        for _, _, kind, bbox, text, xref in combined:
            block_no += 1
            entry = {
                "block_id": f"p{page_idx+1}_b{block_no:02d}", "page": page_idx + 1, "type": kind,
                "bbox": [round(v, 1) for v in bbox],
            }
            if kind == "text":
                entry["text"] = text
            else:
                entry["xref"] = xref
            all_blocks.append(entry)
    return _merge_composite_images(_merge_fragmented_images(all_blocks))


def process_chapter(doc, ch: dict, book_id: str, images_dir: Path, api_key: str, image_cache: list | None = None) -> dict:
    blocks = stage5_build_blocks(doc, ch["start_page"], ch["end_page"])
    text_blocks = [b for b in blocks if b["type"] == "text"]
    image_blocks = [b for b in blocks if b["type"] == "image"]
    by_page = {}
    for b in blocks:
        by_page.setdefault(b["page"], []).append(b)

    text_input = [{"block_id": b["block_id"], "page": b["page"], "bbox": b["bbox"], "text": b["text"]} for b in text_blocks]
    text_decisions = sel.select_text_blocks(text_input, api_key, chapter_number=ch["index"], chapter_title=ch["title"]) if text_input else {}

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
    seen_first_heading = False
    seen_desc_shared = {}  # id(shared judgment dict) -> block_id of this CHAPTER's first
    # occurrence -- resets every call (one call per chapter) by design, so a recurring image
    # (judged once, book-wide, via the perceptual-hash cache) only gets its full description
    # repeated once per chapter; every later occurrence in the SAME chapter becomes a short
    # back-reference instead of the same paragraph again. Deliberately NOT deduped across
    # chapters -- each chapter is a separate published unit with its own content string, and
    # "same as above" wouldn't resolve to anything in a different chapter's text.
    for b in blocks:
        if b["type"] == "text":
            d = text_decisions.get(b["block_id"])
            if not d or not d.get("keep"):
                continue
            text = b["text"]
            if d["type"] == "heading":
                if not seen_first_heading:
                    # The PDF sometimes renders the chapter number right after the opening
                    # title as one continuous text run -- e.g. "Many objects - Different
                    # shapes 1" (confirmed: PyMuPDF returns this as a single block). That
                    # duplicates canonical["title"], which is already carried separately, so
                    # drop this block entirely once it's confirmed to BE the title (only
                    # after stripping the suspected stray suffix, so a real heading that
                    # merely ends in a number is never touched).
                    seen_first_heading = True
                    stray_suffix = f" {ch['index']}"
                    candidate = text[: -len(stray_suffix)].rstrip() if text.endswith(stray_suffix) else text
                    if _normalize_heading(candidate) == _normalize_heading(ch["title"]):
                        continue
                if d.get("topic_number"):
                    # Missing/inconsistent/out-of-order subtopic numbering, corrected --
                    # only ever set for genuine subtopics, never recurring section labels
                    # like "Do These" (see select_text_blocks' docstring). Already-correct
                    # numbering comes back as "" and the original heading text is untouched.
                    text = f"{d['topic_number']}. {text}"
            kept.append({"block_id": b["block_id"], "page": b["page"], "content_type": d["type"], "bbox": b["bbox"], "text": text})
        else:
            d = image_decisions.get(b["block_id"])
            if not d or not d.get("keep"):
                continue
            if d.get("save_image", True):
                shared = d.get("_shared")
                existing_path = shared.get("image_path") if shared else None
                if existing_path:
                    # Same recurring image already cropped and saved for an earlier
                    # occurrence (this chapter or an earlier one) -- reuse that exact
                    # file instead of saving another near-identical copy of it.
                    image_path = existing_path
                else:
                    img_path = images_dir / f"{b['block_id']}.png"
                    images_dir.mkdir(parents=True, exist_ok=True)
                    page = doc[b["page"] - 1]
                    pix = page.get_pixmap(clip=pymupdf.Rect(*b["bbox"]), dpi=150)
                    pix.save(img_path)
                    image_path = str(img_path)
                    if shared is not None:
                        shared["image_path"] = image_path
                kept.append({
                    "block_id": b["block_id"], "page": b["page"], "content_type": "image", "bbox": b["bbox"],
                    "image_path": image_path, "description": d.get("reason", ""), "usage": "direct",
                })
            else:
                shared = d.get("_shared")
                shared_key = id(shared) if shared is not None else None
                if shared_key is not None and shared_key in seen_desc_shared:
                    # This exact recurring image's description already appeared earlier in
                    # THIS chapter (e.g. the same box illustration reused for both a "corner"
                    # and an "edge" example) -- a short back-reference instead of repeating
                    # the same paragraph again.
                    kept.append({
                        "block_id": b["block_id"], "page": b["page"], "content_type": "image_description_ref",
                        "bbox": b["bbox"],
                    })
                    continue
                if shared_key is not None:
                    seen_desc_shared[shared_key] = b["block_id"]
                # Content is real but simple/generic enough that the description alone
                # substitutes for the picture -- no crop, no file, no upload needed at all.
                kept.append({
                    "block_id": b["block_id"], "page": b["page"], "content_type": "image_description",
                    "bbox": b["bbox"], "description": d.get("reason", ""),
                    "usage": d.get("reproduction") or "draw",
                    # "draw" (a teacher reproduces it) is the conservative default when the
                    # model omits/leaves this blank -- safe for anything simple enough to
                    # reach this branch at all, whereas defaulting to "generate" risks an AI
                    # image generator garbling exact text/labels it was never confirmed to
                    # handle reliably.
                })

    kept.sort(key=lambda e: (e["page"], round(e["bbox"][1] / 10), e["bbox"][0]))
    kept = _merge_adjacent_text(kept)
    for i, e in enumerate(kept, start=1):
        e["sequence"] = i

    return {
        "book_id": book_id, "chapter_number": ch["index"], "title": ch["title"],
        "start_page": ch["start_page"], "end_page": ch["end_page"], "content": kept,
        "stats": {"total_blocks": len(blocks), "text_blocks": len(text_blocks), "image_blocks": len(image_blocks), "kept": len(kept)},
    }


def process_book_streaming(
    pdf_path: Path, book_id: str, images_dir: Path, api_key: str,
    on_chapter_done, on_chapters_detected=None, on_progress=None,
) -> dict:
    """Same detection + per-chapter logic as process_book(), but never holds
    more than one chapter's content in memory at a time -- confirmed
    necessary after a real 18-chapter live run crashed Render's 512MB free
    tier with an OOM (via Render's own log: "Ran out of memory (used over
    512MB)"), even after the fragment-merge fix. That fix solved one specific
    failure mode (thousands of tiled fragments in one region); it never
    addressed the plainer problem that process_book() kept every chapter's
    full extracted content sitting in one growing list for the entire run.

    on_chapter_done(canonical) is called immediately after each chapter
    finishes -- the caller (main.py) publishes it right away, so a later
    crash doesn't erase already-finished chapters, only the one in flight --
    after which this function drops its own reference and forces a
    garbage-collection pass before moving to the next chapter, instead of
    letting 18 chapters' worth of retained data quietly accumulate.

    on_chapters_detected(chapters_meta), if given, is called once right
    after chapter detection succeeds, before any chapter is processed --
    lets the caller create (or find an existing) Supabase book row at the
    right time, with a known chapter_count, before any per-chapter publish
    call needs it. It may return a set/iterable of chapter indices to skip
    (already published from an earlier, interrupted run of this same book)
    -- confirmed necessary after a real run got stopped partway through and
    a second upload of the exact same book would otherwise have created a
    second book row and re-paid to reprocess chapters that had already
    succeeded. Chapters in that skip set are neither re-extracted nor
    re-published; they still count toward processed_count."""
    def report(stage, detail=""):
        if on_progress:
            on_progress(stage, detail)

    doc = pymupdf.open(pdf_path)
    report("detecting_chapters", "Scanning for the table of contents")
    detection = toc_detect.detect_chapters(doc, api_key)
    if detection["status"] != "ok":
        report("failed", detection.get("reason", "chapter detection failed"))
        return {"status": "failed", "reason": detection.get("reason"), "chapter_count": 0, "processed_count": 0}

    chapters_meta = detection["chapters"]
    report("processing_chapters", f"{len(chapters_meta)} chapter(s) detected, offset={detection['offset']}")
    skip_chapters = set()
    if on_chapters_detected:
        skip_chapters = set(on_chapters_detected(chapters_meta) or ())

    image_cache = []  # shared across every chapter below -- a recurring image (banner,
    # footer icon) is judged by Gemini once and reused for every later occurrence,
    # matched via tolerant perceptual-hash comparison (see chapter_select.py's _find_cached).
    processed_count = len(skip_chapters)
    for i, ch in enumerate(chapters_meta, start=1):
        if ch["index"] in skip_chapters:
            report("processing_chapters", f"chapter {i}/{len(chapters_meta)}: '{ch['title']}' already published, skipping")
            continue

        report("processing_chapters", f"chapter {i}/{len(chapters_meta)}: {ch['title']}")
        canonical = None
        try:
            canonical = process_chapter(doc, ch, book_id, images_dir, api_key, image_cache)
            on_chapter_done(canonical)
            processed_count += 1
        except Exception as exc:
            report("chapter_failed", f"chapter {i} '{ch['title']}' failed: {exc!r}")
        finally:
            canonical = None  # drop this chapter's content before the next one starts
            gc.collect()

    report("done", f"{processed_count}/{len(chapters_meta)} chapter(s) processed")
    return {"status": "ok", "offset": detection["offset"], "chapter_count": len(chapters_meta), "processed_count": processed_count}


def process_book(pdf_path: Path, book_id: str, images_dir: Path, api_key: str, on_progress=None) -> dict:
    """Kept for existing local one-off scripts (e.g. run_maths_book.py) that
    want the whole book's content back at once, published only at the end --
    fine for a short local test, not for the real service (see
    process_book_streaming's docstring for why). Implemented on top of it so
    the two never drift apart."""
    processed = []
    result = process_book_streaming(pdf_path, book_id, images_dir, api_key, on_chapter_done=processed.append, on_progress=on_progress)
    if result["status"] != "ok":
        return {"status": "failed", "reason": result.get("reason"), "chapters": []}
    return {"status": "ok", "offset": result["offset"], "chapters": processed}
