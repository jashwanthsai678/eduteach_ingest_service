"""
Stage 6, fixed version: chapter-level block selection.

The first version of this stage (see ../benchmark/run_chapter1_test.py) sent
the LLM only bbox + type + size for image blocks -- no pixels. Tested against
real data, that wrongly dropped a genuine content illustration (a children's
group-activity scene, 73x65pt) purely because it was small, alongside
correctly dropping real decorative icons of similar size. A blind size
threshold can't tell those apart; seeing the image can.

Fix, reusing Phase 1's own already-validated thresholds rather than
guessing new ones (see extract_v2.py / build_chapters.py):
  1. QR codes -> detected deterministically (cv2), always dropped, free.
  2. Truly tiny icons (<1600px^2, Phase 1's _TINY_IMAGE_MAX_AREA) -> always
     dropped, free -- badges/bullets/number-circles, never real content at
     this size in practice.
  3. Everything else -> gets an ACTUAL visual judgment: a real Gemini call
     with the cropped thumbnail + nearby text as context, same spirit as
     Phase 1's image_triage.py. This is the only tier that costs anything,
     and it's the minority of images in a typical chapter.

Text blocks still go through one cheap, chapter-wide, text-only call (no
pixels needed for text) -- that part of the original design tested fine.
"""

import base64
import json
import os
import time

import cv2
import numpy as np
import requests

_TINY_IMAGE_MAX_AREA = 1600  # Phase 1's build_chapters.py _TINY_IMAGE_MAX_AREA, reused as-is
_HASH_SIZE = 12  # 12x12 -- Phase 1's validated average-hash size (see apply_image_triage.py)
_HASH_HAMMING_THRESHOLD = 8  # out of 144 bits -- same tolerance Phase 1 validated with zero
# observed false positives across a full book; an exact-match hash would only catch
# byte-identical crops and miss the common case of the same banner/icon recurring with
# minor pixel differences (a different page number baked in, slight compression variance).
_COLOR_GRID = 6  # 6x6, in color -- coarse enough to tolerate compression noise, fine enough
# to catch a genuine recolor of the same shape (see _color_signature).
_COLOR_MAX_MEAN_DIFF = 20  # out of 255 per B/G/R channel -- validated against real data: a
# genuinely recurring banner (same crop, different page) differs by ~0-2; two crops that
# are the same shape but a different color scheme differ by 50+. 20 sits comfortably below
# a real recolor and comfortably above ordinary compression/anti-aliasing noise.
MODEL = "google/gemini-2.5-flash"
_API_URL = "https://openrouter.ai/api/v1/chat/completions"

TEXT_SELECT_PROMPT_TEMPLATE = """You are looking at the TEXT blocks of chapter {chapter_number} of one textbook, page by page, in the order they actually appear. Each has a block_id, page, bbox, and its actual extracted text. This chapter's own known title is: "{chapter_title}".

Classify each block:
- "concept": explains/teaches an idea
- "activity": a task/question/exercise the student does, as part of the lesson itself
- "noise": running headers/footers, page numbers, decorative bars, front matter -- not real content
- "heading": a section/topic heading
- "key_words": the recurring end-of-chapter glossary/key-terms list (often labelled "Key
  words" or similar) -- a short list of this chapter's important terms.
- "summary": the recurring end-of-chapter bullet-point recap (often labelled "What have we
  learnt?" or similar) -- short statements restating the chapter's main points.
- "textbook_question": part of the chapter's FORMAL, clearly-delineated closing assessment
  section -- typically comes after key_words/summary, often organized under this series'
  standard competency headings repeated every chapter ("Conceptual Understanding",
  "Questioning - Hypotheses", "Experiments - Field Observations", "Information Skills,
  Projects", "Communication through Mapping Skills, Drawing Pictures and Making Models",
  "Appreciation, Values and Awareness"), or simply a block under an explicit "Exercise"/
  "Questions" recurring section label. Use this ONLY for that formal, position-clear closing
  block -- an ordinary in-lesson activity/question earlier in the chapter (e.g. "Discuss in
  groups", "Think and Discuss", a question embedded mid-lesson) stays "activity" even though
  it's also phrased as a question. The signal is being part of the chapter's clearly-marked
  closing assessment, not just "is this a question."

ADDITIONALLY, for every block classified "heading": decide whether it is a GENUINE SUBTOPIC
heading -- a real, distinct section of the chapter's subject matter (e.g. "Rectangle", "Square",
"2.1 Rules of the Games") -- as opposed to:
- a RECURRING SECTION LABEL that appears multiple times across the chapter as a generic
  activity/exercise marker (e.g. "Do These", "Think and Discuss", "Exercise", "Key words",
  "Try This", "What have we learnt?"), or
- the CHAPTER'S OWN TITLE ("{chapter_title}") itself, or a fragment/piece of it -- large
  chapter-opener titles are sometimes split across two or more separate blocks purely because of
  page layout/font differences (e.g. "CHANGING FAMILY" and "STRUCTURE" as two blocks that
  together spell the one title "CHANGING FAMILY STRUCTURE"). Any block that is this title, or
  part of it, is the chapter's own name, not a subtopic of it.
Neither recurring section labels nor the chapter's own title/title-fragments are subtopics --
they never get a topic_number, and they must NOT be counted when determining subtopic sequence.

For each GENUINE SUBTOPIC heading, check whether the chapter's subtopic headings, in the order
they actually appear, already carry clean, correctly SEQUENTIAL numbers (e.g. {chapter_number}.1,
then {chapter_number}.2, then {chapter_number}.3 -- no gaps, no repeats, no out-of-order jumps).
If they already do, set "topic_number" to an empty string "" -- it's already correct, leave it
alone. If numbering is missing entirely, inconsistent, or out of order, set "topic_number" to the
CORRECT sequential number this heading should have, formatted as "{chapter_number}.N" (e.g.
"{chapter_number}.1", "{chapter_number}.2"), counting only genuine subtopics in the order they
actually appear in the chapter -- not the order any existing broken numbers might suggest.

For every block that is not a genuine subtopic heading (including recurring section labels, the
chapter's own title, and every non-heading block), "topic_number" is always "".

Return ONLY a JSON array: [{{"block_id": ..., "type": ..., "keep": true/false, "topic_number": ...}}], one per block_id given.
"""

IMAGE_JUDGE_PROMPT = (
    "This is one figure from a school textbook chapter, cropped from the page. "
    "Nearby lesson text is given for context, along with the chapter's title and its "
    "list of subtopics -- use the chapter-wide context (not just the nearby text) for "
    "the relevance scoring and reproduction instructions described below.\n\n"
    "FIRST, check: is this primarily a TITLE OR SECTION-LABEL BANNER -- decorative "
    "artwork with heading-style text rendered inside the graphic itself, whether that's "
    "the CHAPTER'S OWN opening title (even if the artwork is thematically related to the "
    "topic, e.g. a family drawing on a chapter about family) OR a RECURRING SECTION LABEL "
    "that appears the same way in every chapter (e.g. a colored button-style graphic "
    "reading 'Key Words', 'Do These', 'Think and Discuss', 'What have we learnt', 'Try "
    "This', 'Exercise') rendered as a graphic instead of plain text? If so, ALWAYS drop it "
    "either way -- that label text is already captured separately as its own heading/"
    "activity block, so keeping the banner too would serve the same label twice. The "
    "giveaway: the image mostly consists of a short heading-sized text string as its main "
    "content (whether the chapter's title or a recurring section name), not a scene/diagram "
    "illustrating a specific lesson point. THIS RULE ONLY APPLIES WHEN THE BANNER IS "
    "ESSENTIALLY THE WHOLE IMAGE -- if the crop ALSO contains a distinct illustrated scene "
    "(people, objects, a diagram) occupying real space alongside a title/QR/label element, "
    "this is a MIXED crop, not a pure banner: do NOT drop it on this rule. Judge the real "
    "scene on its own merits (drop/keep_description_only/keep_image per the normal criteria "
    "below) and simply ignore the incidental title/QR/label sharing the crop -- never let a "
    "small title or QR code sharing space with real content cause the real content to be "
    "thrown away too. A confirmed real case: a chapter's title banner, its QR code, and its "
    "actual opening illustration (a family looking at a photo) all ended up in one merged "
    "crop with no separating text -- the correct decision keeps the illustration; dropping "
    "the whole crop because a title happened to share the frame lost real, important content.\n\n"
    "SECOND, check: is this a QR code or barcode (a square black-and-white scannable "
    "pattern, sometimes with a short code printed under it)? If so, ALWAYS drop it -- QR "
    "codes link to external digital content and are never real lesson content on their "
    "own, regardless of how visually distinct or 'important to reproduce exactly' the "
    "pattern looks -- UNLESS (same MIXED-crop exception as above) real illustrated content "
    "shares the crop with it, in which case judge that real content normally and ignore the "
    "QR code. (A deterministic check normally catches a standalone QR code for free before "
    "this prompt ever runs; this rule is the backstop for the rare case it misses one, or "
    "for a merged crop the deterministic check correctly skipped for exactly this reason.)\n\n"
    "THIRD, check: is this a standalone decorative mascot/character illustration -- a "
    "smiling child or similar character portrait (head-and-shoulders or full-body), shown "
    "alone with no labeled diagram, scene, or lesson-specific action around it, used purely "
    "as visual filler in the margin or between activity items? If so, ALWAYS drop it, even "
    "though it's a genuine hand-drawn illustration -- it never carries lesson content, "
    "regardless of how its clothing/hair color differs from other occurrences of the same "
    "character elsewhere in the book (this book's recurring 'helper kids' are recolored "
    "between appearances, but that doesn't make any one appearance more lesson-specific than "
    "another). THIS RULE ONLY APPLIES to an isolated character portrait with no other "
    "content; a character actually DOING something lesson-relevant (pointing at a labeled "
    "diagram, part of an illustrated scene the nearby activity refers to) is real content "
    "and must be judged normally instead.\n\n"
    "FOURTH, check: is this a colored, bordered callout/template box (e.g. a light-color "
    "rounded rectangle framing worked-example text like 'Example : 35 = 30 + 5') whose "
    "ENTIRE visible text content is already given to you verbatim in the nearby lesson text "
    "below? If so, ALWAYS drop it -- that content already exists as real, separately-"
    "extracted text; keeping the image too would only produce a redundant 'draw a colored "
    "box containing this same text' instruction that adds no new information. This does NOT "
    "apply if the image contains anything beyond that box's own text (an additional diagram, "
    "illustration, or values not present in the nearby text) -- judge that normally instead.\n\n"
    "OTHERWISE, decide between exactly three outcomes:\n"
    '- "drop": decorative page furniture (a border, a generic bullet icon, a repeated '
    "banner element) -- not real lesson content.\n"
    '- "keep_description_only": genuine lesson content, but SIMPLE and GENERIC enough that '
    "a good text description fully substitutes for the original picture -- a teacher could "
    "redraw it on a blackboard from the description alone, or an AI image generator could "
    "recreate an equivalent illustration from it (e.g. a plain labeled diagram of a few "
    "circles/arrows showing a cycle, a simple line drawing of a common object, a generic "
    "icon-style illustration of a concept).\n"
    '- "keep_image": genuine lesson content that must be kept as the ACTUAL image -- a real '
    "photograph, a map, a diagram with many specific labels/values/shapes that a description "
    "cannot faithfully reproduce, or anything where the exact visual detail matters (a "
    "specific historical picture, a data chart, a diagram whose precise layout is part of "
    "what's being taught).\n\n"
    "IF (and only if) decision is \"keep_description_only\", ALSO set \"reproduction\" to how "
    "this specific image should be recreated downstream:\n"
    '- "generate": purely visual content with no embedded text/labels a viewer needs to be '
    "exactly right -- a plain object, icon, or simple scene. Safe for an AI image generator "
    "to recreate, since there's no text it could get wrong.\n"
    '- "draw": the image depends on specific text, labels, or exact positions to make its '
    "point (a labeled diagram, an arrow pointing at a named part, dots at exact positions) -- "
    "AI image generators are unreliable at rendering embedded text accurately, so a human "
    "(teacher) reproducing it from the description is the safer, correct choice, even though "
    "the image itself is simple enough to skip saving. When genuinely unsure between the two, "
    'choose "draw" -- it always works, "generate" is only better when confidently safe.\n'
    'Leave "reproduction" as an empty string "" for "drop" and "keep_image" -- it does not '
    "apply to either.\n\n"
    "IF decision is NOT \"drop\", ALSO set \"relevance\" to how central this image actually "
    "is to THIS CHAPTER specifically, using the chapter title and subtopic list given above "
    "-- this applies to keep_image just as much as keep_description_only, since even a real, "
    "kept photograph can be core to the lesson or just incidental supporting art:\n"
    '- "core": directly illustrates a specific subtopic this chapter is actually teaching -- '
    "removing it would lose something the chapter specifically needs.\n"
    '- "supporting": genuinely related to the chapter\'s subject area, but not tied to one '
    "specific subtopic being taught -- helpful context, not essential.\n"
    '- "generic": could sit in almost any chapter on almost any topic (a generic classroom '
    "scene, a stock icon, decorative filler) -- present, but not meaningfully tied to what "
    "this specific chapter teaches.\n"
    'Leave "relevance" as an empty string "" for "drop" -- it does not apply.\n\n'
    "IMPORTANT EXCEPTION -- check the nearby lesson text for this before applying the "
    "simplicity test above: if the activity asks the student to COUNT something shown in "
    "the image (e.g. 'how many circles do you see'), or to TRACE, COPY, or JOIN something "
    "shown in it (e.g. 'join the dots given below', 'trace the outline'), always choose "
    '"keep_image" regardless of how visually simple the image itself looks. The activity '
    "depends on the student inspecting the real image themselves -- a text description "
    "would either hand them the answer outright (for a counting task) or leave them unable "
    "to do the activity at all (for a tracing/copying task, where the description can't "
    "reproduce exact dot positions or line paths). This overrides the simplicity test even "
    "for images that would otherwise clearly be keep_description_only -- but it only changes "
    'the "decision", not how "reason" is written: even here, "reason" stays a pure caption '
    "of what's actually drawn (shapes, positions, counts visible), never a sentence "
    "explaining that the image was kept because the activity requires it.\n\n"
    "Default to KEEPING (either keep_image or keep_description_only) if genuinely unsure "
    "between real content and decoration -- a wrongly-kept decorative image costs nothing; a "
    "wrongly-dropped real one loses real content. When unsure whether kept content needs the "
    "real image or just a description, prefer keep_image -- only choose "
    "keep_description_only when you're confident the description alone is sufficient. This "
    "default does NOT apply to the title-banner check above, which is always drop regardless "
    "of uncertainty.\n\n"
    'HOW TO WRITE "reason" DIFFERS BY DECISION -- it is never a justification for your '
    "decision in any case (never phrasing like 'the exact visual details are important' or "
    "'this requires the actual image for context'), but its actual content and audience "
    "differ:\n\n"
    'For "keep_image": a plain, objective caption describing what the image actually shows '
    "(the scene, any labeled elements, names, or values visible) -- a teacher could use it to "
    "reference the figure without needing to look at the file. Nothing more.\n\n"
    'For "keep_description_only": "reason" is a REPRODUCTION INSTRUCTION, not a passive '
    "caption -- it has to actually work for whoever uses it to recreate this image, and its "
    "style depends on which \"reproduction\" you chose:\n"
    '- If "draw": write it as simple, blackboard-friendly guidance a teacher can sketch '
    "quickly by hand in class -- what shapes/figures to draw and roughly how they're "
    "arranged. Skip exact colors or fine texture detail a hand sketch can't capture anyway.\n"
    '- If "generate": write it as a fuller visual prompt suitable for an AI image generator '
    "-- color, style, and composition are fine here, since a generator can actually render "
    "them.\n"
    "GROUND the instruction in the chapter context (title + subtopics) AND the nearby "
    "lesson text when this image is \"core\" or \"supporting\" relevance -- don't just "
    "describe the picture, NAME the specific subtopic or concept it connects to, using the "
    "chapter's own subtopic wording where one applies (e.g. text about grandparents living "
    "with grandchildren, next to an image of exactly that, under subtopic '1.2 Types of "
    "families' -- say so explicitly: 'a grandfather with two grandchildren, illustrating "
    "subtopic 1.2's point about joint families caring for children together', not just 'an "
    "old man with two children'). This keeps a redraw or regeneration in sync with the "
    "chapter instead of producing something that merely resembles the original but no "
    "longer serves the lesson. If relevance is \"generic\" -- genuinely generic supporting "
    "art with no specific tie to any subtopic (could illustrate almost any lesson -- a "
    "stock 'child writing in a notebook' scene, a generic 'people discussing' scene), do "
    "not invent a false connection to sound more relevant than it is -- give the plain, "
    "honest instruction for what to draw/generate instead, with no fabricated topic tie.\n\n"
    'MATCH THE LENGTH OF "reason" TO THE IMAGE\'S ACTUAL COMPLEXITY, for both cases above -- '
    "do not pad every description out to the same length regardless of content, and do not "
    "compress a detailed image down to something that loses what's actually needed. A "
    "simple image (one object, a basic icon, a single clear subject with nothing else "
    "notable) should get a short, plain description, often a single sentence -- e.g. 'A "
    "red toothpaste tube with a white cap.' A genuinely complex image (multiple distinct "
    "elements, labels, values, or things that need to be referenced separately) can "
    "reasonably take 2-4 sentences to cover what's actually needed, but never more than "
    "needed. If decision is \"drop\", reason can stay a brief one-line note on why it's "
    "decorative.\n\n"
    'Return ONLY {"decision": "drop"/"keep_description_only"/"keep_image", "reason": "...", '
    '"reproduction": "generate"/"draw"/"", "relevance": "core"/"supporting"/"generic"/""}'
)

TEXT_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "block_id": {"type": "string"},
            "type": {"type": "string", "enum": ["concept", "activity", "noise", "heading", "key_words", "summary", "textbook_question"]},
            "keep": {"type": "boolean"},
            "topic_number": {"type": "string"},
        },
        "required": ["block_id", "type", "keep", "topic_number"],
    },
}

IMAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["drop", "keep_description_only", "keep_image"]},
        "reason": {"type": "string"},
        "reproduction": {"type": "string", "enum": ["", "generate", "draw"]},
        "relevance": {"type": "string", "enum": ["", "core", "supporting", "generic"]},
    },
    "required": ["decision", "reason", "reproduction", "relevance"],
}


def _is_qr_code(png_bytes: bytes) -> bool:
    """cv2's detector is unreliable on small crops -- a real QR code (~170x185px)
    was observed slipping through undetected, reaching a paid Gemini call that then
    had to catch it instead. Upscaling small crops before detection is the standard
    fix for this class of detector; the IMAGE_JUDGE_PROMPT's explicit QR backstop
    rule covers whatever this still misses."""
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    if max(img.shape) < 400:
        scale = 400 / max(img.shape)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    found, _ = cv2.QRCodeDetector().detect(img)
    return bool(found)


def _perceptual_hash(png_bytes: bytes) -> np.ndarray | None:
    """Average-hash (12x12 grayscale, threshold at the mean) -- same size
    and approach Phase 1 validated with zero observed false positives
    across a full book (a stricter dual-hash variant was tried and
    reverted for regressing -- see apply_image_triage.py's
    find_recurring_images). Used to recognize the SAME recurring image
    (banners, footer icons) reappearing across pages/chapters so it's
    only ever sent to Gemini once. Returns a boolean array compared via
    Hamming distance, not exact equality -- real recurring images vary by
    a few pixels between occurrences (a different page number baked into
    the same banner, minor compression differences), so an exact-match
    hash would miss almost all of them.

    Grayscale-only, so on its own this can't tell two images apart that
    share the same shapes/composition but differ in color (e.g. the same
    line-art icon recolored for a different section) -- that's what
    _color_signature below is for; the two are checked together in
    _find_cached, never on their own."""
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (_HASH_SIZE, _HASH_SIZE), interpolation=cv2.INTER_AREA)
    return (small > small.mean()).flatten()


def _color_signature(png_bytes: bytes) -> np.ndarray | None:
    """A coarse color fingerprint -- average B/G/R per cell of a small 6x6 grid (108
    values), closing the one real gap _perceptual_hash has on its own: it's grayscale,
    so a recolored version of the same shape would otherwise hash identically and get
    wrongly treated as the same recurring image, silently reusing a judgment that was
    never actually made for this specific (differently-colored) picture. Compared by
    mean absolute difference, not exact match, for the same reason the shape hash uses
    a tolerance -- real recurring images vary slightly between occurrences (JPEG
    compression noise, a different page number baked in)."""
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    small = cv2.resize(img, (_COLOR_GRID, _COLOR_GRID), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32).flatten()


def _find_cached(image_cache: list, img_hash: np.ndarray, color_sig: np.ndarray | None) -> dict | None:
    for cached_hash, cached_color, judged in image_cache:
        if np.count_nonzero(img_hash != cached_hash) > _HASH_HAMMING_THRESHOLD:
            continue
        if color_sig is not None and cached_color is not None:
            if float(np.abs(color_sig - cached_color).mean()) > _COLOR_MAX_MEAN_DIFF:
                continue  # same shape, different color -- not actually the same image
        return judged
    return None


def _bbox_area(bbox: list) -> float:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


class IncompleteSelection(RuntimeError):
    """The model returned well-formed JSON that silently omits blocks it was asked
    to classify. Its own class so a partial answer is distinguishable from a
    transport/parse failure both in the retry loop and in a caller's logs."""


def _call_with_retry(payload: dict, api_key: str, timeout: int, max_attempts: int = 3, validate=None):
    """Two real chapters (out of 16, in a live run) failed with JSONDecodeError --
    the model occasionally returns truncated/malformed JSON even under a schema
    constraint. No retry existed before; this is the fix, same backoff pattern
    Phase 1 already uses for its own model calls (extract_v2.py predict_page,
    image_triage.py categorize_image).

    validate(parsed), if given, is called on the decoded response and must raise to
    reject it; a rejected response is retried exactly like a transport or parse
    failure. Needed because that same truncation has a quieter second form than
    JSONDecodeError: a response that parses perfectly but only covers SOME of the
    blocks it was given. `response_format: json_schema` constrains each item's
    SHAPE, never the array's COMPLETENESS, so nothing else in this path catches
    it -- see select_text_blocks' own validator for why that silently lost real
    textbook text."""
    last_exc = RuntimeError("failed on every attempt")
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.post(_API_URL, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json=payload, timeout=timeout)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            parsed = json.loads(raw)
            if validate is not None:
                validate(parsed)
            return parsed
        except Exception as exc:
            last_exc = exc
            print(f"    attempt {attempt}/{max_attempts} failed: {exc!r}")
            if attempt < max_attempts:
                time.sleep(5 * attempt)
    raise last_exc


def select_text_blocks(text_blocks: list[dict], api_key: str, chapter_number: int = 1, chapter_title: str = "") -> dict:
    """text_blocks: [{block_id, page, bbox, text}]. Returns {block_id: {type, keep, topic_number}}.

    topic_number is "" for everything except genuine subtopic headings (never
    for recurring section labels like "Do These"/"Think and Discuss", and
    never for non-heading blocks) -- non-empty only when that heading's
    numbering is missing/inconsistent/out of order and needs a correct
    sequence number attached (e.g. "3.1"); empty when the chapter's existing
    numbering is already correct and should be left alone. Rides along in
    this same call rather than a separate one -- the model already has every
    heading in the chapter, in order, as part of classifying blocks in the
    first place, so this costs nothing extra to ask for."""
    if not text_blocks:
        return {}
    prompt = TEXT_SELECT_PROMPT_TEMPLATE.format(chapter_number=chapter_number, chapter_title=chapter_title)
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt + "\n\n" + json.dumps(text_blocks, ensure_ascii=False)}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "selection", "schema": TEXT_SCHEMA}},
    }
    expected = {b["block_id"] for b in text_blocks}

    def _require_full_coverage(parsed):
        """Every block sent must come back with a decision. Without this check, a
        block the model simply OMITTED was indistinguishable from one it decided to
        drop: pipeline.py's `d = text_decisions.get(block_id); if not d or not
        d.get("keep"): continue` treats a missing entry exactly like keep=False, so
        an incomplete response silently deleted real textbook text from the
        published chapter with no error, no warning, and nothing in the stats to
        show it (`stats.kept` only ever counted what survived). That is the exact
        failure this pipeline exists to avoid -- the whole reason Phase 2 does the
        cutting in Python is so served text is traceable to the real page.

        Deliberately RAISES rather than defaulting the missing blocks to keep=True:
        a chapter that fails loudly is skipped by process_book_streaming's per-
        chapter try/except (reported as "chapter_failed", the rest of the book
        continues) and is picked up by publish.py's resume logic on the next run,
        whereas a silently patched-up chapter would be published with `published=
        true` and served to schools with nobody aware it is wrong."""
        returned = {d["block_id"] for d in parsed if isinstance(d, dict) and "block_id" in d}
        missing = expected - returned
        if missing:
            raise IncompleteSelection(
                f"model classified {len(returned)}/{len(expected)} text blocks; "
                f"{len(missing)} missing (e.g. {sorted(missing)[:5]})"
            )

    decisions = _call_with_retry(payload, api_key, timeout=120, validate=_require_full_coverage)
    return {d["block_id"]: d for d in decisions}


def judge_image(image_bytes: bytes, context_text: str, api_key: str, chapter_context: str = "") -> dict:
    """One real visual judgment call, only for images not resolved by the free tiers.

    chapter_context: a compact, reusable summary (chapter title + its subtopic list --
    see pipeline.py's _build_chapter_context) passed to every image judgment call in the
    same chapter, so it can score how relevant an image actually is to what the chapter
    teaches and name the right subtopic in a reproduction instruction, not just describe
    the picture. Deliberately compact rather than the chapter's full text -- that would
    repeat thousands of words across every image call in the chapter for no matching
    benefit."""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    text = IMAGE_JUDGE_PROMPT
    if chapter_context.strip():
        text += f"\n\nChapter context:\n{chapter_context.strip()}"
    if context_text.strip():
        text += f"\n\nNearby lesson text:\n{context_text.strip()}"
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "judgment", "schema": IMAGE_SCHEMA}},
    }
    return _call_with_retry(payload, api_key, timeout=60)


def select_image_blocks(image_blocks: list[dict], crop_fn, context_fn, api_key: str, image_cache: list | None = None, chapter_context: str = "") -> dict:
    """image_blocks: [{block_id, page, bbox}]. crop_fn(block) -> png bytes.
    context_fn(block) -> nearby text string. Returns {block_id: {keep, save_image, reason, tier}}.

    save_image distinguishes two kinds of "keep": True means the actual
    cropped image must be saved/uploaded (a photo, map, or a diagram whose
    precise visual detail matters); False means the content is genuine but
    simple/generic enough that "reason" alone (a teacher-redrawable /
    AI-regeneratable description) is sufficient, so no file is ever cropped,
    saved, or uploaded for it downstream -- real storage/upload savings for
    the images that don't need to be pixel-faithful.

    image_cache: optional list of (perceptual_hash, judgment_result) pairs,
    shared and mutated across calls (e.g. across every chapter in a book) so
    a recurring image (banner, footer icon) is only ever sent to Gemini
    once -- every later occurrence, even with minor pixel differences (a
    different page number baked into the same banner, slight compression
    variation), reuses the cached decision via a tolerant Hamming-distance
    match instead of paying for another vision call. A plain dict keyed by
    exact hash was tried first and would only catch byte-identical crops --
    real recurring images almost never hash bit-for-bit identical, so this
    is a list scanned with _find_cached's tolerance instead. Pass the SAME
    list into every call for a book to get the cross-chapter benefit; omit
    it to fall back to no caching.

    Each returned decision also carries "_shared": for llm_visual/
    llm_visual_cached tiers, this is the SAME dict object stored inside
    image_cache for that hash -- every occurrence of one recurring image
    points at one shared object. The caller (pipeline.py) uses this to
    record the saved file's path after the FIRST occurrence crops and
    saves it, so every later occurrence of the same image reuses that
    exact file instead of cropping and saving a fresh (nearly-identical,
    but not byte-identical) copy of the same picture again. Without this,
    the judgment cache alone still avoids the Gemini cost per recurrence,
    but not the crop/save/upload cost -- confirmed on real data: a
    recurring illustration that appeared 4 times produced 4 separate
    ~35KB files, all the same picture."""
    if image_cache is None:
        image_cache = []

    decisions = {}
    to_judge = []
    for b in image_blocks:
        area = _bbox_area(b["bbox"])
        if area <= _TINY_IMAGE_MAX_AREA:
            decisions[b["block_id"]] = {"keep": False, "save_image": False, "reason": "tiny icon (<1600px^2), free tier", "tier": "tiny"}
            continue
        try:
            crop = crop_fn(b)
        except Exception as exc:
            # A crop that cannot be rasterised must never cost the whole chapter. The one
            # known cause -- image placements outside the page rect -- is fixed at source
            # in pipeline.py's stage5_build_blocks, so anything reaching here is a PDF
            # shape this pipeline has not seen before or a genuinely corrupt image object.
            # In either case dropping the single image is right, and losing the other ~15
            # pages of the chapter is not -- which is exactly what an uncaught raise here
            # used to do.
            decisions[b["block_id"]] = {
                "keep": False, "save_image": False,
                "reason": f"crop failed ({exc!r}), free tier", "tier": "crop_failed",
            }
            continue
        # merged_from means this crop is the union of several original blocks (fragment or
        # composite merge) -- _is_qr_code's cv2 detector fires on ANY QR pattern found
        # anywhere in the crop, not just a crop that's PURELY a QR code, so a merged block
        # containing a real QR code alongside unrelated real content (confirmed real case:
        # a chapter's title/QR/icon composite-merged with its actual opening illustration,
        # with no real text between them) was being dropped whole for free, losing the real
        # illustration along with the QR code. Only apply this free tier to a genuinely
        # unmerged, standalone block; a merged one always goes to the real judgment call,
        # where the prompt's QR backstop rule can correctly separate "drop the QR part" from
        # "keep the real content" instead of an all-or-nothing pixel-pattern check.
        if not b.get("merged_from") and _is_qr_code(crop):
            decisions[b["block_id"]] = {"keep": False, "save_image": False, "reason": "QR code, free tier", "tier": "qr"}
        else:
            to_judge.append((b, crop))

    for b, crop in to_judge:
        img_hash = _perceptual_hash(crop)
        color_sig = _color_signature(crop)
        cached = _find_cached(image_cache, img_hash, color_sig) if img_hash is not None else None
        if cached is not None:
            decisions[b["block_id"]] = {
                "keep": cached["keep"], "save_image": cached["save_image"], "reason": cached["reason"],
                "reproduction": cached.get("reproduction", ""), "relevance": cached.get("relevance", ""),
                "tier": "llm_visual_cached", "_shared": cached,
            }
            continue
        result = judge_image(crop, context_fn(b), api_key, chapter_context=chapter_context)
        decision = result.get("decision", "drop")
        judged = {
            "keep": decision in ("keep_image", "keep_description_only"),
            "save_image": decision == "keep_image",
            "reason": result.get("reason", ""),
            "reproduction": result.get("reproduction", ""),
            "relevance": result.get("relevance", ""),
        }
        decisions[b["block_id"]] = {**judged, "tier": "llm_visual", "_shared": judged}
        if img_hash is not None:
            image_cache.append((img_hash, color_sig, judged))

    return decisions
