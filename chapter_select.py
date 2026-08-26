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
MODEL = "google/gemini-2.5-flash"
_API_URL = "https://openrouter.ai/api/v1/chat/completions"

TEXT_SELECT_PROMPT = """You are looking at the TEXT blocks of one textbook chapter, page by page. Each has a block_id, page, bbox, and its actual extracted text.

Classify each block:
- "concept": explains/teaches an idea
- "activity": a task/question/exercise the student does
- "noise": running headers/footers, page numbers, decorative bars, front matter -- not real content
- "heading": a section/topic heading

Return ONLY a JSON array: [{"block_id": ..., "type": ..., "keep": true/false}], one per block_id given.
"""

IMAGE_JUDGE_PROMPT = (
    "This is one figure from a school textbook chapter, cropped from the page. "
    "Nearby lesson text is given for context.\n\n"
    "FIRST, check: is this primarily a CHAPTER-OPENER TITLE BANNER -- decorative "
    "artwork with the chapter/section title text rendered inside the graphic itself "
    "(even if the artwork is thematically related to the topic, e.g. a family drawing "
    "on a chapter about family)? If so, ALWAYS drop it -- that title text is already "
    "captured separately as a heading block, so keeping the banner too would serve the "
    "same title twice. The giveaway: the image mostly consists of a title-sized text "
    "string as its main content, not a scene/diagram illustrating a specific lesson point.\n\n"
    "SECOND, check: is this a QR code or barcode (a square black-and-white scannable "
    "pattern, sometimes with a short code printed under it)? If so, ALWAYS drop it -- QR "
    "codes link to external digital content and are never real lesson content on their "
    "own, regardless of how visually distinct or 'important to reproduce exactly' the "
    "pattern looks. (A deterministic check normally catches these for free before this "
    "prompt ever runs; this rule is the backstop for the rare case it misses one.)\n\n"
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
    'Write "reason" as a caption/description a teacher could use to reference or redraw this '
    "figure WITHOUT seeing it -- describe what the image actually shows (the scene, any "
    "labeled elements, names, or values visible), not a justification for your decision. "
    'This applies to EVERY decision, including "keep_image" -- never add a closing sentence '
    "explaining why the image needed to be kept or why a description wouldn't be enough "
    "(e.g. avoid phrasing like 'the exact visual details are important' or 'this requires "
    "the actual image for context') -- 'reason' is always pure descriptive caption text, "
    "never a justification.\n\n"
    'MATCH THE LENGTH OF "reason" TO THE IMAGE\'S ACTUAL COMPLEXITY -- do not pad every '
    "description out to the same length regardless of content, and do not compress a "
    "detailed image down to something that loses what a teacher would actually need. A "
    "simple image (one object, a basic icon, a single clear subject with nothing else "
    "notable) should get a short, plain description, often a single sentence -- e.g. 'A "
    "red toothpaste tube with a white cap.' A genuinely complex image (multiple distinct "
    "elements, labels, values, or things a teacher would need to reference separately) can "
    "reasonably take 2-4 sentences to cover what's actually there, but never more than "
    "needed. If decision is \"drop\", reason can stay a brief one-line note on why it's "
    "decorative.\n\n"
    'Return ONLY {"decision": "drop"/"keep_description_only"/"keep_image", "reason": "..."}'
)

TEXT_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "block_id": {"type": "string"},
            "type": {"type": "string", "enum": ["concept", "activity", "noise", "heading"]},
            "keep": {"type": "boolean"},
        },
        "required": ["block_id", "type", "keep"],
    },
}

IMAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["drop", "keep_description_only", "keep_image"]},
        "reason": {"type": "string"},
    },
    "required": ["decision", "reason"],
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
    hash would miss almost all of them."""
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (_HASH_SIZE, _HASH_SIZE), interpolation=cv2.INTER_AREA)
    return (small > small.mean()).flatten()


def _find_cached(image_cache: list, img_hash: np.ndarray) -> dict | None:
    for cached_hash, judged in image_cache:
        if np.count_nonzero(img_hash != cached_hash) <= _HASH_HAMMING_THRESHOLD:
            return judged
    return None


def _bbox_area(bbox: list) -> float:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _call_with_retry(payload: dict, api_key: str, timeout: int, max_attempts: int = 3):
    """Two real chapters (out of 16, in a live run) failed with JSONDecodeError --
    the model occasionally returns truncated/malformed JSON even under a schema
    constraint. No retry existed before; this is the fix, same backoff pattern
    Phase 1 already uses for its own model calls (extract_v2.py predict_page,
    image_triage.py categorize_image)."""
    last_exc = RuntimeError("failed on every attempt")
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.post(_API_URL, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json=payload, timeout=timeout)
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            return json.loads(raw)
        except Exception as exc:
            last_exc = exc
            print(f"    attempt {attempt}/{max_attempts} failed: {exc!r}")
            if attempt < max_attempts:
                time.sleep(5 * attempt)
    raise last_exc


def select_text_blocks(text_blocks: list[dict], api_key: str) -> dict:
    """text_blocks: [{block_id, page, bbox, text}]. Returns {block_id: {type, keep}}."""
    if not text_blocks:
        return {}
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": TEXT_SELECT_PROMPT + "\n\n" + json.dumps(text_blocks, ensure_ascii=False)}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "selection", "schema": TEXT_SCHEMA}},
    }
    decisions = _call_with_retry(payload, api_key, timeout=120)
    return {d["block_id"]: d for d in decisions}


def judge_image(image_bytes: bytes, context_text: str, api_key: str) -> dict:
    """One real visual judgment call, only for images not resolved by the free tiers."""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    text = IMAGE_JUDGE_PROMPT
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


def select_image_blocks(image_blocks: list[dict], crop_fn, context_fn, api_key: str, image_cache: list | None = None) -> dict:
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
        crop = crop_fn(b)
        if _is_qr_code(crop):
            decisions[b["block_id"]] = {"keep": False, "save_image": False, "reason": "QR code, free tier", "tier": "qr"}
        else:
            to_judge.append((b, crop))

    for b, crop in to_judge:
        img_hash = _perceptual_hash(crop)
        cached = _find_cached(image_cache, img_hash) if img_hash is not None else None
        if cached is not None:
            decisions[b["block_id"]] = {
                "keep": cached["keep"], "save_image": cached["save_image"], "reason": cached["reason"],
                "tier": "llm_visual_cached", "_shared": cached,
            }
            continue
        result = judge_image(crop, context_fn(b), api_key)
        decision = result.get("decision", "drop")
        judged = {
            "keep": decision in ("keep_image", "keep_description_only"),
            "save_image": decision == "keep_image",
            "reason": result.get("reason", ""),
        }
        decisions[b["block_id"]] = {**judged, "tier": "llm_visual", "_shared": judged}
        if img_hash is not None:
            image_cache.append((img_hash, judged))

    return decisions
