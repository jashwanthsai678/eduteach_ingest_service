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
    "on a chapter about family)? If so, ALWAYS drop it (keep=false) -- that title text "
    "is already captured separately as a heading block, so keeping the banner too would "
    "serve the same title twice. The giveaway: the image mostly consists of a title-sized "
    "text string as its main content, not a scene/diagram illustrating a specific lesson point.\n\n"
    "OTHERWISE, decide: is this genuine lesson content (a diagram, a scene illustrating a "
    "specific point in the lesson, labeled information) that should be kept, or is it "
    "decorative page furniture (a border, a generic bullet icon, a repeated banner element)? "
    "Default to KEEPING if genuinely unsure between real content and decoration -- a "
    "wrongly-kept decorative image costs nothing; a wrongly-dropped real one loses real "
    "content. This default does NOT apply to the title-banner check above, which is always "
    "drop regardless of uncertainty.\n\n"
    'Write "reason" as a caption a teacher could use to reference this figure WITHOUT '
    "seeing it -- describe what the image actually shows (2-4 sentences: the scene, any "
    "labeled elements, names, or values visible), not a justification for your keep/drop "
    "call. If keep=false, reason can stay a brief one-line note on why it's decorative.\n\n"
    'Return ONLY {"keep": true/false, "reason": "..."}'
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
    "properties": {"keep": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["keep", "reason"],
}


def _is_qr_code(png_bytes: bytes) -> bool:
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    found, _ = cv2.QRCodeDetector().detect(img)
    return bool(found)


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


def select_image_blocks(image_blocks: list[dict], crop_fn, context_fn, api_key: str) -> dict:
    """image_blocks: [{block_id, page, bbox}]. crop_fn(block) -> png bytes.
    context_fn(block) -> nearby text string. Returns {block_id: {keep, reason, tier}}."""
    decisions = {}
    to_judge = []
    for b in image_blocks:
        area = _bbox_area(b["bbox"])
        crop = crop_fn(b)
        if area <= _TINY_IMAGE_MAX_AREA:
            decisions[b["block_id"]] = {"keep": False, "reason": "tiny icon (<1600px^2), free tier", "tier": "tiny"}
        elif _is_qr_code(crop):
            decisions[b["block_id"]] = {"keep": False, "reason": "QR code, free tier", "tier": "qr"}
        else:
            to_judge.append((b, crop))

    for b, crop in to_judge:
        result = judge_image(crop, context_fn(b), api_key)
        decisions[b["block_id"]] = {**result, "tier": "llm_visual"}

    return decisions
