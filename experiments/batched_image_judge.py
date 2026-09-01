"""VARIANT of chapter_select.select_image_blocks that judges several images per call.

Isolated: nothing in the service imports this. It becomes production only if
ab_batched_images.py shows the batched decisions stay inside the model's own
run-to-run variance.

Why: the judge prompt is ~3,079 tokens and is 74% of an average image call's
input (measured over 40 real calls: 4,139 in / 93 out). Sent once per image it
is paid for on every single image in the book. Sent once per batch it is not.

Measured projection, per 170-page book, image stage only:
    N=1 (today) $0.340    N=3 $0.198    N=5 $0.170    N=10 $0.149
N=5 captures ~85% of the total available saving; going to 10 buys about two
more cents and doubles every risk below. Hence the conservative default.

THREE THINGS THIS HAS TO GET RIGHT, all of them real rather than theoretical:

1. Request size, which bounds the batch before anything else does. Measured
   base64 crop sizes on real chapters: median 261KB, p90 1,256KB, max 3,042KB
   -- wildly skewed. A naive "5 at a time" can post 8.6MB if large crops land
   together, and this service has already hit Render's 512MB ceiling twice. So
   batches are BYTE-BUDGETED: fill to _MAX_BATCH_BYTES then flush, and a crop
   too big to share a batch goes on its own rather than dragging others into a
   doomed request.

2. Blast radius. A malformed response costs one image at N=1 and N images
   here, and a retry re-posts all N crops. So the coverage guard is keyed on
   block_id, never on position, and a batch that still fails after retries
   degrades to per-image calls instead of losing the whole group.

3. Cache interplay. Production judges sequentially, so image 5's judgment can
   be reused by image 6 in the same chapter. Images inside one batch cannot see
   each other. The free tiers still run first and the cache is still consulted
   and populated per batch, so only WITHIN-batch reuse is lost -- small on this
   book (37 of 438 were cache hits) but likely larger on an image-heavy Maths
   book. The A/B reports cache-hit counts for both arms so this is measured
   rather than assumed.
"""

import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import chapter_select as sel

_MAX_BATCH_IMAGES = 5          # hard cap; see the N-vs-saving table above
_MAX_BATCH_BYTES = 2_000_000   # ~2MB of base64 per request, the real constraint

BATCH_PREAMBLE = (
    "You are judging SEVERAL figures from one school textbook chapter in this single "
    "request. Each figure is introduced by a line reading `### FIGURE <id>` followed by "
    "that figure's own nearby lesson text, and then the figure image itself.\n\n"
    "Judge every figure INDEPENDENTLY and on its own merits, exactly as if it were the "
    "only one in front of you. Do not let one figure's decision influence another's, and "
    "do not try to make the decisions look varied or consistent as a set -- several "
    "figures in one chapter genuinely can all be drops, or all be keeps.\n\n"
    "Return one object per figure, and put that figure's `### FIGURE <id>` id in the "
    '"b" field so each judgment can be matched back. Return an object for EVERY figure '
    "given, in the same order they appear.\n\n"
    "The rules for judging each figure follow.\n\n"
)

BATCH_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "b": {"type": "string"},
            "decision": {"type": "string", "enum": ["drop", "keep_description_only", "keep_image"]},
            "reason": {"type": "string"},
            "reproduction": {"type": "string", "enum": ["", "generate", "draw"]},
            "relevance": {"type": "string", "enum": ["", "core", "supporting", "generic"]},
        },
        "required": ["b", "decision", "reason", "reproduction", "relevance"],
    },
}


def _plan_batches(items):
    """items: [(block, crop_bytes)]. Yields lists, byte-budgeted then count-capped.

    A single crop larger than the whole budget is emitted alone rather than
    skipped -- it still has to be judged, it just cannot share a request.
    """
    batch, total = [], 0
    for block, crop in items:
        size = len(crop) * 4 // 3  # base64 expansion
        if batch and (total + size > _MAX_BATCH_BYTES or len(batch) >= _MAX_BATCH_IMAGES):
            yield batch
            batch, total = [], 0
        batch.append((block, crop))
        total += size
    if batch:
        yield batch


def judge_batch(items, context_fn, api_key, chapter_context=""):
    """One call covering several figures. Returns {block_id: judgment dict}."""
    content = [{"type": "text", "text": BATCH_PREAMBLE + sel.IMAGE_JUDGE_PROMPT
                + (f"\n\nChapter context:\n{chapter_context.strip()}" if chapter_context.strip() else "")}]
    for block, crop in items:
        nearby = context_fn(block).strip()
        head = f"\n\n### FIGURE {block['block_id']}\n"
        if nearby:
            head += f"Nearby lesson text:\n{nearby}\n"
        content.append({"type": "text", "text": head})
        content.append({"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(crop).decode("ascii")}})

    payload = {
        "model": sel.MODEL,
        "messages": [{"role": "user", "content": content}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "judgments", "schema": BATCH_SCHEMA}},
    }
    expected = {b["block_id"] for b, _ in items}

    def _require_full_coverage(parsed):
        returned = {d["b"] for d in parsed if isinstance(d, dict) and "b" in d}
        missing = expected - returned
        if missing:
            raise sel.IncompleteSelection(
                f"batch judged {len(returned)}/{len(expected)} figures; missing {sorted(missing)}")

    results = sel._call_with_retry(payload, api_key, timeout=180, validate=_require_full_coverage)
    return {d["b"]: d for d in results}


def select_image_blocks_batched(image_blocks, crop_fn, context_fn, api_key,
                                image_cache=None, chapter_context=""):
    """Drop-in replacement for chapter_select.select_image_blocks.

    Identical free tiers, identical return shape (including the "_shared"
    contract pipeline.py relies on to reuse one saved file across repeats of the
    same recurring image), so adoption is a one-line swap in process_chapter.
    Only the paid tier differs: survivors are judged in byte-budgeted batches.
    """
    if image_cache is None:
        image_cache = []

    decisions, to_judge = {}, []
    for b in image_blocks:
        if sel._bbox_area(b["bbox"]) <= sel._TINY_IMAGE_MAX_AREA:
            decisions[b["block_id"]] = {"keep": False, "save_image": False,
                                        "reason": "tiny icon (<1600px^2), free tier", "tier": "tiny"}
            continue
        try:
            crop = crop_fn(b)
        except Exception as exc:
            decisions[b["block_id"]] = {"keep": False, "save_image": False,
                                        "reason": f"crop failed ({exc!r}), free tier", "tier": "crop_failed"}
            continue
        if not b.get("merged_from") and sel._is_qr_code(crop):
            decisions[b["block_id"]] = {"keep": False, "save_image": False,
                                        "reason": "QR code, free tier", "tier": "qr"}
        else:
            to_judge.append((b, crop))

    # Cache is consulted per batch rather than per image: everything already
    # judged earlier in the book still hits, only within-batch reuse is lost.
    pending = []
    for b, crop in to_judge:
        h = sel._perceptual_hash(crop)
        c = sel._color_signature(crop)
        cached = sel._find_cached(image_cache, h, c) if h is not None else None
        if cached is not None:
            decisions[b["block_id"]] = {
                "keep": cached["keep"], "save_image": cached["save_image"], "reason": cached["reason"],
                "reproduction": cached.get("reproduction", ""), "relevance": cached.get("relevance", ""),
                "tier": "llm_visual_cached", "_shared": cached,
            }
            continue
        pending.append((b, crop, h, c))

    for batch in _plan_batches([(b, crop) for b, crop, _, _ in pending]):
        ids = {b["block_id"] for b, _ in batch}
        try:
            judged_raw = judge_batch(batch, context_fn, api_key, chapter_context)
        except Exception as exc:
            # A batch that cannot be salvaged degrades to per-image calls rather
            # than losing every figure in the group -- the whole point of keeping
            # the single-image path intact.
            print(f"    batch of {len(batch)} failed ({exc!r}); falling back to per-image calls")
            judged_raw = {}
            for b, crop in batch:
                try:
                    judged_raw[b["block_id"]] = sel.judge_image(crop, context_fn(b), api_key, chapter_context)
                except Exception as inner:
                    print(f"      {b['block_id']} also failed solo: {inner!r}")

        for b, crop, h, c in pending:
            if b["block_id"] not in ids:
                continue
            r = judged_raw.get(b["block_id"])
            if r is None:
                decisions[b["block_id"]] = {"keep": False, "save_image": False,
                                            "reason": "no judgment returned", "tier": "batch_missing"}
                continue
            decision = r.get("decision", "drop")
            judged = {
                "keep": decision in ("keep_image", "keep_description_only"),
                "save_image": decision == "keep_image",
                "reason": r.get("reason", ""),
                "reproduction": r.get("reproduction", ""),
                "relevance": r.get("relevance", ""),
            }
            decisions[b["block_id"]] = {**judged, "tier": "llm_batched", "_shared": judged}
            if h is not None:
                image_cache.append((h, c, judged))

    return decisions
