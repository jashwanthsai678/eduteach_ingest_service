"""A/B harness: does batching several images per call change the judgements?

Same method as ab_compact_text.py, for the same reason: the model is not
deterministic, so a variant compared against a single baseline gets blamed for
ordinary variance. Runs the shipped per-image path TWICE (A and B) and the
batched path once, then asks:

    is batched-vs-A disagreement any worse than B-vs-A disagreement?

Disagreement is measured on the decision that actually changes what gets
published -- drop / keep_description_only / keep_image -- not on the wording of
the description, which is free text and never identical between runs.

It separately reports CONTENT LOSS: images the baseline kept that the batch
dropped. That is the failure that matters; a saving is not worth one.

Run:  python experiments/ab_batched_images.py [chapter ...]     (default 5 6)
"""

import io
import json
import os
import sys
import time
from pathlib import Path

SVC = Path(__file__).parent.parent
sys.path.insert(0, str(SVC))
sys.path.insert(0, str(Path(__file__).parent))

for line in io.open(SVC / ".env", encoding="utf-8"):
    if "=" in line:
        k, v = line.strip().split("=", 1)
        os.environ.setdefault(k, v)
API_KEY = os.environ["OPENROUTER_API_KEY"]

import pymupdf

import chapter_select as sel
import pipeline as pl
from batched_image_judge import select_image_blocks_batched

PDF = Path(r"C:\Users\d jashwanth sai\Downloads\paddle_ocr_vl\pdfs\Class 4 - Environmental Studies - 4EM_EVS.pdf")
TOC = Path(r"C:\Users\d jashwanth sai\Downloads\paddle_ocr_vl\phase2_pymupdf_pipeline\staged_test_output\Class 4 - Environmental Studies - 4EM_EVS\_toc_detection.json")
OUT = Path(__file__).parent / "_ab_results"

CALLS = []
_real_post = sel.requests.post


def recording_post(*a, **k):
    t = time.time()
    r = _real_post(*a, **k)
    try:
        CALLS.append({"usage": r.json().get("usage", {}), "secs": round(time.time() - t, 2)})
    except Exception:
        pass
    return r


sel.requests.post = recording_post


def arm(label, fn, blocks, crop_fn, context_fn, ctx):
    before = len(CALLS)
    t = time.time()
    # a FRESH cache per arm, so one arm never inherits another's judgements
    out = fn(blocks, crop_fn, context_fn, API_KEY, [], ctx)
    calls = CALLS[before:]
    info = {
        "label": label, "calls": len(calls),
        "in": sum(c["usage"].get("prompt_tokens", 0) for c in calls),
        "out": sum(c["usage"].get("completion_tokens", 0) for c in calls),
        "cost": sum(c["usage"].get("cost", 0) for c in calls),
        "secs": round(time.time() - t, 1),
        "tiers": {},
    }
    for d in out.values():
        info["tiers"][d["tier"]] = info["tiers"].get(d["tier"], 0) + 1
    print(f"  {label:10s} {info['calls']:3d} calls  in={info['in']:7,}  out={info['out']:6,}  "
          f"${info['cost']:.4f}  {info['secs']:5.1f}s   {info['tiers']}")
    return out, info


def verdict(a, other, blocks):
    """Compare the published-facing decision: drop / description-only / image."""
    def state(d):
        if not d.get("keep"):
            return "drop"
        return "keep_image" if d.get("save_image") else "keep_description_only"

    diffs, lost = [], []
    judged = 0
    for b in blocks:
        bid = b["block_id"]
        da, do = a.get(bid), other.get(bid)
        if not da or not do:
            continue
        if da["tier"] in ("tiny", "qr", "crop_failed"):
            continue                      # free tiers are deterministic, not under test
        judged += 1
        sa, so = state(da), state(do)
        if sa != so:
            diffs.append((bid, sa, so))
            if sa != "drop" and so == "drop":
                lost.append(bid)
    rate = len(diffs) / judged * 100 if judged else 0
    return diffs, lost, rate, judged


def main():
    idx = [int(x) for x in sys.argv[1:]] or [5, 6]
    chapters = [c for c in json.load(open(TOC, encoding="utf-8"))["chapters"] if c["index"] in idx]
    doc = pymupdf.open(PDF)
    OUT.mkdir(exist_ok=True)
    summary = []

    for ch in chapters:
        print(f"\n=== chapter {ch['index']}: {ch['title']} ===")
        blocks = pl.stage5_build_blocks(doc, ch["start_page"], ch["end_page"])
        text_blocks = [b for b in blocks if b["type"] == "text"]
        image_blocks = [b for b in blocks if b["type"] == "image"]
        by_page = {}
        for b in blocks:
            by_page.setdefault(b["page"], []).append(b)

        def crop_fn(b):
            pix = doc[b["page"] - 1].get_pixmap(clip=pymupdf.Rect(*b["bbox"]), dpi=150)
            return pix.tobytes("png")

        def context_fn(b):
            sib = by_page[b["page"]]
            i = sib.index(b)
            return "\n".join(s["text"] for s in sib[max(0, i - 2):i] + sib[i + 1:i + 3] if s["type"] == "text")

        # a cheap chapter context; the text pass is not under test here
        ctx = f"Chapter: {ch['title']}."
        print(f"  {len(image_blocks)} image blocks\n")

        a, ia = arm("baseline A", sel.select_image_blocks, image_blocks, crop_fn, context_fn, ctx)
        b, ib = arm("baseline B", sel.select_image_blocks, image_blocks, crop_fn, context_fn, ctx)
        v, iv = arm("batched", select_image_blocks_batched, image_blocks, crop_fn, context_fn, ctx)

        _, _, noise, n = verdict(a, b, image_blocks)
        diffs, lost, rate, _ = verdict(a, v, image_blocks)

        saving = (ia["cost"] - iv["cost"]) / ia["cost"] * 100 if ia["cost"] else 0
        ok = rate <= noise and not lost
        print(f"\n  judged images compared: {n}")
        print(f"  baseline B vs A (noise)   {noise:5.1f}%")
        print(f"  batched   vs A (signal)   {rate:5.1f}%")
        print(f"  content lost (kept -> dropped): {lost or 'NONE'}")
        print(f"  calls {ia['calls']} -> {iv['calls']}   cost ${ia['cost']:.4f} -> ${iv['cost']:.4f}  ({saving:+.1f}%)")
        print(f"  VERDICT: {'EQUIVALENT' if ok else 'REJECT'}")

        summary.append({"chapter": ch["index"], "compared": n, "noise_rate": noise, "batched_rate": rate,
                        "content_lost": lost, "saving_pct": saving, "verdict": "EQUIVALENT" if ok else "REJECT",
                        "baseline_a": ia, "baseline_b": ib, "batched": iv,
                        "diffs": [{"block_id": d[0], "baseline": d[1], "batched": d[2]} for d in diffs]})

    json.dump(summary, open(OUT / "batched_images.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    total = sum(c["usage"].get("cost", 0) for c in CALLS)
    print(f"\nthis A/B run cost ${total:.4f} across {len(CALLS)} calls -> {OUT / 'batched_images.json'}")


if __name__ == "__main__":
    main()
