"""A/B harness: does the compact output format classify blocks the same way?

The trap this is built to avoid: the model is non-deterministic, so running the
shipped version twice on the same chapter does NOT produce identical answers.
Comparing the variant against a single baseline run would therefore blame the
format for disagreements that are just ordinary model variance.

So this runs the SHIPPED version twice (A and B) and the VARIANT once, then
asks one question:

    is variant-vs-A disagreement any worse than B-vs-A disagreement?

If it is not, the formats are equivalent and the cheaper one wins. If it is,
the format is costing accuracy and should not ship regardless of price.

Run:  python experiments/ab_compact_text.py [chapter_index ...]
      (defaults to chapter 5 -- the largest of the two chapters already
      measured end-to-end, at 208 text blocks)

Writes nothing to Supabase and touches no production code path.
"""

import io
import json
import os
import statistics
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
from compact_text_schema import select_text_blocks_compact

PDF = Path(r"C:\Users\d jashwanth sai\Downloads\paddle_ocr_vl\pdfs\Class 4 - Environmental Studies - 4EM_EVS.pdf")
TOC = Path(r"C:\Users\d jashwanth sai\Downloads\paddle_ocr_vl\phase2_pymupdf_pipeline\staged_test_output\Class 4 - Environmental Studies - 4EM_EVS\_toc_detection.json")
OUT = Path(__file__).parent / "_ab_results"

# --- record every call's real billed usage, without touching chapter_select ---
CALLS = []
_real_post = sel.requests.post


def recording_post(*a, **k):
    t = time.time()
    resp = _real_post(*a, **k)
    try:
        CALLS.append({"usage": resp.json().get("usage", {}), "secs": round(time.time() - t, 2)})
    except Exception:
        pass
    return resp


sel.requests.post = recording_post


def run(label, fn, blocks, ch):
    before = len(CALLS)
    t = time.time()
    decisions = fn(blocks, API_KEY, ch["index"], ch["title"])
    u = CALLS[before]["usage"] if len(CALLS) > before else {}
    info = {"label": label, "in": u.get("prompt_tokens", 0), "out": u.get("completion_tokens", 0),
            "cost": u.get("cost", 0.0), "secs": round(time.time() - t, 1)}
    print(f"  {label:12s} in={info['in']:7,}  out={info['out']:7,}  ${info['cost']:.4f}  {info['secs']:5.1f}s")
    return decisions, info


def compare(name, ref, other, blocks):
    """Disagreement rate on type, plus the specific blocks that differ."""
    diffs = []
    for b in blocks:
        bid = b["block_id"]
        rt = (ref.get(bid) or {}).get("type")
        ot = (other.get(bid) or {}).get("type")
        if rt != ot:
            diffs.append((bid, rt, ot, b["text"][:60]))
    rate = len(diffs) / len(blocks) * 100 if blocks else 0
    print(f"  {name:24s} {len(diffs):3d}/{len(blocks)} blocks differ  ({rate:.1f}%)")
    return diffs, rate


def main():
    indices = [int(a) for a in sys.argv[1:]] or [5]
    chapters = [c for c in json.load(open(TOC, encoding="utf-8"))["chapters"] if c["index"] in indices]
    doc = pymupdf.open(PDF)
    OUT.mkdir(exist_ok=True)
    summary = []

    for ch in chapters:
        print(f"\n=== chapter {ch['index']}: {ch['title']} (pages {ch['start_page']}-{ch['end_page']}) ===")
        blocks = [b for b in pl.stage5_build_blocks(doc, ch["start_page"], ch["end_page"]) if b["type"] == "text"]
        payload = [{"block_id": b["block_id"], "page": b["page"], "bbox": b["bbox"], "text": b["text"]} for b in blocks]
        print(f"  {len(payload)} text blocks\n")

        a, ia = run("baseline A", sel.select_text_blocks, payload, ch)
        b, ib = run("baseline B", sel.select_text_blocks, payload, ch)
        v, iv = run("compact", select_text_blocks_compact, payload, ch)

        print()
        _, noise_rate = compare("baseline B vs A (noise)", a, b, payload)
        diffs, var_rate = compare("compact  vs A (signal)", a, v, payload)

        unknown = [bid for bid, d in v.items() if d["type"] == "noise"
                   and (a.get(bid) or {}).get("type") not in (None, "noise")]
        saving = (ia["cost"] - iv["cost"]) / ia["cost"] * 100 if ia["cost"] else 0
        verdict = "EQUIVALENT" if var_rate <= noise_rate else "WORSE THAN NOISE"
        print(f"\n  cost: baseline ${ia['cost']:.4f} -> compact ${iv['cost']:.4f}  ({saving:+.1f}%)")
        print(f"  output tokens: {ia['out']:,} -> {iv['out']:,}")
        print(f"  VERDICT: {verdict}  (variant {var_rate:.1f}% vs model noise {noise_rate:.1f}%)")

        summary.append({"chapter": ch["index"], "blocks": len(payload), "baseline_a": ia,
                        "baseline_b": ib, "compact": iv, "noise_rate": noise_rate,
                        "variant_rate": var_rate, "verdict": verdict,
                        "reclassified_as_noise": unknown,
                        "diffs": [{"block_id": d[0], "baseline": d[1], "compact": d[2], "text": d[3]} for d in diffs]})

    json.dump(summary, open(OUT / "compact_text.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    total = sum(c["usage"].get("cost", 0) for c in CALLS)
    print(f"\nthis A/B run cost ${total:.4f} across {len(CALLS)} calls -> {OUT / 'compact_text.json'}")


if __name__ == "__main__":
    main()
