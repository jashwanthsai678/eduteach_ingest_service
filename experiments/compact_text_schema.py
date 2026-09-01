"""VARIANT of chapter_select.select_text_blocks with a cheaper output format.

Isolated on purpose: nothing in the service imports this. It only becomes
production code once ab_compact_text.py shows it classifies the same blocks the
same way the shipped version does.

Why it is worth trying, from a real measured run (chapters 5-6 of Class 4 EVS,
recorded in staged_test_output/.../_livetest/_run.json): the two text calls
billed $0.0591, and 81% of that ($0.0479) was OUTPUT tokens -- 19,177 of them
to answer about 417 blocks. The answers cost the money, not the chapter text
going in. Three things make each answer bigger than it needs to be:

  1. Three spelled-out key names per block ("block_id"/"type"/"topic_number").
  2. Spelled-out category words ("textbook_question" is several tokens alone).
  3. A "keep" boolean carrying no information -- noise is the only category the
     pipeline ever drops, so keep is always (type != "noise").

This variant shortens the keys to b/t/n, replaces categories with single
letters, and drops "keep", deriving it in Python. It deliberately does NOT drop
block_id: that is what lets the full-coverage check stay in place, and
positional alignment on a 200+ item array is exactly the kind of assumption
that fails silently.

Identical to production in every other respect -- same model, same rules, same
seven categories, same subtopic-numbering logic -- so any A/B difference can
only come from the output format.

This is a CORRECTED version of the older 4-letter prototype in
paddle_ocr_vl/phase2_pymupdf_pipeline/experiments/compact_text_schema_test.py,
which predates the 7-category taxonomy. Shipping that one as-is would silently
drop key_words / summary / textbook_question -- i.e. the [KEY WORDS],
[WHAT HAVE WE LEARNT] and [TEXTBOOK QUESTION] tags -- from published output.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import chapter_select as sel

# All seven production categories. A letter added here must also be added to
# COMPACT_SCHEMA's enum and described in the prompt below.
TYPE_DECODE = {
    "c": "concept",
    "a": "activity",
    "n": "noise",
    "h": "heading",
    "k": "key_words",
    "s": "summary",
    "q": "textbook_question",
}

COMPACT_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "b": {"type": "string"},
            "t": {"type": "string", "enum": sorted(TYPE_DECODE)},
            "n": {"type": "string"},
        },
        "required": ["b", "t", "n"],
    },
}

COMPACT_PROMPT_TEMPLATE = """You are looking at the TEXT blocks of chapter {chapter_number} of one textbook, page by page, in the order they actually appear. Each has a block_id, page, bbox, and its actual extracted text. This chapter's own known title is: "{chapter_title}".

Classify each block with a SINGLE LETTER:
- "c" concept: explains/teaches an idea
- "a" activity: a task/question/exercise the student does, as part of the lesson itself
- "n" noise: running headers/footers, page numbers, decorative bars, front matter -- not real content
- "h" heading: a section/topic heading
- "k" key_words: the recurring end-of-chapter glossary/key-terms list (often labelled "Key
  words" or similar) -- a short list of this chapter's important terms.
- "s" summary: the recurring end-of-chapter bullet-point recap (often labelled "What have we
  learnt?" or similar) -- short statements restating the chapter's main points.
- "q" textbook_question: part of the chapter's FORMAL, clearly-delineated closing assessment
  section -- typically comes after the key-words/summary sections, often organized under this
  series' standard competency headings repeated every chapter ("Conceptual Understanding",
  "Questioning - Hypotheses", "Experiments - Field Observations", "Information Skills,
  Projects", "Communication through Mapping Skills, Drawing Pictures and Making Models",
  "Appreciation, Values and Awareness"), or simply a block under an explicit "Exercise"/
  "Questions" recurring section label. Use this ONLY for that formal, position-clear closing
  block -- an ordinary in-lesson activity/question earlier in the chapter (e.g. "Discuss in
  groups", "Think and Discuss", a question embedded mid-lesson) stays "a" even though it is
  also phrased as a question. The signal is being part of the chapter's clearly-marked
  closing assessment, not just "is this a question."

ADDITIONALLY, for every block classified "h": decide whether it is a GENUINE SUBTOPIC
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
they never get a topic number, and they must NOT be counted when determining subtopic sequence.

For each GENUINE SUBTOPIC heading, check whether the chapter's subtopic headings, in the order
they actually appear, already carry clean, correctly SEQUENTIAL numbers (e.g. {chapter_number}.1,
then {chapter_number}.2, then {chapter_number}.3 -- no gaps, no repeats, no out-of-order jumps).
If they already do, set "n" to an empty string "" -- it is already correct, leave it alone. If
numbering is missing entirely, inconsistent, or out of order, set "n" to the CORRECT sequential
number this heading should have, formatted as "{chapter_number}.N" (e.g. "{chapter_number}.1",
"{chapter_number}.2"), counting only genuine subtopics in the order they actually appear in the
chapter -- not the order any existing broken numbers might suggest.

For every block that is not a genuine subtopic heading (including recurring section labels, the
chapter's own title, and every non-heading block), "n" is always "".

Return ONLY a JSON array, one object per block_id given, where "b" is the block_id, "t" is the
single-letter type, and "n" is the topic number: [{{"b": ..., "t": ..., "n": ...}}]
"""


def select_text_blocks_compact(text_blocks, api_key, chapter_number=1, chapter_title=""):
    """Drop-in replacement for chapter_select.select_text_blocks.

    Returns the identical shape -- {block_id: {"type", "keep", "topic_number"}}
    with spelled-out type names -- so process_chapter needs no change at all if
    this is adopted. The compact wire format never escapes this function.
    """
    if not text_blocks:
        return {}
    prompt = COMPACT_PROMPT_TEMPLATE.format(chapter_number=chapter_number, chapter_title=chapter_title)
    payload = {
        "model": sel.MODEL,
        "messages": [{"role": "user", "content": prompt + "\n\n" + json.dumps(text_blocks, ensure_ascii=False)}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "selection", "schema": COMPACT_SCHEMA}},
    }
    expected = {b["block_id"] for b in text_blocks}

    def _require_full_coverage(parsed):
        """The same guard the shipped version uses, and the reason "b" stays in
        the schema: json_schema constrains each item's shape, never the array's
        completeness, so a block omitted from the answer is otherwise
        indistinguishable from one deliberately dropped."""
        returned = {d["b"] for d in parsed if isinstance(d, dict) and "b" in d}
        missing = expected - returned
        if missing:
            raise sel.IncompleteSelection(
                f"model classified {len(returned)}/{len(expected)} text blocks; "
                f"{len(missing)} missing (e.g. {sorted(missing)[:5]})"
            )

    decisions = sel._call_with_retry(payload, api_key, timeout=120, validate=_require_full_coverage)
    out = {}
    for d in decisions:
        # An unrecognised letter becomes noise rather than crashing the chapter;
        # the A/B harness reports these separately so they can never hide.
        t = TYPE_DECODE.get(d["t"], "noise")
        out[d["b"]] = {"type": t, "keep": t != "noise", "topic_number": d.get("n", "")}
    return out
