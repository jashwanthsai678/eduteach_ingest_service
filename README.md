# EduTeach Ingest Service (Phase 2)

Upload a textbook PDF, get back a `job_id`, poll it until the pipeline
finishes. Runs the full Phase 2 pipeline (PyMuPDF extraction, automated
chapter detection, per-chapter LLM content/image selection) and publishes
into the **same Supabase project and tables** Phase 1 uses -- content
published here is servable through the exact same `eduteach-textbook-api`
with zero changes there.

## Endpoints

```
POST /jobs
  multipart form: file (PDF), board, grade, subject, language, school_id
  -> {"job_id": "...", "status": "queued"}

GET /jobs/{job_id}
  -> {"status": "queued" | "processing" | "done" | "failed",
      "stage": ..., "detail": ...,
      "book_uuid": ... (once done), "chapter_count": ... (once done),
      "reason": ... (if failed)}
```

## How it works

1. `toc_detect.py` -- fully automatic chapter-boundary detection: finds
   TOC-candidate pages (cheap keyword scan), parses them with one LLM
   call into `{title, printed_start_page}`, then confirms the real
   printed-vs-actual page offset by searching for chapter titles' real
   occurrences after the TOC and taking the offset the majority agree on.
   No hardcoded per-book chapter list -- validated against two real books
   whose correct answers were already known from manual testing, and it
   reproduced both exactly.
2. `chapter_select.py` -- per-chapter LLM selection: one text call per
   chapter (concept/activity/heading/noise), and a tiered image judgment
   (free rules for QR codes and tiny icons, a real visual LLM call with
   the actual cropped thumbnail for everything else -- this is the fix
   for a real bug found in testing, where a genuine content image got
   wrongly dropped for being small).
3. `pipeline.py` -- orchestrates 1+2 into full canonical chapter data.
4. `publish.py` -- adapts that data into the same shape Phase 1's
   `chapter_content.py` produces, then calls the same Supabase
   insert/image-upload/validation logic Phase 1's `publish_book.py` uses
   (copied here, not imported across repos, since this deploys standalone).

## Environment variables required

```
SUPABASE_URL
SUPABASE_SERVICE_ROLE_KEY
OPENROUTER_API_KEY
```

## Known, real limitations (not hidden)

- **Job state is in-memory.** If the service restarts mid-job (a deploy,
  or Render's free tier spinning down an idle instance), that job's
  progress is lost. Acceptable for now; a real fix would persist job
  state to Supabase or use a real queue.
- **Processing takes real time** -- roughly 20 minutes for a 200-page,
  16-chapter book, measured on a real run. The frontend needs to poll
  patiently, not assume a fast response.
- **Chapter detection can report `status: "needs_review"`** if it can't
  confidently parse a TOC or confirm a consistent page offset -- this is
  deliberate (it never silently guesses a wrong offset), but it means
  some books will need a manual fallback path that doesn't exist yet.

## Local run

```
pip install -r requirements.txt
export SUPABASE_URL=... SUPABASE_SERVICE_ROLE_KEY=... OPENROUTER_API_KEY=...
uvicorn main:app --port 8000
```

## Deploying on Render

**New +** -> **Web Service** (not a Static Site -- this needs to run
Python). Build command: `pip install -r requirements.txt`. Start command:
`uvicorn main:app --host 0.0.0.0 --port $PORT`. Add the three environment
variables above in the service's Environment settings.
