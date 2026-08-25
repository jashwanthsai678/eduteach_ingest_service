"""
Upload a PDF, get back a job_id; poll it until the Phase 2 pipeline finishes
and the book is published to the same Supabase project Phase 1 uses.

Known, real limitation (documented, not hidden): jobs are tracked in an
in-memory dict. If this service restarts mid-job (Render free tier can
spin down an idle instance, and a deploy always restarts it), that job's
status is lost -- the pipeline itself isn't resumable across a restart
yet. Acceptable for an initial version; a real job queue (or persisting
job state to Supabase) is the fix if this becomes unreliable in practice.
"""

import os
import shutil
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import pipeline
import publish

APP_DIR = Path(__file__).parent
UPLOAD_DIR = APP_DIR / "uploads"
IMAGES_DIR = APP_DIR / "images"
UPLOAD_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]

app = FastAPI(title="EduTeach Ingest Service (Phase 2)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _update_job(job_id: str, **fields):
    with _jobs_lock:
        _jobs[job_id].update(fields)


def _run_job(job_id: str, pdf_path: Path, book_id: str, board: str, grade: str, subject: str, language: str, school_id: str):
    """Publishes each chapter to Supabase immediately after it finishes,
    instead of collecting the whole book first and publishing at the very
    end -- confirmed necessary after a real 18-chapter live run hit Render's
    512MB OOM ceiling with NOTHING published, even though most/all chapters
    had already been extracted and paid for. With this, a crash midway
    through only loses the one chapter in flight; every chapter already
    finished is already sitting in Supabase by the time it happens."""
    def on_progress(stage: str, detail: str):
        _update_job(job_id, stage=stage, detail=detail)

    uploaded_cache: dict[str, tuple] = {}
    state = {"book_uuid": None, "published_count": 0, "problem_chapters": []}

    def on_chapters_detected(chapters_meta):
        _update_job(job_id, stage="publishing", detail="Creating book record")
        state["book_uuid"] = publish.create_book_row(pdf_path, book_id, board, grade, subject, language, school_id, len(chapters_meta))

    def on_chapter_done(canonical):
        status = publish.publish_one_chapter(state["book_uuid"], book_id, school_id, canonical, uploaded_cache)
        state["published_count"] += 1
        if status["problems"]:
            state["problem_chapters"].append(status["chapter_number"])
        _update_job(job_id, stage="processing_chapters", detail=f"chapter {status['chapter_number']} published ({state['published_count']} so far)")

    try:
        _update_job(job_id, status="processing", stage="starting")
        result = pipeline.process_book_streaming(
            pdf_path, book_id, IMAGES_DIR / job_id, OPENROUTER_API_KEY,
            on_chapter_done=on_chapter_done, on_chapters_detected=on_chapters_detected, on_progress=on_progress,
        )

        if result["status"] != "ok":
            # Chapter detection itself failed -- nothing could have been published yet either way.
            _update_job(job_id, status="failed", reason=result.get("reason", "chapter detection failed"))
            return

        _update_job(
            job_id, status="done", book_uuid=state["book_uuid"], book_id=book_id,
            chapter_count=state["published_count"], problem_chapters=state["problem_chapters"],
        )
    except Exception as exc:
        # Even on a hard failure, surface whatever DID make it to Supabase before the
        # crash -- book_uuid/chapter_count here reflect real, already-published chapters,
        # not zero, since publishing now happens incrementally rather than at the end.
        _update_job(
            job_id, status="failed", reason=repr(exc), book_uuid=state["book_uuid"],
            book_id=book_id, chapter_count=state["published_count"],
        )
    finally:
        pdf_path.unlink(missing_ok=True)


@app.get("/")
async def root():
    return {"service": "EduTeach Ingest Service (Phase 2)", "jobs_endpoint": "/jobs"}


@app.post("/jobs")
async def create_job(
    file: UploadFile = File(...),
    board: str = Form(...),
    grade: str = Form(...),
    subject: str = Form(...),
    language: str = Form("en"),
    school_id: str = Form(...),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")

    job_id = str(uuid.uuid4())
    pdf_path = UPLOAD_DIR / f"{job_id}.pdf"
    with pdf_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    book_id = f"{board.lower().replace(' ', '_')}_class{grade}_{subject.lower().replace(' ', '_')}_{language}"
    with _jobs_lock:
        _jobs[job_id] = {"status": "queued", "stage": None, "detail": None}

    thread = threading.Thread(
        target=_run_job, args=(job_id, pdf_path, book_id, board, grade, subject, language, school_id), daemon=True,
    )
    thread.start()

    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"unknown job_id {job_id!r}")
    return job
