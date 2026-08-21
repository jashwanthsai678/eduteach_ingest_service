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
    def on_progress(stage: str, detail: str):
        _update_job(job_id, stage=stage, detail=detail)

    try:
        _update_job(job_id, status="processing", stage="starting")
        result = pipeline.process_book(pdf_path, book_id, IMAGES_DIR / job_id, OPENROUTER_API_KEY, on_progress=on_progress)

        if result["status"] != "ok":
            _update_job(job_id, status="failed", reason=result.get("reason", "chapter detection failed"))
            return

        _update_job(job_id, stage="publishing", detail="Uploading to Supabase")
        book_uuid = publish.publish_phase2_book(pdf_path, result, board, grade, subject, language, school_id)
        _update_job(job_id, status="done", book_uuid=book_uuid, book_id=book_id, chapter_count=len(result["chapters"]))
    except Exception as exc:
        _update_job(job_id, status="failed", reason=repr(exc))
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
