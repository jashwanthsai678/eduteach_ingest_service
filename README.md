# EduTeach Ingest Service (Phase 2)

Upload a textbook PDF, get back a `job_id`, poll it until the pipeline
finishes. Runs the full Phase 2 pipeline (PyMuPDF extraction, automated
chapter detection, per-chapter LLM content/image selection) and publishes
into Supabase. This is the **only** one of the three repos that writes to
the database — the other two only ever read from it.

## How the three repos fit together

```
                 uploads a PDF, polls job status
edu_teach_textbook_api_interface  ─────────────────►  eduteach-ingest-service  (this repo)
   (the "Find a textbook" site)                              │
         │                                                    │ writes books / chapters / images
         │ reads published content                            ▼
         └──────────────────────────────────────►         Supabase
                                                        (Postgres + Storage)
                                                                ▲
                                                                │ reads published content
                                                      eduteach-textbook-api
                                                    (read-only REST API)
```

- **`edu_teach_textbook_api_interface`** ([repo](https://github.com/jashwanthsai678/edu_teach_textbook_api_interface)) — the frontend. Posts a PDF to this service's `POST /jobs`, polls `GET /jobs/{job_id}` for progress, and separately reads already-published content from `eduteach-textbook-api`. Never touches Supabase directly.
- **`eduteach-textbook-api`** ([repo](https://github.com/jashwanthsai678/eduteach-textbook-api)) — a separate, read-only service. Never talks to this repo directly; it only reads whatever this repo has already written to Supabase.
- **This repo** — the only one with write access to the database (via `SUPABASE_SERVICE_ROLE_KEY`). Everything in `publish.py` is the single source of truth for what the schema actually looks like.

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

DELETE /books/{book_id}?password=...
  -> cascading delete: image rows + their Storage files -> chapter rows -> the book row.
     The password is a fat-finger guard for the UI, not real access control -- anyone
     who can call this API directly could already delete a book with or without it.
```

## How it works

1. `toc_detect.py` -- fully automatic chapter-boundary detection: finds
   TOC-candidate pages (cheap keyword scan), parses them with one LLM
   call into `{title, printed_start_page}`, then confirms the real
   printed-vs-actual page offset by searching for chapter titles' real
   occurrences after the TOC and taking the offset the majority agree on.
   No hardcoded per-book chapter list.
2. `chapter_select.py` -- per-chapter LLM selection: one text-classification
   call per chapter (concept/activity/heading/noise/key_words/summary/
   textbook_question, plus subtopic numbering), and a tiered image
   judgment -- free deterministic tiers first (tiny icons, QR codes on
   standalone blocks, full-page-width decorative background strips, and a
   perceptual-hash-plus-color-signature cache for recurring images), a real
   visual LLM call for everything that survives those.
3. `pipeline.py` -- orchestrates 1+2 into full canonical chapter data:
   interleaves text and image blocks into true reading order, merges PDF
   fragmentation artifacts and genuine small-object collages (with
   safeguards against merging unrelated content together), merges
   consecutive same-type text fragments into one tag, and drops the
   chapter's duplicated opening title.
4. `publish.py` -- writes each chapter to Supabase **immediately** as it
   finishes (not batched at the end -- a crash partway through only loses
   the one chapter in flight, not the whole book) and uploads real images
   to Supabase Storage, deduplicated so a recurring image is only ever
   uploaded once.

## The Supabase schema

Three tables, all in the default `public` schema, plus one Storage bucket.
This repo is the only writer; `eduteach-textbook-api` only ever reads.

### `textbook_books`
| column | notes |
|---|---|
| `id` | uuid, primary key |
| `school_id` | uuid -- a single shared-library placeholder value today (`04b5b4aa-37cc-4790-955e-e995da9b80c7`), not real per-school auth yet |
| `book_id` | deterministic slug: `{board}_class{grade}_{subject}_{language}`, lowercased, spaces -> underscores -- e.g. `ts_scert_class5_maths_en` |
| `board`, `grade`, `subject`, `language` | as given at upload time |
| `source_pdf`, `source_pdf_sha256` | provenance of the original file |
| `chapter_count` | the *detected* total from the PDF's table of contents -- can be more than how many chapters are actually published yet |

### `textbook_chapters`
| column | notes |
|---|---|
| `id` | uuid, primary key |
| `book_uuid` | FK -> `textbook_books.id` |
| `school_id` | |
| `chapter_number`, `chapter_title`, `page_start`, `page_end` | |
| `content_markdown` | the tagged content string (`[CONCEPT]`/`[ACTIVITY]`/`[HEADING]`/`[KEY WORDS]`/`[WHAT HAVE WE LEARNT]`/`[TEXTBOOK QUESTION]`, plus `<img id="..."/>` placeholders) |
| `content_sha256` | |
| `published` | bool -- the read API only ever serves rows where this is `true` |

### `textbook_images`
| column | notes |
|---|---|
| `id` | uuid, primary key |
| `chapter_uuid` | FK -> `textbook_chapters.id` |
| `school_id` | |
| `image_id` | matches an `<img id="...">` placeholder in that chapter's `content_markdown` |
| `caption` | a plain description for a real file, or a reproduction instruction (see `usage`) when there's no file |
| `storage_path` | path inside the `textbook-images` bucket, **nullable** -- null when `usage` is `"generate"` or `"draw"` (no file was ever saved) |
| `source_page`, `width`, `height`, `bytes` | `width`/`height`/`bytes` are null whenever `storage_path` is null |
| `order_index` | position within the chapter |
| `usage` | `"direct"` (real file, use `storage_path`), `"generate"` (caption is safe to hand an AI image generator), or `"draw"` (caption should go to a teacher instead -- depends on exact text/labels an image generator can't reliably render) |
| `drive_file_id` | legacy, nullable -- from an earlier pre-Supabase-Storage version of the pipeline. Not written by any code in this repo; only `eduteach-textbook-api` still reads it as a fallback for old rows. |

### Storage bucket
- `textbook-images`, public bucket
- path pattern: `{book_id}/ch{chapter_number:02d}/{image_id}.jpg`
- images are compressed on upload (max 1600px dimension, JPEG quality 70) and
  deduplicated -- a recurring image is only ever uploaded once, every later
  occurrence reuses the same `storage_path`.

## If you're changing the database

Every write in this repo goes through `publish.py`. Update these specific
spots to match a schema change, then check the other two repos' READMEs
for what else depends on the old shape:

- `create_book_row()` -- the `textbook_books` insert
- `publish_one_chapter()` -- the `textbook_chapters` insert and the
  `textbook_images` insert (the loop building `image_rows`)
- `delete_book()` -- deletes across all three tables plus the matching
  Storage files; a new table with a chapter-level or book-level foreign
  key needs to be added here too, in dependency order (children before
  parents), or deletes will start leaving orphaned rows.
- `find_existing_book()` and `publish_one_chapter`'s resume logic -- read
  `textbook_books`/`textbook_chapters` to figure out what's already
  published; a column rename here needs to match what `create_book_row`/
  `publish_one_chapter` actually wrote.

A schema change here is only half done until `eduteach-textbook-api`'s
`select=` lists (what it reads) and `edu_teach_textbook_api_interface`'s
`app.js` (what fields it displays) are updated to match -- see their
READMEs.

## Environment variables required

```
SUPABASE_URL
SUPABASE_SERVICE_ROLE_KEY
OPENROUTER_API_KEY
```

## Known, real limitations (not hidden)

- **Job state is in-memory.** If the service restarts mid-job (a deploy,
  or Render's free tier spinning down an idle instance), that job's
  progress is lost. The pipeline itself is resume-safe (it skips already-
  published chapters on retry), but the frontend loses track of the job ID
  to poll. A real fix would persist job state to Supabase or use a real
  queue.
- **Processing takes real time** -- up to ~20 minutes for a full book,
  since images that aren't caught by the free tiers each need a real LLM
  call. The frontend needs to poll patiently, not assume a fast response.
- **Large PDF uploads can fail at the browser/network level** before this
  service ever sees them -- a plain multipart upload of a 100MB+ file is
  fragile over an imperfect connection. Not yet mitigated (a PDF
  pre-compression step was validated -- ~50% size reduction, zero content
  loss -- but isn't wired in yet).

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
