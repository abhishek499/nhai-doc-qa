"""NHAI Document Q&A Assistant — FastAPI application."""

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

load_dotenv()

# Upload guardrails (Heroku: 30 s request timeout, ~512 MB dyno memory).
# Keeping uploads small ensures synchronous embedding finishes well within 30 s.
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "50"))
MAX_PAGES = int(os.getenv("MAX_PAGES", "500"))

STATIC_DIR = Path(__file__).parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the pre-built base index once at startup — never rebuild at runtime.
    # Heroku's filesystem is ephemeral; the serialised artifact lives in the slug.
    from app import store
    # Pre-load the embedding model so the first request doesn't hit the
    # 30-second Heroku timeout waiting for fastembed to download its ONNX model.
    from app.embeddings import embed_texts
    embed_texts(["warmup"])

    store.load_base_index()
    yield


app = FastAPI(title="NHAI Document Q&A Assistant", lifespan=lifespan)


# ── Schemas ───────────────────────────────────────────────────────────────────

class HistoryMessage(BaseModel):
    role: str       # "user" or "assistant"
    content: str

class AskRequest(BaseModel):
    question: str
    history: list[HistoryMessage] = []


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    """Serve the single-page chat UI."""
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return FileResponse(str(html_path), media_type="text/html")
    # Fallback while static/index.html is not yet built
    return JSONResponse({"status": "ok", "message": "NHAI Document Q&A Assistant is running."})


@app.post("/ask")
async def ask(request: AskRequest):
    """
    Answer a natural-language question grounded in the indexed documents.
    Returns {answer, sources} where each source has filename, page, and snippet.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    from app import rag
    result = rag.answer(
        request.question,
        history=[m.model_dump() for m in request.history],
    )
    return result


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    Accept a PDF upload, chunk + embed it, and add it to the in-memory index.
    Guards: non-PDF rejected, >15 MB rejected, >150 pages rejected.
    Uploaded documents persist only for this dyno's lifetime (Heroku ephemeral FS).
    """
    filename = file.filename or "upload.pdf"

    # Validate file type by extension (content-type header can be spoofed)
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    content = await file.read()

    # Size guard — rejects files that would push embedding time past 30 s
    max_bytes = int(MAX_UPLOAD_MB * 1024 * 1024)
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {MAX_UPLOAD_MB:.0f} MB.",
        )

    # Write to a temp file so pypdf can open it by path
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        from pypdf import PdfReader
        reader = PdfReader(tmp_path)
        if len(reader.pages) > MAX_PAGES:
            raise HTTPException(
                status_code=413,
                detail=f"PDF has {len(reader.pages)} pages. Maximum allowed is {MAX_PAGES}.",
            )

        from app.ingest import load_pdf
        from app import store

        chunks = load_pdf(tmp_path)
        # Override the source_filename with the original upload name
        for chunk in chunks:
            chunk["source_filename"] = filename

        store.add_chunks(chunks)
    finally:
        os.unlink(tmp_path)

    return {"filename": filename, "chunks_added": len(chunks)}


@app.get("/documents")
async def documents():
    """List all currently indexed documents with their chunk counts."""
    from app import store
    return {"documents": store.list_documents()}
