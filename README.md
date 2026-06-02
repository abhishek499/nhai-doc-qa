# NHAI Document Q&A Assistant

A RAG-based document Q&A web app for NHAI public documents, deployable to Heroku.
Users ask natural-language questions and get answers grounded **only** in the indexed
documents, with inline citations back to source file and page number.

## Tech stack

| Layer | Choice |
|---|---|
| Backend | FastAPI + uvicorn |
| Generation | Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) |
| Embeddings | Voyage AI `voyage-3-lite` or OpenAI `text-embedding-3-small` |
| Vector store | FAISS `IndexFlatIP` (exact cosine similarity) |
| Frontend | Vanilla HTML/JS — no framework |

---

## Local setup

### 1. Prerequisites

- Python 3.12
- An [Anthropic API key](https://console.anthropic.com/)
- Either a [Voyage AI key](https://www.voyageai.com/) or an [OpenAI key](https://platform.openai.com/)

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

### 4. Add base documents

Place your NHAI PDF files in the `data/` directory.

### 5. Build the base index (run once, re-run when PDFs change)

```bash
python scripts/build_index.py
```

This writes `app/index/base.index` and `app/index/base_metadata.json`.
**Commit these files** — they are the pre-built artifact shipped to Heroku.

### 6. Run locally

```bash
uvicorn app.main:app --reload
```

Open http://localhost:8000

---

## Heroku deployment

```bash
# One-time setup
heroku create your-app-name
heroku config:set ANTHROPIC_API_KEY=your_key
heroku config:set EMBEDDING_PROVIDER=voyage
heroku config:set VOYAGE_API_KEY=your_key

# Optional tunables
heroku config:set TOP_K=5
heroku config:set MAX_UPLOAD_MB=15
heroku config:set MAX_PAGES=150

# Deploy
git push heroku main
```

### Heroku notes

**Ephemeral filesystem**
Heroku dynos have a read-write filesystem that is wiped on every restart, sleep,
or deploy. The base index (`app/index/`) is committed to git and therefore baked
into the slug — it is always present. Documents uploaded at runtime are added to
the in-memory FAISS index only; they are lost when the dyno restarts. This is
a known and acceptable trade-off for a free/hobby tier deployment.

**30-second request timeout**
Heroku terminates requests after 30 seconds. The upload guards (15 MB / 150 page
limits) ensure that embedding a PDF finishes well within this window.

**Dyno memory (~512 MB)**
The FAISS index and metadata live in process memory. For a typical base corpus of
a few hundred NHAI PDFs (tens of thousands of chunks) this is well within limits.
If you scale to hundreds of thousands of chunks, upgrade to a Standard-2X dyno.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required |
| `EMBEDDING_PROVIDER` | `voyage` | `voyage` or `openai` |
| `VOYAGE_API_KEY` | — | Required when provider is `voyage` |
| `OPENAI_API_KEY` | — | Required when provider is `openai` |
| `TOP_K` | `5` | Chunks retrieved per query |
| `CHUNK_SIZE` | `3500` | Characters per chunk (~875 tokens) |
| `CHUNK_OVERLAP` | `350` | Overlap between consecutive chunks (~88 tokens) |
| `MAX_UPLOAD_MB` | `15` | Maximum upload file size |
| `MAX_PAGES` | `150` | Maximum pages per uploaded PDF |

---

## Architecture

```
User question
    │
    ▼
embed_texts([question])          ← Voyage / OpenAI API
    │
    ▼
FAISS IndexFlatIP.search(k=5)   ← exact cosine similarity (dot product on unit vectors)
    │
    ▼
Top-K chunks  {text, filename, page}
    │
    ▼
Grounding prompt  →  Claude Haiku 4.5
    │
    ▼
{answer, sources: [{filename, page, snippet}]}
```

### Key design decisions

**Cosine similarity via `IndexFlatIP`**
Vectors are L2-normalised before insertion. With unit vectors, inner product equals
cosine similarity, so `IndexFlatIP` gives exact cosine search with no approximation
overhead — ideal for corpora below ~100 k chunks.

**Chunk size ~875 tokens with 88-token overlap**
Large enough to contain a complete clause or table row; small enough that retrieved
chunks stay on-topic. The overlap prevents answers from being split across chunk
boundaries when a sentence straddles two consecutive windows.

**Cite-or-refuse grounding**
Claude is instructed to respond with a fixed phrase when context is insufficient.
This eliminates hallucination from prior knowledge. Citation tags (`[file p.X]`) are
embedded directly in the context so Claude can reproduce them verbatim.

**Offline index build**
Building and committing the base index means zero embedding API calls at boot time
and no cold-start latency on Heroku.
