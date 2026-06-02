"""
FAISS vector store for the NHAI Document Q&A Assistant.

Design choices:
- IndexFlatIP: exact inner-product search; with L2-normalised vectors this is
  exact cosine similarity. No approximation needed for corpora <~100 k chunks.
- The base index is built offline (scripts/build_index.py), committed, and loaded
  once at startup. This avoids cold-start latency and embedding API calls on boot.
- Uploaded documents are added to the same in-memory index at request time.
  They are lost on dyno restart — acceptable per the project spec.
- A threading.Lock guards all mutations (add) and reads (search) because FastAPI
  may handle concurrent requests in the same process.
"""

import json
import threading
from pathlib import Path
from typing import Any

import faiss
import numpy as np

INDEX_DIR = Path(__file__).parent / "index"
INDEX_FILE = INDEX_DIR / "base.index"
METADATA_FILE = INDEX_DIR / "base_metadata.json"

# Module-level state — shared across all requests in the same dyno process
_index: faiss.IndexFlatIP | None = None
_metadata: list[dict[str, Any]] = []  # 1-to-1 aligned with FAISS vector ids
_lock = threading.Lock()


def load_base_index() -> None:
    """Load the pre-built FAISS index and metadata from the committed artifact."""
    global _index, _metadata

    if not INDEX_FILE.exists() or not METADATA_FILE.exists():
        print("Warning: No base index found at app/index/. Run scripts/build_index.py first.")
        print("Starting with an empty in-memory vector store.")
        return

    with _lock:
        _index = faiss.read_index(str(INDEX_FILE))
        with open(METADATA_FILE, encoding="utf-8") as f:
            _metadata = json.load(f)

    print(f"Base index loaded: {_index.ntotal} vectors, {len(_metadata)} chunks.")


def add_chunks(chunks: list[dict[str, Any]]) -> None:
    """Embed chunks and add them to the in-memory index (thread-safe)."""
    global _index, _metadata

    if not chunks:
        return

    # Local import avoids a circular dependency at module-load time
    from app.embeddings import embed_texts

    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts)  # shape (N, D), already L2-normalised

    with _lock:
        if _index is None:
            dim = vectors.shape[1]
            _index = faiss.IndexFlatIP(dim)
        _index.add(vectors)
        _metadata.extend(
            {"text": c["text"], "source_filename": c["source_filename"], "page": c["page"]}
            for c in chunks
        )


def search(query: str, k: int = 5) -> list[dict[str, Any]]:
    """
    Embed a query and return the top-k most similar chunks.
    Each result dict: {text, source_filename, page, score}.
    """
    global _index, _metadata

    if _index is None or _index.ntotal == 0:
        return []

    from app.embeddings import embed_texts

    query_vec = embed_texts([query])          # shape (1, D), normalised
    actual_k = min(k, _index.ntotal)

    with _lock:
        if query_vec.shape[1] != _index.d:
            raise RuntimeError(
                f"Embedding dimension mismatch: query has {query_vec.shape[1]} dims "
                f"but the loaded index has {_index.d} dims. "
                f"The index was built with a different embedding provider. "
                f"Re-run scripts/build_index.py with the current EMBEDDING_PROVIDER "
                f"and restart the server."
            )
        scores, indices = _index.search(query_vec, actual_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        meta = _metadata[idx]
        results.append({
            "text": meta["text"],
            "source_filename": meta["source_filename"],
            "page": meta["page"],
            "score": float(score),
        })
    return results


def list_documents() -> list[dict[str, Any]]:
    """Return a deduplicated list of indexed documents with chunk counts."""
    counts: dict[str, int] = {}
    for meta in _metadata:
        fname = meta["source_filename"]
        counts[fname] = counts.get(fname, 0) + 1
    return [{"filename": fname, "chunks": count} for fname, count in sorted(counts.items())]
