"""
NHAI Document Q&A Assistant — vector + keyword store with hybrid retrieval.

WHY HYBRID RETRIEVAL
--------------------
Pure semantic (dense) search excels at paraphrased questions but misses exact
token matches.  NHAI documents are full of tokens that carry meaning only when
matched exactly:

  • Clause / section numbers  — "4.1.2", "Schedule-II"
  • Fee / penalty amounts     — "₹150", "Rs.500"
  • Acronyms & initialisms    — "FASTag", "RFID", "NHAI", "NH-48"
  • Circular / notification   — "Circular No. RW/NH-33048/1/2015"

BM25 (sparse) retrieves by term frequency and catches these verbatim tokens
that a dense embedding might conflate with semantically similar but distinct
chunks.  Running both and fusing the results covers both failure modes.

RECIPROCAL RANK FUSION (RRF)
-----------------------------
RRF fuses two ranked lists without needing comparable score scales:

    score(doc) = Σ  1 / (k_rrf + rank_i)      k_rrf = 60 (per original paper)
                lists

A document ranked #1 in one list and #5 in the other beats a document that
appears in only one list.  k_rrf = 60 dampens the advantage of top-1 hits so
mid-list overlap is still rewarded.  The result is returned as the "score"
field so the call-site (rag.py) needs no changes.

FAISS DESIGN
------------
IndexFlatIP: exact inner-product search; with L2-normalised vectors this is
exact cosine similarity. No approximation needed for corpora <~100 k chunks.
The base index is built offline, committed, and loaded once at startup.
Uploaded documents are added to the in-memory index at request time (ephemeral).
A threading.Lock guards all mutations and reads for concurrency safety.
"""

import json
import threading
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

INDEX_DIR = Path(__file__).parent / "index"
INDEX_FILE = INDEX_DIR / "base.index"
METADATA_FILE = INDEX_DIR / "base_metadata.json"

# Candidates drawn from each retriever before fusion.
# Larger than default top_k so RRF has enough overlap to work with.
_CANDIDATE_K = 20

# RRF dampening constant (Cormack et al. 2009).
_RRF_K = 60

# Module-level state — shared across all requests in the same process
_index: faiss.IndexFlatIP | None = None
_metadata: list[dict[str, Any]] = []   # 1-to-1 aligned with FAISS vector ids
_bm25: BM25Okapi | None = None         # mirrors _metadata exactly
_lock = threading.Lock()


# ── Internal helpers ──────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Whitespace tokenizer.  No stemming — preserves clause numbers, amounts,
    acronyms, and other exact tokens that BM25 must match verbatim."""
    return text.lower().split()


def _rebuild_bm25() -> None:
    """Rebuild the BM25 index from the current _metadata list.
    Must be called while _lock is held (rank_bm25 has no incremental add)."""
    global _bm25
    if not _metadata:
        _bm25 = None
        return
    corpus = [_tokenize(m["text"]) for m in _metadata]
    _bm25 = BM25Okapi(corpus)


# ── Public API ────────────────────────────────────────────────────────────────

def load_base_index() -> None:
    """Load the pre-built FAISS index and metadata from the committed artifact,
    then build the BM25 index over the same corpus."""
    global _index, _metadata

    if not INDEX_FILE.exists() or not METADATA_FILE.exists():
        print("Warning: No base index found at app/index/. Run scripts/build_index.py first.")
        print("Starting with an empty in-memory vector store.")
        return

    with _lock:
        _index = faiss.read_index(str(INDEX_FILE))
        with open(METADATA_FILE, encoding="utf-8") as f:
            _metadata = json.load(f)
        _rebuild_bm25()

    print(f"Base index loaded: {_index.ntotal} vectors, {len(_metadata)} chunks.")


def add_chunks(chunks: list[dict[str, Any]]) -> None:
    """Embed chunks, add to FAISS, extend metadata, and rebuild BM25 (thread-safe)."""
    global _index, _metadata

    if not chunks:
        return

    from app.embeddings import embed_texts  # local import avoids circular dep

    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts)  # shape (N, D), already L2-normalised

    with _lock:
        if _index is None:
            _index = faiss.IndexFlatIP(vectors.shape[1])
        _index.add(vectors)
        _metadata.extend(
            {"text": c["text"], "source_filename": c["source_filename"], "page": c["page"]}
            for c in chunks
        )
        # BM25 has no incremental add — cheapest correct option is a full rebuild.
        # For corpora <100 k chunks this takes well under 1 s.
        _rebuild_bm25()


def search(query: str, k: int = 5) -> list[dict[str, Any]]:
    """
    Hybrid retrieval: dense (FAISS cosine) + sparse (BM25), fused with RRF.

    1. Retrieve up to _CANDIDATE_K (20) results from each retriever.
    2. Score every candidate with RRF: 1 / (_RRF_K + rank), summed across lists.
    3. Return the top-k by RRF score.

    Return shape is identical to the previous dense-only implementation:
      [{text, source_filename, page, score}, ...]
    so /ask requires no changes.
    """
    global _index, _metadata, _bm25

    if _index is None or _index.ntotal == 0:
        return []

    from app.embeddings import embed_texts

    query_vec = embed_texts([query])  # shape (1, D), normalised
    cand_k = min(_CANDIDATE_K, _index.ntotal)

    with _lock:
        # ── dimension guard ───────────────────────────────────────────────────
        if query_vec.shape[1] != _index.d:
            raise RuntimeError(
                f"Embedding dimension mismatch: query has {query_vec.shape[1]} dims "
                f"but the loaded index has {_index.d} dims. "
                f"Re-run scripts/build_index.py with the current EMBEDDING_PROVIDER "
                f"and restart the server."
            )

        # ── dense retrieval ───────────────────────────────────────────────────
        faiss_scores, faiss_indices = _index.search(query_vec, cand_k)
        faiss_ranked: list[int] = [int(i) for i in faiss_indices[0] if i >= 0]

        # ── sparse retrieval ──────────────────────────────────────────────────
        bm25_ranked: list[int] = []
        if _bm25 is not None:
            bm25_scores = _bm25.get_scores(_tokenize(query))
            bm25_top = min(_CANDIDATE_K, len(_metadata))
            bm25_ranked = list(np.argsort(bm25_scores)[::-1][:bm25_top].astype(int))

    # ── Reciprocal Rank Fusion ────────────────────────────────────────────────
    rrf: dict[int, float] = {}
    for ranked_list in (faiss_ranked, bm25_ranked):
        for rank, idx in enumerate(ranked_list):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (_RRF_K + rank + 1)

    fused = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:k]

    return [
        {
            "text": _metadata[idx]["text"],
            "source_filename": _metadata[idx]["source_filename"],
            "page": _metadata[idx]["page"],
            "score": rrf_score,
        }
        for idx, rrf_score in fused
    ]


def list_documents() -> list[dict[str, Any]]:
    """Return a deduplicated list of indexed documents with chunk counts."""
    counts: dict[str, int] = {}
    for meta in _metadata:
        fname = meta["source_filename"]
        counts[fname] = counts.get(fname, 0) + 1
    return [{"filename": fname, "chunks": count} for fname, count in sorted(counts.items())]
