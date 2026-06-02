"""
Offline index builder for the NHAI Document Q&A Assistant.

Run this script whenever base documents in /data change:

    python scripts/build_index.py

Reads all PDFs from /data, embeds them in batches, builds a FAISS IndexFlatIP,
and writes two artifacts to app/index/:
    base.index          — the serialised FAISS index
    base_metadata.json  — aligned list of {text, source_filename, page}

These artifacts are committed to git and shipped inside the Heroku slug.
The app loads them at startup — no embedding API calls happen at runtime.
"""

import json
import os
import sys
import time
from pathlib import Path

# Make project root importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import faiss
import numpy as np

from app.embeddings import embed_texts
from app.ingest import load_pdf

DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "app" / "index"
INDEX_FILE = INDEX_DIR / "base.index"
METADATA_FILE = INDEX_DIR / "base_metadata.json"

BATCH_SIZE = 64              # default for fastembed / openai
VOYAGE_FREE_TIER_DELAY = 21  # seconds between batches on free tier (3 RPM limit)

# Embed this many chunks per API call to stay within rate limits
BATCH_SIZE = 64


def main() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {DATA_DIR}/")
        print("Add your base NHAI PDFs to the data/ directory and re-run.")
        sys.exit(0)

    print(f"Found {len(pdf_files)} PDF(s):")
    for p in pdf_files:
        print(f"  {p.name}")

    all_chunks = []
    for pdf_path in pdf_files:
        print(f"\nIngesting {pdf_path.name} ...")
        chunks = load_pdf(pdf_path)
        all_chunks.extend(chunks)
        print(f"  -> {len(chunks)} chunks")

    total = len(all_chunks)
    provider = os.getenv("EMBEDDING_PROVIDER", "fastembed").lower()

    # Voyage free tier: 3 RPM / 10K TPM — use small batches with delays
    batch_size = 5 if provider == "voyage" else BATCH_SIZE
    delay = VOYAGE_FREE_TIER_DELAY if provider == "voyage" else 0

    print(f"\nTotal chunks: {total}")
    print(f"Embedding with provider={provider}, batch_size={batch_size} ...")

    all_vectors: list[np.ndarray] = []
    for i in range(0, total, batch_size):
        batch = all_chunks[i : i + batch_size]
        vectors = embed_texts([c["text"] for c in batch])
        all_vectors.append(vectors)
        done = min(i + batch_size, total)
        print(f"  Embedded {done}/{total}")
        if delay and done < total:
            print(f"  (rate-limit pause {delay}s …)")
            time.sleep(delay)

    vectors_np = np.vstack(all_vectors).astype(np.float32)
    dim = vectors_np.shape[1]

    # Build exact cosine-similarity index (dot product on unit vectors)
    index = faiss.IndexFlatIP(dim)
    index.add(vectors_np)

    faiss.write_index(index, str(INDEX_FILE))

    metadata = [
        {"text": c["text"], "source_filename": c["source_filename"], "page": c["page"]}
        for c in all_chunks
    ]
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nIndex built: {index.ntotal} vectors, dim={dim}")
    print(f"  {INDEX_FILE}")
    print(f"  {METADATA_FILE}")
    print("\nCommit app/index/ to include it in the Heroku slug.")


if __name__ == "__main__":
    main()
