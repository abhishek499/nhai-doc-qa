"""
Provider-agnostic text embedding for the NHAI Document Q&A Assistant.

Cosine similarity is used for retrieval because it measures directional (topical)
similarity independent of text length, making it robust across chunks of varying size.

Implementation: vectors are L2-normalised before being stored in a FAISS IndexFlatIP.
With unit vectors, inner product == cosine similarity, so IndexFlatIP gives exact
cosine search without a separate normalisation step at query time.

EMBEDDING_PROVIDER options:
  fastembed (default) — local ONNX inference, no API key needed, completely free.
                        Model (~130 MB) is downloaded once and cached locally.
  voyage              — Voyage AI API (voyage-3-lite). Requires VOYAGE_API_KEY.
  openai              — OpenAI API (text-embedding-3-small). Requires OPENAI_API_KEY.
"""

import os

import numpy as np

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "fastembed").lower()

# fastembed model cache — loaded once, reused across all calls
_fastembed_model = None


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a batch of texts and return an L2-normalised float32 array of shape (N, D).
    The normalisation enables cosine similarity via dot product in IndexFlatIP.
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    if EMBEDDING_PROVIDER == "fastembed":
        return _embed_fastembed(texts)
    if EMBEDDING_PROVIDER == "voyage":
        return _embed_voyage(texts)
    if EMBEDDING_PROVIDER == "openai":
        return _embed_openai(texts)
    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER!r}. "
        "Choose 'fastembed', 'voyage', or 'openai'."
    )


def _embed_fastembed(texts: list[str]) -> np.ndarray:
    """
    Local ONNX-based embeddings via fastembed. No API key required.
    The model is downloaded once (~130 MB) and cached at ~/.cache/fastembed/.
    Uses BAAI/bge-small-en-v1.5 by default (384 dims, fast, good quality).
    Override with FASTEMBED_MODEL env var if needed.
    """
    global _fastembed_model
    if _fastembed_model is None:
        from fastembed import TextEmbedding  # type: ignore
        model_name = os.getenv("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")
        print(f"Loading fastembed model: {model_name} (downloads on first use) ...")
        _fastembed_model = TextEmbedding(model_name)

    embeddings = list(_fastembed_model.embed(texts))
    vectors = np.array(embeddings, dtype=np.float32)
    return _l2_normalize(vectors)


def _embed_voyage(texts: list[str]) -> np.ndarray:
    import voyageai  # type: ignore
    client = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
    result = client.embed(texts, model="voyage-3-lite", input_type="document")
    vectors = np.array(result.embeddings, dtype=np.float32)
    return _l2_normalize(vectors)


def _embed_openai(texts: list[str]) -> np.ndarray:
    from openai import OpenAI  # type: ignore
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    result = client.embeddings.create(input=texts, model="text-embedding-3-small")
    vectors = np.array([item.embedding for item in result.data], dtype=np.float32)
    return _l2_normalize(vectors)


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Normalise each row to unit length; safe against zero-norm edge cases."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms
