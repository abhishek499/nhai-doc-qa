#!/usr/bin/env python3
"""
Evaluation harness for the NHAI Document Q&A Assistant.

Metrics computed per golden-set item:
  Hit@k       — 1 if any retrieved chunk comes from the expected source file, else 0
  RR          — Reciprocal Rank: 1/rank of the first chunk from the expected source (0 if not found)
  Faithfulness — Claude-as-judge score 0-1: are the answer's claims supported by the context?

Summary: Hit Rate@k, MRR, and mean Faithfulness are printed as a table.

Usage:
  python scripts/evaluate.py                          # hybrid retrieval (default)
  python scripts/evaluate.py --dense-only             # pure FAISS cosine, no BM25/RRF
  python scripts/evaluate.py --k 10                   # evaluate at top-10 (default: 5)
  python scripts/evaluate.py --golden eval/golden_set.json
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# ── Bootstrap app modules (loads FAISS index + embedding model) ────────────────
print("Loading index and embedding model…")
from app import store
from app.embeddings import embed_texts

store.load_base_index()
embed_texts(["warmup"])  # pre-load ONNX model so first query isn't slow
print("Ready.\n")

import anthropic

# ── Retrieval helpers ──────────────────────────────────────────────────────────

def search_hybrid(query: str, k: int) -> list[dict]:
    """Hybrid retrieval: FAISS + BM25 fused with RRF (the default app behaviour)."""
    return store.search(query, k=k)


def search_dense_only(query: str, k: int) -> list[dict]:
    """Pure FAISS cosine similarity search, bypassing BM25 and RRF entirely."""
    query_vec = embed_texts([query])

    with store._lock:
        if store._index is None or store._index.ntotal == 0:
            return []
        if query_vec.shape[1] != store._index.d:
            raise RuntimeError(
                f"Embedding dimension mismatch: query={query_vec.shape[1]}, "
                f"index={store._index.d}"
            )
        cand_k = min(k, store._index.ntotal)
        scores, indices = store._index.search(query_vec, cand_k)

    return [
        {
            "text": store._metadata[int(i)]["text"],
            "source_filename": store._metadata[int(i)]["source_filename"],
            "page": store._metadata[int(i)]["page"],
            "score": float(s),
        }
        for s, i in zip(scores[0], indices[0])
        if i >= 0
    ]


# ── Generation ─────────────────────────────────────────────────────────────────

def generate_answer(question: str, chunks: list[dict]) -> tuple[str, str]:
    """
    Run the same RAG prompt as app/rag.py but with a pre-retrieved chunk list
    so the evaluate script can swap retrieval strategies without touching rag.py.

    Returns (answer_text, context_text).
    """
    if not chunks:
        return "I don't know based on the provided documents.", ""

    context_parts = [
        f"[{c['source_filename']} p.{c['page']}]\n{c['text']}"
        for c in chunks
    ]
    context = "\n\n---\n\n".join(context_parts)

    system = (
        "You are a precise document assistant for NHAI documents.\n\n"
        "Rules:\n"
        "1. Answer ONLY from the context excerpts provided.\n"
        "2. Cite every claim inline using the exact tags present, e.g. [file.pdf p.3].\n"
        "3. If the context does not contain enough information, respond EXACTLY with: "
        '"I don\'t know based on the provided documents."\n'
        "4. Do not speculate or add information beyond the text."
    )

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        temperature=0.1,
        system=system,
        messages=[{"role": "user", "content": f"Context:\n\n{context}\n\nQuestion: {question}"}],
    )
    return response.content[0].text, context


# ── Faithfulness judge ─────────────────────────────────────────────────────────

JUDGE_SYSTEM = (
    "You are a faithfulness evaluator for a RAG (retrieval-augmented generation) system.\n\n"
    "Given a retrieved context and a generated answer, score how faithfully the answer "
    "is grounded in the context.\n\n"
    "Scoring guide:\n"
    "  1.0 — Every factual claim is directly supported by the context.\n"
    "  0.7 — Most claims are supported; minor inferences or omissions present.\n"
    "  0.4 — Mixed: some supported claims, some unsupported or speculative.\n"
    "  0.0 — Claims are fabricated or contradict the context.\n\n"
    "Reply with ONLY valid JSON, no prose: {\"score\": 0.0, \"reason\": \"one sentence\"}"
)


def judge_faithfulness(context: str, answer: str) -> tuple[float, str]:
    """Ask Claude Haiku to score faithfulness. Returns (score 0-1, reason str)."""
    # A correct refusal needs no further evaluation — it's maximally faithful.
    if "I don't know based on the provided documents" in answer:
        return 1.0, "Correct refusal — no unsupported claims made."

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=120,
        temperature=0,
        system=JUDGE_SYSTEM,
        messages=[{
            "role": "user",
            "content": (
                f"Context (truncated to 3000 chars):\n{context[:3000]}\n\n"
                f"Answer:\n{answer}"
            ),
        }],
    )
    raw = msg.content[0].text.strip()
    # Strip markdown code fences if the model wraps the JSON
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        data = json.loads(raw)
        return float(data["score"]), str(data.get("reason", ""))
    except (json.JSONDecodeError, KeyError, ValueError):
        import re
        m = re.search(r'"score"\s*:\s*([0-9.]+)', raw)
        score = float(m.group(1)) if m else 0.5
        return score, f"(parse error - raw: {raw[:80]})"


# ── Metric helpers ─────────────────────────────────────────────────────────────

def hit_at_k(chunks: list[dict], expected_source: str) -> int:
    """1 if any chunk came from expected_source, else 0."""
    return int(any(c["source_filename"] == expected_source for c in chunks))


def reciprocal_rank(chunks: list[dict], expected_source: str) -> float:
    """1/rank of the first relevant chunk (1-indexed); 0.0 if not found."""
    for rank, chunk in enumerate(chunks, start=1):
        if chunk["source_filename"] == expected_source:
            return 1.0 / rank
    return 0.0


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NHAI Q&A retrieval + generation.")
    parser.add_argument(
        "--golden",
        default=str(ROOT / "eval" / "golden_set.json"),
        help="Path to golden_set.json (default: eval/golden_set.json)",
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="Number of chunks to retrieve per question (default: 5)",
    )
    parser.add_argument(
        "--dense-only", action="store_true",
        help="Use pure FAISS cosine search instead of hybrid BM25+FAISS+RRF",
    )
    args = parser.parse_args()

    golden = json.loads(Path(args.golden).read_text(encoding="utf-8"))
    search_fn = search_dense_only if args.dense_only else search_hybrid
    mode_label = "dense-only" if args.dense_only else "hybrid (BM25+FAISS+RRF)"

    print(f"Mode      : {mode_label}")
    print(f"k         : {args.k}")
    print(f"Questions : {len(golden)}")
    print(f"Golden set: {args.golden}")
    print("=" * 100)

    col_w = [4, 52, 10, 6, 6, 13, 40]  # id, question, category, hit, rr, faith, reason
    header = (
        f"{'ID':<{col_w[0]}}  "
        f"{'Question':<{col_w[1]}}  "
        f"{'Category':<{col_w[2]}}  "
        f"{'Hit':>{col_w[3]}}  "
        f"{'RR':>{col_w[4]}}  "
        f"{'Faithful':>{col_w[5]}}  "
        f"{'Judge note':<{col_w[6]}}"
    )
    print(header)
    print("-" * 100)

    hits, rrs, faiths = [], [], []

    for item in golden:
        qid      = item["id"]
        question = item["question"]
        expected = item["expected_source"]
        category = item["category"]

        # ── Retrieval ─────────────────────────────────────────────────────────
        chunks = search_fn(question, k=args.k)

        hit = hit_at_k(chunks, expected)
        rr  = reciprocal_rank(chunks, expected)

        # ── Generation ────────────────────────────────────────────────────────
        answer, context = generate_answer(question, chunks)

        # ── Faithfulness judge ────────────────────────────────────────────────
        faith_score, faith_reason = judge_faithfulness(context, answer)

        # Brief pause to respect API rate limits when running many items
        time.sleep(0.3)

        hits.append(hit)
        rrs.append(rr)
        faiths.append(faith_score)

        q_short  = (question[:49] + "…") if len(question) > 50 else question
        r_short  = (faith_reason[:39] + "…") if len(faith_reason) > 40 else faith_reason

        print(
            f"{qid:<{col_w[0]}}  "
            f"{q_short:<{col_w[1]}}  "
            f"{category:<{col_w[2]}}  "
            f"{'Y' if hit else 'N':>{col_w[3]}}  "
            f"{rr:>{col_w[4]}.2f}  "
            f"{faith_score:>{col_w[5]}.2f}  "
            f"{r_short:<{col_w[6]}}"
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    n = len(hits)
    avg_hit   = sum(hits)   / n
    avg_mrr   = sum(rrs)    / n
    avg_faith = sum(faiths) / n

    print("=" * 100)
    print(f"\nSummary  [{mode_label}]  (n={n})\n")
    print(f"  Hit Rate@{args.k}   : {avg_hit:.3f}   ({sum(hits)}/{n} questions had a relevant chunk in top-{args.k})")
    print(f"  MRR             : {avg_mrr:.3f}   (mean reciprocal rank of first relevant chunk)")
    print(f"  Faithfulness    : {avg_faith:.3f}   (avg Claude-judge score; 1.0 = fully grounded)\n")

    # Per-category breakdown
    categories = sorted({item["category"] for item in golden})
    if len(categories) > 1:
        print("  Per-category breakdown:")
        for cat in categories:
            idx = [i for i, item in enumerate(golden) if item["category"] == cat]
            c_hit   = sum(hits[i]   for i in idx) / len(idx)
            c_mrr   = sum(rrs[i]    for i in idx) / len(idx)
            c_faith = sum(faiths[i] for i in idx) / len(idx)
            print(f"    {cat:<18}  Hit@k={c_hit:.2f}  MRR={c_mrr:.2f}  Faith={c_faith:.2f}  (n={len(idx)})")
        print()

    # ── Save results to evals/ ────────────────────────────────────────────────
    evals_dir = ROOT / "evals"
    evals_dir.mkdir(exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    mode_slug = "dense" if args.dense_only else "hybrid"
    out_path = evals_dir / f"results_{ts}_{mode_slug}_k{args.k}.json"

    per_question = []
    for i, item in enumerate(golden):
        per_question.append({
            "id":                 item["id"],
            "category":           item["category"],
            "question":           item["question"],
            "expected_source":    item["expected_source"],
            "hit":                hits[i],
            "rr":                 rrs[i],
            "faithfulness_score": faiths[i],
        })

    output = {
        "run_at":          ts,
        "mode":            mode_slug,
        "k":               args.k,
        "golden_set":      args.golden,
        "n":               n,
        "summary": {
            f"hit_rate_at_{args.k}": round(avg_hit,   4),
            "mrr":                   round(avg_mrr,   4),
            "mean_faithfulness":     round(avg_faith, 4),
        },
        "per_category": {
            cat: {
                f"hit_rate_at_{args.k}": round(sum(hits[i]   for i in [j for j, it in enumerate(golden) if it["category"] == cat]) / len([j for j, it in enumerate(golden) if it["category"] == cat]), 4),
                "mrr":                   round(sum(rrs[i]    for i in [j for j, it in enumerate(golden) if it["category"] == cat]) / len([j for j, it in enumerate(golden) if it["category"] == cat]), 4),
                "mean_faithfulness":     round(sum(faiths[i] for i in [j for j, it in enumerate(golden) if it["category"] == cat]) / len([j for j, it in enumerate(golden) if it["category"] == cat]), 4),
                "n":                     len([j for j, it in enumerate(golden) if it["category"] == cat]),
            }
            for cat in categories
        },
        "results": per_question,
    }

    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Results saved -> {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
