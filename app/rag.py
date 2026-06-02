"""
RAG answer generation for the NHAI Document Q&A Assistant.

Grounding strategy:
- Each retrieved chunk is prefixed with its [filename p.X] citation tag before
  being concatenated into the context. This means Claude can reproduce exact
  citations without fabricating references — it just copies from the context.
- The system prompt forbids answering from prior knowledge and mandates a fixed
  refusal phrase ("I don't know based on the provided documents.") when context
  is insufficient. This eliminates hallucination on out-of-scope questions.
- Temperature 0.1 keeps responses faithful to source text and reduces creative
  paraphrasing that could distort the original meaning.
- Model: claude-haiku-4-5-20251001 — fast and cost-efficient; fully capable for
  grounded extraction tasks where the hard work is retrieval, not reasoning.
"""

import os

import anthropic

from app import store

MODEL_ID = "claude-haiku-4-5-20251001"
TOP_K = int(os.getenv("TOP_K", "5"))

SYSTEM_PROMPT = (
    "You are a precise document assistant for NHAI (National Highways Authority of India) documents. "
    "Your ONLY job is to answer questions using the context excerpts provided by the user.\n\n"
    "Rules you MUST follow without exception:\n"
    "1. Answer ONLY from the provided context. Never use outside knowledge.\n"
    "2. Cite every claim inline using the exact tags already present in the context, "
    "e.g. [Annual_Report_2023.pdf p.12].\n"
    "3. If the context does not contain enough information to answer the question, "
    'respond EXACTLY with: "I don\'t know based on the provided documents."\n'
    "4. Do not speculate, infer beyond the text, add disclaimers, or pad your response."
)


def answer(question: str) -> dict:
    """
    Retrieve top-K chunks, build a grounded prompt, call Claude, and return
    {answer: str, sources: list[{filename, page, snippet}]}.
    """
    chunks = store.search(question, k=TOP_K)

    if not chunks:
        return {
            "answer": "I don't know based on the provided documents.",
            "sources": [],
        }

    # Prefix each chunk with its citation tag so Claude can reproduce it verbatim
    context_parts = []
    for chunk in chunks:
        tag = f"[{chunk['source_filename']} p.{chunk['page']}]"
        context_parts.append(f"{tag}\n{chunk['text']}")
    context = "\n\n---\n\n".join(context_parts)

    user_message = f"Context:\n\n{context}\n\nQuestion: {question}"

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=MODEL_ID,
        max_tokens=1024,
        temperature=0.1,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    answer_text = response.content[0].text

    # Build sources from the retrieved chunks (snippet = first 200 chars of the chunk)
    sources = []
    for chunk in chunks:
        snippet = chunk["text"][:200]
        if len(chunk["text"]) > 200:
            snippet += "..."
        sources.append({
            "filename": chunk["source_filename"],
            "page": chunk["page"],
            "snippet": snippet,
        })

    return {"answer": answer_text, "sources": sources}
