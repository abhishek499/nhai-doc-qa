"""
PDF ingestion and chunking for the NHAI Document Q&A Assistant.

Chunk size (~875 tokens) is chosen to balance two competing needs:
- Large enough to contain a complete idea / table row / clause
- Small enough that retrieved chunks stay on-topic and fit several into the prompt

Character-based chunking (~3500 chars ≈ 875 tokens at ~4 chars/token average) avoids
heavy tokenizer dependencies while giving a reasonable approximation for English text.

Overlap (~350 chars ≈ 88 tokens) prevents answers from being split across chunk
boundaries when a sentence straddles two consecutive windows.
"""

import os
from pathlib import Path
from typing import TypedDict

from pypdf import PdfReader

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "3500"))      # characters ≈ ~875 tokens
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "350"))  # characters ≈ ~88 tokens


class Chunk(TypedDict):
    text: str
    source_filename: str
    page: int  # 1-indexed


def load_pdf(path: str | Path) -> list[Chunk]:
    """
    Extract text from a PDF, sliding-window chunk it, and return Chunk objects.
    Page numbers are preserved by tracking character offsets per page.
    """
    path = Path(path)
    reader = PdfReader(str(path))
    filename = path.name

    # Build a flat string from all pages, recording each page's character range
    full_text = ""
    page_boundaries: list[tuple[int, int, int]] = []  # (start_char, end_char, page_num)

    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        start = len(full_text)
        full_text += text + "\n"
        end = len(full_text)
        page_boundaries.append((start, end, page_num))

    if not full_text.strip():
        return []

    def char_to_page(char_idx: int) -> int:
        """Map a character index back to its 1-indexed page number."""
        for start, end, page_num in page_boundaries:
            if start <= char_idx < end:
                return page_num
        return page_boundaries[-1][2] if page_boundaries else 1

    # Sliding-window chunking over the full document text
    chunks: list[Chunk] = []
    start = 0
    text_len = len(full_text)

    while start < text_len:
        end = min(start + CHUNK_SIZE, text_len)
        chunk_text = full_text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(
                text=chunk_text,
                source_filename=filename,
                page=char_to_page(start),
            ))
        if end == text_len:
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks
