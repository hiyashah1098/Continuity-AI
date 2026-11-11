"""Split documents into citation-friendly overlapping chunks."""

from __future__ import annotations

from .models import Chunk, Document


def chunk_documents(
    documents: list[Document], words_per_chunk: int = 180, overlap: int = 35
) -> list[Chunk]:
    if words_per_chunk <= overlap:
        raise ValueError("words_per_chunk must be greater than overlap")

    chunks: list[Chunk] = []
    for document in documents:
        words = document.text.split()
        if not words:
            continue
        step = words_per_chunk - overlap
        for start in range(0, len(words), step):
            end = min(start + words_per_chunk, len(words))
            chunks.append(
                Chunk(
                    filename=document.filename,
                    text=" ".join(words[start:end]),
                    location=f"words {start + 1}-{end}",
                )
            )
            if end == len(words):
                break
    return chunks
