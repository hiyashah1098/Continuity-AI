"""Simple lexical retrieval using TF-IDF cosine similarity."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import Chunk


def retrieve_chunks(
    question: str, chunks: list[Chunk], top_k: int = 5
) -> list[tuple[Chunk, float]]:
    """Return the highest-scoring chunks, omitting zero-similarity results."""
    if not question.strip() or not chunks:
        return []
    texts = [chunk.text for chunk in chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts + [question])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
    return [(chunks[index], float(score)) for index, score in ranked[:top_k] if score > 0]


def retrieve_context(
    question: str,
    chunks: list[Chunk],
    top_k: int = 5,
    full_context_limit: int = 12,
) -> list[Chunk]:
    """Retrieve matching chunks, or use a bounded full corpus when lexical search misses.

    Small handoff corpora fit comfortably in one grounded Gemini request. Sending the
    complete small corpus lets the model handle morphology and synonyms that word-level
    TF-IDF cannot connect, while the existing prompt and citation checks still require
    an explicit fallback when the records do not contain the answer.
    """
    ranked = retrieve_chunks(question, chunks, top_k=top_k)
    if ranked:
        return [chunk for chunk, _score in ranked]
    if question.strip() and 0 < len(chunks) <= full_context_limit:
        return list(chunks)
    return []
