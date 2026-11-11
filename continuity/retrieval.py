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
