"""
embeddings.py — Semantic embedding utility.

Uses sentence-transformers (all-MiniLM-L6-v2) when available.
Falls back to a simple TF-IDF bag-of-words vector if the library
is not installed, so the rest of the module still works without it.
"""

from __future__ import annotations
import math
import re
from collections import Counter
from typing import Optional

# Attempt to load sentence-transformers; degrade gracefully if absent.
try:
    from sentence_transformers import SentenceTransformer as _ST
    _MODEL: Optional[_ST] = None          # lazy-loaded on first use

    def _get_model() -> _ST:
        global _MODEL
        if _MODEL is None:
            _MODEL = _ST("all-MiniLM-L6-v2")
        return _MODEL

    SEMANTIC_AVAILABLE = True

except ImportError:
    SEMANTIC_AVAILABLE = False
    _get_model = None  # type: ignore


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed(text: str) -> list[float]:
    """Return a normalised embedding vector for `text`."""
    if SEMANTIC_AVAILABLE:
        return _semantic_embed(text)
    return _tfidf_embed(text)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors in [−1, 1]."""
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def top_k_similar(
    query_vec: list[float],
    candidates: list[tuple[str, list[float]]],   # (id, vector)
    k: int = 5,
    threshold: float = 0.0,
) -> list[tuple[str, float]]:
    """
    Return the top-k (id, score) pairs from `candidates` sorted by
    descending cosine similarity to `query_vec`.
    Entries below `threshold` are excluded.
    """
    scored = [
        (cid, cosine_similarity(query_vec, vec))
        for cid, vec in candidates
        if vec
    ]
    scored = [(cid, s) for cid, s in scored if s >= threshold]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


# ---------------------------------------------------------------------------
# Private implementations
# ---------------------------------------------------------------------------

def _semantic_embed(text: str) -> list[float]:
    vec = _get_model().encode(text, normalize_embeddings=True)
    return vec.tolist()


# Simple vocabulary-free TF-IDF fallback (hashed feature space of 512 dims)
_VOCAB_SIZE = 512

def _tfidf_embed(text: str) -> list[float]:
    tokens = re.findall(r"[a-z]+", text.lower())
    counts = Counter(tokens)
    vec = [0.0] * _VOCAB_SIZE
    for token, cnt in counts.items():
        idx = hash(token) % _VOCAB_SIZE
        vec[idx] += cnt
    # L2-normalise
    mag = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / mag for v in vec]
