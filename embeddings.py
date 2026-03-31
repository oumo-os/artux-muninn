"""
embeddings.py — Embedding backend for Muninn.

Priority chain (highest to lowest):

  1. llama-cpp-python + a GGUF embedding model
       Configured via MUNINN_EMBEDDING_MODEL env var or explicit configure() call.
       Any GGUF works, but dedicated embedding models give the best results:
         nomic-embed-text-v1.5  (768-dim, strong general-purpose)
         mxbai-embed-large-v1   (1024-dim, MTEB SOTA at time of writing)
         bge-small-en-v1.5      (384-dim, fast, good quality)
         all-MiniLM-L6-v2       (384-dim, widely available)
       Install: pip install llama-cpp-python

  2. sentence-transformers
       Downloads all-MiniLM-L6-v2 on first use (~80 MB, cached).
       Install: pip install sentence-transformers

  3. TF-IDF bag-of-words (512-dim hashed)
       No external dependencies. Reasonable for exact-match and structured
       recall signals. Poor at bridging vocabulary gaps ("Betty" → "car").
       Automatically used when neither of the above is available.

Configuration
-------------
Set the GGUF path before first use — either via environment variable:

    export MUNINN_EMBEDDING_MODEL=/models/nomic-embed-text-v1.5.Q8_0.gguf

Or explicitly in code (call before creating any MemoryAgent):

    from memory_module.embeddings import configure
    configure(model_path="/models/nomic-embed-text-v1.5.Q8_0.gguf",
              n_gpu_layers=-1)    # -1 = all layers on GPU, 0 = CPU only

The MemoryAgent constructor also accepts embedding_model_path directly:

    agent = MemoryAgent("agent.db",
                        embedding_model_path="/models/nomic-embed.gguf")
"""

from __future__ import annotations
import math
import os
import re
from collections import Counter
from typing import Optional


# ---------------------------------------------------------------------------
# Backend state
# ---------------------------------------------------------------------------

_backend: str = "tfidf"          # "llamacpp" | "transformers" | "tfidf"
_llama_model = None               # llama_cpp.Llama instance
_st_model    = None               # SentenceTransformer instance

SEMANTIC_AVAILABLE: bool = False  # True when llamacpp or transformers is active


def configure(
    model_path:    Optional[str] = None,
    n_gpu_layers:  int           = 0,
    n_ctx:         int           = 512,
    n_threads:     Optional[int] = None,
) -> str:
    """
    Explicitly configure the embedding backend.

    Parameters
    ----------
    model_path   : Path to a GGUF file.  If None, falls back to
                   MUNINN_EMBEDDING_MODEL env var, then sentence-transformers,
                   then TF-IDF.
    n_gpu_layers : Layers to offload to GPU. 0 = CPU only, -1 = all on GPU.
    n_ctx        : Context window for the embedding model (default 512).
                   Increase if you embed very long texts.
    n_threads    : CPU threads. None = llama-cpp default (usually num cores / 2).

    Returns
    -------
    The name of the backend that was activated: "llamacpp" | "transformers" | "tfidf"
    """
    global _backend, _llama_model, _st_model, SEMANTIC_AVAILABLE

    # ── Try llama-cpp-python ────────────────────────────────────────────
    resolved_path = model_path or os.environ.get("MUNINN_EMBEDDING_MODEL")

    if resolved_path:
        try:
            from llama_cpp import Llama  # type: ignore
            kwargs: dict = dict(
                model_path  = resolved_path,
                embedding   = True,
                n_ctx       = n_ctx,
                n_gpu_layers= n_gpu_layers,
                verbose     = False,
            )
            if n_threads is not None:
                kwargs["n_threads"] = n_threads

            _llama_model      = Llama(**kwargs)
            _backend          = "llamacpp"
            SEMANTIC_AVAILABLE = True
            return "llamacpp"

        except ImportError:
            pass   # llama-cpp-python not installed
        except Exception as e:
            # Model file missing, corrupt, etc. — don't silently swallow
            raise RuntimeError(
                f"Failed to load GGUF embedding model at '{resolved_path}': {e}\n"
                "Check that the file exists and llama-cpp-python is installed."
            ) from e

    # ── Try sentence-transformers ────────────────────────────────────────
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        _st_model         = SentenceTransformer("all-MiniLM-L6-v2")
        _backend          = "transformers"
        SEMANTIC_AVAILABLE = True
        return "transformers"

    except ImportError:
        pass

    # ── TF-IDF fallback ──────────────────────────────────────────────────
    _backend           = "tfidf"
    SEMANTIC_AVAILABLE = False
    return "tfidf"


# Run the auto-configure on module import so callers don't have to.
# If the user calls configure() explicitly, it overrides this.
configure()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed(text: str) -> list[float]:
    """Return a normalised embedding vector for `text`."""
    if _backend == "llamacpp":
        return _llamacpp_embed(text)
    if _backend == "transformers":
        return _transformers_embed(text)
    return _tfidf_embed(text)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def top_k_similar(
    query_vec:  list[float],
    candidates: list[tuple[str, list[float]]],
    k:          int   = 5,
    threshold:  float = 0.0,
) -> list[tuple[str, float]]:
    """Return the top-k (id, score) pairs sorted by descending cosine similarity."""
    scored = [
        (cid, cosine_similarity(query_vec, vec))
        for cid, vec in candidates
        if vec
    ]
    scored = [(cid, s) for cid, s in scored if s >= threshold]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def backend_info() -> dict:
    """Return a dict describing the active embedding backend."""
    return {
        "backend":          _backend,
        "semantic_available": SEMANTIC_AVAILABLE,
        "model_path":       (
            getattr(_llama_model, "model_path", None)
            if _llama_model else None
        ),
    }


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------

def _llamacpp_embed(text: str) -> list[float]:
    result = _llama_model.create_embedding(text)
    vec    = result["data"][0]["embedding"]
    return _l2_normalize(vec)


def _transformers_embed(text: str) -> list[float]:
    vec = _st_model.encode(text, normalize_embeddings=True)
    return vec.tolist()


# TF-IDF bag-of-words — 512-dimensional hashed feature space
_VOCAB_SIZE = 512

def _tfidf_embed(text: str) -> list[float]:
    tokens = re.findall(r"[a-z]+", text.lower())
    counts = Counter(tokens)
    vec    = [0.0] * _VOCAB_SIZE
    for token, cnt in counts.items():
        vec[hash(token) % _VOCAB_SIZE] += cnt
    return _l2_normalize(vec)


def _l2_normalize(vec: list[float]) -> list[float]:
    mag = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / mag for v in vec]
