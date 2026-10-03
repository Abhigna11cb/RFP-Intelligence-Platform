"""
Embedding provider — routes to OpenAI text-embedding-3-large or local sentence-transformers.
Configured via EMBEDDING_PROVIDER in .env.

text-embedding-3-large:
  - 1536 dimensions (truncated from 3072 — still excellent quality)
  - API-based, costs ~$0.02 for the full bid corpus
  - Far better retrieval quality than all-mpnet-base-v2 for domain-specific text

Usage:
    from rfp_platform.search.embeddings import get_embeddings, get_query_embedding
    vectors = get_embeddings(["text1", "text2"])   # list[list[float]]
    vector  = get_query_embedding("submission deadline")  # list[float]
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from rfp_platform.core.config import get_settings

logger = logging.getLogger(__name__)


# ── OpenAI embedding client (lazy singleton) ─────────────────────────────────

@lru_cache(maxsize=1)
def _openai_client():
    from openai import OpenAI
    cfg = get_settings()
    return OpenAI(api_key=cfg.openai_api_key)


# ── Local sentence-transformers (lazy singleton) ──────────────────────────────

@lru_cache(maxsize=1)
def _local_model():
    from sentence_transformers import SentenceTransformer
    cfg = get_settings()
    logger.info(f"Loading local embedding model: {cfg.embedding_model_name}")
    return SentenceTransformer(cfg.embedding_model_name)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a batch of texts.
    Routes to OpenAI or local model based on EMBEDDING_PROVIDER env var.

    Args:
        texts: List of strings to embed.

    Returns:
        List of embedding vectors (each a list of floats).
    """
    cfg = get_settings()

    if cfg.embedding_provider == "openai":
        return _openai_embeddings(texts)
    else:
        return _local_embeddings(texts)


def get_query_embedding(query: str) -> list[float]:
    """
    Embed a single query string.
    Slightly different call path for OpenAI (uses search_query prefix internally).
    """
    return get_embeddings([query])[0]


# ── OpenAI provider ───────────────────────────────────────────────────────────

def _openai_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Call OpenAI text-embedding-3-large in batches of 100.
    Uses dimensions=1536 (truncated from 3072) for smaller storage + same quality.
    """
    cfg = get_settings()
    client = _openai_client()
    all_embeddings: list[list[float]] = []
    batch_size = 100  # OpenAI allows up to 2048 per call

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        # Clean texts — OpenAI rejects empty strings
        batch = [t.strip() or "empty" for t in batch]

        response = client.embeddings.create(
            model=cfg.embedding_model_name,          # text-embedding-3-large
            input=batch,
            dimensions=cfg.embedding_dimensions,     # 1536 (truncation)
        )
        batch_vecs = [item.embedding for item in response.data]
        all_embeddings.extend(batch_vecs)
        logger.debug(
            f"Embedded batch {i // batch_size + 1} "
            f"({len(batch)} texts) via OpenAI"
        )

    return all_embeddings


# ── Local provider ────────────────────────────────────────────────────────────

def _local_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Use sentence-transformers locally (no API key needed).
    Falls back to this when EMBEDDING_PROVIDER=local.
    """
    model = _local_model()
    batch_size = 32
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        vecs = model.encode(
            batch,
            batch_size=batch_size,
            convert_to_tensor=False,
            show_progress_bar=False,
        )
        all_embeddings.extend(v.tolist() for v in vecs)

    return all_embeddings
