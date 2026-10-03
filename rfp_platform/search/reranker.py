"""
Cross-encoder re-ranker — re-ranks hybrid search results using a cross-encoder model.

Why cross-encoders beat bi-encoders for re-ranking:
  Bi-encoders (like text-embedding-3-large) encode query and doc SEPARATELY,
  losing fine-grained interaction signals.
  Cross-encoders process (query, doc) JOINTLY → much higher precision.

Model used: cross-encoder/ms-marco-MiniLM-L-6-v2
  - 22M params, very fast (CPU-friendly)
  - Trained on MS-MARCO passage ranking
  - Returns a relevance logit (higher = more relevant)

Usage:
    from rfp_platform.search.reranker import rerank
    results = hybrid_search(query, bid_id=bid_id, top_k=20)
    results = rerank(query, results, top_k=5)
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _get_reranker():
    """Lazy-load the cross-encoder (downloaded once, cached in memory)."""
    try:
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading re-ranker: {RERANKER_MODEL}")
        model = CrossEncoder(RERANKER_MODEL, max_length=512)
        logger.info("Re-ranker loaded.")
        return model
    except ImportError:
        logger.warning("sentence-transformers not installed — re-ranking disabled.")
        return None
    except Exception as e:
        logger.warning(f"Re-ranker load failed ({e}) — re-ranking disabled.")
        return None


def rerank(
    query: str,
    results: list[dict[str, Any]],
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """
    Re-rank hybrid search results using a cross-encoder.

    Takes the top candidates from hybrid_search (typically top-20) and
    re-scores each (query, chunk_text) pair with the cross-encoder,
    then returns the top_k by cross-encoder score.

    Falls back gracefully to original ordering if cross-encoder unavailable.

    Args:
        query:   The user's query string.
        results: List of CitedResult dicts from hybrid_search / semantic_search.
        top_k:   How many to return after re-ranking (default: return all re-ranked).

    Returns:
        Re-ranked list of CitedResult dicts with an added 'rerank_score' key.
    """
    if not results:
        return results

    model = _get_reranker()
    if model is None:
        logger.warning("Re-ranker unavailable — returning original order.")
        return results[:top_k] if top_k else results

    # Build (query, passage) pairs
    pairs = [(query, r.get("chunk_text", "")) for r in results]

    try:
        scores = model.predict(pairs, show_progress_bar=False)
    except Exception as e:
        logger.error(f"Re-ranking prediction failed: {e} — returning original order.")
        return results[:top_k] if top_k else results

    # Attach re-rank scores and sort
    for r, score in zip(results, scores):
        r["rerank_score"] = float(score)

    reranked = sorted(results, key=lambda x: x.get("rerank_score", 0.0), reverse=True)

    # Update rank field
    for i, r in enumerate(reranked):
        r["rank"] = i + 1

    k = top_k or len(reranked)
    logger.info(
        f"Re-ranked {len(results)} → top {k} | "
        f"top score={reranked[0]['rerank_score']:.3f} | "
        f"query='{query[:40]}'"
    )
    return reranked[:k]


def hybrid_search_reranked(
    query: str,
    bid_id: str | None = None,
    doc_type: str | None = None,
    top_k: int = 5,
    candidate_k: int = 20,
    file_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Convenience wrapper: hybrid search → cross-encoder re-rank.

    Retrieves `candidate_k` candidates via RRF, then re-ranks to top `top_k`.
    This is the highest-quality retrieval configuration.

    Args:
        query:       User query.
        bid_id:      Optional bid filter.
        doc_type:    Optional doc_type filter.
        top_k:       Final number of results after re-ranking.
        candidate_k: How many candidates to fetch before re-ranking (default 20).
        file_names:  Optional filename filter.

    Returns:
        Top-k re-ranked CitedResult dicts with rerank_score.
    """
    from rfp_platform.search.tools import hybrid_search
    candidates = hybrid_search(
        query=query,
        bid_id=bid_id,
        doc_type=doc_type,
        top_k=candidate_k,
        file_names=file_names,
    )
    return rerank(query, candidates, top_k=top_k)
