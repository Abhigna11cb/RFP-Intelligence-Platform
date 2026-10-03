"""
Search tools for the RFP Intelligence Platform.

Analysis of your existing tools vs what's needed:

YOUR TOOLS            DECISION      REASON
──────────────────────────────────────────────────────────────────
semantic_search       KEEP+EXTEND   Core RAG tool — add bid_id/doc_type filter
keyword_search        KEEP+EXTEND   BM25 exact match — critical for bid #, model #
retrieve_nearby_chunks KEEP+EXTEND  Context expansion — great for RFP sections
retrieve_pdf_page_image KEEP        Visual fallback for garbled/table-heavy pages
summary_agent         REMOVED       Not a tool in our system — agents use LangGraph
                                    state + structured Pydantic outputs instead

NEW RFP-SPECIFIC TOOLS ADDED:
hybrid_search         NEW (MAIN)    RRF of vector + BM25 — primary search tool
get_bid_documents     NEW           List all indexed files per bid
get_addendum_changes  NEW           Fetch all addendum content for reconciliation
search_by_field       NEW           Targeted search for a specific extraction field
──────────────────────────────────────────────────────────────────

All tools return a consistent CitedResult schema:
  {chunk_text, file_name, page_number, bid_id, doc_type, score, chunk_index}
"""
from __future__ import annotations

import logging
from typing import Any

from psycopg2.extras import RealDictCursor

from rfp_platform.core.config import get_settings
from rfp_platform.core.database import get_conn
from rfp_platform.search.embeddings import get_query_embedding

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result schema (consistent across all tools)
# ─────────────────────────────────────────────────────────────────────────────

def _make_result(row: dict, score: float, rank: int) -> dict[str, Any]:
    return {
        "chunk_text":   row["chunk_text"],
        "file_name":    row["file_name"],
        "page_number":  row["page_number"],
        "chunk_index":  row["chunk_index"],
        "bid_id":       row["bid_id"],
        "doc_type":     row["doc_type"],
        "addendum_number": row.get("addendum_number"),
        "score":        round(score, 4),
        "rank":         rank,
    }


def _build_filter_clause(
    bid_id: str | None,
    doc_type: str | None,
    addendum_number: int | None,
    file_names: list[str] | None,
) -> tuple[str, list]:
    """Build a WHERE clause from optional metadata filters."""
    clauses, params = [], []

    if bid_id:
        clauses.append("c.bid_id = %s")
        params.append(bid_id)
    if doc_type:
        clauses.append("c.doc_type = %s")
        params.append(doc_type)
    if addendum_number is not None:
        clauses.append("c.addendum_number = %s")
        params.append(addendum_number)
    if file_names:
        placeholders = ",".join(["%s"] * len(file_names))
        clauses.append(f"d.file_name IN ({placeholders})")
        params.extend(file_names)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    return where, params


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1 — hybrid_search  (PRIMARY TOOL — use this by default)
# ─────────────────────────────────────────────────────────────────────────────

def hybrid_search(
    query: str,
    bid_id: str | None = None,
    doc_type: str | None = None,
    top_k: int | None = None,
    file_names: list[str] | None = None,
    addendum_number: int | None = None,
) -> list[dict[str, Any]]:
    """
    PRIMARY SEARCH TOOL — Hybrid semantic + keyword search with Reciprocal Rank Fusion.

    Combines:
      - Dense vector search (text-embedding-3-large, cosine similarity)
      - BM25 full-text search (PostgreSQL tsvector)
    Merges results via RRF for the best of both worlds.

    Use for: all general queries, field extraction, Q&A over bids.

    Args:
        query:            Natural language or keyword query.
        bid_id:           Filter to a specific bid ('Bid1', 'Bid2').
        doc_type:         Filter by document type ('rfp','addendum','specs','affidavit','bid_page').
        top_k:            Number of results (default from config).
        file_names:       Optional list of specific filenames to search within.
        addendum_number:  Filter to a specific addendum number.

    Returns:
        List of CitedResult dicts sorted by RRF score (descending).
    """
    cfg = get_settings()
    k = top_k or cfg.default_top_k
    rrf_k = cfg.rrf_k
    where, filter_params = _build_filter_clause(bid_id, doc_type, addendum_number, file_names)

    query_vec = get_query_embedding(query)
    fetch = k * 3  # fetch more candidates before merging

    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:

            # ── Semantic leg ──────────────────────────────────────────────
            sem_sql = f"""
                SELECT c.id, c.chunk_text, c.page_number, c.chunk_index,
                       c.bid_id, c.doc_type, c.addendum_number,
                       d.file_name,
                       1 - (c.embedding <=> %s::vector) AS sem_score,
                       ROW_NUMBER() OVER (ORDER BY c.embedding <=> %s::vector) AS sem_rank
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                {where}
                AND c.embedding IS NOT NULL
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
            """
            cur.execute(sem_sql, [query_vec, query_vec] + filter_params + [query_vec, fetch])
            sem_rows = {row["id"]: dict(row) for row in cur.fetchall()}

            # ── Keyword leg (BM25) ────────────────────────────────────────
            kw_sql = f"""
                SELECT c.id, c.chunk_text, c.page_number, c.chunk_index,
                       c.bid_id, c.doc_type, c.addendum_number,
                       d.file_name,
                       ts_rank_cd(c.fts_vector, plainto_tsquery('english', %s)) AS kw_score,
                       ROW_NUMBER() OVER (
                           ORDER BY ts_rank_cd(c.fts_vector, plainto_tsquery('english', %s)) DESC
                       ) AS kw_rank
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                {where}
                AND c.fts_vector @@ plainto_tsquery('english', %s)
                ORDER BY kw_score DESC
                LIMIT %s
            """
            cur.execute(kw_sql, [query, query] + filter_params + [query, fetch])
            kw_rows = {row["id"]: dict(row) for row in cur.fetchall()}

    # ── RRF merge ────────────────────────────────────────────────────────────
    all_ids = set(sem_rows) | set(kw_rows)
    rrf_scores: dict[int, float] = {}
    for chunk_id in all_ids:
        sem_rank = sem_rows[chunk_id]["sem_rank"] if chunk_id in sem_rows else 1000
        kw_rank  = kw_rows[chunk_id]["kw_rank"]  if chunk_id in kw_rows  else 1000
        rrf_scores[chunk_id] = 1.0 / (rrf_k + sem_rank) + 1.0 / (rrf_k + kw_rank)

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]

    results = []
    for rank, (chunk_id, score) in enumerate(ranked, 1):
        row = sem_rows.get(chunk_id) or kw_rows.get(chunk_id)
        results.append(_make_result(row, score, rank))

    logger.info(
        f"hybrid_search '{query[:40]}' → {len(results)} results "
        f"(sem={len(sem_rows)}, kw={len(kw_rows)})"
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2 — semantic_search  (your tool, extended with metadata filters)
# ─────────────────────────────────────────────────────────────────────────────

def semantic_search(
    query: str,
    bid_id: str | None = None,
    doc_type: str | None = None,
    top_k: int | None = None,
    file_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Pure vector/semantic search using text-embedding-3-large + pgvector cosine similarity.

    Use when:
      - Query is conceptual/meaning-based
      - You want broad thematic matches
      - Keyword search returned nothing

    From your existing tools: adapted with bid_id/doc_type metadata filters
    and upgraded to text-embedding-3-large.

    Args:
        query:      Natural language query.
        bid_id:     Restrict to a specific bid.
        doc_type:   Restrict by document type.
        top_k:      Number of results.
        file_names: Optional list of specific PDF filenames to search.
    """
    cfg = get_settings()
    k = top_k or cfg.default_top_k
    where, filter_params = _build_filter_clause(bid_id, doc_type, None, file_names)
    query_vec = get_query_embedding(query)

    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            sql = f"""
                SELECT c.id, c.chunk_text, c.page_number, c.chunk_index,
                       c.bid_id, c.doc_type, c.addendum_number,
                       d.file_name,
                       1 - (c.embedding <=> %s::vector) AS score
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                {where}
                AND c.embedding IS NOT NULL
                AND 1 - (c.embedding <=> %s::vector) > %s
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
            """
            threshold = 1 - cfg.similarity_threshold
            cur.execute(
                sql,
                [query_vec] + filter_params + [query_vec, threshold, query_vec, k],
            )
            rows = cur.fetchall()

    results = [_make_result(dict(r), r["score"], i + 1) for i, r in enumerate(rows)]
    logger.info(f"semantic_search '{query[:40]}' → {len(results)} results")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3 — keyword_search  (your tool, extended)
# ─────────────────────────────────────────────────────────────────────────────

def keyword_search(
    query: str,
    bid_id: str | None = None,
    doc_type: str | None = None,
    top_k: int | None = None,
    file_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    BM25 keyword / full-text search using PostgreSQL tsvector.

    Use when:
      - Looking for exact bid numbers (JA-207652, E20P4600040)
      - Looking for model numbers (CC7802, WD22TB4, 210-BLYZ)
      - Looking for part SKUs, dates, specific names
      - Exact phrase matching

    From your existing tools: same concept, now uses PostgreSQL tsvector
    instead of a separate BM25 index.

    Args:
        query:      Keywords or phrase to search.
        bid_id:     Restrict to a specific bid.
        doc_type:   Restrict by document type.
        top_k:      Number of results.
        file_names: Optional list of specific PDF filenames.
    """
    cfg = get_settings()
    k = top_k or cfg.default_top_k
    where, filter_params = _build_filter_clause(bid_id, doc_type, None, file_names)

    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            sql = f"""
                SELECT c.id, c.chunk_text, c.page_number, c.chunk_index,
                       c.bid_id, c.doc_type, c.addendum_number,
                       d.file_name,
                       ts_rank_cd(c.fts_vector, plainto_tsquery('english', %s)) AS score
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                {where}
                AND c.fts_vector @@ plainto_tsquery('english', %s)
                ORDER BY score DESC
                LIMIT %s
            """
            cur.execute(sql, [query] + filter_params + [query, k])
            rows = cur.fetchall()

    results = [_make_result(dict(r), r["score"], i + 1) for i, r in enumerate(rows)]
    logger.info(f"keyword_search '{query[:40]}' → {len(results)} results")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Tool 4 — retrieve_nearby_chunks  (your tool, kept almost identical)
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_nearby_chunks(
    file_name: str,
    chunk_index: int,
    context_window: int = 2,
    bid_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieves the N chunks before and after a specific chunk for context expansion.

    Use when:
      - A search result is interesting but lacks full context
      - An extraction agent needs surrounding sentences/paragraphs
      - A sentence mentions a due date but you need the full section

    From your existing tools: retrieve_nearby_chunks — kept identical,
    adapted to use file_name + chunk_index instead of document_id.

    Args:
        file_name:      Exact filename from a previous search result.
        chunk_index:    The chunk_index from a previous search result.
        context_window: Number of chunks before and after to fetch (default: 2).
        bid_id:         Optional bid filter for extra safety.
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:

            # Get the target chunk's document_id first
            cur.execute(
                """
                SELECT c.document_id, c.page_number, c.bid_id, c.doc_type, c.addendum_number
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE d.file_name = %s AND c.chunk_index = %s
                LIMIT 1
                """,
                (file_name, chunk_index),
            )
            anchor = cur.fetchone()
            if not anchor:
                logger.warning(f"retrieve_nearby_chunks: chunk not found ({file_name}, {chunk_index})")
                return []

            doc_id = anchor["document_id"]

            # Fetch the window of chunks
            cur.execute(
                """
                SELECT c.chunk_text, c.page_number, c.chunk_index,
                       c.bid_id, c.doc_type, c.addendum_number,
                       d.file_name,
                       0.0 AS score
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.document_id = %s
                  AND c.chunk_index BETWEEN %s AND %s
                ORDER BY c.chunk_index
                """,
                (doc_id, chunk_index - context_window, chunk_index + context_window),
            )
            rows = cur.fetchall()

    results = [_make_result(dict(r), 0.0, i + 1) for i, r in enumerate(rows)]
    logger.info(
        f"retrieve_nearby_chunks '{file_name}' chunk={chunk_index} "
        f"window={context_window} → {len(results)} chunks"
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Tool 5 — retrieve_pdf_page_image  (your tool, kept — visual fallback)
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_pdf_page_image(
    file_name: str,
    page_numbers: list[int],
    bid_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Renders specific PDF pages as base64-encoded PNG images.

    Use when:
      - Extracted text looks garbled, scrambled, or missing
      - A page contains a complex table/diagram that didn't parse well
      - You need to visually verify a value before citing it

    From your existing tools: retrieve_pdf_page_image — adapted to look up
    file path from the documents table rather than requiring a file upload.

    Returns:
        List of {page_number, image_b64, file_name} dicts.
        Pass image_b64 to a vision-capable LLM for OCR/visual analysis.
    """
    import base64
    import fitz  # PyMuPDF

    # Look up the actual file path from DB (stored during ingestion)
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT file_name FROM documents WHERE file_name = %s LIMIT 1;",
                (file_name,),
            )
            row = cur.fetchone()

    if not row:
        return [{"error": f"File '{file_name}' not found in database"}]

    # We stored file_name but not path — look in bid folders relative to cwd
    import os
    file_path = None
    for root, _, files in os.walk("."):
        if file_name in files:
            file_path = os.path.join(root, file_name)
            break

    if not file_path or not os.path.exists(file_path):
        return [{"error": f"Physical file '{file_name}' not found on disk"}]

    results = []
    doc = fitz.open(file_path)
    for page_num in page_numbers:
        idx = page_num - 1  # 0-indexed
        if idx < 0 or idx >= len(doc):
            results.append({"page_number": page_num, "error": "Page out of range"})
            continue
        page = doc[idx]
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        results.append({
            "page_number": page_num,
            "file_name":   file_name,
            "image_b64":   img_b64,
            "width":       pix.width,
            "height":      pix.height,
        })
    doc.close()

    logger.info(f"retrieve_pdf_page_image '{file_name}' pages={page_numbers}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Tool 6 — get_bid_documents  (NEW — RFP-specific)
# ─────────────────────────────────────────────────────────────────────────────

def get_bid_documents(bid_id: str) -> list[dict[str, Any]]:
    """
    Lists all indexed documents for a given bid, with their metadata.

    Use when:
      - Orchestrator needs to know what files exist before routing
      - Addendum Reconciliation Agent needs to discover addendum numbers
      - You need to check if all expected files are indexed

    Returns:
        List of {file_name, doc_type, addendum_number, file_size, chunk_count} dicts.
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT d.file_name, d.doc_type, d.addendum_number,
                       d.file_size, d.created_at,
                       COUNT(c.id) AS chunk_count
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.id
                WHERE d.bid_id = %s
                GROUP BY d.id
                ORDER BY d.doc_type, d.addendum_number NULLS LAST
                """,
                (bid_id,),
            )
            rows = cur.fetchall()

    return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# Tool 7 — get_addendum_changes  (NEW — critical for addendum reconciliation)
# ─────────────────────────────────────────────────────────────────────────────

def get_addendum_changes(bid_id: str) -> list[dict[str, Any]]:
    """
    Fetches ALL addendum content for a bid, ordered by addendum number.
    Used by the Addendum Reconciliation Agent to detect field changes.

    Returns all chunks from addendum documents so the agent can compare
    original RFP values vs addendum updates (e.g., due date extensions).

    Args:
        bid_id: The bid to fetch addendums for ('Bid1', 'Bid2', etc.)

    Returns:
        List of chunks from all addendums, ordered by addendum_number then chunk_index.
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT c.chunk_text, c.page_number, c.chunk_index,
                       c.bid_id, c.doc_type, c.addendum_number,
                       d.file_name, 1.0 AS score
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.bid_id = %s
                  AND c.doc_type = 'addendum'
                ORDER BY c.addendum_number, c.chunk_index
                """,
                (bid_id,),
            )
            rows = cur.fetchall()

    results = [_make_result(dict(r), 1.0, i + 1) for i, r in enumerate(rows)]
    logger.info(f"get_addendum_changes bid='{bid_id}' → {len(results)} addendum chunks")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Tool 8 — search_by_field  (NEW — targeted field extraction search)
# ─────────────────────────────────────────────────────────────────────────────

# Pre-defined query expansions for each of the 20 extraction fields.
# Queries are intentionally verbose to maximise recall for each field.
_FIELD_QUERIES: dict[str, str] = {
    "Bid Number":           "solicitation number RFP IFB ITB bid number reference identifier JA- E20P",
    "Title":                "title name of bid solicitation request for proposal subject heading",
    "Due Date":             "due date submission deadline closing date proposals responses addendum",
    "Bid Submission Type":  "how to submit bid electronic portal sealed envelope email eMMA eMaryland",
    "Term of Bid":          "contract term duration period years renewal options extensions",
    "Pre Bid Meeting":      "pre-bid pre-proposal conference meeting date time location mandatory optional",
    "Installation":         "installation deployment imaging asset tagging etching services required vendor",
    "Bid Bond Requirement": "bid bond surety security deposit performance bond required amount percentage",
    "Delivery Date":        "delivery date required window business days after award ship FOB",
    "Payment Terms":        "payment terms net 30 invoice submitted accepted conditions interest penalty",
    "Any Additional Documentation Required": "affidavit certificate W-9 W9 form MWBE mercury required documentation submit",
    "MFG for Registration": "manufacturer Dell HP Lenovo Apple authorized reseller OEM brand registration",
    "Contract or Cooperative to use": "cooperative contract vehicle piggyback state contract number TIPS NCPA",
    "Model_no":             "model number SI# CC product model line item configuration table",
    "Part_no":              "part number SKU item number 210- catalog specification sheet",
    "Product":              "product name quantity items requested computing devices laptops chromebooks",
    "contact_info":         "contact person name email phone procurement officer buyer purchasing",
    "company_name":         "issuing organization agency school district state office department",
    # Richer queries for summary and specs — pull more content
    "Bid Summary":          "purpose scope overview background RFP summary district procurement objective deliverables",
    "Product Specification": "technical specifications minimum requirements CPU processor RAM memory storage SSD display screen OS Windows warranty year Chromebook laptop",
}

# Secondary queries for warranty — fetched as supplemental evidence for Product Specification
_WARRANTY_QUERY = "warranty minimum years service repair replacement business days cost Chromebook laptop Windows"


def search_by_field(
    field_name: str,
    bid_id: str | None = None,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    """
    Targeted hybrid search for a specific extraction field.
    Uses pre-defined query expansions for each of the 20 required fields
    so the Extraction Agent gets the most relevant chunks without needing
    to craft perfect queries itself.

    Args:
        field_name: One of the 20 required fields (e.g., 'Due Date', 'Bid Number').
        bid_id:     Restrict to a specific bid.
        top_k:      Number of results (default: 8 for extraction tasks).

    Returns:
        Hybrid search results ranked by RRF score.
    """
    expanded_query = _FIELD_QUERIES.get(field_name, field_name)
    logger.info(f"search_by_field '{field_name}' → query: '{expanded_query[:50]}'")
    return hybrid_search(query=expanded_query, bid_id=bid_id, top_k=top_k)


# ─────────────────────────────────────────────────────────────────────────────
# Tool registry — used by LangGraph agents
# ─────────────────────────────────────────────────────────────────────────────

TOOL_REGISTRY: dict[str, callable] = {
    "hybrid_search":          hybrid_search,
    "semantic_search":        semantic_search,
    "keyword_search":         keyword_search,
    "retrieve_nearby_chunks": retrieve_nearby_chunks,
    "retrieve_pdf_page_image":retrieve_pdf_page_image,
    "get_bid_documents":      get_bid_documents,
    "get_addendum_changes":   get_addendum_changes,
    "search_by_field":        search_by_field,
}


def call_tool(name: str, **kwargs) -> Any:
    """
    Dispatch a tool call by name. Used by agents to invoke tools dynamically.
    Raises KeyError if tool not found.
    """
    if name not in TOOL_REGISTRY:
        raise KeyError(f"Unknown tool: '{name}'. Available: {list(TOOL_REGISTRY)}")
    return TOOL_REGISTRY[name](**kwargs)
