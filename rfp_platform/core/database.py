"""
PostgreSQL connection pool + schema initialisation.
Completely replaces the old get_db_connection() from app.py with:
  - A proper connection pool (no new connection per query)
  - Context manager for safe auto-close/rollback
  - Full schema for the RFP platform
"""
import logging
from contextlib import contextmanager
from typing import Generator

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor

from rfp_platform.core.config import get_settings

logger = logging.getLogger(__name__)

_pool: pool.SimpleConnectionPool | None = None


def _get_pool() -> pool.SimpleConnectionPool:
    """Lazy-initialise the connection pool (singleton)."""
    global _pool
    if _pool is None:
        cfg = get_settings()
        _pool = pool.SimpleConnectionPool(
            minconn=1,
            maxconn=10,
            **cfg.db_config_dict,
        )
        logger.info("PostgreSQL connection pool created.")
    return _pool


@contextmanager
def get_conn() -> Generator:
    """
    Context manager that checks out a connection from the pool,
    commits on success and rolls back + returns on any exception.

    Usage:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(...)
    """
    p = _get_pool()
    conn = p.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        p.putconn(conn)


# ─────────────────────────────────────────────────────────────────────────────
# Schema — fully stripped and rebuilt for the RFP platform
# ─────────────────────────────────────────────────────────────────────────────
SCHEMA_SQL = """
-- Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ── 1. bids ────────────────────────────────────────────────────────────────
-- One row per bid folder (Bid1, Bid2, …).  Incremental: inserting a new bid
-- folder never touches existing rows.
CREATE TABLE IF NOT EXISTS bids (
    id              SERIAL PRIMARY KEY,
    bid_id          TEXT UNIQUE NOT NULL,           -- 'Bid1', 'Bid2', …
    display_name    TEXT,                           -- 'Student and Staff Computing Devices'
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── 2. documents ──────────────────────────────────────────────────────────
-- One row per file in a bid folder.
CREATE TABLE IF NOT EXISTS documents (
    id              SERIAL PRIMARY KEY,
    bid_id          TEXT NOT NULL REFERENCES bids(bid_id) ON DELETE CASCADE,
    file_name       TEXT NOT NULL,
    file_hash       VARCHAR(64) NOT NULL,           -- SHA-256, used for dedup
    file_size       BIGINT NOT NULL,
    doc_type        TEXT NOT NULL,                  -- 'rfp' | 'addendum' | 'specs' | 'affidavit' | 'bid_page'
    addendum_number INTEGER,                        -- NULL unless doc_type = 'addendum'
    doc_date        DATE,                           -- extracted document date if available
    source_url      TEXT,                           -- for HTML pages
    raw_pages       JSONB,                          -- [{page_num, content}, …]
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (bid_id, file_hash)
);

CREATE INDEX IF NOT EXISTS idx_documents_bid_id     ON documents (bid_id);
CREATE INDEX IF NOT EXISTS idx_documents_doc_type   ON documents (doc_type);
CREATE INDEX IF NOT EXISTS idx_documents_file_hash  ON documents (file_hash);

-- ── 3. chunks ─────────────────────────────────────────────────────────────
-- Core RAG table: one row per chunk, holds both dense vector and BM25 tsvector.
CREATE TABLE IF NOT EXISTS chunks (
    id              SERIAL PRIMARY KEY,
    document_id     INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    bid_id          TEXT NOT NULL,                  -- denormalised for fast filter
    doc_type        TEXT NOT NULL,                  -- denormalised for fast filter
    addendum_number INTEGER,
    page_number     INTEGER NOT NULL,
    chunk_index     INTEGER NOT NULL,               -- within document
    chunk_text      TEXT NOT NULL,
    token_count     INTEGER,                        -- tiktoken cl100k_base count
    embedding       vector(3072),                   -- text-embedding-3-large, full 3072-dim
    fts_vector      tsvector                        -- BM25-style full-text index (auto-maintained)
        GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED,
    metadata        JSONB DEFAULT '{}',             -- any extra k/v pairs
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- NOTE: pgvector ANN indexes (ivfflat, hnsw) support max 2000 dims.
-- At 3072-dim we use EXACT nearest-neighbor (sequential scan with <=> operator).
-- For our corpus size (~200 chunks), exact scan is sub-millisecond and MORE accurate.
-- ANN indexes only matter at 100,000+ vectors.

-- Full-text GIN index for BM25 / keyword search
CREATE INDEX IF NOT EXISTS idx_chunks_fts
    ON chunks USING GIN (fts_vector);

-- Metadata filter indexes
CREATE INDEX IF NOT EXISTS idx_chunks_bid_id         ON chunks (bid_id);
CREATE INDEX IF NOT EXISTS idx_chunks_doc_type       ON chunks (doc_type);
CREATE INDEX IF NOT EXISTS idx_chunks_addendum       ON chunks (addendum_number);

-- ── 4. bid_extractions ────────────────────────────────────────────────────
-- Final structured output: one row per bid, stores all 20 fields as JSONB.
CREATE TABLE IF NOT EXISTS bid_extractions (
    id               SERIAL PRIMARY KEY,
    bid_id           TEXT UNIQUE NOT NULL REFERENCES bids(bid_id) ON DELETE CASCADE,
    fields           JSONB NOT NULL DEFAULT '{}',   -- {field_name: {value, sources, confidence, notes}}
    addendum_changes JSONB DEFAULT '[]',
    validation       JSONB DEFAULT '{}',            -- {passed, failed, not_found}
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    updated_at       TIMESTAMPTZ DEFAULT NOW()
);

-- ── 5. agent_logs ─────────────────────────────────────────────────────────
-- Observability: every agent step is logged here.
CREATE TABLE IF NOT EXISTS agent_logs (
    id          SERIAL PRIMARY KEY,
    run_id      UUID NOT NULL,
    bid_id      TEXT,
    agent_name  TEXT NOT NULL,
    step        TEXT,
    input       JSONB,
    output      JSONB,
    tool_calls  JSONB,
    tokens_used INTEGER,
    latency_ms  INTEGER,
    status      TEXT DEFAULT 'success',             -- 'success' | 'error' | 'retry'
    error_msg   TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_logs_run_id    ON agent_logs (run_id);
CREATE INDEX IF NOT EXISTS idx_agent_logs_bid_id    ON agent_logs (bid_id);

-- ── 6. eval_questions ────────────────────────────────────────────────────
-- Evaluation dataset: question → expected source passage.
CREATE TABLE IF NOT EXISTS eval_questions (
    id               SERIAL PRIMARY KEY,
    bid_id           TEXT,
    question         TEXT NOT NULL,
    expected_passage TEXT,
    expected_file    TEXT,
    expected_page    INTEGER,
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

-- ── 7. eval_results ──────────────────────────────────────────────────────
-- Stores retrieval metrics per question per config run.
CREATE TABLE IF NOT EXISTS eval_results (
    id                SERIAL PRIMARY KEY,
    eval_question_id  INTEGER REFERENCES eval_questions(id),
    config_name       TEXT NOT NULL,                -- 'vector_only' | 'hybrid' | 'hybrid+rerank'
    retrieved_chunks  JSONB,
    recall_at_5       FLOAT,
    recall_at_10      FLOAT,
    mrr               FLOAT,
    created_at        TIMESTAMPTZ DEFAULT NOW()
);
"""


def init_database() -> bool:
    """
    Create all tables, extensions and indexes.
    Safe to call multiple times (all statements are IF NOT EXISTS).
    Returns True on success, False on failure.
    """
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(SCHEMA_SQL)
        logger.info("✅ Database schema initialised successfully.")
        return True
    except Exception as e:
        logger.error(f"❌ Database initialisation failed: {e}")
        return False


def close_pool() -> None:
    """Close all connections in the pool (call on app shutdown)."""
    global _pool
    if _pool:
        _pool.closeall()
        _pool = None
        logger.info("Connection pool closed.")
