"""
FastAPI application — REST API for the RFP Intelligence Platform.

Endpoints:
  POST /index          — Ingest a bid folder
  GET  /bids           — List all indexed bids
  POST /extract        — Run extraction pipeline on a bid
  POST /ask            — Q&A over bid documents
  GET  /results/{bid}  — Get saved extraction results
  POST /search         — Direct search (for testing)
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rfp_platform.core.config import get_settings
from rfp_platform.core.database import init_database, close_pool, get_conn
from rfp_platform.ingestion.pipeline import ingest_bid_folder
from rfp_platform.search.tools import (
    hybrid_search, get_bid_documents, get_addendum_changes
)

logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RFP Platform API...")
    init_database()
    yield
    close_pool()
    logger.info("RFP Platform API shut down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RFP Intelligence Platform",
    description=(
        "AI-powered RAG system for extracting structured fields from "
        "RFP bid documents using hybrid search + multi-agent extraction."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response Models ───────────────────────────────────────────────────

class IndexRequest(BaseModel):
    folder_path:  str
    bid_id:       str | None = None
    display_name: str | None = None
    force:        bool       = False

class IndexResponse(BaseModel):
    bid_id:       str
    files_indexed: int
    files_skipped: int
    files_errored: int
    results:      list[dict]

class ExtractRequest(BaseModel):
    bid_id: str

class AskRequest(BaseModel):
    question: str
    bid_id:   str | None = None

class SearchRequest(BaseModel):
    query:    str
    bid_id:   str | None = None
    doc_type: str | None = None
    top_k:    int        = Field(default=10, ge=1, le=50)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Health check — returns DB status and chunk count."""
    try:
        from psycopg2.extras import RealDictCursor
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM chunks;")
                chunk_count = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM bids;")
                bid_count = cur.fetchone()[0]
        return {"status": "ok", "version": "0.1.0", "chunks": chunk_count, "bids": bid_count}
    except Exception as e:
        return {"status": "degraded", "error": str(e), "chunks": 0, "bids": 0}


@app.post("/index", response_model=IndexResponse)
async def index_bid(req: IndexRequest):
    """
    Ingest all PDF/HTML files in a bid folder into PostgreSQL.
    Deduplication is automatic — safe to call multiple times.
    """
    try:
        results = await ingest_bid_folder(
            folder_path=req.folder_path,
            bid_id=req.bid_id,
            display_name=req.display_name,
            force_reindex=req.force,
        )
        bid_id  = req.bid_id or req.folder_path.rstrip("/\\").split("\\")[-1].split("/")[-1]
        indexed = sum(1 for r in results if r["status"] == "indexed")
        skipped = sum(1 for r in results if r["status"] == "skipped")
        errored = sum(1 for r in results if r["status"] == "error")
        return IndexResponse(
            bid_id=bid_id,
            files_indexed=indexed,
            files_skipped=skipped,
            files_errored=errored,
            results=results,
        )
    except Exception as e:
        logger.error(f"/index failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/bids")
def list_bids():
    """List all indexed bids with their document counts."""
    from psycopg2.extras import RealDictCursor
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT b.bid_id, b.display_name, b.created_at,
                       COUNT(DISTINCT d.id) AS doc_count,
                       COUNT(c.id) AS chunk_count
                FROM bids b
                LEFT JOIN documents d ON d.bid_id = b.bid_id
                LEFT JOIN chunks    c ON c.bid_id  = b.bid_id
                GROUP BY b.bid_id, b.display_name, b.created_at
                ORDER BY b.created_at DESC
            """)
            return [dict(r) for r in cur.fetchall()]


@app.get("/bids/{bid_id}/documents")
def get_documents(bid_id: str):
    """List all indexed documents for a bid."""
    return get_bid_documents(bid_id)


@app.post("/extract")
async def extract_fields(req: ExtractRequest, background_tasks: BackgroundTasks):
    """
    Run the full multi-agent extraction pipeline for a bid.
    Returns the 20 structured fields with citations, addendum changes, and validation.
    """
    try:
        from rfp_platform.agents.orchestrator import run_extraction
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, run_extraction, req.bid_id)
        return result
    except Exception as e:
        logger.error(f"/extract failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/results/{bid_id}")
def get_results(bid_id: str):
    """Get previously saved extraction results for a bid."""
    from psycopg2.extras import RealDictCursor
    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM bid_extractions WHERE bid_id = %s;",
                (bid_id,)
            )
            row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"No results found for bid '{bid_id}'")
    return dict(row)


@app.post("/ask")
async def ask_question(req: AskRequest):
    """
    Answer a free-form question about bid documents with citations.
    Optionally restrict to a specific bid_id.
    """
    try:
        from rfp_platform.agents.orchestrator import run_qa
        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, run_qa, req.question, req.bid_id)
        return result
    except Exception as e:
        logger.error(f"/ask failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
def search(req: SearchRequest):
    """
    Direct hybrid search endpoint — for testing retrieval quality.
    Returns results with citations.
    """
    results = hybrid_search(
        query=req.query,
        bid_id=req.bid_id,
        doc_type=req.doc_type,
        top_k=req.top_k,
    )
    return {"results": results, "total": len(results), "query": req.query}


@app.get("/addendums/{bid_id}")
def get_addendums(bid_id: str):
    """Get all addendum content for a bid."""
    return get_addendum_changes(bid_id)
