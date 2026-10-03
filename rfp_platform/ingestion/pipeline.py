"""
Ingestion pipeline — Stage 1 of RAG.

Handles both file types:
  - PDF  → pymupdf4llm (markdown per-page) + your async chunking pattern
  - HTML → BeautifulSoup structured extraction

Flow per file:
  1. Detect doc_type from filename
  2. Parse → page text
  3. OCR fallback for empty pages (if enabled)
  4. smart_chunk_page() per page  (your logic)
  5. merge_small_chunks()         (your logic)
  6. Insert document + chunks to PostgreSQL (dedup via file_hash)
  7. Generate embeddings in batches via text-embedding-3-large → update chunks

Called by:
  - CLI:  python main.py index --bid ./Bid1
  - API:  POST /index
  - main.py entry point
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Any

import pymupdf4llm
from bs4 import BeautifulSoup

from rfp_platform.core.config import get_settings
from rfp_platform.core.database import get_conn
from rfp_platform.ingestion.chunker import PDFChunkingHelpers
from rfp_platform.search.embeddings import get_embeddings  # OpenAI text-embedding-3-large

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Doc-type detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_doc_type(filename: str) -> tuple[str, int | None]:
    """
    Infer doc_type and addendum_number from filename (case-insensitive).

    Returns (doc_type, addendum_number)
    doc_type ∈ {'rfp', 'addendum', 'specs', 'affidavit', 'bid_page'}
    """
    name = filename.lower()

    # Check for addendum first (most specific)
    add_match = re.search(r"addendum[^\d]*(\d+)", name)
    if add_match:
        return "addendum", int(add_match.group(1))

    if any(kw in name for kw in ("affidavit", "mercury", "contract_affidavit")):
        return "affidavit", None

    if any(kw in name for kw in ("spec", "specs", "specifications", "laptop_specs")):
        return "specs", None

    if name.endswith(".html") or name.endswith(".htm"):
        return "bid_page", None

    # Default: treat as main RFP
    return "rfp", None


# ─────────────────────────────────────────────────────────────────────────────
# File hash (dedup — from your app.py)
# ─────────────────────────────────────────────────────────────────────────────

def compute_file_hash(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha.update(block)
    return sha.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# OCR fallback (optional, requires pytesseract + tesseract binary)
# ─────────────────────────────────────────────────────────────────────────────

async def _ocr_page(pdf_path: str, page_index: int) -> str:
    """
    OCR a single page using pytesseract.
    Runs in a thread executor so it doesn't block the event loop.
    """
    cfg = get_settings()
    if not cfg.ocr_enabled:
        return ""
    try:
        import pytesseract
        import fitz  # PyMuPDF

        def _do_ocr() -> str:
            doc = fitz.open(pdf_path)
            page = doc[page_index]
            pix = page.get_pixmap(dpi=300)
            img_bytes = pix.tobytes("png")
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(img_bytes))
            return pytesseract.image_to_string(img, lang=cfg.ocr_language)

        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, _do_ocr)
        logger.info(f"OCR extracted {len(text)} chars from page {page_index + 1}")
        return text
    except Exception as e:
        logger.warning(f"OCR failed for page {page_index}: {e}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# PDF ingestion  (your async pattern, extended)
# ─────────────────────────────────────────────────────────────────────────────

async def _extract_and_chunk_pdf(
    saved_path: str,
    filename: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """
    Your original async chunking pattern — kept intact, extended with:
      - proper OCR fallback gate (config-driven)
      - heading/table-aware chunking via PDFChunkingHelpers
      - total_tokens returned for logging

    Returns: (all_chunks, pages_data, total_tokens)
    """
    try:
        chunker = PDFChunkingHelpers(saved_path)

        # Extract per-page markdown using pymupdf4llm (your chosen tool)
        md_pages = pymupdf4llm.to_markdown(saved_path, page_chunks=True)

        # Build pages_data for storage in documents.raw_pages
        pages_data = [
            {
                "page_num": i + 1,
                "content": page.get("text", ""),
            }
            for i, page in enumerate(md_pages)
        ]

        all_chunks: list[dict[str, Any]] = []

        for i, page in enumerate(md_pages):
            page_number = i + 1
            page_text = page.get("text", "")

            # OCR fallback for empty pages (your pattern)
            if not page_text.strip():
                logger.info(f"Page {page_number} has no text — attempting OCR")
                page_text = await _ocr_page(saved_path, i)
                pages_data[i]["content"] = page_text

            if not page_text.strip():
                logger.debug(f"Page {page_number} still empty after OCR — skipping")
                continue

            # Your smart chunking logic
            page_chunks = chunker.smart_chunk_page(page_text, page_number)
            all_chunks.extend(page_chunks)

        # Your merge logic
        all_chunks = chunker.merge_small_chunks(all_chunks)

        if not all_chunks:
            raise ValueError(
                "No valid text content found in PDF. "
                "Provide a PDF with text contents or enable OCR."
            )

        # Token count (your tiktoken pattern)
        import tiktoken
        tokenizer = tiktoken.get_encoding("cl100k_base")
        total_tokens = sum(
            len(tokenizer.encode(c["page_content"])) for c in all_chunks
        )

        logger.info(
            f"✅ PDF '{filename}': {len(all_chunks)} chunks from "
            f"{len(pages_data)} pages, {total_tokens} tokens"
        )
        return all_chunks, pages_data, total_tokens

    except Exception as e:
        logger.error(f"Error extracting/chunking PDF '{filename}': {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# HTML ingestion
# ─────────────────────────────────────────────────────────────────────────────

def _extract_and_chunk_html(
    file_path: str,
    filename: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """
    Parse BidNet Direct HTML pages (and similar portals).
    Extracts structured fields from the page, then creates a small
    number of meaningful chunks.
    """
    try:
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f, "html.parser")

        # Remove nav, script, style noise
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        full_text = soup.get_text(separator="\n", strip=True)
        # Collapse excess whitespace
        full_text = re.sub(r"\n{3,}", "\n\n", full_text)

        # Treat the whole HTML as one synthetic "page"
        pages_data = [{"page_num": 1, "content": full_text}]

        chunker = PDFChunkingHelpers(file_path)
        page_chunks = chunker.smart_chunk_page(full_text, page_number=1)
        all_chunks = chunker.merge_small_chunks(page_chunks)

        if not all_chunks:
            # Fallback: single chunk with full text
            import tiktoken
            tokenizer = tiktoken.get_encoding("cl100k_base")
            all_chunks = [
                {
                    "page_content": full_text[:4000],
                    "page_number": 1,
                    "chunk_index": 0,
                    "token_count": len(tokenizer.encode(full_text[:4000])),
                }
            ]

        import tiktoken
        tokenizer = tiktoken.get_encoding("cl100k_base")
        total_tokens = sum(
            len(tokenizer.encode(c["page_content"])) for c in all_chunks
        )

        logger.info(
            f"✅ HTML '{filename}': {len(all_chunks)} chunks, {total_tokens} tokens"
        )
        return all_chunks, pages_data, total_tokens

    except Exception as e:
        logger.error(f"Error extracting HTML '{filename}': {e}")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Database helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_bid(bid_id: str, display_name: str | None = None) -> None:
    """Insert bid row if it doesn't exist yet (incremental indexing)."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO bids (bid_id, display_name)
                VALUES (%s, %s)
                ON CONFLICT (bid_id) DO NOTHING;
                """,
                (bid_id, display_name),
            )


def _document_exists(bid_id: str, file_hash: str) -> int | None:
    """
    Check dedup by (bid_id, file_hash). Returns existing document_id or None.
    Adapted from your document_exists() in app.py.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM documents WHERE bid_id = %s AND file_hash = %s;",
                (bid_id, file_hash),
            )
            row = cur.fetchone()
            return row[0] if row else None


def _insert_document(
    bid_id: str,
    filename: str,
    file_hash: str,
    file_size: int,
    doc_type: str,
    addendum_number: int | None,
    pages_data: list[dict],
    source_url: str | None = None,
) -> int:
    """Insert a document row and return its id."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            import json
            cur.execute(
                """
                INSERT INTO documents
                    (bid_id, file_name, file_hash, file_size, doc_type,
                     addendum_number, raw_pages, source_url)
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                RETURNING id;
                """,
                (
                    bid_id, filename, file_hash, file_size, doc_type,
                    addendum_number, json.dumps(pages_data), source_url,
                ),
            )
            return cur.fetchone()[0]


def _insert_chunks(
    document_id: int,
    bid_id: str,
    doc_type: str,
    addendum_number: int | None,
    chunks: list[dict[str, Any]],
) -> list[int]:
    """
    Bulk-insert chunks (without embeddings yet).
    Returns list of inserted chunk IDs.
    Batch commit pattern adapted from your store_document_and_chunks().
    """
    chunk_ids: list[int] = []
    batch_size = 100

    with get_conn() as conn:
        with conn.cursor() as cur:
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i: i + batch_size]
                for chunk in batch:
                    cur.execute(
                        """
                        INSERT INTO chunks
                            (document_id, bid_id, doc_type, addendum_number,
                             page_number, chunk_index, chunk_text, token_count)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id;
                        """,
                        (
                            document_id,
                            bid_id,
                            doc_type,
                            addendum_number,
                            chunk["page_number"],
                            chunk["chunk_index"],
                            chunk["page_content"],
                            chunk.get("token_count", 0),
                        ),
                    )
                    chunk_ids.append(cur.fetchone()[0])
                conn.commit()

    return chunk_ids


def _update_embeddings(
    chunk_ids: list[int],
    chunks: list[dict[str, Any]],
) -> None:
    """
    Generate embeddings via text-embedding-3-large (OpenAI) and write to DB.
    Uses get_embeddings() from search/embeddings.py which batches to 100 per API call.
    """
    texts = [c["page_content"] for c in chunks]

    # Generate all embeddings (batched internally by get_embeddings)
    embeddings = get_embeddings(texts)

    with get_conn() as conn:
        with conn.cursor() as cur:
            for chunk_id, emb in zip(chunk_ids, embeddings):
                cur.execute(
                    "UPDATE chunks SET embedding = %s WHERE id = %s;",
                    (emb, chunk_id),
                )
        conn.commit()

    # Refresh IVFFlat index statistics after bulk load
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("ANALYZE chunks;")
    logger.info(f"Embeddings written for {len(chunk_ids)} chunks via text-embedding-3-large.")


# ─────────────────────────────────────────────────────────────────────────────
# Main public entry point
# ─────────────────────────────────────────────────────────────────────────────

async def ingest_file(
    file_path: str,
    bid_id: str,
    display_name: str | None = None,
    force_reindex: bool = False,
) -> dict[str, Any]:
    """
    Ingest a single file (PDF or HTML) into the RFP platform.

    Args:
        file_path:     Absolute or relative path to the file.
        bid_id:        Bid folder identifier, e.g. 'Bid1'.
        display_name:  Human-readable bid name (optional).
        force_reindex: If True, re-process even if file_hash already exists.

    Returns:
        {
          "status":       "indexed" | "skipped" | "error",
          "document_id":  int | None,
          "chunks":       int,
          "tokens":       int,
          "message":      str,
        }
    """
    path = Path(file_path)
    filename = path.name
    file_size = path.stat().st_size
    file_hash = compute_file_hash(str(path))
    doc_type, addendum_number = detect_doc_type(filename)

    logger.info(
        f"Ingesting '{filename}' → bid='{bid_id}' "
        f"doc_type='{doc_type}' addendum={addendum_number}"
    )

    # Ensure bid exists (incremental)
    _ensure_bid(bid_id, display_name)

    # Dedup check (your pattern from app.py)
    if not force_reindex:
        existing_id = _document_exists(bid_id, file_hash)
        if existing_id:
            logger.info(f"⏭ Skipping '{filename}' — already indexed (doc_id={existing_id})")
            return {
                "status": "skipped",
                "document_id": existing_id,
                "chunks": 0,
                "tokens": 0,
                "message": f"Already indexed as document_id={existing_id}",
            }

    try:
        # ── Parse & chunk ─────────────────────────────────────────────────
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            all_chunks, pages_data, total_tokens = await _extract_and_chunk_pdf(
                str(path), filename
            )
        elif suffix in (".html", ".htm"):
            # HTML is synchronous — wrap in executor so we don't block
            loop = asyncio.get_event_loop()
            all_chunks, pages_data, total_tokens = await loop.run_in_executor(
                None, _extract_and_chunk_html, str(path), filename
            )
        else:
            return {
                "status": "error",
                "document_id": None,
                "chunks": 0,
                "tokens": 0,
                "message": f"Unsupported file type: {suffix}",
            }

        # ── Store document row ─────────────────────────────────────────────
        document_id = _insert_document(
            bid_id=bid_id,
            filename=filename,
            file_hash=file_hash,
            file_size=file_size,
            doc_type=doc_type,
            addendum_number=addendum_number,
            pages_data=pages_data,
        )

        # ── Store chunks (without embeddings first) ────────────────────────
        chunk_ids = _insert_chunks(
            document_id=document_id,
            bid_id=bid_id,
            doc_type=doc_type,
            addendum_number=addendum_number,
            chunks=all_chunks,
        )

        # ── Generate & store embeddings ────────────────────────────────────
        # Run in executor so embedding doesn't block event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, _update_embeddings, chunk_ids, all_chunks
        )

        logger.info(
            f"✅ Indexed '{filename}': doc_id={document_id}, "
            f"{len(chunk_ids)} chunks, {total_tokens} tokens"
        )
        return {
            "status": "indexed",
            "document_id": document_id,
            "chunks": len(chunk_ids),
            "tokens": total_tokens,
            "message": "OK",
        }

    except Exception as e:
        logger.error(f"❌ Ingestion failed for '{filename}': {e}", exc_info=True)
        return {
            "status": "error",
            "document_id": None,
            "chunks": 0,
            "tokens": 0,
            "message": str(e),
        }


async def ingest_bid_folder(
    folder_path: str,
    bid_id: str | None = None,
    display_name: str | None = None,
    force_reindex: bool = False,
) -> list[dict[str, Any]]:
    """
    Ingest all supported files in a bid folder.
    Processes files concurrently where possible.

    Args:
        folder_path:   Path to the bid folder (e.g., './Bid1').
        bid_id:        Override bid ID; defaults to folder name.
        display_name:  Human-readable bid title.
        force_reindex: Re-index even if already present.

    Returns:
        List of per-file ingest results.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise NotADirectoryError(f"'{folder_path}' is not a directory")

    _bid_id = bid_id or folder.name
    supported = {".pdf", ".html", ".htm"}

    files = [f for f in sorted(folder.iterdir()) if f.suffix.lower() in supported]
    if not files:
        logger.warning(f"No supported files found in '{folder_path}'")
        return []

    logger.info(f"Ingesting {len(files)} files from '{folder_path}' → bid='{_bid_id}'")

    # Process all files concurrently
    tasks = [
        ingest_file(
            file_path=str(f),
            bid_id=_bid_id,
            display_name=display_name,
            force_reindex=force_reindex,
        )
        for f in files
    ]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    # Summary log
    indexed = sum(1 for r in results if r["status"] == "indexed")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors  = sum(1 for r in results if r["status"] == "error")
    logger.info(
        f"Bid '{_bid_id}' ingestion complete: "
        f"{indexed} indexed, {skipped} skipped, {errors} errors"
    )
    return list(results)
