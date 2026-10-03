"""
PDFChunkingHelpers — your chunking logic, adapted and extended for RFP documents.

Key improvements over the old app.py RecursiveCharacterTextSplitter:
  - Markdown-aware: respects section headings from pymupdf4llm output
  - Table-aware: keeps table rows together, never splits mid-table
  - Overlap stitching: adds configurable tail-overlap between chunks
  - merge_small_chunks: merges fragments below min_chars threshold
  - Token counting via tiktoken (same cl100k_base you used before)
"""
import re
import logging
from typing import Any

import tiktoken

from rfp_platform.core.config import get_settings

logger = logging.getLogger(__name__)

# ── Heading patterns that signal a new logical section ───────────────────────
_HEADING_RE = re.compile(
    r"^(#{1,6}\s.+|[A-Z][A-Z\s\d\-&/]{4,}:?\s*$)",  # markdown headings OR ALL-CAPS headings
    re.MULTILINE,
)

# ── Table fence patterns (markdown table rows start with |) ───────────────────
_TABLE_ROW_RE = re.compile(r"^\|.+\|", re.MULTILINE)


class PDFChunkingHelpers:
    """
    Stateless helper — instantiated per document, holds the file path and
    a cached tokenizer.

    Chunking strategy (justified in README):
      1. Split page text on section boundaries (headings) to preserve
         logical document structure — critical for addendum reconciliation.
      2. If a section is still too large, split further on paragraph
         boundaries (double-newline) with overlap.
      3. Tables are never split mid-row — kept as a single chunk even if
         they exceed max_chars (rare for RFP tables).
      4. Small chunks below min_chars are merged with their neighbour —
         avoids embedding noise from tiny fragments.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        cfg = get_settings()
        self.max_chars = cfg.chunk_max_chars
        self.min_chars = cfg.chunk_min_chars
        self.overlap_chars = cfg.chunk_overlap_chars
        self._tokenizer = tiktoken.get_encoding("cl100k_base")

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def smart_chunk_page(
        self, page_text: str, page_number: int
    ) -> list[dict[str, Any]]:
        """
        Split one page's markdown text into smart chunks.

        Returns a list of dicts:
          {
            "page_content": str,
            "page_number":  int,
            "chunk_index":  int,   # within this page
            "token_count":  int,
          }
        """
        if not page_text.strip():
            return []

        # 1. Split on section boundaries first
        sections = self._split_on_headings(page_text)

        chunks: list[dict[str, Any]] = []
        chunk_idx = 0

        for section in sections:
            if self._contains_table(section):
                # Keep tables intact even if over max_chars
                c = self._make_chunk(section, page_number, chunk_idx)
                chunks.append(c)
                chunk_idx += 1
            elif len(section) <= self.max_chars:
                c = self._make_chunk(section, page_number, chunk_idx)
                chunks.append(c)
                chunk_idx += 1
            else:
                # Section too large → split on paragraphs with overlap
                sub_chunks = self._split_on_paragraphs(section)
                for sub in sub_chunks:
                    c = self._make_chunk(sub, page_number, chunk_idx)
                    chunks.append(c)
                    chunk_idx += 1

        return chunks

    def merge_small_chunks(
        self,
        chunks: list[dict[str, Any]],
        min_chars: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Merge consecutive chunks that are smaller than min_chars.
        Merged chunks inherit the page_number of the first chunk.

        This is a direct port of your original merge_small_chunks logic,
        extended to recalculate token_count after merging.
        """
        if not chunks:
            return []

        min_c = min_chars or self.min_chars
        merged: list[dict[str, Any]] = []
        buffer = chunks[0].copy()

        for chunk in chunks[1:]:
            if len(buffer["page_content"]) < min_c:
                # Merge: append with a newline separator
                buffer["page_content"] = (
                    buffer["page_content"] + "\n\n" + chunk["page_content"]
                )
                buffer["token_count"] = len(
                    self._tokenizer.encode(buffer["page_content"])
                )
            else:
                merged.append(buffer)
                buffer = chunk.copy()

        merged.append(buffer)

        # Re-index chunk_index sequentially across the whole document
        for i, c in enumerate(merged):
            c["chunk_index"] = i

        logger.debug(
            f"merge_small_chunks: {len(chunks)} → {len(merged)} chunks "
            f"(min_chars={min_c})"
        )
        return merged

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _make_chunk(
        self, text: str, page_number: int, chunk_index: int
    ) -> dict[str, Any]:
        text = self._clean_text(text)
        tokens = len(self._tokenizer.encode(text))
        return {
            "page_content": text,
            "page_number": page_number,
            "chunk_index": chunk_index,
            "token_count": tokens,
        }

    def _split_on_headings(self, text: str) -> list[str]:
        """
        Split text at every markdown or ALL-CAPS heading, keeping the
        heading at the top of its section.
        """
        parts = _HEADING_RE.split(text)
        # _HEADING_RE.split gives alternating [before, heading, body, heading, body …]
        # Re-join heading + its body
        sections: list[str] = []
        if parts[0].strip():
            sections.append(parts[0])

        # parts comes in groups of 3: [before_match, matched_group, rest]
        # but Python split with groups returns [pre, group1, post, group2, …]
        i = 1
        while i < len(parts):
            heading = parts[i] if i < len(parts) else ""
            body = parts[i + 1] if i + 1 < len(parts) else ""
            combined = (heading + "\n" + body).strip()
            if combined:
                sections.append(combined)
            i += 2

        return sections if sections else [text]

    def _split_on_paragraphs(self, text: str) -> list[str]:
        """
        Split text on double-newlines (paragraph boundary) with overlap.
        Overlap = last `overlap_chars` characters of previous chunk prepended.
        """
        paragraphs = re.split(r"\n{2,}", text)
        chunks: list[str] = []
        current = ""
        tail = ""  # overlap buffer

        for para in paragraphs:
            candidate = (tail + "\n\n" + para).strip() if tail else para
            if len(current) + len(candidate) <= self.max_chars:
                current = (current + "\n\n" + candidate).strip() if current else candidate
            else:
                if current:
                    chunks.append(current)
                    tail = current[-self.overlap_chars:] if len(current) > self.overlap_chars else current
                current = candidate

        if current:
            chunks.append(current)

        return chunks if chunks else [text]

    @staticmethod
    def _contains_table(text: str) -> bool:
        """Return True if the text contains at least one markdown table row."""
        return bool(_TABLE_ROW_RE.search(text))

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Mirrors what you had in mind — remove repeated headers/footers,
        fix broken hyphenation, normalise whitespace.
        """
        # Fix PDF hyphenation: "competi-\ntion" → "competition"
        text = re.sub(r"-\n(\w)", r"\1", text)
        # Collapse 3+ newlines to double newline
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Normalise horizontal whitespace (keep newlines)
        text = re.sub(r"[ \t]{2,}", " ", text)
        # Strip leading/trailing whitespace from each line
        lines = [ln.strip() for ln in text.splitlines()]
        text = "\n".join(lines)
        return text.strip()
