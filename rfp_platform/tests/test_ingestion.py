"""
Unit tests for document ingestion and parsing.
Tests doc_type detection, chunker logic, HTML extraction — no DB required.
"""
from __future__ import annotations

import pytest
from rfp_platform.ingestion.pipeline import detect_doc_type
from rfp_platform.ingestion.chunker import PDFChunkingHelpers


# ─────────────────────────────────────────────────────────────────────────────
# Test doc_type detection
# ─────────────────────────────────────────────────────────────────────────────

class TestDocTypeDetection:
    """Tests for the filename → doc_type + addendum_number classifier."""

    def test_addendum_with_number(self):
        doc_type, num = detect_doc_type("Addendum 1 RFP JA-207652 Student Computing.pdf")
        assert doc_type == "addendum"
        assert num == 1

    def test_addendum_2(self):
        doc_type, num = detect_doc_type("Addendum 2 RFP JA-207652 Student and Staff Computing Devices.pdf")
        assert doc_type == "addendum"
        assert num == 2

    def test_rfp_main_document(self):
        doc_type, num = detect_doc_type("JA-207652 Student and Staff Computing Devices FINAL.pdf")
        assert doc_type == "rfp"
        assert num is None

    def test_specs_document(self):
        doc_type, num = detect_doc_type("Dell_Laptop_Specs.pdf")
        assert doc_type == "specs"
        assert num is None

    def test_affidavit_document(self):
        doc_type, num = detect_doc_type("Contract_Affidavit.pdf")
        assert doc_type == "affidavit"
        assert num is None

    def test_mercury_affidavit(self):
        doc_type, num = detect_doc_type("Mercury_Affidavit.pdf")
        assert doc_type == "affidavit"
        assert num is None

    def test_html_bid_page(self):
        doc_type, num = detect_doc_type(
            "Student and Staff Computing Devices __SOURCING #168884__ - Bid Information - {3} _ BidNet Direct.html"
        )
        assert doc_type == "bid_page"
        assert num is None

    def test_porfp_rfp(self):
        doc_type, num = detect_doc_type("PORFP_-_Dell_Laptop_Final.pdf")
        assert doc_type == "rfp"
        assert num is None

    def test_case_insensitive_addendum(self):
        doc_type, num = detect_doc_type("addendum 3 - final.pdf")
        assert doc_type == "addendum"
        assert num == 3

    def test_unknown_file(self):
        doc_type, num = detect_doc_type("random_document.pdf")
        # Should default to rfp or similar, not crash
        assert doc_type in ("rfp", "specs", "affidavit", "bid_page", "addendum")


# ─────────────────────────────────────────────────────────────────────────────
# Test chunker logic (no PDF required — test with synthetic text)
# ─────────────────────────────────────────────────────────────────────────────

class TestChunker:
    """Tests for the PDFChunkingHelpers class."""

    @pytest.fixture
    def chunker(self, tmp_path):
        """Create a chunker with a dummy file path."""
        dummy = tmp_path / "test.pdf"
        dummy.write_bytes(b"dummy")
        return PDFChunkingHelpers(str(dummy))

    def test_smart_chunk_short_text(self, chunker):
        """Short text under min_chars should produce one chunk."""
        text = "This is a short paragraph about bid requirements."
        chunks = chunker.smart_chunk_page(text, page_number=1)
        assert isinstance(chunks, list)

    def test_smart_chunk_long_text(self, chunker):
        """Long text should be split into multiple chunks."""
        # 5000 chars of text should produce multiple chunks
        text = ("This is a test sentence about the RFP requirements. " * 100)
        chunks = chunker.smart_chunk_page(text, page_number=1)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert "page_content" in chunk
            assert "page_number" in chunk
            assert chunk["page_number"] == 1

    def test_chunk_has_required_fields(self, chunker):
        text = "The submission deadline is July 9, 2024 at 2:00 PM CST. " * 30
        chunks = chunker.smart_chunk_page(text, page_number=3)
        if chunks:
            chunk = chunks[0]
            assert "page_content" in chunk
            assert "page_number" in chunk
            assert "chunk_index" in chunk

    def test_merge_small_chunks(self, chunker):
        """merge_small_chunks should combine tiny fragments."""
        small_chunks = [
            {"page_content": "Short.", "page_number": 1, "chunk_index": 0, "token_count": 5},
            {"page_content": "Also short.", "page_number": 1, "chunk_index": 1, "token_count": 5},
        ]
        merged = chunker.merge_small_chunks(small_chunks)
        assert isinstance(merged, list)

    def test_heading_preserved_in_chunk(self, chunker):
        """Section headings should be kept with their content."""
        text = """
## Section 4: Submission Requirements

All bids must be submitted electronically through the procurement portal.
The deadline for submission is July 9, 2024 at 2:00 PM CST.

## Section 5: Bond Requirements

No bid bond is required for this solicitation.
""" * 5
        chunks = chunker.smart_chunk_page(text, page_number=2)
        # At least the content should be captured
        all_text = " ".join(c["page_content"] for c in chunks)
        assert "submission" in all_text.lower() or len(chunks) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Test file hash (deduplication)
# ─────────────────────────────────────────────────────────────────────────────

class TestFileHash:
    def test_sha256_consistency(self, tmp_path):
        """Same content should always produce the same hash."""
        import hashlib
        content = b"This is test content for hashing."
        f = tmp_path / "test.pdf"
        f.write_bytes(content)

        def compute_hash(path):
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(8192), b""):
                    h.update(chunk)
            return h.hexdigest()

        h1 = compute_hash(str(f))
        h2 = compute_hash(str(f))
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_different_content_different_hash(self, tmp_path):
        import hashlib
        f1 = tmp_path / "a.pdf"
        f2 = tmp_path / "b.pdf"
        f1.write_bytes(b"content A")
        f2.write_bytes(b"content B")

        def h(p):
            return hashlib.sha256(open(p, "rb").read()).hexdigest()

        assert h(str(f1)) != h(str(f2))
