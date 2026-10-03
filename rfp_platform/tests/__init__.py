"""
Unit tests for the RFP Intelligence Platform.

Covers (as required by Part 9):
  - Parsing & ingestion (doc_type detection, chunker logic)
  - Search (hybrid_search, keyword_search, semantic_search, reranker)
  - Agent state (Pydantic models)
  - API endpoints (FastAPI TestClient)

Run all:
    python -m pytest rfp_platform/tests/ -v

Run just parsing tests (no DB required):
    python -m pytest rfp_platform/tests/test_ingestion.py -v

Run with coverage:
    python -m pytest rfp_platform/tests/ --cov=rfp_platform --cov-report=term-missing
"""
from __future__ import annotations

import pytest
import sys
import os

# Make sure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
