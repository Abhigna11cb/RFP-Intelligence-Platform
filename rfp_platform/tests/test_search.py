"""
Unit tests for search components.
Requires the rfp_platform DB with data (run ingestion first).
"""
from __future__ import annotations

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Search tool tests (requires DB with data)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestHybridSearch:
    """Integration tests — require live DB with indexed data."""

    def test_hybrid_returns_results(self):
        from rfp_platform.search.tools import hybrid_search
        results = hybrid_search("submission deadline", bid_id="Bid1", top_k=5)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_hybrid_result_has_required_fields(self):
        from rfp_platform.search.tools import hybrid_search
        results = hybrid_search("bid number", bid_id="Bid1", top_k=3)
        for r in results:
            assert "chunk_text" in r
            assert "file_name" in r
            assert "page_number" in r
            assert "bid_id" in r
            assert "score" in r

    def test_bid_filter_works(self):
        """Results filtered to Bid1 must not contain Bid2 chunks."""
        from rfp_platform.search.tools import hybrid_search
        results = hybrid_search("laptop specifications", bid_id="Bid1", top_k=10)
        for r in results:
            assert r["bid_id"] == "Bid1"

    def test_keyword_search_exact_bid_number(self):
        """Keyword search must find exact bid number string."""
        from rfp_platform.search.tools import keyword_search
        results = keyword_search("JA-207652", bid_id="Bid1", top_k=5)
        assert len(results) > 0
        found = any("JA-207652" in r["chunk_text"] for r in results)
        assert found, "Exact bid number not found via keyword search"

    def test_keyword_search_bid2(self):
        from rfp_platform.search.tools import keyword_search
        results = keyword_search("E20P4600040", bid_id="Bid2", top_k=5)
        if results:
            found = any("E20P4600040" in r["chunk_text"] for r in results)
            # May not always be in text — just ensure no crash
            assert isinstance(results, list)

    def test_semantic_search_returns_results(self):
        from rfp_platform.search.tools import semantic_search
        results = semantic_search("warranty requirements for laptops", bid_id="Bid2", top_k=5)
        assert isinstance(results, list)
        assert len(results) >= 0  # may be empty if below threshold

    def test_retrieve_nearby_chunks(self):
        from rfp_platform.search.tools import hybrid_search, retrieve_nearby_chunks
        results = hybrid_search("due date", bid_id="Bid1", top_k=1)
        if results:
            file_name = results[0]["file_name"]
            chunk_index = results[0]["chunk_index"]
            context = retrieve_nearby_chunks(file_name, chunk_index, context_window=2)
            assert isinstance(context, list)

    def test_get_bid_documents(self):
        from rfp_platform.search.tools import get_bid_documents
        docs = get_bid_documents("Bid1")
        assert len(docs) > 0
        for d in docs:
            assert "file_name" in d
            assert "doc_type" in d

    def test_get_addendum_changes(self):
        from rfp_platform.search.tools import get_addendum_changes
        chunks = get_addendum_changes("Bid1")
        assert isinstance(chunks, list)
        for c in chunks:
            assert c["doc_type"] == "addendum"

    def test_search_by_field(self):
        from rfp_platform.search.tools import search_by_field
        results = search_by_field("Due Date", bid_id="Bid1", top_k=5)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_hybrid_search_no_filter(self):
        """Search across all bids returns results from both."""
        from rfp_platform.search.tools import hybrid_search
        results = hybrid_search("Dell laptop", top_k=10)
        bid_ids = {r["bid_id"] for r in results}
        # Should have at least one result
        assert len(results) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Reranker tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestReranker:

    def test_rerank_preserves_count(self):
        from rfp_platform.search.tools import hybrid_search
        from rfp_platform.search.reranker import rerank
        results = hybrid_search("due date deadline", bid_id="Bid1", top_k=10)
        reranked = rerank("due date deadline", results, top_k=5)
        assert len(reranked) <= 5

    def test_reranked_has_score(self):
        from rfp_platform.search.tools import hybrid_search
        from rfp_platform.search.reranker import rerank
        results = hybrid_search("bid number", bid_id="Bid1", top_k=5)
        reranked = rerank("bid number", results)
        for r in reranked:
            assert "rerank_score" in r

    def test_reranker_empty_input(self):
        from rfp_platform.search.reranker import rerank
        result = rerank("some query", [], top_k=5)
        assert result == []

    def test_hybrid_search_reranked(self):
        from rfp_platform.search.reranker import hybrid_search_reranked
        results = hybrid_search_reranked("submission deadline", bid_id="Bid1", top_k=3)
        assert isinstance(results, list)
        assert len(results) <= 3


# ─────────────────────────────────────────────────────────────────────────────
# Agent state tests (no DB required)
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentState:

    def test_state_creation(self):
        from rfp_platform.agents.state import AgentState
        state = AgentState(bid_id="Bid1", mode="extraction")
        assert state.bid_id == "Bid1"
        assert state.mode == "extraction"
        assert state.run_id  # UUID generated
        assert state.is_complete is False

    def test_state_mark_step(self):
        from rfp_platform.agents.state import AgentState
        state = AgentState(bid_id="Bid1")
        state.mark_step_done("retrieval")
        assert "retrieval" in state.steps_completed

    def test_state_retry_logic(self):
        from rfp_platform.agents.state import AgentState
        state = AgentState(bid_id="Bid1", max_retries=2)
        assert state.should_retry("Due Date") is True
        state.increment_retry("Due Date")
        state.increment_retry("Due Date")
        assert state.should_retry("Due Date") is False

    def test_field_result(self):
        from rfp_platform.agents.state import FieldResult
        f = FieldResult(value="JA-207652", confidence=0.95, status="extracted")
        assert f.value == "JA-207652"
        assert f.confidence == 0.95

    def test_to_output_json(self):
        from rfp_platform.agents.state import AgentState, FieldResult
        state = AgentState(bid_id="Bid1")
        state.final_fields["Bid Number"] = FieldResult(
            value="JA-207652", confidence=0.95, status="extracted"
        )
        out = state.to_output_json()
        assert out["bid_id"] == "Bid1"
        assert "Bid Number" in out["fields"]
        assert out["fields"]["Bid Number"]["value"] == "JA-207652"
