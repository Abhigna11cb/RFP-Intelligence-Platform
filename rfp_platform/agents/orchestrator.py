"""
LangGraph orchestrator — wires all agent nodes into a directed graph.

Graph topology:
                    ┌──────────────┐
                    │ orchestrator │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   retrieval  │
                    └──────┬───────┘
                           │
               ┌───────────┼───────────┐
          (extraction)            (qa)
               │                      │
       ┌───────▼────────┐      ┌──────▼──────┐
       │   extraction   │      │     qa      │
       └───────┬────────┘      └──────┬──────┘
               │                     │
      ┌────────▼────────┐           END
      │    addendum     │
      └────────┬────────┘
               │
      ┌────────▼────────┐
      │    validator    │
      └────────┬────────┘
               │
    ┌──────────┴──────────┐
  (retry)            (finalize)
    │                    │
    └──► retrieval   ┌───▼───┐
                     │  qa   │ (finalize)
                     └───┬───┘
                         │
                        END
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from langgraph.graph import StateGraph, END

from rfp_platform.agents.state import AgentState
from rfp_platform.agents.nodes import (
    orchestrator_node,
    retrieval_node,
    extraction_node,
    addendum_node,
    validator_node,
    qa_node,
    should_retry,
    route_after_orchestrator,
    route_after_retrieval,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph wrapper (AgentState → dict for LangGraph compatibility)
# ─────────────────────────────────────────────────────────────────────────────

def _wrap(node_fn):
    """Wrap a node function so it accepts and returns full state dicts.

    LangGraph's StateGraph(dict) REPLACES the entire state with whatever the
    node returns — so every node must return the COMPLETE state, not just changed keys.
    """
    def wrapper(state_dict: dict) -> dict:
        state = AgentState(**state_dict)
        result = node_fn(state)

        # Start from the full current state (JSON-safe)
        updated = state.model_dump(mode="json")

        # Merge node updates
        for k, v in result.items():
            if hasattr(v, "model_dump"):
                updated[k] = v.model_dump(mode="json")
            elif isinstance(v, dict):
                # May contain FieldResult objects (draft_fields / final_fields)
                updated[k] = {
                    dk: dv.model_dump(mode="json") if hasattr(dv, "model_dump") else dv
                    for dk, dv in v.items()
                }
            elif isinstance(v, list):
                # May contain Pydantic models (addendum_changes, etc.)
                updated[k] = [
                    item.model_dump(mode="json") if hasattr(item, "model_dump") else item
                    for item in v
                ]
            else:
                updated[k] = v

        return updated
    return wrapper



def _wrap_router(router_fn):
    def wrapper(state_dict: dict) -> str:
        state = AgentState(**state_dict)
        return router_fn(state)
    return wrapper


# ─────────────────────────────────────────────────────────────────────────────
# Build the graph
# ─────────────────────────────────────────────────────────────────────────────

def build_graph():
    """
    Construct and compile the LangGraph agent pipeline.
    Returns a compiled graph ready to invoke.
    """
    graph = StateGraph(dict)

    # Add nodes
    graph.add_node("orchestrator", _wrap(orchestrator_node))
    graph.add_node("retrieval",    _wrap(retrieval_node))
    graph.add_node("extraction",   _wrap(extraction_node))
    graph.add_node("addendum",     _wrap(addendum_node))
    graph.add_node("validator",    _wrap(validator_node))
    graph.add_node("qa",           _wrap(qa_node))

    # Entry point
    graph.set_entry_point("orchestrator")

    # Fixed edges
    graph.add_edge("orchestrator", "retrieval")
    graph.add_edge("extraction",   "addendum")
    graph.add_edge("addendum",     "validator")

    # Conditional: after retrieval → extraction or qa
    graph.add_conditional_edges(
        "retrieval",
        _wrap_router(route_after_retrieval),
        {"extraction": "extraction", "qa": "qa"},
    )

    # Conditional: after validation → retry (retrieval) or finalize (qa node)
    graph.add_conditional_edges(
        "validator",
        _wrap_router(should_retry),
        {"retry": "retrieval", "finalize": "qa"},
    )

    # Terminal edges
    graph.add_edge("qa", END)

    return graph.compile()


# Singleton compiled graph (reset if you change _wrap or nodes)
_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_extraction(bid_id: str) -> dict[str, Any]:
    """
    Run the full extraction pipeline for a bid.
    Returns the final JSON with all 20 fields + addendum changes + validation.

    Args:
        bid_id: e.g. 'Bid1', 'Bid2'

    Returns:
        Final output dict (see AgentState.to_output_json())
    """
    logger.info(f"Starting extraction pipeline for bid='{bid_id}'")
    run_id = str(uuid.uuid4())

    initial_state = AgentState(
        run_id=run_id,
        bid_id=bid_id,
        mode="extraction",
    ).model_dump()

    graph  = get_graph()
    result = graph.invoke(initial_state)

    final_state = AgentState(**result)
    output = final_state.to_output_json()

    # Persist to DB
    _save_extraction(bid_id, output)
    logger.info(f"Extraction complete for bid='{bid_id}' run_id={run_id}")
    return output


def run_qa(question: str, bid_id: str | None = None) -> dict[str, Any]:
    """
    Answer a free-form question about one or all bids.

    Args:
        question: Natural language question.
        bid_id:   Optional — restrict to a specific bid.

    Returns:
        {"answer": str, "bid_id": str|None, "run_id": str}
    """
    logger.info(f"Starting Q&A pipeline: '{question[:60]}'")
    run_id = str(uuid.uuid4())

    initial_state = AgentState(
        run_id=run_id,
        bid_id=bid_id or "",
        mode="qa",
        user_query=question,
    ).model_dump()

    graph  = get_graph()
    result = graph.invoke(initial_state)

    final_state = AgentState(**result)
    return {
        "answer": final_state.final_answer,
        "bid_id": bid_id,
        "run_id": run_id,
    }


def _save_extraction(bid_id: str, output: dict) -> None:
    """Persist extraction results to bid_extractions table."""
    import json
    from rfp_platform.core.database import get_conn
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO bid_extractions (bid_id, fields, addendum_changes, validation)
                    VALUES (%s, %s::jsonb, %s::jsonb, %s::jsonb)
                    ON CONFLICT (bid_id) DO UPDATE
                        SET fields           = EXCLUDED.fields,
                            addendum_changes = EXCLUDED.addendum_changes,
                            validation       = EXCLUDED.validation,
                            updated_at       = NOW();
                    """,
                    (
                        bid_id,
                        json.dumps(output.get("fields", {})),
                        json.dumps(output.get("addendum_changes", [])),
                        json.dumps(output.get("validation", {})),
                    ),
                )
        logger.info(f"Saved extraction results for bid='{bid_id}'")
    except Exception as e:
        logger.error(f"Failed to save extraction: {e}")
