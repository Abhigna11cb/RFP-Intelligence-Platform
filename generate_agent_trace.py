"""
generate_agent_trace.py
-----------------------
Runs the full extraction pipeline for Bid1 and captures every agent step
(node name, tool calls, LLM input/output, latency) into agent_trace_example.json.

Usage:
    python generate_agent_trace.py
"""
from __future__ import annotations
import json
import sys
import time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rfp_platform.agents.orchestrator import run_extraction
from rfp_platform.core.database import get_conn
from psycopg2.extras import RealDictCursor

print("Running extraction for Bid1 to capture agent trace...")
t0 = time.time()
result = run_extraction("Bid1")
elapsed = round(time.time() - t0, 2)
print(f"Extraction complete in {elapsed}s")

# Fetch the agent_logs rows written during this run
run_id = result.get("run_id")
agent_steps = []
if run_id:
    try:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT node_name, step_index, input_summary, output_summary,
                           tool_calls, llm_tokens, latency_ms, created_at
                    FROM agent_logs
                    WHERE run_id = %s
                    ORDER BY step_index
                    """,
                    (run_id,),
                )
                agent_steps = [dict(r) for r in cur.fetchall()]
    except Exception as e:
        print(f"Could not fetch agent_logs (may not exist): {e}")

trace = {
    "bid_id": "Bid1",
    "run_id": run_id,
    "total_elapsed_seconds": elapsed,
    "extraction_summary": {
        "fields_extracted": len(result.get("fields", {})),
        "addendum_changes": len(result.get("addendum_changes", [])),
        "validation": result.get("validation", {}),
    },
    "agent_steps": agent_steps if agent_steps else [
        {
            "note": "agent_logs table not available — showing extraction result summary only",
            "fields": {
                k: {"value": v.get("value"), "confidence": v.get("confidence")}
                for k, v in result.get("fields", {}).items()
            },
            "addendum_changes": result.get("addendum_changes", []),
        }
    ],
    "full_result": result,
}

with open("agent_trace_example.json", "w", encoding="utf-8") as f:
    json.dump(trace, f, indent=2, ensure_ascii=False, default=str)

print("agent_trace_example.json written!")
