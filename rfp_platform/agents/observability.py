"""
Observability — structured logging of every agent step to the agent_logs table.
Satisfies Part C 7.2 requirement: "log each agent step (input, tool calls, output, tokens, latency)".

Usage inside any agent node:
    from rfp_platform.agents.observability import log_step
    log_step(run_id=state.run_id, agent_name="extraction",
              step="extract_fields", input={...}, output={...},
              tokens_used=1200, latency_ms=840)
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)


def log_step(
    run_id: str,
    agent_name: str,
    step: str,
    bid_id: str | None = None,
    input_data: Any = None,
    output_data: Any = None,
    tool_calls: list[dict] | None = None,
    tokens_used: int = 0,
    latency_ms: int = 0,
    status: str = "success",
    error_msg: str | None = None,
) -> None:
    """
    Persist one agent step to the agent_logs table.
    Silently swallows DB errors so a logging failure never crashes an agent.
    """
    from rfp_platform.core.database import get_conn

    def _safe_json(obj: Any) -> str:
        try:
            return json.dumps(obj, default=str)
        except Exception:
            return json.dumps({"error": "unserializable"})

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO agent_logs
                        (run_id, bid_id, agent_name, step,
                         input, output, tool_calls,
                         tokens_used, latency_ms, status, error_msg)
                    VALUES
                        (%s::uuid, %s, %s, %s,
                         %s::jsonb, %s::jsonb, %s::jsonb,
                         %s, %s, %s, %s)
                    """,
                    (
                        run_id,
                        bid_id,
                        agent_name,
                        step,
                        _safe_json(input_data),
                        _safe_json(output_data),
                        _safe_json(tool_calls or []),
                        tokens_used,
                        latency_ms,
                        status,
                        error_msg,
                    ),
                )
    except Exception as e:
        logger.debug(f"agent_logs write failed (non-fatal): {e}")


@contextmanager
def timed_step(
    run_id: str,
    agent_name: str,
    step: str,
    bid_id: str | None = None,
    input_data: Any = None,
):
    """
    Context manager that automatically logs start + end of an agent step
    with latency measurement.

    Usage:
        with timed_step(run_id, "extraction", "extract_due_date", bid_id) as ctx:
            result = do_work()
            ctx["output"] = result
            ctx["tokens"] = 500
    """
    ctx: dict[str, Any] = {"output": None, "tokens": 0, "tool_calls": [], "status": "success", "error": None}
    start = time.time()
    try:
        yield ctx
    except Exception as e:
        ctx["status"] = "error"
        ctx["error"] = str(e)
        raise
    finally:
        elapsed = int((time.time() - start) * 1000)
        log_step(
            run_id=run_id,
            agent_name=agent_name,
            step=step,
            bid_id=bid_id,
            input_data=input_data,
            output_data=ctx.get("output"),
            tool_calls=ctx.get("tool_calls"),
            tokens_used=ctx.get("tokens", 0),
            latency_ms=elapsed,
            status=ctx.get("status", "success"),
            error_msg=ctx.get("error"),
        )


def get_run_trace(run_id: str) -> list[dict]:
    """
    Retrieve the full agent trace for a run_id from agent_logs.
    Use this to produce the 'one full example trace' required by the assignment.
    """
    from rfp_platform.core.database import get_conn
    from psycopg2.extras import RealDictCursor

    with get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT agent_name, step, bid_id, tokens_used, latency_ms,
                       status, error_msg, created_at,
                       output, tool_calls
                FROM agent_logs
                WHERE run_id = %s::uuid
                ORDER BY created_at
                """,
                (run_id,),
            )
            return [dict(r) for r in cur.fetchall()]
