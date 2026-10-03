"""
Base LLM client — routes to OpenAI or Anthropic based on config.
All agents import get_llm_response() for chat completions.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from rfp_platform.core.config import get_settings
from rfp_platform.search.tools import call_tool

logger = logging.getLogger(__name__)


# ── Lazy OpenAI client ────────────────────────────────────────────────────────
from functools import lru_cache

@lru_cache(maxsize=1)
def _openai_client():
    from openai import OpenAI
    return OpenAI(api_key=get_settings().openai_api_key)


@lru_cache(maxsize=1)
def _anthropic_client():
    import anthropic
    return anthropic.Anthropic(api_key=get_settings().anthropic_api_key)


# ── Core LLM call with tool-use support ──────────────────────────────────────

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def get_llm_response(
    system_prompt: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    """
    Unified LLM call that supports tool/function calling.

    Returns:
        {
          "content":    str | None,
          "tool_calls": list[{name, arguments}] | None,
          "usage":      {prompt_tokens, completion_tokens},
        }
    """
    cfg = get_settings()
    start = time.time()

    if cfg.llm_provider == "openai":
        result = _call_openai(system_prompt, messages, tools, temperature, max_tokens)
    elif cfg.llm_provider == "anthropic":
        result = _call_anthropic(system_prompt, messages, tools, temperature, max_tokens)
    else:
        raise ValueError(f"Unknown LLM provider: {cfg.llm_provider}")

    elapsed = int((time.time() - start) * 1000)
    logger.debug(f"LLM call: {elapsed}ms, tokens={result.get('usage', {})}")
    return result


def _call_openai(system_prompt, messages, tools, temperature, max_tokens):
    cfg = get_settings()
    client = _openai_client()

    msgs = [{"role": "system", "content": system_prompt}] + messages

    kwargs = dict(
        model=cfg.openai_model,
        messages=msgs,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    resp = client.chat.completions.create(**kwargs)
    msg  = resp.choices[0].message

    tool_calls = None
    if msg.tool_calls:
        tool_calls = [
            {
                "id":        tc.id,
                "name":      tc.function.name,
                "arguments": json.loads(tc.function.arguments),
            }
            for tc in msg.tool_calls
        ]

    return {
        "content":    msg.content,
        "tool_calls": tool_calls,
        "usage": {
            "prompt_tokens":     resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
        },
    }


def _call_anthropic(system_prompt, messages, tools, temperature, max_tokens):
    cfg = get_settings()
    client = _anthropic_client()

    # Convert OpenAI tool format → Anthropic format
    ant_tools = None
    if tools:
        ant_tools = [
            {
                "name":        t["function"]["name"],
                "description": t["function"]["description"],
                "input_schema": t["function"]["parameters"],
            }
            for t in tools
        ]

    resp = client.messages.create(
        model=cfg.anthropic_model,
        system=system_prompt,
        messages=messages,
        tools=ant_tools or [],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    content_text = None
    tool_calls   = None

    for block in resp.content:
        if block.type == "text":
            content_text = block.text
        elif block.type == "tool_use":
            if tool_calls is None:
                tool_calls = []
            tool_calls.append({
                "id":        block.id,
                "name":      block.name,
                "arguments": block.input,
            })

    return {
        "content":    content_text,
        "tool_calls": tool_calls,
        "usage": {
            "prompt_tokens":     resp.usage.input_tokens,
            "completion_tokens": resp.usage.output_tokens,
        },
    }


# ── Agentic tool execution loop ───────────────────────────────────────────────

def run_agent_with_tools(
    system_prompt: str,
    messages: list[dict],
    tools: list[dict],
    max_iterations: int = 5,
) -> tuple[str, list[dict]]:
    """
    Run an LLM agent loop until it produces a final text response (no more tool calls).

    Returns:
        (final_text_response, updated_messages_list)
    """
    current_messages = list(messages)

    for iteration in range(max_iterations):
        response = get_llm_response(
            system_prompt=system_prompt,
            messages=current_messages,
            tools=tools,
        )

        tool_calls = response.get("tool_calls")
        content    = response.get("content", "")

        if not tool_calls:
            # Agent finished — return the text response
            return content or "", current_messages

        # Execute each tool call and append results
        current_messages.append({
            "role": "assistant",
            "content": content or "",
            "tool_calls": [
                {
                    "id":       tc["id"],
                    "type":     "function",
                    "function": {
                        "name":      tc["name"],
                        "arguments": json.dumps(tc["arguments"]),
                    },
                }
                for tc in tool_calls
            ],
        })

        for tc in tool_calls:
            try:
                result = call_tool(tc["name"], **tc["arguments"])
                result_str = json.dumps(result, default=str)
            except Exception as e:
                result_str = json.dumps({"error": str(e)})
                logger.error(f"Tool '{tc['name']}' failed: {e}")

            current_messages.append({
                "role":         "tool",
                "tool_call_id": tc["id"],
                "content":      result_str,
            })

        logger.debug(f"Agent iteration {iteration + 1}: ran {len(tool_calls)} tool(s)")

    # Max iterations hit — return last content
    logger.warning("Agent reached max_iterations without a final response")
    return content or "Max iterations reached", current_messages
