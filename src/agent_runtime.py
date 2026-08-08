"""
Shared execution helper for every agent that: issues a system-prompted call
(optionally with the web_search tool), drives the search tool-use loop to
completion if used, validates the final JSON against a schema, and retries
once via the schema-validator/retry-correction prompt on failure.

query_parser.py does NOT use this — it has to try two different schemas
against the same raw output (QueryParserOutput or QueryParserError), which
is genuinely different validation logic, not a variant of this shape.
market_sizing.py, competitor_landscape.py, financial_feasibility.py, and
synthesis.py all follow this exact shape, so it's factored out here rather
than copy-pasted four times.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Type, TypeVar

from pydantic import BaseModel

from src import validator

T = TypeVar("T", bound=BaseModel)

WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}
MAX_TOOL_TURNS = 6  # guards against a runaway search loop; each turn is one API call


@dataclass
class AgentRunResult:
    status: str  # "ok" | "failed"
    output: Optional[BaseModel]
    raw_attempts: List[str] = field(default_factory=list)
    search_calls_made: int = 0
    error: Optional[str] = None


def _final_text(response) -> str:
    return "".join(block.text for block in response.content if block.type == "text")


async def _run_to_final_text(client, system_prompt: str, messages: list, model: str, use_search: bool):
    """Drive a conversation to a final non-tool-use text response.

    Returns (final_text, search_calls_made, messages_including_assistant_turns).
    Anthropic's web search tool is server-executed but still surfaces as a
    tool_use / tool_result pair in the message stream, so we loop turns
    until stop_reason isn't "tool_use" — same pattern as any client-side
    tool, even though search execution itself happens server-side.
    """
    search_calls = 0
    tools = [WEB_SEARCH_TOOL] if use_search else None

    for _ in range(MAX_TOOL_TURNS):
        kwargs = dict(model=model, max_tokens=2048, system=system_prompt, messages=messages)
        if tools:
            kwargs["tools"] = tools
        response = await client.messages.create(**kwargs)

        search_calls += sum(1 for b in response.content if getattr(b, "type", None) == "server_tool_use")

        if response.stop_reason != "tool_use":
            return _final_text(response), search_calls, messages

        messages = messages + [{"role": "assistant", "content": response.content}]

        has_client_tool_use = any(getattr(b, "type", None) == "tool_use" for b in response.content)
        if not has_client_tool_use:
            # Only server-side tool use happened; nothing for us to answer
            # with a tool_result. Continue the loop so the next turn can
            # keep searching or move on to its final answer.
            messages = messages + [{"role": "user", "content": "Continue."}]
            continue

        # Defensive: these agents only ever get a server tool. If a future
        # edit adds a client-side tool, don't hang the loop indefinitely.
        return _final_text(response), search_calls, messages

    return "", search_calls, messages


async def run_json_agent(
    client,
    system_prompt: str,
    user_prompt: str,
    schema_cls: Type[T],
    model: str,
    use_search: bool = False,
) -> AgentRunResult:
    messages = [{"role": "user", "content": user_prompt}]
    raw_attempts: List[str] = []

    raw_text, search_calls, messages = await _run_to_final_text(client, system_prompt, messages, model, use_search)
    raw_attempts.append(raw_text)

    ok, parsed, err = validator.validate_agent_output(raw_text, schema_cls)

    if not ok:
        retry_payload = validator.build_retry_payload(raw_text, err or "unknown validation error")
        retry_messages = messages + [{"role": "user", "content": retry_payload}]
        retry_kwargs = dict(model=model, max_tokens=2048, system=system_prompt, messages=retry_messages)
        if use_search:
            retry_kwargs["tools"] = [WEB_SEARCH_TOOL]
        retry_response = await client.messages.create(**retry_kwargs)
        raw_text_2 = _final_text(retry_response)
        raw_attempts.append(raw_text_2)
        ok, parsed, err = validator.validate_agent_output(raw_text_2, schema_cls)

    if not ok:
        return AgentRunResult(
            status="failed", output=None, raw_attempts=raw_attempts, search_calls_made=search_calls, error=err
        )

    return AgentRunResult(status="ok", output=parsed, raw_attempts=raw_attempts, search_calls_made=search_calls)
