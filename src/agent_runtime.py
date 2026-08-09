"""
Shared execution helper for every agent that: issues a Gemini
generate_content call (optionally with the Google Search grounding tool),
validates the final JSON against a schema, and retries once via the
schema-validator/retry-correction prompt on failure.

Ported from an Anthropic Claude version. One genuine simplification fell
out of the port, not just a mechanical swap: Claude's web_search tool
needed a multi-turn tool-use loop (model calls the tool, server executes it,
model gets the result back, model continues) that agent_runtime.py used to
drive by hand. Gemini's Google Search grounding is fully server-managed
within a single generate_content call — the model just returns final text
with grounding_metadata attached describing what it searched. No client-
visible tool_use/tool_result turns to loop over.

One real constraint this port has to respect: Gemini does not support
combining a `tools` list with `response_schema`-based structured output in
the same call. So, same as the Claude version, JSON-shaped output here is
still enforced by prompt instruction + validator.py's extraction/retry path,
not by the API's native structured-output feature — that's not a limitation
introduced by this port, it's inherent to using search grounding at all.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import List, Optional, Type, TypeVar

from google.genai import errors, types
from pydantic import BaseModel

from src import validator

T = TypeVar("T", bound=BaseModel)

GOOGLE_SEARCH_TOOL = types.Tool(google_search=types.GoogleSearch())

# Rate-limit retry is deliberately separate from validator.py's retry: that
# one exists because the *model's own output* was malformed and gets a
# corrective prompt; this one exists because the *request itself* was
# rejected by infrastructure (429 RESOURCE_EXHAUSTED) before the model ever
# saw it. Same request, unchanged, just resent after a short wait — nothing
# here should ever touch prompt content.
RATE_LIMIT_MAX_RETRIES = 3
RATE_LIMIT_BASE_DELAY_SECONDS = 5.0


async def generate_with_backoff(client, model: str, contents, config):
    """client.aio.models.generate_content, retried with exponential backoff
    specifically on 429 (RESOURCE_EXHAUSTED). Any other error — a bad model
    name, an invalid key, a 500 — is raised immediately on the first
    attempt, since retrying those just wastes 3x the wait for the same
    guaranteed failure.
    """
    last_error: Optional[errors.APIError] = None
    for attempt in range(RATE_LIMIT_MAX_RETRIES + 1):
        try:
            return await client.aio.models.generate_content(model=model, contents=contents, config=config)
        except errors.APIError as e:
            is_rate_limit = getattr(e, "code", None) == 429
            if not is_rate_limit or attempt == RATE_LIMIT_MAX_RETRIES:
                raise
            last_error = e
            await asyncio.sleep(RATE_LIMIT_BASE_DELAY_SECONDS * (2**attempt))
    raise last_error  # unreachable in practice; satisfies type checkers


@dataclass
class AgentRunResult:
    status: str  # "ok" | "failed"
    output: Optional[BaseModel]
    raw_attempts: List[str] = field(default_factory=list)
    search_calls_made: int = 0
    error: Optional[str] = None


def _search_call_count(response) -> int:
    """Gemini doesn't expose a literal call count; approximate it as the
    number of distinct search queries grounding_metadata reports the model
    actually issued. 0 if search wasn't used or nothing came back.
    """
    try:
        metadata = response.candidates[0].grounding_metadata
        if metadata and metadata.web_search_queries:
            return len(metadata.web_search_queries)
    except (AttributeError, IndexError, TypeError):
        pass
    return 0


async def _generate(client, model: str, contents, system_prompt: str, use_search: bool):
    config_kwargs = {"system_instruction": system_prompt}
    if use_search:
        config_kwargs["tools"] = [GOOGLE_SEARCH_TOOL]
    config = types.GenerateContentConfig(**config_kwargs)
    return await generate_with_backoff(client, model, contents, config)


def _retry_contents(user_prompt: str, previous_raw_text: str, retry_payload: str) -> list:
    """Gemini's `contents` is a flat, role-tagged conversation list, unlike
    Claude's separate system/messages split reused turn-by-turn. Rebuild the
    short conversation explicitly for the correction turn.
    """
    return [
        {"role": "user", "parts": [{"text": user_prompt}]},
        {"role": "model", "parts": [{"text": previous_raw_text}]},
        {"role": "user", "parts": [{"text": retry_payload}]},
    ]


async def run_json_agent(
    client,
    system_prompt: str,
    user_prompt: str,
    schema_cls: Type[T],
    model: str,
    use_search: bool = False,
) -> AgentRunResult:
    raw_attempts: List[str] = []

    response = await _generate(client, model, user_prompt, system_prompt, use_search)
    raw_text = response.text or ""
    raw_attempts.append(raw_text)
    search_calls = _search_call_count(response)

    ok, parsed, err = validator.validate_agent_output(raw_text, schema_cls)

    if not ok:
        retry_payload = validator.build_retry_payload(raw_text, err or "unknown validation error")
        retry_response = await _generate(
            client, model, _retry_contents(user_prompt, raw_text, retry_payload), system_prompt, use_search
        )
        raw_text_2 = retry_response.text or ""
        raw_attempts.append(raw_text_2)
        search_calls += _search_call_count(retry_response)
        ok, parsed, err = validator.validate_agent_output(raw_text_2, schema_cls)

    if not ok:
        return AgentRunResult(
            status="failed", output=None, raw_attempts=raw_attempts, search_calls_made=search_calls, error=err
        )

    return AgentRunResult(status="ok", output=parsed, raw_attempts=raw_attempts, search_calls_made=search_calls)
