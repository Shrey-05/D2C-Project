"""
Query Parser Agent (prompts.md Section 0).

Deliberately the "dumbest" agent in the pipeline: extraction only, zero
analysis. It has two possible valid output shapes, not one:

- QueryParserOutput, the normal case.
- QueryParserError ({"error": "not_a_business_question"}), when the input
  isn't a business decision question at all.

Both are attempted during validation before falling back to the retry path.

This agent talks to the Gemini API directly (rather than through
agent_runtime.run_json_agent) because the dual-schema validation above is
genuinely different logic from every other agent's single-schema case —
it's not a variant of the shared shape, so it isn't forced into it. Uses no
search tool, same as before.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from google.genai import types

from src import prompt_loader, validator
from src.agent_runtime import generate_with_backoff
from src.schemas import QueryParserError, QueryParserOutput

AGENT_NAME = "query_parser"
MODEL = "gemini-flash-latest"


@dataclass
class QueryParserResult:
    status: str  # "ok" | "rejected" | "failed"
    output: Optional[Union[QueryParserOutput, QueryParserError]]
    raw_attempts: list  # raw text of every attempt, successful or not, for tracing
    error: Optional[str] = None


def _validate_either(raw_text: str):
    """Try QueryParserOutput first, then QueryParserError — the prompt can
    legitimately return either shape, and pydantic validation of the wrong
    one will fail even on well-formed JSON, so we can't just call
    validator.validate_agent_output once with a single schema.
    """
    ok, parsed, err = validator.validate_agent_output(raw_text, QueryParserOutput)
    if ok:
        return True, parsed, None
    ok_err, parsed_err, err_err = validator.validate_agent_output(raw_text, QueryParserError)
    if ok_err:
        return True, parsed_err, None
    # Neither shape matched — report the QueryParserOutput error, since
    # that's the far more common intended shape and the more useful message.
    return False, None, err


async def run_query_parser(client, raw_user_input: str, model: str = MODEL) -> QueryParserResult:
    system_prompt = prompt_loader.get_system_prompt(AGENT_NAME)
    user_prompt = prompt_loader.render_user_template(AGENT_NAME, raw_user_input=raw_user_input)

    raw_attempts = []
    config = types.GenerateContentConfig(system_instruction=system_prompt)

    response = await generate_with_backoff(client, model, user_prompt, config)
    raw_text = response.text or ""
    raw_attempts.append(raw_text)

    ok, parsed, err = _validate_either(raw_text)

    if not ok:
        # One retry: same short conversation, corrective user turn appended.
        retry_payload = validator.build_retry_payload(raw_text, err or "unknown validation error")
        retry_contents = [
            {"role": "user", "parts": [{"text": user_prompt}]},
            {"role": "model", "parts": [{"text": raw_text}]},
            {"role": "user", "parts": [{"text": retry_payload}]},
        ]
        retry_response = await generate_with_backoff(client, model, retry_contents, config)
        raw_text_2 = retry_response.text or ""
        raw_attempts.append(raw_text_2)
        ok, parsed, err = _validate_either(raw_text_2)

    if not ok:
        return QueryParserResult(status="failed", output=None, raw_attempts=raw_attempts, error=err)

    if isinstance(parsed, QueryParserError):
        return QueryParserResult(status="rejected", output=parsed, raw_attempts=raw_attempts)

    return QueryParserResult(status="ok", output=parsed, raw_attempts=raw_attempts)
