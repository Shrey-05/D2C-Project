"""
Failure Summary meta-prompt (prompts.md Section 6).

Only invoked when 2+ of {market_sizing, competitor_landscape,
financial_feasibility} failed even after their own single retry —
orchestrator.py's job, not this module's, to decide when that threshold is
hit. Outputs plain text, not JSON: there's no schema to validate here, and
per prompts.md's own design note this is meant to be a short, direct,
user-facing status message, not another structured artifact.

One wrinkle prompts.md's template has: the three status lines all use the
*same* placeholder name, {{status_and_reason}}, repeated three times —

    - Market Sizing: {{status_and_reason}}
    - Competitor Landscape: {{status_and_reason}}
    - Financial Feasibility: {{status_and_reason}}

prompt_loader.render_text() fills a named placeholder with one value
everywhere it appears, which can't express three different values under one
name. So this module fills that specific template positionally instead of
going through render_text — first occurrence gets the first value passed in,
second gets the second, third gets the third, in the fixed order the
template lists them (Market Sizing, Competitor Landscape, Financial
Feasibility).
"""

from __future__ import annotations

from google.genai import types

from src import prompt_loader
from src.agent_runtime import generate_with_backoff

AGENT_NAME = "failure_summary"
MODEL = "gemini-flash-latest"

_PLACEHOLDER = "{{status_and_reason}}"


def _fill_positional(template: str, values: list) -> str:
    parts = template.split(_PLACEHOLDER)
    if len(parts) - 1 != len(values):
        raise ValueError(
            f"expected {len(values)} occurrences of {_PLACEHOLDER!r} in the "
            f"failure_summary template, found {len(parts) - 1} — prompts.md "
            f"Section 6 may have changed shape"
        )
    out = parts[0]
    for part, value in zip(parts[1:], values):
        out += value + part
    return out


def status_reason(result) -> str:
    """Render one agent's AgentRunResult as the short status string the
    template expects, e.g. "ok" or "failed: <validation error>"."""
    if result.status == "ok":
        return "ok"
    return f"failed: {result.error or 'unknown error'}"


async def run_failure_summary(
    client,
    market_sizing_status: str,
    competitor_landscape_status: str,
    financial_feasibility_status: str,
    model: str = MODEL,
) -> str:
    system_prompt = prompt_loader.get_system_prompt(AGENT_NAME)
    raw_template = prompt_loader.get_user_template(AGENT_NAME)
    user_prompt = _fill_positional(
        raw_template,
        [market_sizing_status, competitor_landscape_status, financial_feasibility_status],
    )

    config = types.GenerateContentConfig(system_instruction=system_prompt)
    response = await generate_with_backoff(client, model, user_prompt, config)
    return (response.text or "").strip()
