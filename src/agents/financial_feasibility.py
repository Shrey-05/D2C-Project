"""
Financial Feasibility Agent (prompts.md Section 3).

The one non-parallel dependency in the pipeline: it consumes Market
Sizing's SOM value and confidence directly, per prompts.md's "confidence
inheritance" rule — if the input confidence is "low", this agent must
widen its own ranges and say so rather than presenting false precision; if
SOM is unavailable, it must null out its scenario values rather than
fabricate numbers.

Note the uploaded financial_feasibility.py built its user turn with
`som_value` / `som_confidence` placeholders — prompts.md's actual template
(Section 3) uses `{{agent1_som_value}}` / `{{agent1_confidence}}`. This
version matches prompts.md exactly, since prompt_loader.render_user_template
fails loudly on an unfilled placeholder rather than silently leaving one in.
"""

from __future__ import annotations

from typing import Optional

from src import prompt_loader
from src.agent_runtime import AgentRunResult, run_json_agent
from src.schemas import FinancialFeasibilityOutput

AGENT_NAME = "financial_feasibility"
MODEL = "gemini-2.5-flash"

FinancialFeasibilityResult = AgentRunResult


async def run_financial_feasibility(
    client,
    industry: str,
    geography: str,
    som_value_usd: Optional[float],
    som_confidence: str,  # "high" | "medium" | "low" | "unavailable"
    model: str = MODEL,
) -> AgentRunResult:
    system_prompt = prompt_loader.get_system_prompt(AGENT_NAME)
    # Explicit "unavailable" text rather than "None"/"null" as a raw string —
    # the model reads this as English, and the system prompt's own
    # instruction ("if SAM/SOM is null (unavailable)...") is keyed on that word.
    som_value_str = "unavailable" if som_value_usd is None else str(som_value_usd)
    user_prompt = prompt_loader.render_user_template(
        AGENT_NAME,
        industry=industry,
        geography=geography,
        agent1_som_value=som_value_str,
        agent1_confidence=som_confidence,
    )
    return await run_json_agent(
        client, system_prompt, user_prompt, FinancialFeasibilityOutput, model, use_search=True
    )
