"""
Market Sizing Agent (prompts.md Section 1).

Grounded in Anthropic's native web search tool per README_architecture.md
Section 2. Thin wrapper around agent_runtime.run_json_agent now — the
tool-loop/retry logic that used to live here directly moved to
src/agent_runtime.py so competitor_landscape.py and financial_feasibility.py
(which need the exact same shape) don't duplicate it.
"""

from __future__ import annotations

from src import prompt_loader
from src.agent_runtime import AgentRunResult, run_json_agent
from src.schemas import MarketSizingOutput

AGENT_NAME = "market_sizing"
MODEL = "gemini-2.5-flash"

# Kept as an alias: orchestrator.py and existing tests reference this name.
MarketSizingResult = AgentRunResult


async def run_market_sizing(
    client, industry: str, geography: str, target_customer: str, model: str = MODEL
) -> AgentRunResult:
    system_prompt = prompt_loader.get_system_prompt(AGENT_NAME)
    user_prompt = prompt_loader.render_user_template(
        AGENT_NAME, industry=industry, geography=geography, target_customer=target_customer
    )
    return await run_json_agent(client, system_prompt, user_prompt, MarketSizingOutput, model, use_search=True)
