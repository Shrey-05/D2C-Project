"""
Competitor Landscape Agent (prompts.md Section 2).

The uploaded competitor_landscape.py used Build #1's `run_agent()` helper
(offline/fixture_key, synchronous) — incompatible with Build #2's async
AsyncAnthropic-client design. This is the equivalent built against Build #2's
actual runtime: same job (identify 4-6 real, search-grounded competitors),
same input shape (industry, geography), rebuilt on agent_runtime.run_json_agent
so it shares the exact tool-loop and single-retry logic market_sizing.py uses.

Per prompts.md's own design note, this is deliberately the agent most likely
to legitimately return fewer than 4 competitors with an honest
search_coverage_note on a thin-coverage query — that's correct behavior, not
a bug, and orchestrator.py must not treat "confidence: low" here as a
pipeline failure.
"""

from __future__ import annotations

from src import prompt_loader
from src.agent_runtime import AgentRunResult, run_json_agent
from src.schemas import CompetitorLandscapeOutput

AGENT_NAME = "competitor_landscape"
MODEL = "gemini-flash-latest"

CompetitorLandscapeResult = AgentRunResult


async def run_competitor_landscape(client, industry: str, geography: str, model: str = MODEL) -> AgentRunResult:
    system_prompt = prompt_loader.get_system_prompt(AGENT_NAME)
    user_prompt = prompt_loader.render_user_template(AGENT_NAME, industry=industry, geography=geography)
    return await run_json_agent(
        client, system_prompt, user_prompt, CompetitorLandscapeOutput, model, use_search=True
    )
