"""
Synthesis Agent (prompts.md Section 4).

Combines the three sub-agent outputs into the final one-page memo. Does no
research of its own — the whole point of the traceability_check field in
its output is that every number in the memo must be traceable to one of the
three inputs.

If an upstream agent failed (status == "failed" after its own retry), its
output is not silently omitted — it's passed to Synthesis as an explicit
{"status": "failed", "reason": ...} object, so Synthesis's own rule 3
("explicitly name what is missing in a Data Gaps section") has something
concrete to name. This only runs at all when fewer than 2 of the 3 upstream
agents failed — orchestrator.py routes to failure_summary.py instead once
2+ have failed, since Synthesis working with two-thirds-missing input isn't
a synthesis job anymore.
"""

from __future__ import annotations

import json

from src import prompt_loader
from src.agent_runtime import AgentRunResult, run_json_agent
from src.schemas import QueryParserOutput, SynthesisOutput

AGENT_NAME = "synthesis"
MODEL = "claude-sonnet-5"

SynthesisResult = AgentRunResult


def _agent_output_json(result: AgentRunResult) -> str:
    if result.status == "ok" and result.output is not None:
        return json.dumps(result.output.model_dump(mode="json"))
    return json.dumps({"status": "failed", "reason": result.error or "unknown failure"})


async def run_synthesis(
    client,
    raw_user_input: str,
    parsed_query: QueryParserOutput,
    market_sizing_result: AgentRunResult,
    competitor_landscape_result: AgentRunResult,
    financial_feasibility_result: AgentRunResult,
    model: str = MODEL,
) -> AgentRunResult:
    system_prompt = prompt_loader.get_system_prompt(AGENT_NAME)
    user_prompt = prompt_loader.render_user_template(
        AGENT_NAME,
        raw_user_input=raw_user_input,
        parsed_query_json=json.dumps(parsed_query.model_dump(mode="json")),
        agent1_output_json=_agent_output_json(market_sizing_result),
        agent2_output_json=_agent_output_json(competitor_landscape_result),
        agent3_output_json=_agent_output_json(financial_feasibility_result),
    )
    return await run_json_agent(client, system_prompt, user_prompt, SynthesisOutput, model, use_search=False)
