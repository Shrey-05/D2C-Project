"""
End-to-end orchestration tests against a single mocked Gemini client
(client.aio.models.generate_content) shared across all five agents in one
run. Verifies:

1. The happy path reaches synthesis and produces a final memo.
2. Market Sizing's SOM value/confidence actually flow into Financial
   Feasibility's prompt (the one non-parallel dependency in the system).
3. When 2+ of {market_sizing, competitor_landscape, financial_feasibility}
   fail even after their own retry, the orchestrator skips synthesis and
   calls failure_summary instead.
4. A rejected (non-business) input stops the chain at query_parser and
   never calls any other agent.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.orchestrator import ConsultingOrchestrator


def gemini_response(text):
    candidate = SimpleNamespace(grounding_metadata=None)
    return SimpleNamespace(text=text, candidates=[candidate])


VALID_QP = {
    "industry": "specialty coffee wholesale",
    "geography": "United States",
    "target_customer": "independent specialty cafes",
    "decision_type": "market_entry",
    "ambiguities": [],
}

VALID_MS = {
    "TAM": {"value_usd": 5_000_000_000, "tag": "sourced", "source": "Some Report 2025"},
    "SAM": {"value_usd": 500_000_000, "tag": "derived", "filter_logic": "specialty-only slice"},
    "SOM": {"value_usd": 10_000_000, "tag": "derived", "capture_rate_pct": 2.0, "rationale": "x"},
    "method_used": "top-down",
    "sources": [],
    "key_assumptions": [],
    "confidence": "medium",
}

VALID_CL = {
    "competitors": [
        {"name": "Roast & Co", "positioning": "premium", "pricing_tier": "premium", "differentiator": "x", "source": "y"}
    ],
    "search_coverage_note": "adequate coverage",
    "white_space_hypothesis": {"text": "gap in mid-tier", "is_inference": True},
    "confidence": "medium",
}

VALID_FF = {
    "inherited_som_confidence": "medium",
    "cac_estimate_range_usd": [50.0, 120.0],
    "gross_margin_range_pct": [30.0, 45.0],
    "scenarios": [
        {"name": "conservative", "key_assumptions": ["x"], "breakeven_estimate": "24-30 months", "precision_note": "n/a"},
        {"name": "base", "key_assumptions": ["x"], "breakeven_estimate": "18-24 months", "precision_note": "n/a"},
        {"name": "aggressive", "key_assumptions": ["x"], "breakeven_estimate": "12-18 months", "precision_note": "n/a"},
    ],
    "recommendation_lean": "favorable",
    "confidence": "medium",
}

VALID_SYNTHESIS = {
    "memo_markdown": "# Memo\n\nRecommend proceeding.",
    "overall_confidence": "medium",
    "recommendation": "Recommend proceeding",
    "traceability_check": {"all_figures_sourced_from_inputs": True, "notes": "ok"},
}


def make_sequenced_client(mapping):
    """mapping: dict of agent_name -> response.

    Since market_sizing and competitor_landscape run concurrently via
    asyncio.gather, their call order relative to each other isn't
    guaranteed. Route by inspecting config.system_instruction instead of
    relying on call order — anchored on each agent's own opening sentence
    in prompts.md, which is unique to that agent (synthesis's own prompt,
    for example, mentions "market sizing" and "financial feasibility" as
    plain words, so a loose substring match on those alone would misroute
    it — full opening phrases avoid that collision).
    """
    client = SimpleNamespace()

    async def _generate_content(*, model, contents, config):
        system = (config.system_instruction or "").lower()
        if "you are a query parser" in system:
            return mapping["query_parser"]
        if "you are a market-sizing analyst" in system:
            return mapping["market_sizing"]
        if "you are a competitive-landscape analyst" in system:
            return mapping["competitor_landscape"]
        if "you are a financial feasibility analyst" in system:
            return mapping["financial_feasibility"]
        if "you are a senior analyst producing" in system:
            return mapping["synthesis"]
        if "you are generating a short, honest status message" in system:
            return mapping["failure_summary"]
        raise AssertionError(f"unrouted system_instruction: {system[:200]!r}")

    generate_content = AsyncMock(side_effect=_generate_content)
    client.aio = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    return client


@pytest.mark.asyncio
async def test_full_pipeline_happy_path_reaches_synthesis():
    client = make_sequenced_client(
        {
            "query_parser": gemini_response(json.dumps(VALID_QP)),
            "market_sizing": gemini_response(json.dumps(VALID_MS)),
            "competitor_landscape": gemini_response(json.dumps(VALID_CL)),
            "financial_feasibility": gemini_response(json.dumps(VALID_FF)),
            "synthesis": gemini_response(json.dumps(VALID_SYNTHESIS)),
            "failure_summary": gemini_response("should not be called"),
        }
    )

    events = []
    orchestrator = ConsultingOrchestrator(client, progress_callback=lambda s, st: events.append((s, st)))
    result = await orchestrator.run("Should we enter the US wholesale coffee market?")

    assert result.query_parser.status == "ok"
    assert result.market_sizing.status == "ok"
    assert result.competitor_landscape.status == "ok"
    assert result.financial_feasibility.status == "ok"
    assert result.synthesis.status == "ok"
    assert result.final_memo == "# Memo\n\nRecommend proceeding."
    assert result.recommendation == "Recommend proceeding"
    assert result.overall_confidence == "medium"
    assert result.failure_summary_text is None

    assert ("synthesis", "running") in events
    assert ("failure_summary", "running") not in events


@pytest.mark.asyncio
async def test_financial_feasibility_receives_market_sizings_som():
    client = make_sequenced_client(
        {
            "query_parser": gemini_response(json.dumps(VALID_QP)),
            "market_sizing": gemini_response(json.dumps(VALID_MS)),
            "competitor_landscape": gemini_response(json.dumps(VALID_CL)),
            "financial_feasibility": gemini_response(json.dumps(VALID_FF)),
            "synthesis": gemini_response(json.dumps(VALID_SYNTHESIS)),
            "failure_summary": gemini_response("n/a"),
        }
    )
    orchestrator = ConsultingOrchestrator(client)
    await orchestrator.run("Should we enter the US wholesale coffee market?")

    ff_call = next(
        c for c in client.aio.models.generate_content.call_args_list
        if "financial feasibility" in (c.kwargs["config"].system_instruction or "").lower()
    )
    sent_text = ff_call.kwargs["contents"]
    assert "10000000" in sent_text  # VALID_MS SOM value_usd
    assert "medium" in sent_text  # VALID_MS confidence


@pytest.mark.asyncio
async def test_two_failures_triggers_failure_summary_not_synthesis():
    # market_sizing and competitor_landscape both fail twice (initial + retry);
    # financial_feasibility succeeds -> failed_count == 2 -> failure_summary path.
    bad = gemini_response("not valid json at all")

    client = make_sequenced_client(
        {
            "query_parser": gemini_response(json.dumps(VALID_QP)),
            "market_sizing": bad,
            "competitor_landscape": bad,
            "financial_feasibility": gemini_response(json.dumps(VALID_FF)),
            "synthesis": gemini_response(json.dumps(VALID_SYNTHESIS)),
            "failure_summary": gemini_response(
                "Market sizing and competitor analysis could not be completed due to "
                "malformed output. Financial feasibility completed. Try a narrower query."
            ),
        }
    )

    events = []
    orchestrator = ConsultingOrchestrator(client, progress_callback=lambda s, st: events.append((s, st)))
    result = await orchestrator.run("Should we enter the US wholesale coffee market?")

    assert result.market_sizing.status == "failed"
    assert result.competitor_landscape.status == "failed"
    assert result.financial_feasibility.status == "ok"
    assert result.synthesis is None
    assert result.final_memo is None
    assert result.failure_summary_text is not None
    assert "malformed output" in result.failure_summary_text

    assert ("synthesis", "skipped") in events
    assert ("failure_summary", "running") in events
    assert not any(s == "synthesis" and st == "running" for s, st in events)


@pytest.mark.asyncio
async def test_rejected_input_stops_before_any_other_agent():
    client = make_sequenced_client(
        {
            "query_parser": gemini_response(json.dumps({"error": "not_a_business_question"})),
            "market_sizing": gemini_response("should not be called"),
            "competitor_landscape": gemini_response("should not be called"),
            "financial_feasibility": gemini_response("should not be called"),
            "synthesis": gemini_response("should not be called"),
            "failure_summary": gemini_response("should not be called"),
        }
    )
    orchestrator = ConsultingOrchestrator(client)
    result = await orchestrator.run("lol hi how are you")

    assert result.stopped_reason == "input was not a business question"
    assert result.market_sizing is None
    assert result.competitor_landscape is None
    assert result.financial_feasibility is None
    assert result.synthesis is None
    assert client.aio.models.generate_content.call_count == 1  # only query_parser was ever called
