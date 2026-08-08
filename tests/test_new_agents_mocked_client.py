"""
Mocked-client tests for competitor_landscape.py, financial_feasibility.py,
synthesis.py, and failure_summary.py — same style as
test_agents_mocked_client.py: no real API key or network needed.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.agent_runtime import AgentRunResult
from src.agents.competitor_landscape import run_competitor_landscape
from src.agents.failure_summary import _fill_positional, run_failure_summary, status_reason
from src.agents.financial_feasibility import run_financial_feasibility
from src.agents.synthesis import run_synthesis
from src.schemas import ConfidenceLevel, MarketSizingOutput, QueryParserOutput, ValueTag


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def make_client(*responses):
    client = SimpleNamespace()
    client.messages = SimpleNamespace(create=AsyncMock(side_effect=list(responses)))
    return client


VALID_CL = {
    "competitors": [
        {
            "name": "Roast & Co",
            "positioning": "premium single-origin",
            "pricing_tier": "premium",
            "differentiator": "direct-trade sourcing story",
            "source": "search result 1",
        }
    ],
    "search_coverage_note": "found 1 clearly matching competitor across 2 searches",
    "white_space_hypothesis": {"text": "no strong mid-tier player in this niche", "is_inference": True},
    "confidence": "low",
}

VALID_FF = {
    "inherited_som_confidence": "medium",
    "cac_estimate_range_usd": [50.0, 120.0],
    "gross_margin_range_pct": [30.0, 45.0],
    "scenarios": [
        {
            "name": "conservative",
            "key_assumptions": ["slow account acquisition"],
            "breakeven_estimate": "24-30 months",
            "precision_note": "widened range due to inherited medium confidence",
        },
        {
            "name": "base",
            "key_assumptions": ["moderate acquisition"],
            "breakeven_estimate": "18-24 months",
            "precision_note": "n/a",
        },
        {
            "name": "aggressive",
            "key_assumptions": ["fast acquisition"],
            "breakeven_estimate": "12-18 months",
            "precision_note": "n/a",
        },
    ],
    "recommendation_lean": "mixed",
    "confidence": "medium",
}

VALID_SYNTHESIS = {
    "memo_markdown": "# Memo\n\nExecutive summary...",
    "overall_confidence": "medium",
    "recommendation": "Recommend proceeding with caveats",
    "traceability_check": {"all_figures_sourced_from_inputs": True, "notes": "all numbers traced"},
}


# ---- competitor_landscape ----------------------------------------------


@pytest.mark.asyncio
async def test_competitor_landscape_happy_path():
    resp = SimpleNamespace(content=[text_block(json.dumps(VALID_CL))], stop_reason="end_turn")
    client = make_client(resp)
    result = await run_competitor_landscape(client, "specialty coffee", "USA")
    assert result.status == "ok"
    assert result.output.confidence.value == "low"
    assert len(result.output.competitors) == 1


@pytest.mark.asyncio
async def test_competitor_landscape_retries_on_bad_json():
    bad = SimpleNamespace(content=[text_block("not json")], stop_reason="end_turn")
    good = SimpleNamespace(content=[text_block(json.dumps(VALID_CL))], stop_reason="end_turn")
    client = make_client(bad, good)
    result = await run_competitor_landscape(client, "specialty coffee", "USA")
    assert result.status == "ok"
    assert client.messages.create.call_count == 2


# ---- financial_feasibility ----------------------------------------------


@pytest.mark.asyncio
async def test_financial_feasibility_happy_path_with_som():
    resp = SimpleNamespace(content=[text_block(json.dumps(VALID_FF))], stop_reason="end_turn")
    client = make_client(resp)
    result = await run_financial_feasibility(client, "specialty coffee", "USA", 10_000_000.0, "medium")
    assert result.status == "ok"
    assert len(result.output.scenarios) == 3
    assert result.output.recommendation_lean.value == "mixed"

    # Confirm the SOM value/confidence actually made it into the prompt sent.
    sent_messages = client.messages.create.call_args.kwargs["messages"]
    sent_user_text = sent_messages[0]["content"]
    assert "10000000.0" in sent_user_text
    assert "medium" in sent_user_text


@pytest.mark.asyncio
async def test_financial_feasibility_passes_unavailable_when_som_is_none():
    resp = SimpleNamespace(content=[text_block(json.dumps(VALID_FF))], stop_reason="end_turn")
    client = make_client(resp)
    await run_financial_feasibility(client, "specialty coffee", "USA", None, "unavailable")
    sent_messages = client.messages.create.call_args.kwargs["messages"]
    sent_user_text = sent_messages[0]["content"]
    assert "unavailable" in sent_user_text


# ---- synthesis ------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesis_happy_path_all_inputs_ok():
    resp = SimpleNamespace(content=[text_block(json.dumps(VALID_SYNTHESIS))], stop_reason="end_turn")
    client = make_client(resp)

    parsed_query = QueryParserOutput(
        industry="specialty coffee wholesale",
        geography="USA",
        target_customer="independent cafes",
        decision_type="market_entry",
        ambiguities=[],
    )
    ms = AgentRunResult(status="ok", output=MarketSizingOutput.model_validate({
        "TAM": {"value_usd": 1.0, "tag": "sourced", "source": "x"},
        "SAM": {"value_usd": 1.0, "tag": "derived", "filter_logic": "x"},
        "SOM": {"value_usd": 1.0, "tag": "derived", "capture_rate_pct": 1.0, "rationale": "x"},
        "method_used": "x",
        "sources": [],
        "key_assumptions": [],
        "confidence": "medium",
    }))
    cl = AgentRunResult(status="failed", output=None, error="validation failed twice")
    ff = AgentRunResult(status="ok", output=None)  # status ok but output None shouldn't happen in practice; use real below

    result = await run_synthesis(client, "Should we enter the US market?", parsed_query, ms, cl, ff)
    assert result.status == "ok"
    assert result.output.recommendation.value == "Recommend proceeding with caveats"

    # The failed competitor_landscape result must be visible to the model as
    # a data gap, not silently dropped.
    sent_messages = client.messages.create.call_args.kwargs["messages"]
    sent_user_text = sent_messages[0]["content"]
    assert '"status": "failed"' in sent_user_text
    assert "validation failed twice" in sent_user_text


# ---- failure_summary --------------------------------------------------


def test_fill_positional_fills_in_order():
    template = "MS: {{status_and_reason}}\nCL: {{status_and_reason}}\nFF: {{status_and_reason}}"
    out = _fill_positional(template, ["ok", "failed: x", "failed: y"])
    assert out == "MS: ok\nCL: failed: x\nFF: failed: y"


def test_fill_positional_raises_on_count_mismatch():
    template = "MS: {{status_and_reason}}\nCL: {{status_and_reason}}"
    with pytest.raises(ValueError):
        _fill_positional(template, ["ok", "failed", "failed"])


def test_status_reason_ok():
    result = AgentRunResult(status="ok", output=object())
    assert status_reason(result) == "ok"


def test_status_reason_failed():
    result = AgentRunResult(status="failed", output=None, error="schema mismatch")
    assert status_reason(result) == "failed: schema mismatch"


@pytest.mark.asyncio
async def test_run_failure_summary_sends_three_distinct_statuses():
    resp = SimpleNamespace(content=[text_block("Market sizing and competitor analysis completed. Financial feasibility could not be completed after two attempts due to malformed output. Try narrowing the geography.")])
    client = make_client(resp)
    text = await run_failure_summary(client, "ok", "ok", "failed: malformed output twice")
    assert "narrowing" in text.lower() or "completed" in text.lower()

    sent_messages = client.messages.create.call_args.kwargs["messages"]
    sent_user_text = sent_messages[0]["content"]
    assert sent_user_text.count("ok") >= 2
    assert "failed: malformed output twice" in sent_user_text
