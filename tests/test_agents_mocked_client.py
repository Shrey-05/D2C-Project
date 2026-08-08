"""
Exercises query_parser.py and market_sizing.py end-to-end against a mocked
AsyncAnthropic client — no real API key or network needed. This is what
actually proves the retry wiring and the search tool-use loop work, since
test_validator.py and test_schemas.py only cover the pieces they call.

The mock mimics just enough of the SDK's response shape (response.content
as a list of objects with .type / .text, response.stop_reason) to drive
query_parser.py and market_sizing.py's real code paths.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.agents.market_sizing import run_market_sizing
from src.agents.query_parser import run_query_parser


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


VALID_QP = {
    "industry": "specialty coffee wholesale",
    "geography": "United States",
    "target_customer": "independent specialty cafes",
    "decision_type": "market_entry",
    "ambiguities": [{"field": "target_customer", "assumption": "cafes not grocery"}],
}

VALID_MS = {
    "TAM": {"value_usd": 5_000_000_000, "tag": "sourced", "source": "Some Report 2025"},
    "SAM": {"value_usd": 500_000_000, "tag": "derived", "filter_logic": "specialty-only slice"},
    "SOM": {
        "value_usd": 10_000_000,
        "tag": "derived",
        "capture_rate_pct": 2.0,
        "rationale": "conservative new-entrant rate",
    },
    "method_used": "top-down",
    "sources": [{"name": "Some Report", "figure_cited": "$5B", "year": "2025"}],
    "key_assumptions": ["x"],
    "confidence": "medium",
}


def make_client(*responses):
    client = SimpleNamespace()
    client.messages = SimpleNamespace(create=AsyncMock(side_effect=list(responses)))
    return client


@pytest.mark.asyncio
async def test_query_parser_happy_path():
    resp = SimpleNamespace(content=[text_block(json.dumps(VALID_QP))])
    client = make_client(resp)
    result = await run_query_parser(client, "Should we enter the US market?")
    assert result.status == "ok"
    assert result.output.industry == "specialty coffee wholesale"
    assert client.messages.create.call_count == 1


@pytest.mark.asyncio
async def test_query_parser_recovers_on_retry():
    bad = SimpleNamespace(content=[text_block("not json at all, sorry")])
    good = SimpleNamespace(content=[text_block(json.dumps(VALID_QP))])
    client = make_client(bad, good)
    result = await run_query_parser(client, "Should we enter the US market?")
    assert result.status == "ok"
    assert len(result.raw_attempts) == 2
    assert client.messages.create.call_count == 2


@pytest.mark.asyncio
async def test_query_parser_rejects_non_business_question():
    resp = SimpleNamespace(content=[text_block(json.dumps({"error": "not_a_business_question"}))])
    client = make_client(resp)
    result = await run_query_parser(client, "lol hi how are you")
    assert result.status == "rejected"


@pytest.mark.asyncio
async def test_query_parser_fails_after_one_retry_capped():
    bad = SimpleNamespace(content=[text_block("garbage")])
    client = make_client(bad, bad)
    result = await run_query_parser(client, "Should we enter the US market?")
    assert result.status == "failed"
    assert client.messages.create.call_count == 2  # capped: never a third attempt


@pytest.mark.asyncio
async def test_market_sizing_happy_path_no_search():
    resp = SimpleNamespace(content=[text_block(json.dumps(VALID_MS))], stop_reason="end_turn")
    client = make_client(resp)
    result = await run_market_sizing(client, "specialty coffee", "USA", "cafes")
    assert result.status == "ok"
    assert result.output.TAM.value_usd == 5_000_000_000


@pytest.mark.asyncio
async def test_market_sizing_drives_search_tool_loop_to_completion():
    search_block = SimpleNamespace(type="server_tool_use", name="web_search")
    tool_turn = SimpleNamespace(content=[search_block], stop_reason="tool_use")
    final_turn = SimpleNamespace(content=[text_block(json.dumps(VALID_MS))], stop_reason="end_turn")
    client = make_client(tool_turn, final_turn)
    result = await run_market_sizing(client, "specialty coffee", "USA", "cafes")
    assert result.status == "ok"
    assert result.search_calls_made == 1
    assert client.messages.create.call_count == 2
