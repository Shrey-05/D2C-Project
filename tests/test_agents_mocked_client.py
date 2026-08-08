"""
Exercises query_parser.py and market_sizing.py end-to-end against a mocked
Gemini client (google.genai.Client-shaped) — no real API key or network
needed. This is what actually proves the retry wiring and the search-
grounding path work, since test_validator.py and test_schemas.py only cover
the pieces they call directly.

The mock mimics just enough of the SDK's response shape (response.text,
response.candidates[0].grounding_metadata.web_search_queries) to drive
query_parser.py and market_sizing.py's real code paths, and
client.aio.models.generate_content as the call site both files use.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.agents.market_sizing import run_market_sizing
from src.agents.query_parser import run_query_parser


def gemini_response(text, search_queries=None):
    grounding_metadata = SimpleNamespace(web_search_queries=search_queries) if search_queries else None
    candidate = SimpleNamespace(grounding_metadata=grounding_metadata)
    return SimpleNamespace(text=text, candidates=[candidate])


def make_client(*responses):
    client = SimpleNamespace()
    generate_content = AsyncMock(side_effect=list(responses))
    client.aio = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    return client


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


@pytest.mark.asyncio
async def test_query_parser_happy_path():
    client = make_client(gemini_response(json.dumps(VALID_QP)))
    result = await run_query_parser(client, "Should we enter the US market?")
    assert result.status == "ok"
    assert result.output.industry == "specialty coffee wholesale"
    assert client.aio.models.generate_content.call_count == 1


@pytest.mark.asyncio
async def test_query_parser_recovers_on_retry():
    bad = gemini_response("not json at all, sorry")
    good = gemini_response(json.dumps(VALID_QP))
    client = make_client(bad, good)
    result = await run_query_parser(client, "Should we enter the US market?")
    assert result.status == "ok"
    assert len(result.raw_attempts) == 2
    assert client.aio.models.generate_content.call_count == 2


@pytest.mark.asyncio
async def test_query_parser_rejects_non_business_question():
    client = make_client(gemini_response(json.dumps({"error": "not_a_business_question"})))
    result = await run_query_parser(client, "lol hi how are you")
    assert result.status == "rejected"


@pytest.mark.asyncio
async def test_query_parser_fails_after_one_retry_capped():
    bad = gemini_response("garbage")
    client = make_client(bad, bad)
    result = await run_query_parser(client, "Should we enter the US market?")
    assert result.status == "failed"
    assert client.aio.models.generate_content.call_count == 2  # capped: never a third attempt


@pytest.mark.asyncio
async def test_market_sizing_happy_path_no_search_reported():
    client = make_client(gemini_response(json.dumps(VALID_MS)))
    result = await run_market_sizing(client, "specialty coffee", "USA", "cafes")
    assert result.status == "ok"
    assert result.output.TAM.value_usd == 5_000_000_000
    assert result.search_calls_made == 0


@pytest.mark.asyncio
async def test_market_sizing_reports_search_calls_from_grounding_metadata():
    resp = gemini_response(
        json.dumps(VALID_MS),
        search_queries=["US specialty coffee market size 2025", "TAM specialty coffee wholesale"],
    )
    client = make_client(resp)
    result = await run_market_sizing(client, "specialty coffee", "USA", "cafes")
    assert result.status == "ok"
    assert result.search_calls_made == 2

    # Confirm the Google Search grounding tool was actually attached to the call.
    sent_config = client.aio.models.generate_content.call_args.kwargs["config"]
    assert sent_config.tools is not None
    assert len(sent_config.tools) == 1
