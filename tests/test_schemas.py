from pydantic import ValidationError

from src.schemas import (
    AGENT_SCHEMAS,
    MarketSizingOutput,
    QueryParserError,
    QueryParserOutput,
)


def test_query_parser_output_parses_clean_json():
    data = {
        "industry": "specialty coffee wholesale",
        "geography": "United States",
        "target_customer": "independent specialty cafes",
        "decision_type": "market_entry",
        "ambiguities": [{"field": "target_customer", "assumption": "cafes, not grocery"}],
    }
    parsed = QueryParserOutput.model_validate(data)
    assert parsed.industry == "specialty coffee wholesale"
    assert parsed.decision_type.value == "market_entry"
    assert len(parsed.ambiguities) == 1


def test_query_parser_output_rejects_bad_decision_type():
    data = {
        "industry": "x",
        "geography": "y",
        "target_customer": "z",
        "decision_type": "not_a_real_type",
        "ambiguities": [],
    }
    try:
        QueryParserOutput.model_validate(data)
        assert False, "should have raised"
    except ValidationError:
        pass


def test_query_parser_error_shape():
    parsed = QueryParserError.model_validate({"error": "not_a_business_question"})
    assert parsed.error == "not_a_business_question"


def test_market_sizing_output_parses_clean_json():
    data = {
        "TAM": {"value_usd": 5_000_000_000, "tag": "sourced", "source": "Some Report 2025"},
        "SAM": {"value_usd": 500_000_000, "tag": "derived", "filter_logic": "specialty-only slice"},
        "SOM": {
            "value_usd": 10_000_000,
            "tag": "derived",
            "capture_rate_pct": 2.0,
            "rationale": "conservative new-entrant capture rate",
        },
        "method_used": "top-down from industry report, filtered to specialty segment",
        "sources": [{"name": "Some Report", "figure_cited": "$5B", "year": "2025"}],
        "key_assumptions": ["specialty segment is 10% of total market"],
        "confidence": "medium",
    }
    parsed = MarketSizingOutput.model_validate(data)
    assert parsed.TAM.value_usd == 5_000_000_000
    assert parsed.confidence.value == "medium"


def test_market_sizing_output_allows_null_values_with_unavailable_tag():
    data = {
        "TAM": {"value_usd": None, "tag": "unavailable", "source": None},
        "SAM": {"value_usd": None, "tag": "unavailable", "filter_logic": "n/a — no usable source found"},
        "SOM": {
            "value_usd": None,
            "tag": "unavailable",
            "capture_rate_pct": None,
            "rationale": "no source found after 2 searches",
        },
        "method_used": "attempted top-down, no usable source retrieved",
        "sources": [],
        "key_assumptions": ["searched X and Y, both returned no usable figures"],
        "confidence": "low",
    }
    parsed = MarketSizingOutput.model_validate(data)
    assert parsed.TAM.value_usd is None
    assert parsed.confidence.value == "low"


def test_agent_schemas_registry_has_all_five_agents():
    assert set(AGENT_SCHEMAS.keys()) == {
        "query_parser",
        "market_sizing",
        "competitor_landscape",
        "financial_feasibility",
        "synthesis",
    }
