import json

import pytest

from src import validator
from src.schemas import QueryParserOutput


VALID_QP_JSON = {
    "industry": "specialty coffee wholesale",
    "geography": "United States",
    "target_customer": "independent specialty cafes",
    "decision_type": "market_entry",
    "ambiguities": [],
}


# ---- extract_json ----------------------------------------------------


def test_extract_json_plain():
    text = json.dumps(VALID_QP_JSON)
    assert json.loads(validator.extract_json(text)) == VALID_QP_JSON


def test_extract_json_labelled_fence():
    text = f"Here you go:\n```json\n{json.dumps(VALID_QP_JSON)}\n```"
    assert json.loads(validator.extract_json(text)) == VALID_QP_JSON


def test_extract_json_unlabelled_fence():
    text = f"```\n{json.dumps(VALID_QP_JSON)}\n```"
    assert json.loads(validator.extract_json(text)) == VALID_QP_JSON


def test_extract_json_leading_prose_no_fence():
    text = f"Sure, here is the JSON: {json.dumps(VALID_QP_JSON)}"
    assert json.loads(validator.extract_json(text)) == VALID_QP_JSON


def test_extract_json_brace_inside_string_not_mistaken_for_close():
    data = {**VALID_QP_JSON, "target_customer": "cafes (segment: {niche})"}
    text = json.dumps(data)
    assert json.loads(validator.extract_json(text)) == data


def test_extract_json_empty_raises():
    with pytest.raises(validator.JSONExtractionError):
        validator.extract_json("")


def test_extract_json_no_json_raises():
    with pytest.raises(validator.JSONExtractionError):
        validator.extract_json("sorry, I can't help with that")


def test_extract_json_unbalanced_raises():
    with pytest.raises(validator.JSONExtractionError):
        validator.extract_json('{"industry": "coffee"')


# ---- validate_agent_output --------------------------------------------


def test_validate_agent_output_success():
    ok, parsed, err = validator.validate_agent_output(json.dumps(VALID_QP_JSON), QueryParserOutput)
    assert ok is True
    assert isinstance(parsed, QueryParserOutput)
    assert err is None


def test_validate_agent_output_bad_schema_reports_error():
    bad = {**VALID_QP_JSON, "decision_type": "not_a_real_enum_value"}
    ok, parsed, err = validator.validate_agent_output(json.dumps(bad), QueryParserOutput)
    assert ok is False
    assert parsed is None
    assert err is not None and "decision_type" in err


def test_validate_agent_output_malformed_json_reports_error():
    ok, parsed, err = validator.validate_agent_output("not json at all", QueryParserOutput)
    assert ok is False
    assert parsed is None
    assert err is not None


# ---- build_retry_payload ------------------------------------------------


def test_build_retry_payload_fills_both_placeholders():
    payload = validator.build_retry_payload("{bad json", "Expecting value: line 1 column 1")
    assert "{bad json" in payload
    assert "Expecting value: line 1 column 1" in payload
    assert "{{" not in payload
