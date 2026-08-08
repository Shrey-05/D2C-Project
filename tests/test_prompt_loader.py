import textwrap

import pytest

from src import prompt_loader


def test_parses_real_prompts_md_for_every_agent():
    for agent in [
        "query_parser",
        "market_sizing",
        "competitor_landscape",
        "financial_feasibility",
        "synthesis",
        "failure_summary",
    ]:
        sp = prompt_loader.get_system_prompt(agent)
        assert isinstance(sp, str)
        assert len(sp) > 50


def test_query_parser_system_prompt_contains_schema_fields():
    sp = prompt_loader.get_system_prompt("query_parser")
    assert "industry" in sp
    assert "decision_type" in sp
    assert "not_a_business_question" in sp


def test_market_sizing_user_template_renders_placeholders():
    rendered = prompt_loader.render_user_template(
        "market_sizing", industry="specialty coffee", geography="USA", target_customer="cafes"
    )
    assert "specialty coffee" in rendered
    assert "USA" in rendered
    assert "cafes" in rendered
    assert "{{" not in rendered


def test_missing_placeholder_raises():
    with pytest.raises(prompt_loader.PromptLoadError):
        prompt_loader.render_user_template("market_sizing", industry="x")  # missing geography, target_customer


def test_retry_correction_template_has_both_placeholders():
    template = prompt_loader.get_retry_correction_template()
    assert "{{validation_error_message}}" in template
    assert "{{previous_raw_output}}" in template


def test_missing_agent_section_raises(tmp_path):
    bad_prompts = tmp_path / "prompts.md"
    bad_prompts.write_text(
        textwrap.dedent(
            """
            ## 0. Orchestrator — Query Parser Prompt
            **System Prompt:**
            ```
            You are a query parser.
            ```
            **User turn (template):**
            ```
            Business question: "{{raw_user_input}}"
            ```
            """
        )
    )
    prompt_loader.clear_cache()
    with pytest.raises(prompt_loader.PromptLoadError):
        prompt_loader.get_system_prompt("market_sizing", prompts_path=bad_prompts)
    prompt_loader.clear_cache()


def test_section_with_no_system_prompt_fence_raises(tmp_path):
    bad_prompts = tmp_path / "prompts.md"
    bad_prompts.write_text(
        textwrap.dedent(
            """
            ## 0. Orchestrator — Query Parser Prompt
            (no system prompt fence here at all)

            ## 1. Agent 1 — Market Sizing Agent
            **System Prompt:**
            ```
            You are a market-sizing analyst.
            ```

            ## 2. Agent 2 — Competitor Landscape Agent
            **System Prompt:**
            ```
            competitors
            ```

            ## 3. Agent 3 — Financial Feasibility Agent
            **System Prompt:**
            ```
            feasibility
            ```

            ## 4. Synthesis Agent
            **System Prompt:**
            ```
            synthesis
            ```

            ## 5. Schema Validator / Retry-Correction Prompt
            **System Prompt (appended as a corrective user turn, not a new system prompt):**
            ```
            retry {{validation_error_message}} {{previous_raw_output}}
            ```

            ## 6. Orchestrator-Level Meta-Prompt (Failure Summary, optional but recommended)
            **System Prompt:**
            ```
            failure summary
            ```
            """
        )
    )
    prompt_loader.clear_cache()
    with pytest.raises(prompt_loader.PromptLoadError):
        prompt_loader.get_system_prompt("query_parser", prompts_path=bad_prompts)
    prompt_loader.clear_cache()
