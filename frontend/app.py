"""
Streamlit frontend — README_architecture.md Section 5, step 7 ("wrap the
working backend in a UI — this should be quick once the orchestration logic
actually works").

Run locally with:
    export ANTHROPIC_API_KEY=sk-ant-...
    streamlit run frontend/app.py

Deployed (e.g. Streamlit Community Cloud), the key comes from st.secrets
instead of the environment — see the deploy steps in README.md. If neither
is set, the app falls back to asking the visitor to paste in their own key
for that session only, so a public deployment doesn't have to expose a
shared key to every visitor's usage.
"""

from __future__ import annotations

import asyncio
import os
import sys

import streamlit as st
from anthropic import AsyncAnthropic

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.orchestrator import ConsultingOrchestrator, result_to_json

st.set_page_config(page_title="Multi-Agent Consulting Analyst", page_icon="📊", layout="centered")


def _resolve_app_password() -> str | None:
    try:
        return st.secrets.get("APP_PASSWORD", None)
    except Exception:
        return os.environ.get("APP_PASSWORD")


app_password = _resolve_app_password()

if app_password:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("📊 Multi-Agent Consulting Analyst")
        st.caption("This demo is password-protected to prevent public use of the deployed API key.")
        entered = st.text_input("Password", type="password")
        if st.button("Enter"):
            if entered == app_password:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password.")
        st.stop()

st.title("📊 Multi-Agent Consulting Analyst")
st.caption(
    "Turns a business question into a source-traceable one-page recommendation "
    "memo by running it through 5 specialized agents: Query Parser, Market "
    "Sizing, Competitor Landscape, Financial Feasibility, and Synthesis."
)

STAGES = [
    ("query_parser", "Query Parser"),
    ("market_sizing", "Market Sizing Agent"),
    ("competitor_landscape", "Competitor Landscape Agent"),
    ("financial_feasibility", "Financial Feasibility Agent"),
    ("synthesis", "Synthesis Agent"),
    ("failure_summary", "Failure Summary"),
]
STAGE_LABELS = dict(STAGES)
STATUS_ICON = {"running": "⏳", "completed": "✅", "failed": "❌", "rejected": "🚫", "skipped": "⏭️"}


def _resolve_api_key() -> tuple[str | None, str]:
    """Returns (key, source_description)."""
    try:
        secret_key = st.secrets.get("ANTHROPIC_API_KEY", None)
    except Exception:
        secret_key = None
    if secret_key:
        return secret_key, "deployment secrets"
    env_key = os.environ.get("ANTHROPIC_API_KEY")
    if env_key:
        return env_key, "environment variable"
    return None, "none"


configured_key, key_source = _resolve_api_key()

with st.sidebar:
    st.subheader("API key")
    if configured_key:
        st.success(f"Using key from {key_source}.")
        api_key = configured_key
    else:
        st.info("No key configured for this deployment. Paste your own to run a query.")
        api_key = st.text_input("Anthropic API key", type="password", help="Used only for this session, never stored.")
    st.divider()
    st.caption(
        "Every run makes real API calls, including web search on 3 of the 5 "
        "agents. Expect this to take 30-90 seconds and to cost a small "
        "amount against the key in use."
    )

question = st.text_area(
    "Business question",
    placeholder='e.g. "Should we enter the US wholesale coffee market?"',
    height=100,
)
material = st.text_area(
    "Optional supporting context",
    placeholder="Paste any extra notes or context here (optional)",
    height=80,
)

run_clicked = st.button("Analyze", type="primary", disabled=not question.strip() or not api_key)

if run_clicked:
    placeholders = {key: st.empty() for key, _ in STAGES}
    for key, label in STAGES:
        placeholders[key].markdown(f"⬜ {label}")

    def progress_callback(stage: str, status: str) -> None:
        label = STAGE_LABELS.get(stage, stage)
        icon = STATUS_ICON.get(status, "•")
        if stage in placeholders:
            placeholders[stage].markdown(f"{icon} **{label}** — {status}")

    async def _run():
        client = AsyncAnthropic(api_key=api_key)
        orchestrator = ConsultingOrchestrator(client, progress_callback=progress_callback)
        raw_question = question
        if material.strip():
            raw_question += f"\n\nAdditional context/material provided by the user:\n{material}"
        return await orchestrator.run(raw_question)

    with st.spinner("Running the agent pipeline..."):
        try:
            result = asyncio.run(_run())
        except Exception as e:
            st.error(f"Run failed: {e}")
            st.stop()

    st.divider()

    if result.final_memo:
        st.success(f"**Recommendation:** {result.recommendation}  ·  **Confidence:** {result.overall_confidence}")
        st.markdown(result.final_memo)
        st.download_button("Download memo (.md)", result.final_memo, file_name="analysis.md", mime="text/markdown")
    elif result.failure_summary_text:
        st.warning("The pipeline could not complete in full.")
        st.write(result.failure_summary_text)
    else:
        st.error(f"Run stopped before producing output: {result.stopped_reason}")

    with st.expander("Raw agent outputs (JSON)"):
        st.json(result_to_json(result))
