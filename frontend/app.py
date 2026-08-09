"""
Streamlit frontend — README_architecture.md Section 5, step 7.

Visual design concept: an analyst's desk, not a chat app. The pipeline
renders as a vertical "signal trace" connecting all 6 agent stages, filling
gold as each completes — that's a literal encoding of the pipeline's real
dependency structure, not decoration. The finished memo renders on a
parchment "document" surface against the dark chrome around it, so it reads
as an actual deliverable landing on the desk, not another UI panel.

Run locally with:
    export GEMINI_API_KEY=AIza...
    streamlit run frontend/app.py

Deployed (e.g. Streamlit Community Cloud), the key comes from st.secrets
instead of the environment — see README.md. If neither is set, the app
falls back to asking the visitor to paste in their own key for that session
only, so a public deployment doesn't have to expose a shared key to every
visitor's usage.
"""

from __future__ import annotations

import asyncio
import os
import sys

import streamlit as st
from google import genai

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.orchestrator import ConsultingOrchestrator, result_to_json

st.set_page_config(page_title="Multi-Agent Consulting Analyst", page_icon="◈", layout="centered")

# --------------------------------------------------------------------------
# Design tokens + global styling
# --------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,wght@0,400;0,500;0,600;1,400&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --ink: #0E1526;
        --surface: #161F35;
        --surface-2: #1D2843;
        --parchment: #F6F1E7;
        --parchment-ink: #2B2620;
        --signal: #C9A227;
        --signal-dim: #6B5A1E;
        --text: #E8EAF0;
        --text-muted: #7C88A0;
        --ok: #6FA98A;
        --danger: #C97066;
        --border: #263257;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    [data-testid="stAppViewContainer"], [data-testid="stHeader"] { background: var(--ink); }
    [data-testid="stSidebar"] { background: var(--surface); border-right: 1px solid var(--border); }

    /* Header block */
    .eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--signal);
        margin-bottom: 0.4rem;
    }
    .display-title {
        font-family: 'Newsreader', serif;
        font-weight: 500;
        font-size: 2.3rem;
        color: var(--text);
        margin: 0 0 0.6rem 0;
        line-height: 1.15;
    }
    .rule {
        height: 1px;
        background: linear-gradient(90deg, var(--signal) 0%, var(--border) 40%);
        margin: 0.4rem 0 1.4rem 0;
    }
    .subcaption { color: var(--text-muted); font-size: 0.92rem; margin-bottom: 1.6rem; }

    /* Inputs */
    [data-testid="stTextArea"] textarea, [data-testid="stTextInput"] input {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        border-radius: 6px !important;
    }
    [data-testid="stTextArea"] textarea:focus, [data-testid="stTextInput"] input:focus {
        border-color: var(--signal) !important;
        box-shadow: 0 0 0 1px var(--signal) !important;
    }
    [data-testid="stWidgetLabel"] label p {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: var(--text-muted) !important;
    }

    /* Primary button */
    [data-testid="stButton"] button[kind="primary"] {
        background: var(--signal) !important;
        border: none !important;
        color: var(--ink) !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
    }
    [data-testid="stButton"] button[kind="primary"]:hover { background: #E0B830 !important; }
    [data-testid="stButton"] button[kind="primary"]:disabled { background: var(--signal-dim) !important; color: #9a915f !important; }

    /* Signal trace (the 6-stage pipeline) */
    .trace { position: relative; margin: 1.6rem 0 2rem 0; padding-left: 2.2rem; }
    .trace-line {
        position: absolute; left: 0.55rem; top: 0.3rem; bottom: 0.3rem;
        width: 2px; background: var(--border);
    }
    .trace-node { position: relative; padding-bottom: 1.15rem; }
    .trace-dot {
        position: absolute; left: -2.2rem; top: 0.05rem;
        width: 13px; height: 13px; border-radius: 50%;
        background: var(--surface-2); border: 2px solid var(--border);
        transition: all 0.3s ease;
    }
    .trace-dot.running { border-color: var(--signal); background: var(--surface-2); box-shadow: 0 0 0 3px rgba(201,162,39,0.18); }
    .trace-dot.completed { border-color: var(--signal); background: var(--signal); }
    .trace-dot.failed { border-color: var(--danger); background: var(--danger); }
    .trace-dot.skipped { border-color: var(--border); background: var(--border); }
    .trace-label {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.86rem;
        color: var(--text); font-weight: 500;
    }
    .trace-status {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.74rem;
        color: var(--text-muted); margin-left: 0.5rem;
    }
    .trace-status.completed { color: var(--ok); }
    .trace-status.failed { color: var(--danger); }
    .trace-status.running { color: var(--signal); }

    /* Result banner */
    .result-banner {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.86rem;
        background: var(--surface); border: 1px solid var(--signal);
        border-radius: 6px; padding: 0.85rem 1.1rem; margin: 1.2rem 0;
        color: var(--text);
    }
    .result-banner b { color: var(--signal); }

    /* The memo itself: parchment document surface, applied to Streamlit's
       bordered container (st.container(border=True)) since that's a real
       DOM parent — unlike a hand-opened <div> split across separate
       st.markdown calls, which Streamlit renders as isolated siblings. */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--parchment) !important;
        border: none !important;
        border-radius: 4px !important;
        box-shadow: 0 12px 32px rgba(0,0,0,0.35);
    }
    [data-testid="stVerticalBlockBorderWrapper"] * { color: var(--parchment-ink) !important; }
    [data-testid="stVerticalBlockBorderWrapper"] h1,
    [data-testid="stVerticalBlockBorderWrapper"] h2,
    [data-testid="stVerticalBlockBorderWrapper"] h3 { font-family: 'Newsreader', serif; }
    [data-testid="stVerticalBlockBorderWrapper"] p,
    [data-testid="stVerticalBlockBorderWrapper"] li {
        font-family: 'Newsreader', serif; font-size: 1rem; line-height: 1.6;
    }
    [data-testid="stVerticalBlockBorderWrapper"] code { background: rgba(43,38,32,0.08) !important; }

    [data-testid="stCaptionContainer"] { color: var(--text-muted) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


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
        st.markdown('<div class="eyebrow">Access</div>', unsafe_allow_html=True)
        st.markdown('<h1 class="display-title">Multi-Agent Consulting Analyst</h1>', unsafe_allow_html=True)
        st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
        st.caption("This demo is password-protected to prevent public use of the deployed API key.")
        entered = st.text_input("Password", type="password")
        if st.button("Enter", type="primary"):
            if entered == app_password:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password.")
        st.stop()

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.markdown('<div class="eyebrow">Five-Agent Research Pipeline</div>', unsafe_allow_html=True)
st.markdown('<h1 class="display-title">Multi-Agent Consulting Analyst</h1>', unsafe_allow_html=True)
st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subcaption">Turns a business question into a source-traceable one-page '
    'recommendation memo — Query Parser, Market Sizing, Competitor Landscape, '
    'Financial Feasibility, and Synthesis, each agent handing off to the next.</div>',
    unsafe_allow_html=True,
)

STAGES = [
    ("query_parser", "01 · Query Parser"),
    ("market_sizing", "02 · Market Sizing"),
    ("competitor_landscape", "02 · Competitor Landscape"),
    ("financial_feasibility", "03 · Financial Feasibility"),
    ("synthesis", "04 · Synthesis"),
    ("failure_summary", "— · Failure Summary"),
]
STATUS_TEXT = {
    "running": "running",
    "completed": "done",
    "failed": "failed",
    "rejected": "rejected",
    "skipped": "skipped",
}


def _resolve_api_key() -> tuple[str | None, str]:
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY", None)
    except Exception:
        secret_key = None
    if secret_key:
        return secret_key, "deployment secrets"
    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key:
        return env_key, "environment variable"
    return None, "none"


configured_key, key_source = _resolve_api_key()

with st.sidebar:
    st.markdown('<div class="eyebrow">API Key</div>', unsafe_allow_html=True)
    if configured_key:
        st.success(f"Using key from {key_source}.")
        api_key = configured_key
    else:
        st.info("No key configured for this deployment. Paste your own to run a query.")
        api_key = st.text_input("Gemini API key", type="password", help="Used only for this session, never stored.")
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    st.caption(
        "Every run makes real API calls, including web search on 3 of the 5 "
        "agents. Expect this to take 1-2 minutes and to cost a small amount "
        "against the key in use."
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
    stage_status: dict[str, str | None] = {key: None for key, _ in STAGES}
    trace_placeholder = st.empty()

    def render_trace() -> str:
        nodes = []
        for key, label in STAGES:
            status = stage_status[key]
            dot_class = status or ""
            status_text = STATUS_TEXT.get(status, "") if status else ""
            status_html = f'<span class="trace-status {dot_class}">{status_text}</span>' if status_text else ""
            nodes.append(
                f'<div class="trace-node"><span class="trace-dot {dot_class}"></span>'
                f'<span class="trace-label">{label}</span>{status_html}</div>'
            )
        return '<div class="trace"><div class="trace-line"></div>' + "".join(nodes) + "</div>"

    trace_placeholder.markdown(render_trace(), unsafe_allow_html=True)

    def progress_callback(stage: str, status: str) -> None:
        if stage in stage_status:
            stage_status[stage] = status
            trace_placeholder.markdown(render_trace(), unsafe_allow_html=True)

    async def _run():
        client = genai.Client(api_key=api_key)
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

    if result.final_memo:
        st.markdown(
            f'<div class="result-banner"><b>Recommendation:</b> {result.recommendation} '
            f'&nbsp;·&nbsp; <b>Confidence:</b> {result.overall_confidence}</div>',
            unsafe_allow_html=True,
        )
        # st.container(border=True) is used deliberately instead of a raw
        # <div>...</div> opened/closed across separate st.markdown calls —
        # Streamlit renders each markdown call as its own isolated DOM
        # sibling, so a hand-opened div wouldn't actually wrap the memo
        # content between calls. A bordered container is a real nesting
        # parent, so the .memo-doc-target CSS below actually applies to
        # everything rendered inside the `with` block.
        with st.container(border=True):
            st.markdown(result.final_memo)
        st.download_button("Download memo (.md)", result.final_memo, file_name="analysis.md", mime="text/markdown")
    elif result.failure_summary_text:
        st.warning("The pipeline could not complete in full.")
        st.write(result.failure_summary_text)
    else:
        st.error(f"Run stopped before producing output: {result.stopped_reason}")

    with st.expander("Raw agent outputs (JSON)"):
        st.json(result_to_json(result))
