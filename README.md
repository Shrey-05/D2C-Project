# Multi-Agent Consulting Analyst — Gemini Edition

All 5 agents wired, matching `prompts.md`'s data-flow diagram, running on
**Google's Gemini API** instead of Claude — switched over specifically
because Gemini has a genuine free tier (no credit card, ~1,500 requests/day
on Flash models), which Anthropic's API doesn't.

```
Query Parser
   |
   +--> Market Sizing ---------\
   +--> Competitor Landscape    |  (parallel, asyncio.gather)
                                v
        Financial Feasibility <-- waits on Market Sizing's SOM + confidence
                |
    2+ of those three failed?  no -> Synthesis -> final memo
                                yes -> Failure Summary -> plain-text status
```

## What changed in the Gemini port

- **SDK:** `anthropic.AsyncAnthropic` → `google.genai.Client` (via its
  `.aio` async namespace)
- **Model:** `claude-sonnet-5` → `gemini-2.5-flash` everywhere
- **Search tool:** Claude's `web_search` needed a multi-turn tool-use loop
  (model requests search, server executes, model continues) that
  `agent_runtime.py` used to drive by hand. Gemini's Google Search
  grounding is fully server-managed within a *single* `generate_content`
  call — genuinely simpler, not just swapped syntax. `agent_runtime.py`'s
  tool loop is gone; it's now one call, optionally two if a retry fires.
- **JSON enforcement:** unchanged. Gemini can't combine `tools` with
  `response_schema`-based structured output, so — same as the Claude
  version — JSON shape is still enforced by prompt instruction +
  `validator.py`'s extraction/retry path, not a native API feature.
- **Env var:** `ANTHROPIC_API_KEY` → `GEMINI_API_KEY` everywhere (CLI,
  orchestrator, Streamlit app, `.env.example`)

`prompts.md` itself is untouched — the actual agent instructions are
model-agnostic English, so nothing there needed to change.

## Getting a free key

1. Go to **aistudio.google.com**
2. Sign in with any Google account
3. Click **Get API Key** → **Create API key**
4. No credit card required — copy the key (starts with `AIza...`)

## How to test

**Without a key:**
```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```
45/45 passing, all against a mocked `google.genai.Client` — no real API key
used anywhere in this build.

**With a real key:**
```bash
export GEMINI_API_KEY=AIza...
python consulting_analyst.py --brief "Should a mid-market European specialty coffee roaster enter the US wholesale market?" --out ./coffee_run
```

**As the web app (local):**
```bash
streamlit run frontend/app.py
```

**Deployed:** see the deploy steps discussed earlier in this project's
conversation — same Streamlit Community Cloud process, just add
`GEMINI_API_KEY` (not `ANTHROPIC_API_KEY`) and `APP_PASSWORD` to the
deployment's Secrets.

## What I verified vs. didn't

**Verified by running it here:** all 45 tests pass against the new mocked
Gemini client shape (`response.text`, `response.candidates[0].grounding_metadata`).
Every module imports cleanly. The CLI fails gracefully with no key set. The
Streamlit app boots (HTTP 200) with the new imports.

**Not verified — no real Gemini key in this environment:** no actual Gemini
API call was made anywhere in this port. The mocked tests prove the
orchestration and retry logic is correct against Gemini's *documented*
response shape; they can't prove a real `gemini-2.5-flash` call returns
clean JSON on the first try, or that Google Search grounding behaves
exactly as mocked. Run the live command above before trusting this further.

**Known limits of the free tier worth knowing:**
- Flash-class models only — Gemini's stronger "Pro" models aren't included
  on the free tier as of this port
- Rate-limited (~15 requests/minute on Flash) — fine for a resume demo used
  occasionally, not for heavy concurrent traffic
- Google's terms allow free-tier prompts/outputs to be used to improve their
  models — worth knowing if you ever put non-public business context into
  the "Optional supporting context" field
