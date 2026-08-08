# Multi-Agent Consulting Analyst — Build #2, Complete

All 5 agents wired, per `prompts.md`'s own "How these prompts fit together" diagram:

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

## A note on how this happened

You uploaded three files (`consulting_analyst.py`, `financial_feasibility.py`,
`competitor_landscape.py`) written for **Build #1**'s internals — synchronous,
`run_agent(name, user_turn, offline, fixture_key)`, `--offline`/`--material`
flags. Build #2 is async, calls `AsyncAnthropic` directly, and validates
against `schemas.py`/`prompt_loader.py`/`validator.py`. The uploaded files
don't run against Build #2's code as-is. I used them as a spec of intent —
matched against what `prompts.md` Sections 2–4 and 6 actually say — and wrote
the real Build #2 equivalents from that, not from the uploaded files directly.

## What's here now

```
src/
├── schemas.py                    # all 5 agents' output schemas (built in milestone 1)
├── prompt_loader.py               # parses prompts.md (milestone 1)
├── validator.py                   # JSON extraction + retry payload (milestone 1)
├── agent_runtime.py                # NEW: shared tool-loop/retry runner, used by
│                                    #   market_sizing, competitor_landscape,
│                                    #   financial_feasibility, synthesis
├── orchestrator.py                 # UPDATED: now wires all 5 agents, not just 2
└── agents/
    ├── query_parser.py            # Agent 0 (milestone 1, unchanged)
    ├── market_sizing.py           # Agent 1 (milestone 1, refactored onto agent_runtime.py)
    ├── competitor_landscape.py     # NEW: Agent 2
    ├── financial_feasibility.py    # NEW: Agent 3
    ├── synthesis.py                 # NEW: Agent 4
    └── failure_summary.py           # NEW: orchestrator-level meta-prompt

consulting_analyst.py              # NEW: top-level CLI, saves analysis.md + run_log.json
                                     # (the file you meant by "output memo")

tests/  — 45 tests total, all passing, all against a mocked AsyncAnthropic
          client (still no real API key used anywhere in this build)
├── test_schemas.py
├── test_prompt_loader.py
├── test_validator.py
├── test_agents_mocked_client.py         # query_parser, market_sizing (milestone 1)
├── test_new_agents_mocked_client.py     # NEW: competitor_landscape, financial_feasibility,
│                                          #      synthesis, failure_summary
└── test_orchestrator_mocked.py           # NEW: full 5-agent wiring — happy path,
                                            #      SOM hand-off, 2-failure -> failure_summary,
                                            #      rejected-input short-circuit
```

## How to test

**Without a key** (proves the logic, not the model):
```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```
45/45 passing as of this writing.

**With a real key** (the actual thing):
```bash
export ANTHROPIC_API_KEY=sk-ant-...
python consulting_analyst.py --brief "Should a mid-market European specialty coffee roaster enter the US wholesale market?" --out ./coffee_run
```
Saves `coffee_run/analysis.md` (the memo, or the failure summary, or a stopped-reason
stub) and `coffee_run/run_log.json` (every agent's full validated output, for audit).

Optional: `--material ./notes.txt` appends a text file's contents as extra
context before the question reaches Query Parser.

Try a deliberately thin query (a narrow niche + narrow geography) to see
Competitor Landscape legitimately return fewer than 4 competitors with an
honest `search_coverage_note`, and Market Sizing legitimately return
`confidence: "low"` — this is Section 6's "thin-coverage" failure-mode test
case, and it's correct behavior, not a bug.

## What I verified vs. didn't — same caveat as milestone 1

**Verified by running it here:** all 45 tests pass, including the full
5-agent wiring against a mocked client — the parallel Market
Sizing/Competitor Landscape gather, the SOM value/confidence actually
reaching Financial Feasibility's prompt, the 2-failure threshold correctly
routing to `failure_summary` instead of `synthesis`, and a rejected
non-business input correctly short-circuiting before any other agent runs.

**Not verified — no API key in this environment:** no real model call was
made anywhere in this build, same as milestone 1. The mocked tests prove the
orchestration logic is right; they can't prove the real model's JSON comes
back clean on the first try for all 5 agents, or that `claude-sonnet-5` and
the web search tool behave as assumed. Run the live command above before
trusting this further — and note the retry-doubles-token-cost caveat from
milestone 1 now applies across 4 search-grounded/JSON-validated agents, not
1.

## Explicitly not built (per README_architecture.md Section 7's non-goals,
and Section 5 step 7 not yet reached)

- Streamlit frontend
- `test_cases/` saved fixture runs (clean / thin-coverage / forced-malformed) —
  the *logic* for all three is tested via mocks in `tests/`, but the
  interview-demo artifact of saving three real API run JSONs to
  `test_cases/` per Section 6 hasn't been done, since it needs a live key
- Tavily/Serper fallback search — only the native Anthropic web_search tool
  is wired
