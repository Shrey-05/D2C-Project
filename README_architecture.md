# Multi-Agent Consulting Analyst — Build Brief & Architecture

Hand this file to Claude Code alongside `multi_agent_consulting_analyst_prompts.md`.
This document defines the system architecture, schemas, file structure, and
build order. The prompts file defines what each agent says; this file defines
how the pieces connect and run.

---

## 1. What this system does

Takes a natural-language business question (e.g. "Should Client X enter the
Indian D2C skincare export market?") and produces a one-page, source-traceable
recommendation memo by running the question through 4 LLM agents:

1. **Query Parser** — extracts structured fields from the raw question
2. **Market Sizing Agent** — TAM/SAM/SOM with sourced-vs-derived tagging
3. **Competitor Landscape Agent** — 4-6 real, search-grounded competitors
4. **Financial Feasibility Agent** — breakeven scenarios, depends on Agent 2's output
5. **Synthesis Agent** — combines everything into the final memo

Agents 2 and 3 run in parallel. Agent 4 waits on Agent 2's SOM output and
confidence level. All agent outputs are strict JSON validated against
schemas; failures get one retry, then are flagged as data gaps rather than
silently dropped.

---

## 2. Tech stack

- **Language:** Python 3.11+
- **LLM calls:** Anthropic API (`anthropic` Python SDK), using tool use /
  structured output to enforce JSON schemas — do not rely on prompt
  instructions alone to guarantee valid JSON
- **Web search:** Anthropic's web search tool (preferred, since it's native
  to the API) — if unavailable, fall back to Tavily or Serper API
- **Orchestration:** plain Python (asyncio for parallel agent calls) — no
  LangGraph or other framework for the MVP. Add a framework later only if
  the plain version becomes unwieldy.
- **Frontend:** Streamlit — single-page app with a text input and a live
  status panel per agent ("Market sizing... ✓ done" / "Competitor scan...
  running")
- **Storage:** none required — each run is stateless. Optionally log runs to
  local JSON files for building your demo examples.
- **Testing:** a small `test_cases/` folder with 3-4 saved example queries,
  including at least one deliberately thin/edge-case query (see Section 6)

---

## 3. File structure

```
multi-agent-consulting-analyst/
├── README_architecture.md          (this file)
├── prompts.md                      (the prompt library — source of truth for all agent prompts)
├── requirements.txt
├── .env.example                    (ANTHROPIC_API_KEY, etc. — never commit real keys)
├── src/
│   ├── orchestrator.py             (state machine, runs agents, handles retries)
│   ├── schemas.py                  (JSON schema / pydantic models for every agent's output)
│   ├── agents/
│   │   ├── query_parser.py
│   │   ├── market_sizing.py
│   │   ├── competitor_landscape.py
│   │   ├── financial_feasibility.py
│   │   └── synthesis.py
│   ├── validator.py                (schema validation + retry-correction logic)
│   └── prompt_loader.py            (loads system prompts from prompts.md so prompts stay editable without touching code)
├── frontend/
│   └── app.py                      (Streamlit UI)
├── test_cases/
│   ├── clean_run_example.json
│   ├── thin_coverage_example.json  (deliberately hard query — see Section 6)
│   └── malformed_retry_example.json
└── outputs/
    └── (generated memos saved here, gitignored)
```

---

## 4. Data flow diagram

```
User question (raw text)
        │
        ▼
┌───────────────────┐
│  Query Parser      │  → parsed_query: {industry, geography, target_customer,
└───────────────────┘     decision_type, ambiguities[]}
        │
        ├─────────────────────────┬──────────────────────────┐
        ▼                         ▼                           │
┌───────────────────┐   ┌─────────────────────┐               │
│ Market Sizing      │   │ Competitor Landscape │              │
│ Agent              │   │ Agent                │              │
└───────────────────┘   └─────────────────────┘               │
        │  (SOM + confidence)                                  │
        └───────────────────────┐                              │
                                 ▼                              │
                     ┌───────────────────────┐                 │
                     │ Financial Feasibility  │                 │
                     │ Agent                  │                 │
                     └───────────────────────┘                 │
                                 │                               │
        ┌────────────────────────┴──────────────────────────────┘
        ▼
┌───────────────────┐
│  Schema Validator   │  → any agent output that fails validation gets
│  (all agent outputs)│     ONE retry via the correction prompt, then is
└───────────────────┘     marked "failed" and passed forward as a data gap
        │
        ▼
┌───────────────────┐
│  Synthesis Agent    │  → final memo (markdown) + overall_confidence +
└───────────────────┘     traceability_check
        │
        ▼
   Rendered in Streamlit + saved to outputs/
```

Note: Market Sizing and Competitor Landscape run in parallel (`asyncio.gather`).
Financial Feasibility only starts once Market Sizing's output is validated,
since it consumes Market Sizing's SOM and confidence fields directly.

---

## 5. Build order (do not build all agents before testing the first one)

1. **Schemas first.** Define pydantic models for every agent's output in
   `schemas.py` before writing any agent code — this is what the validator
   checks against, and it forces you to think through edge cases (null
   values, confidence levels) up front.
2. **Query Parser + Market Sizing Agent + Validator/Retry.** Get this
   three-piece chain fully working end-to-end with real API + search calls
   before touching anything else. This alone demonstrates the core
   grounding/validation story.
3. **Competitor Landscape Agent.** Add it running in parallel with Market
   Sizing via `asyncio.gather`.
4. **Financial Feasibility Agent.** Wire the dependency on Market Sizing's
   output — this is the one non-parallel step, test it separately with a
   hardcoded Market Sizing output first before connecting it live.
5. **Synthesis Agent.** Only build once all three upstream agents are
   producing real validated output — test with hand-crafted fake upstream
   JSON first so you're not debugging two things at once.
6. **Failure-path testing.** Deliberately feed thin/edge-case queries to
   confirm graceful degradation (see Section 6) before building the
   frontend.
7. **Streamlit frontend last.** Wrap the working backend in a UI — this
   should be quick once the orchestration logic actually works.

---

## 6. Required failure-mode test cases

Build and save these three specific test runs — they are your interview
demo material, not just QA:

1. **Clean run:** a well-known industry/geography (e.g. "premium coffee
   chains, urban India") that should produce high-confidence output from
   all agents.
2. **Thin-coverage run:** a genuinely niche industry/geography combo where
   web search will struggle (e.g. a very specific micro-category in a small
   market) — confirm the Competitor Agent returns fewer than 4 competitors
   with an honest `search_coverage_note` instead of inventing names, and
   confirm Market Sizing returns `confidence: low` rather than a
   fabricated-sounding number.
3. **Forced malformed-output run:** temporarily break one agent's prompt
   (e.g. remove the "output ONLY valid JSON" instruction) to trigger the
   retry path on purpose, confirm the retry succeeds or the run correctly
   falls back to marking that agent "failed" and the synthesis memo
   reflects the gap.

Save the raw JSON output of each of these three runs to `test_cases/` —
these become your "here's where I caught a failure mode and how I handled
it" interview story, which is more valuable than the clean run alone.

---

## 7. Explicit non-goals (keep scope tight)

- No user accounts, no persistence beyond local file logging
- No support for follow-up/conversational refinement of the memo — one
  question in, one memo out
- No fine-tuning or custom models — API calls to a hosted model only
- No general-purpose agent framework — the 4-agent structure is fixed and
  hardcoded, not configurable/extensible by design
- No real-money or brokerage integration of any kind (not applicable here,
  but stated for clarity if reused for other agent projects)

---

## 8. How to brief Claude Code with this

Suggested first message to Claude Code once both files are in the project
folder:

> "Read README_architecture.md and prompts.md in this folder. Build this
> project following the build order in Section 5 — start with schemas.py,
> then the Query Parser + Market Sizing Agent + Validator/Retry chain only,
> and stop there so I can test it before you continue. Use the exact system
> prompts from prompts.md via prompt_loader.py rather than hardcoding
> prompt text into the agent files."

Keeping prompts in a separate file that gets loaded at runtime (rather than
hardcoded inline in each agent's Python file) means you can tune the
prompts later without touching code — worth mentioning if asked about your
design choices.
