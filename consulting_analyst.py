#!/usr/bin/env python3
"""
consulting_analyst.py — Build #2 entrypoint.

This mirrors the shape of the consulting_analyst.py you uploaded (same
--brief/--material/--out idea, same "save the memo + a full run log" output
contract) but drives Build #2's actual 5-agent async pipeline
(ConsultingOrchestrator) instead of Build #1's run_pipeline().

Two things from the uploaded version are deliberately NOT carried over,
because Build #2 doesn't have the machinery behind them:

- --offline: Build #2 has no fixture/offline mode. Every run here is a real
  API + web-search run. (Build #1, at ./consulting_analyst.py in your
  original upload, still has --offline if you want that.)
- --max-revisions: prompts.md Section 5 is explicit that the retry is
  "intentionally a single retry, not a loop" — that's hardcoded in
  agent_runtime.py, not a knob. Passing this flag would imply a
  configurability this build doesn't have, so it's left out rather than
  silently ignored.

Usage:

  python consulting_analyst.py --brief "Should we open a second warehouse
      in Poland, or expand the existing one?" --material ./notes.txt --out ./warehouse_q

  ANTHROPIC_API_KEY must be set in the environment.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from anthropic import AsyncAnthropic

from src.orchestrator import ConsultingOrchestrator, result_to_json


def _print_progress(stage: str, status: str) -> None:
    print(f"  {stage}: {status}", file=sys.stderr)


async def _run(brief: str, material_text: str) -> dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    raw_question = brief
    if material_text:
        raw_question = f"{brief}\n\nAdditional context/material provided by the user:\n{material_text}"

    client = AsyncAnthropic(api_key=api_key)
    orchestrator = ConsultingOrchestrator(client, progress_callback=_print_progress)
    result = await orchestrator.run(raw_question)
    return result, result_to_json(result)


def main():
    parser = argparse.ArgumentParser(description="Multi-agent consulting analyst (Build #2, full 5-agent pipeline)")
    parser.add_argument("--brief", type=str, required=True, help="The business question to analyze")
    parser.add_argument("--material", type=str, default=None, help="Path to a plain text file with supporting context")
    parser.add_argument("--out", type=str, default="./run_output", help="Directory to save the memo and run log")
    args = parser.parse_args()

    material_text = ""
    if args.material:
        if not os.path.exists(args.material):
            print(f"Error: material file not found: {args.material}")
            sys.exit(1)
        with open(args.material, "r", encoding="utf-8") as f:
            material_text = f.read()

    print(f"Running: \"{args.brief}\"\n", file=sys.stderr)
    orch_result, payload = asyncio.run(_run(args.brief, material_text))

    os.makedirs(args.out, exist_ok=True)

    # Full run log: every agent's status/output, for debugging/audit —
    # same role as Build #1's run_log.json.
    log_path = os.path.join(args.out, "run_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # The memo itself, if synthesis succeeded; the failure summary if not;
    # a stub explaining why if the pipeline stopped before either.
    memo_path = os.path.join(args.out, "analysis.md")
    with open(memo_path, "w", encoding="utf-8") as f:
        if orch_result.final_memo:
            f.write(orch_result.final_memo)
        elif orch_result.failure_summary_text:
            f.write(f"# Analysis could not be completed\n\n{orch_result.failure_summary_text}\n")
        else:
            f.write(f"# Run did not complete\n\nReason: {orch_result.stopped_reason}\n")

    print("--- Run summary ---", file=sys.stderr)
    print(f"  query_parser: {orch_result.query_parser.status}", file=sys.stderr)
    if orch_result.market_sizing:
        print(f"  market_sizing: {orch_result.market_sizing.status}", file=sys.stderr)
    if orch_result.competitor_landscape:
        print(f"  competitor_landscape: {orch_result.competitor_landscape.status}", file=sys.stderr)
    if orch_result.financial_feasibility:
        print(f"  financial_feasibility: {orch_result.financial_feasibility.status}", file=sys.stderr)
    if orch_result.synthesis:
        print(f"  synthesis: {orch_result.synthesis.status}", file=sys.stderr)
    elif orch_result.failure_summary_text:
        print("  synthesis: skipped (2+ upstream agents failed) -> failure_summary ran instead", file=sys.stderr)
    print(file=sys.stderr)

    if orch_result.final_memo:
        print(f"Recommendation: {orch_result.recommendation}", file=sys.stderr)
        print(f"Overall confidence: {orch_result.overall_confidence}", file=sys.stderr)
        print(f"\nMemo saved to: {memo_path}", file=sys.stderr)
    elif orch_result.failure_summary_text:
        print("Run degraded gracefully — see failure summary in the memo file.", file=sys.stderr)
        print(f"Memo saved to: {memo_path}", file=sys.stderr)
    else:
        print(f"Run did not produce a memo. Reason: {orch_result.stopped_reason}", file=sys.stderr)
    print(f"Full run log saved to: {log_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
