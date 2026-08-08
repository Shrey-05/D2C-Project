"""
ConsultingOrchestrator — pipeline controller for the full 5-agent chain,
matching prompts.md's own "How these prompts fit together" diagram:

    Query Parser
       |
       +--> Market Sizing ---------\
       +--> Competitor Landscape    |  (run in parallel via asyncio.gather)
                                    v
            Financial Feasibility <-- waits on Market Sizing's SOM + confidence
                    |
        (2+ of the three agents above failed even after retry?)
           no  -> Synthesis Agent -> final memo
           yes -> Failure Summary  -> plain-text status message instead

Milestone 1 only wired Query Parser -> Market Sizing. This wires everything.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from typing import Callable, Optional

from anthropic import AsyncAnthropic

from src.agent_runtime import AgentRunResult
from src.agents.competitor_landscape import run_competitor_landscape
from src.agents.failure_summary import run_failure_summary, status_reason
from src.agents.financial_feasibility import run_financial_feasibility
from src.agents.market_sizing import run_market_sizing
from src.agents.query_parser import QueryParserResult, run_query_parser
from src.agents.synthesis import run_synthesis

FAILURE_THRESHOLD = 2  # 2+ of {market_sizing, competitor_landscape, financial_feasibility} failing -> failure_summary


@dataclass
class OrchestratorResult:
    raw_question: str
    query_parser: QueryParserResult
    market_sizing: Optional[AgentRunResult] = None
    competitor_landscape: Optional[AgentRunResult] = None
    financial_feasibility: Optional[AgentRunResult] = None
    synthesis: Optional[AgentRunResult] = None
    failure_summary_text: Optional[str] = None
    stopped_reason: Optional[str] = None  # set when the chain didn't reach synthesis/failure_summary

    @property
    def final_memo(self) -> Optional[str]:
        if self.synthesis is not None and self.synthesis.status == "ok":
            return self.synthesis.output.memo_markdown
        return None

    @property
    def recommendation(self) -> Optional[str]:
        if self.synthesis is not None and self.synthesis.status == "ok":
            return self.synthesis.output.recommendation.value
        return None

    @property
    def overall_confidence(self) -> Optional[str]:
        if self.synthesis is not None and self.synthesis.status == "ok":
            return self.synthesis.output.overall_confidence.value
        return None


ProgressCallback = Callable[[str, str], None]


class ConsultingOrchestrator:
    """progress_callback(stage_name, status) — status is one of
    "running" | "completed" | "failed" | "rejected" | "skipped".
    Matches the shape the eventual Streamlit frontend expects (build order
    Section 5 step 7, not built yet).
    """

    def __init__(self, client: AsyncAnthropic, model: str = "claude-sonnet-5", progress_callback: Optional[ProgressCallback] = None):
        self.client = client
        self.model = model
        self._progress: ProgressCallback = progress_callback or (lambda stage, status: None)

    async def run(self, raw_question: str) -> OrchestratorResult:
        self._progress("query_parser", "running")
        qp_result = await run_query_parser(self.client, raw_question, model=self.model)

        if qp_result.status == "failed":
            self._progress("query_parser", "failed")
            return OrchestratorResult(
                raw_question=raw_question,
                query_parser=qp_result,
                stopped_reason=f"query_parser failed validation twice: {qp_result.error}",
            )

        if qp_result.status == "rejected":
            self._progress("query_parser", "rejected")
            return OrchestratorResult(
                raw_question=raw_question,
                query_parser=qp_result,
                stopped_reason="input was not a business question",
            )

        self._progress("query_parser", "completed")
        parsed = qp_result.output  # QueryParserOutput

        # --- Market Sizing + Competitor Landscape, in parallel ---
        self._progress("market_sizing", "running")
        self._progress("competitor_landscape", "running")
        ms_result, cl_result = await asyncio.gather(
            run_market_sizing(self.client, parsed.industry, parsed.geography, parsed.target_customer, model=self.model),
            run_competitor_landscape(self.client, parsed.industry, parsed.geography, model=self.model),
        )
        self._progress("market_sizing", "completed" if ms_result.status == "ok" else "failed")
        self._progress("competitor_landscape", "completed" if cl_result.status == "ok" else "failed")

        # --- Financial Feasibility, dependent on Market Sizing's SOM ---
        if ms_result.status == "ok":
            som_value = ms_result.output.SOM.value_usd
            som_confidence = ms_result.output.confidence.value
        else:
            # Per prompts.md Section 3: this agent handles a null SOM itself
            # (returns null scenario values rather than fabricating), so we
            # still call it — we don't skip it just because Market Sizing failed.
            som_value = None
            som_confidence = "unavailable"

        self._progress("financial_feasibility", "running")
        ff_result = await run_financial_feasibility(
            self.client, parsed.industry, parsed.geography, som_value, som_confidence, model=self.model
        )
        self._progress("financial_feasibility", "completed" if ff_result.status == "ok" else "failed")

        failed_count = sum(1 for r in (ms_result, cl_result, ff_result) if r.status != "ok")

        if failed_count >= FAILURE_THRESHOLD:
            self._progress("synthesis", "skipped")
            self._progress("failure_summary", "running")
            summary_text = await run_failure_summary(
                self.client,
                status_reason(ms_result),
                status_reason(cl_result),
                status_reason(ff_result),
                model=self.model,
            )
            self._progress("failure_summary", "completed")
            return OrchestratorResult(
                raw_question=raw_question,
                query_parser=qp_result,
                market_sizing=ms_result,
                competitor_landscape=cl_result,
                financial_feasibility=ff_result,
                synthesis=None,
                failure_summary_text=summary_text,
            )

        self._progress("synthesis", "running")
        syn_result = await run_synthesis(
            self.client, raw_question, parsed, ms_result, cl_result, ff_result, model=self.model
        )
        self._progress("synthesis", "completed" if syn_result.status == "ok" else "failed")

        return OrchestratorResult(
            raw_question=raw_question,
            query_parser=qp_result,
            market_sizing=ms_result,
            competitor_landscape=cl_result,
            financial_feasibility=ff_result,
            synthesis=syn_result,
        )


def _print_progress(stage: str, status: str) -> None:
    print(f"  [{stage}] {status}", file=sys.stderr)


def _agent_result_json(result: Optional[AgentRunResult]) -> Optional[dict]:
    if result is None:
        return None
    return {
        "status": result.status,
        "output": result.output.model_dump(mode="json") if result.output is not None else None,
        "search_calls_made": result.search_calls_made,
        "error": result.error,
    }


def result_to_json(result: OrchestratorResult) -> dict:
    return {
        "raw_question": result.raw_question,
        "stopped_reason": result.stopped_reason,
        "query_parser": {
            "status": result.query_parser.status,
            "output": result.query_parser.output.model_dump(mode="json") if result.query_parser.output else None,
            "error": result.query_parser.error,
        },
        "market_sizing": _agent_result_json(result.market_sizing),
        "competitor_landscape": _agent_result_json(result.competitor_landscape),
        "financial_feasibility": _agent_result_json(result.financial_feasibility),
        "synthesis": _agent_result_json(result.synthesis),
        "failure_summary_text": result.failure_summary_text,
        "final_memo": result.final_memo,
        "recommendation": result.recommendation,
        "overall_confidence": result.overall_confidence,
    }


async def _main_async(question: str, out_path: Optional[str]) -> int:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY is not set — this makes real API + web-search calls.", file=sys.stderr)
        return 1

    client = AsyncAnthropic(api_key=api_key)
    orchestrator = ConsultingOrchestrator(client, progress_callback=_print_progress)

    print(f"Running: {question!r}", file=sys.stderr)
    result = await orchestrator.run(question)

    payload = result_to_json(result)
    text = json.dumps(payload, indent=2)

    if out_path:
        with open(out_path, "w") as f:
            f.write(text)
        print(f"Wrote {out_path}", file=sys.stderr)
    else:
        print(text)

    return 0 if result.final_memo or result.failure_summary_text else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full 5-agent chain directly (bypasses consulting_analyst.py's file-saving).")
    parser.add_argument("--question", required=True, help="The raw business question.")
    parser.add_argument("--out", default=None, help="Optional path to write JSON result to.")
    args = parser.parse_args()

    sys.exit(asyncio.run(_main_async(args.question, args.out)))


if __name__ == "__main__":
    main()
