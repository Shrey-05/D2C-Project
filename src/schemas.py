"""
Pydantic v2 models for every agent's output in the pipeline.

Built first, per README_architecture.md Section 5 ("Schemas first ... this is
what the validator checks against, and it forces you to think through edge
cases up front"). Every field here maps directly to a schema block in
prompts.md — if you change a prompt's schema, change it here too, they must
stay in lockstep since validator.py trusts this file as ground truth.

Only QueryParserOutput and MarketSizingOutput are wired into orchestrator.py
in this first milestone. The remaining three (CompetitorLandscapeOutput,
FinancialFeasibilityOutput, SynthesisOutput) are defined now, per the build
order, but have no agent code calling them yet.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
# Shared enums
# --------------------------------------------------------------------------


class DecisionType(str, Enum):
    MARKET_ENTRY = "market_entry"
    PRODUCT_LAUNCH = "product_launch"
    EXPANSION = "expansion"
    PRICING_STRATEGY = "pricing_strategy"
    OTHER = "other"


class ValueTag(str, Enum):
    SOURCED = "sourced"
    DERIVED = "derived"
    UNAVAILABLE = "unavailable"


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PricingTier(str, Enum):
    BUDGET = "budget"
    MID = "mid"
    PREMIUM = "premium"
    LUXURY = "luxury"
    UNCLEAR = "unclear"


class RecommendationLean(str, Enum):
    FAVORABLE = "favorable"
    UNFAVORABLE = "unfavorable"
    MIXED = "mixed"
    INSUFFICIENT_DATA = "insufficient_data"


class Recommendation(str, Enum):
    PROCEED = "Recommend proceeding"
    PROCEED_WITH_CAVEATS = "Recommend proceeding with caveats"
    FURTHER_RESEARCH = "Recommend further research before deciding"
    AGAINST = "Recommend against proceeding"


class InheritedConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNAVAILABLE = "unavailable"


# --------------------------------------------------------------------------
# 0. Query Parser
# --------------------------------------------------------------------------


class AmbiguityItem(BaseModel):
    field: str
    assumption: str


class QueryParserOutput(BaseModel):
    industry: str
    geography: str
    target_customer: str
    decision_type: DecisionType
    ambiguities: List[AmbiguityItem] = Field(default_factory=list)


class QueryParserError(BaseModel):
    """The parser's escape hatch for non-business-question input.

    Kept as a separate model rather than an Optional field on
    QueryParserOutput, because the prompt says a rejected input returns
    *only* {"error": "not_a_business_question"} — a wholly different shape,
    not a variant with nulled-out fields.
    """

    error: str = "not_a_business_question"


# --------------------------------------------------------------------------
# 1. Market Sizing Agent
# --------------------------------------------------------------------------


class TAMField(BaseModel):
    value_usd: Optional[float] = None
    tag: ValueTag
    source: Optional[str] = None


class SAMField(BaseModel):
    value_usd: Optional[float] = None
    tag: ValueTag
    filter_logic: str


class SOMField(BaseModel):
    value_usd: Optional[float] = None
    tag: ValueTag
    capture_rate_pct: Optional[float] = None
    rationale: str


class SourceItem(BaseModel):
    name: str
    figure_cited: str
    year: str


class MarketSizingOutput(BaseModel):
    TAM: TAMField
    SAM: SAMField
    SOM: SOMField
    method_used: str
    sources: List[SourceItem] = Field(default_factory=list)
    key_assumptions: List[str] = Field(default_factory=list)
    confidence: ConfidenceLevel


# --------------------------------------------------------------------------
# 2. Competitor Landscape Agent  (schema only — no agent code yet)
# --------------------------------------------------------------------------


class CompetitorItem(BaseModel):
    name: str
    positioning: str
    pricing_tier: PricingTier
    differentiator: str
    source: str


class WhiteSpaceHypothesis(BaseModel):
    text: str
    is_inference: bool = True


class CompetitorLandscapeOutput(BaseModel):
    competitors: List[CompetitorItem] = Field(default_factory=list)
    search_coverage_note: str
    white_space_hypothesis: WhiteSpaceHypothesis
    confidence: ConfidenceLevel


# --------------------------------------------------------------------------
# 3. Financial Feasibility Agent  (schema only — no agent code yet)
# --------------------------------------------------------------------------


class ScenarioName(str, Enum):
    CONSERVATIVE = "conservative"
    BASE = "base"
    AGGRESSIVE = "aggressive"


class ScenarioItem(BaseModel):
    name: ScenarioName
    key_assumptions: List[str] = Field(default_factory=list)
    breakeven_estimate: str
    precision_note: str


class FinancialFeasibilityOutput(BaseModel):
    inherited_som_confidence: InheritedConfidence
    cac_estimate_range_usd: Optional[Tuple[float, float]] = None
    gross_margin_range_pct: Optional[Tuple[float, float]] = None
    scenarios: List[ScenarioItem] = Field(default_factory=list)
    recommendation_lean: RecommendationLean
    confidence: ConfidenceLevel


# --------------------------------------------------------------------------
# 4. Synthesis Agent  (schema only — no agent code yet)
# --------------------------------------------------------------------------


class TraceabilityCheck(BaseModel):
    all_figures_sourced_from_inputs: bool
    notes: str


class SynthesisOutput(BaseModel):
    memo_markdown: str
    overall_confidence: ConfidenceLevel
    recommendation: Recommendation
    traceability_check: TraceabilityCheck


# --------------------------------------------------------------------------
# Registry: agent name -> its output schema.
#
# validator.py and orchestrator.py key off this rather than hardcoding
# per-agent branches, so adding Agent 2/3/4's wiring later is a registry
# entry, not a new if/elif chain.
# --------------------------------------------------------------------------

AGENT_SCHEMAS = {
    "query_parser": QueryParserOutput,
    "market_sizing": MarketSizingOutput,
    "competitor_landscape": CompetitorLandscapeOutput,
    "financial_feasibility": FinancialFeasibilityOutput,
    "synthesis": SynthesisOutput,
}
