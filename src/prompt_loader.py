"""
Parses `prompts.md` into per-agent system prompts and user-turn templates.

Why this exists at all (README_architecture.md Section 8): prompts stay
editable without touching Python. Nobody should have to open orchestrator.py
to fix a typo in the Market Sizing system prompt.

Parsing strategy
-----------------
`prompts.md` is a fixed hand-authored document, not arbitrary markdown, so
this is a small structural parser rather than a general markdown parser:

1. Split the file on top-level "## " headers into sections.
2. Match each section's header text against AGENT_HEADER_KEYWORDS to find
   which agent (if any) it belongs to. Sections that match nothing (the
   title, section 7 "how these fit together", etc.) are ignored.
3. Within a matched section, pull the first fenced code block following a
   "**System Prompt:**" line, and the first fenced code block following a
   "**User turn (template):**" line.

If a keyword match finds two headers (shouldn't happen with this file, but
cheap to guard), the first one wins and a warning is not raised — fixing a
duplicate heading is a prompts.md authoring problem, not a runtime one this
loader should paper over silently forever, so tests catch it instead
(see tests/test_prompt_loader.py).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

DEFAULT_PROMPTS_PATH = Path(__file__).resolve().parent.parent / "prompts.md"

# Header text (lowercased) -> canonical agent key.
# Matched with "in" (substring), not equality, so minor header wording
# changes in prompts.md don't silently break the loader.
AGENT_HEADER_KEYWORDS: Dict[str, str] = {
    "query parser prompt": "query_parser",
    "market sizing agent": "market_sizing",
    "competitor landscape agent": "competitor_landscape",
    "financial feasibility agent": "financial_feasibility",
    "synthesis agent": "synthesis",
    "retry-correction prompt": "retry_correction",
    "failure summary": "failure_summary",
}


class PromptLoadError(RuntimeError):
    """Raised when prompts.md is missing an agent's system or user prompt."""


@dataclass(frozen=True)
class AgentPrompt:
    agent: str
    system_prompt: str
    user_template: Optional[str]  # None for failure_summary's templated-only case is not applicable here


_HEADER_RE = re.compile(r"^## (.+)$", re.MULTILINE)
_FENCE_RE = re.compile(r"```(?:\w*\n)?(.*?)```", re.DOTALL)


def _split_sections(text: str) -> Dict[str, str]:
    """Split on '## ' headers -> {header_text: section_body}."""
    matches = list(_HEADER_RE.finditer(text))
    sections: Dict[str, str] = {}
    for i, m in enumerate(matches):
        header = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[header] = text[start:end]
    return sections


def _agent_key_for_header(header: str) -> Optional[str]:
    lowered = header.lower()
    for keyword, agent_key in AGENT_HEADER_KEYWORDS.items():
        if keyword in lowered:
            return agent_key
    return None


def _extract_labelled_fence(body: str, label: str) -> Optional[str]:
    """Find the first fenced block that follows a '**{label}:**' line."""
    label_idx = body.find(label)
    if label_idx == -1:
        return None
    remainder = body[label_idx:]
    fence_match = _FENCE_RE.search(remainder)
    if fence_match is None:
        return None
    return fence_match.group(1).strip("\n")


@lru_cache(maxsize=None)
def _load_all(prompts_path: str) -> Dict[str, AgentPrompt]:
    path = Path(prompts_path)
    if not path.exists():
        raise PromptLoadError(f"prompts file not found: {path}")

    text = path.read_text(encoding="utf-8")
    sections = _split_sections(text)

    prompts: Dict[str, AgentPrompt] = {}
    for header, body in sections.items():
        agent_key = _agent_key_for_header(header)
        if agent_key is None:
            continue

        system_prompt = _extract_labelled_fence(body, "**System Prompt")
        user_template = _extract_labelled_fence(body, "**User turn (template):**")

        if system_prompt is None:
            raise PromptLoadError(
                f"section '{header}' matched agent '{agent_key}' but has no "
                f"'**System Prompt:**' fenced block"
            )

        prompts[agent_key] = AgentPrompt(
            agent=agent_key,
            system_prompt=system_prompt,
            user_template=user_template,
        )

    missing = set(AGENT_HEADER_KEYWORDS.values()) - set(prompts.keys())
    if missing:
        raise PromptLoadError(f"prompts.md is missing sections for: {sorted(missing)}")

    return prompts


def get_system_prompt(agent_name: str, prompts_path: Path = DEFAULT_PROMPTS_PATH) -> str:
    prompts = _load_all(str(prompts_path))
    if agent_name not in prompts:
        raise PromptLoadError(f"no such agent: {agent_name!r}")
    return prompts[agent_name].system_prompt


def get_user_template(agent_name: str, prompts_path: Path = DEFAULT_PROMPTS_PATH) -> str:
    prompts = _load_all(str(prompts_path))
    if agent_name not in prompts:
        raise PromptLoadError(f"no such agent: {agent_name!r}")
    template = prompts[agent_name].user_template
    if template is None:
        raise PromptLoadError(f"agent {agent_name!r} has no user-turn template")
    return template


def get_retry_correction_template(prompts_path: Path = DEFAULT_PROMPTS_PATH) -> str:
    """The retry-correction section (prompts.md Section 5) is a single fenced
    block, not a System Prompt / User turn pair like the agent sections — it
    gets sent as a corrective *user* turn in the failing agent's own
    conversation, per its own design note. It parses into `system_prompt`
    purely because that's the label `_extract_labelled_fence` matched on;
    this accessor exists so callers don't need to know that implementation
    detail.
    """
    return get_system_prompt("retry_correction", prompts_path=prompts_path)


_PLACEHOLDER_RE = re.compile(r"\{\{(\w+)\}\}")


def render_text(template: str, **kwargs: str) -> str:
    """Fill a {{placeholder}}-style template. Raises if a placeholder is left unfilled."""

    def _sub(match: re.Match) -> str:
        key = match.group(1)
        if key not in kwargs:
            raise PromptLoadError(f"template needs {{{{{key}}}}} but it wasn't provided")
        return str(kwargs[key])

    return _PLACEHOLDER_RE.sub(_sub, template)


def render_user_template(agent_name: str, prompts_path: Path = DEFAULT_PROMPTS_PATH, **kwargs: str) -> str:
    """Fill a named agent's {{placeholder}}-style user-turn template."""
    template = get_user_template(agent_name, prompts_path=prompts_path)
    return render_text(template, **kwargs)


def clear_cache() -> None:
    """Mainly for tests that write a temp prompts.md and want a fresh parse."""
    _load_all.cache_clear()
