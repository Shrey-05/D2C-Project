"""
Schema validation and retry-correction logic. No network calls here — this
module only ever looks at text that's already come back from the model.

Two entry points, matching implementation_plan.md's Phase 2 spec:

- validate_agent_output(raw_text, schema_cls) -> (ok, parsed, error_message)
- build_retry_payload(raw_text, error_message) -> str

The retry payload is built from prompts.md's own "Schema Validator /
Retry-Correction Prompt" section (via prompt_loader), not hardcoded here —
per README_architecture.md, prompt text lives in prompts.md, always.
"""

from __future__ import annotations

import json
from typing import Optional, Tuple, Type, TypeVar

from pydantic import BaseModel, ValidationError

from src import prompt_loader

T = TypeVar("T", bound=BaseModel)


class JSONExtractionError(ValueError):
    pass


def extract_json(raw_text: str) -> str:
    """Salvage a JSON object/array out of a model reply.

    Models say "output ONLY valid JSON" and then wrap it in a code fence or
    a sentence of preamble anyway. Order of attempts:

    1. The raw text as-is (the common, well-behaved case).
    2. The contents of the first fenced code block (``` or ```json).
    3. A hand-rolled scan for the first balanced {...} or [...], respecting
       string literals and escapes — not a regex, because a regex closing
       brace inside a quoted string will truncate the match early.
    """
    text = raw_text.strip()
    if not text:
        raise JSONExtractionError("empty input")

    # 1. Raw text.
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # 2. Fenced code block.
    if "```" in text:
        parts = text.split("```")
        # parts alternate: prose, fence-content, prose, fence-content, ...
        for i in range(1, len(parts), 2):
            candidate = parts[i]
            # Strip a leading language tag like "json\n"
            if "\n" in candidate:
                first_line, rest = candidate.split("\n", 1)
                if first_line.strip().isalpha():
                    candidate = rest
            candidate = candidate.strip()
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                continue

    # 3. Balanced-brace scan.
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = text.find(open_ch)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        break  # try the next open_ch/close_ch pair

    raise JSONExtractionError(f"could not find valid JSON in: {raw_text[:200]!r}")


def validate_agent_output(
    raw_text: str, schema_cls: Type[T]
) -> Tuple[bool, Optional[T], Optional[str]]:
    """Attempt JSON extraction + Pydantic validation.

    Returns (ok, parsed_model_or_None, error_message_or_None).
    error_message is human-readable and safe to drop straight into the
    retry-correction prompt.
    """
    try:
        json_text = extract_json(raw_text)
    except JSONExtractionError as e:
        return False, None, f"No valid JSON found in response: {e}"

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        return False, None, f"JSON parse error: {e}"

    try:
        parsed = schema_cls.model_validate(data)
    except ValidationError as e:
        return False, None, e.json(indent=2)

    return True, parsed, None


def build_retry_payload(raw_text: str, error_message: str) -> str:
    """Render prompts.md's retry-correction template with this failure's detail.

    Sent as a follow-up *user* turn in the same conversation (per prompts.md
    Section 5's design note) — not a fresh call with a new system prompt.
    """
    template = prompt_loader.get_retry_correction_template()
    return prompt_loader.render_text(
        template,
        validation_error_message=error_message,
        previous_raw_output=raw_text,
    )
