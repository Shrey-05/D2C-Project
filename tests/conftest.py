"""
Shared test fixtures.

generate_with_backoff() in src/agent_runtime.py now enforces a process-wide
minimum interval between calls (MIN_CALL_INTERVAL_SECONDS, default 13s) to
avoid tripping Gemini's free-tier rate limit during real runs. Every mocked
test in this suite routes through that same function, so without this
fixture the full test suite would take real minutes of wall-clock sleeping
instead of running in milliseconds.

This resets the module-level throttle clock before each test (so every test
starts as if no call has happened recently — the realistic case for a fresh
pipeline run) and patches asyncio.sleep globally for the test's duration, so
any throttle/backoff logic that *does* fire is still exercised and
assertable, it just never actually blocks.

Individual tests that want to inspect sleep call counts/arguments still
nest their own `with patch("src.agent_runtime.asyncio.sleep", ...)` inside
this — that's expected and works fine (the inner patch just shadows this
outer one for the duration of its own `with` block).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

import src.agent_runtime as agent_runtime


@pytest.fixture(autouse=True)
def _reset_throttle_and_stub_sleep():
    agent_runtime._last_call_at = 0.0
    with patch("src.agent_runtime.asyncio.sleep", new=AsyncMock()):
        yield
    agent_runtime._last_call_at = 0.0
