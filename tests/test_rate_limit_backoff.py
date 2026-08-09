"""
Tests for generate_with_backoff — the 429-specific retry wrapper.

asyncio.sleep is patched throughout so these run in milliseconds instead of
actually waiting out the real backoff delays (5s, 10s, 20s).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from google.genai import errors

from src.agent_runtime import RATE_LIMIT_MAX_RETRIES, generate_with_backoff


def make_api_error(code: int, status: str = "ERROR") -> errors.APIError:
    return errors.APIError(code=code, response_json={"error": {"code": code, "status": status, "message": "x"}})


def make_client(*side_effects):
    client = SimpleNamespace()
    generate_content = AsyncMock(side_effect=list(side_effects))
    client.aio = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    return client


@pytest.mark.asyncio
async def test_succeeds_immediately_with_no_errors():
    ok_response = SimpleNamespace(text="fine")
    client = make_client(ok_response)
    with patch("src.agent_runtime.asyncio.sleep", new=AsyncMock()) as sleep_mock:
        result = await generate_with_backoff(client, "gemini-flash-latest", "hi", config=None)
    assert result is ok_response
    sleep_mock.assert_not_called()


@pytest.mark.asyncio
async def test_retries_on_429_then_succeeds():
    ok_response = SimpleNamespace(text="fine")
    client = make_client(make_api_error(429, "RESOURCE_EXHAUSTED"), ok_response)
    with patch("src.agent_runtime.asyncio.sleep", new=AsyncMock()) as sleep_mock:
        result = await generate_with_backoff(client, "gemini-flash-latest", "hi", config=None)
    assert result is ok_response
    assert client.aio.models.generate_content.call_count == 2
    sleep_mock.assert_called_once()


@pytest.mark.asyncio
async def test_gives_up_after_max_retries_all_429():
    client = make_client(*[make_api_error(429) for _ in range(RATE_LIMIT_MAX_RETRIES + 1)])
    with patch("src.agent_runtime.asyncio.sleep", new=AsyncMock()):
        with pytest.raises(errors.APIError):
            await generate_with_backoff(client, "gemini-flash-latest", "hi", config=None)
    assert client.aio.models.generate_content.call_count == RATE_LIMIT_MAX_RETRIES + 1


@pytest.mark.asyncio
async def test_does_not_retry_non_429_errors():
    """A bad model name or invalid key should fail fast, not burn 3 retries
    waiting out a guaranteed-identical failure."""
    client = make_client(make_api_error(404, "NOT_FOUND"), SimpleNamespace(text="should never be reached"))
    with patch("src.agent_runtime.asyncio.sleep", new=AsyncMock()) as sleep_mock:
        with pytest.raises(errors.APIError):
            await generate_with_backoff(client, "gemini-flash-latest", "hi", config=None)
    assert client.aio.models.generate_content.call_count == 1
    sleep_mock.assert_not_called()


@pytest.mark.asyncio
async def test_backoff_delay_grows_between_attempts():
    client = make_client(make_api_error(429), make_api_error(429), SimpleNamespace(text="fine"))
    with patch("src.agent_runtime.asyncio.sleep", new=AsyncMock()) as sleep_mock:
        await generate_with_backoff(client, "gemini-flash-latest", "hi", config=None)
    delays = [call.args[0] for call in sleep_mock.call_args_list]
    assert delays == sorted(delays)
    assert len(set(delays)) == len(delays)  # each wait is strictly longer than the last
