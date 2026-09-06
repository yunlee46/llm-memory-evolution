"""Retry classification and token accounting — no network calls."""

from __future__ import annotations

import pytest
from evolution.llm import (
    TokenLedger,
    TokenUsage,
    is_daily_quota_exhausted,
    is_retryable,
)
from google.genai import errors


class _Resp:
    def __init__(self, code: int, message: str = "boom") -> None:
        self.status_code = code
        self.headers: dict[str, str] = {}
        self._message = message

    def json(self) -> dict:
        return {"error": {"code": self.status_code, "message": self._message}}


def _api_error(code: int, message: str = "boom") -> errors.APIError:
    resp = _Resp(code, message)
    return errors.APIError(code, resp.json(), resp)


DAILY_QUOTA_MESSAGE = (
    "You exceeded your current quota. Quota exceeded for metric: "
    "generate_content_free_tier_requests, quotaId: "
    "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
)


@pytest.mark.parametrize("code", [408, 429, 500, 502, 503, 504])
def test_transient_statuses_are_retried(code: int) -> None:
    assert is_retryable(_api_error(code))


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
def test_fatal_statuses_are_not_retried(code: int) -> None:
    assert not is_retryable(_api_error(code))


def test_connection_failures_are_retried() -> None:
    assert is_retryable(ConnectionError("reset"))
    assert is_retryable(TimeoutError())


def test_programming_errors_are_not_retried() -> None:
    assert not is_retryable(ValueError("bad argument"))


class TestDailyQuota:
    """A per-day 429 must not be retried — it will not clear in 20 seconds."""

    def test_daily_quota_is_detected(self) -> None:
        assert is_daily_quota_exhausted(_api_error(429, DAILY_QUOTA_MESSAGE))

    def test_daily_quota_is_not_retried(self) -> None:
        assert not is_retryable(_api_error(429, DAILY_QUOTA_MESSAGE))

    def test_per_minute_rate_limit_is_still_retried(self) -> None:
        per_minute = _api_error(
            429, "Quota exceeded, quotaId: GenerateRequestsPerMinutePerProject"
        )
        assert not is_daily_quota_exhausted(per_minute)
        assert is_retryable(per_minute)


class _Usage:
    prompt_token_count = 100
    candidates_token_count = 30
    thoughts_token_count = 12
    cached_content_token_count = 5


def test_usage_parsed_from_response() -> None:
    usage = TokenUsage.from_response(type("R", (), {"usage_metadata": _Usage()})())
    assert (usage.prompt, usage.output, usage.thinking, usage.cached) == (100, 30, 12, 5)
    assert usage.total == 142


def test_usage_absent_is_zero_not_crash() -> None:
    assert TokenUsage.from_response(object()).total == 0


def test_ledger_accumulates_per_role() -> None:
    ledger = TokenLedger()
    ledger.record("worker", TokenUsage(prompt=10, output=5))
    ledger.record("worker", TokenUsage(prompt=20, output=1))
    ledger.record("curator", TokenUsage(prompt=7, output=2))

    assert ledger.usage("worker").prompt == 30
    assert ledger.usage("curator").output == 2
    assert ledger.total.total == 45
    assert ledger.as_dict()["total"]["total_tokens"] == 45
