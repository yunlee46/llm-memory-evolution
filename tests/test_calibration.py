"""Request budgeting and resumable calibration state.

Both exist because the free tier caps requests per *day*: a run spans days, so
losing progress or overshooting the quota are the two failure modes that matter.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from evolution.calibration import CalibrationState, TaskCalibration, TrialResult
from evolution.config import ApiConfig
from evolution.llm import GeminiClient, RequestBudgetExceeded


def _trial(task_id: str, trial: int, *, pass_fraction: float = 0.5, error: str | None = None):
    return TrialResult(
        task_id=task_id,
        trial=trial,
        pass_fraction=pass_fraction,
        held_out_pass_fraction=None,
        turns=3,
        stop_reason="finished",
        total_tokens=1000,
        thinking_tokens=100,
        wall_clock_s=12.0,
        error=error,
    )


class TestRequestBudget:
    @pytest.fixture
    def api(self) -> ApiConfig:
        return ApiConfig(
            concurrency=1,
            max_retries=3,
            initial_backoff_s=0.1,
            max_backoff_s=1.0,
            daily_request_budget=3,
        )

    def test_charges_are_counted_per_model(self, api: ApiConfig):
        client = GeminiClient("dummy-key", api, request_budget=5)
        client._charge_request("flash")
        client._charge_request("flash")
        client._charge_request("pro")

        assert client.requests_by_model == {"flash": 2, "pro": 1}
        assert client.request_count == 3
        assert client.remaining_budget() == 2

    def test_budget_stops_before_exceeding(self, api: ApiConfig):
        client = GeminiClient("dummy-key", api, request_budget=2)
        client._charge_request("flash")
        client._charge_request("flash")

        with pytest.raises(RequestBudgetExceeded):
            client._charge_request("flash")

    def test_no_budget_means_unlimited(self, api: ApiConfig):
        client = GeminiClient("dummy-key", api, request_budget=None)
        for _ in range(50):
            client._charge_request("flash")
        assert client.remaining_budget() is None


class TestCalibrationState:
    def test_completed_trials_are_skipped_on_resume(self, tmp_path: Path):
        path = tmp_path / "trials.jsonl"
        first = CalibrationState(path, "gemini-x")
        first.record(_trial("semver", 1))

        resumed = CalibrationState(path, "gemini-x")
        assert resumed.has("semver", 1)
        assert not resumed.has("semver", 2)

    def test_quota_failures_are_not_persisted_as_results(self, tmp_path: Path):
        """A quota failure is an absence of data, not a score of zero."""
        path = tmp_path / "trials.jsonl"
        state = CalibrationState(path, "gemini-x")
        state.record(_trial("semver", 1, pass_fraction=0.0, error="RequestBudgetExceeded: ..."))

        resumed = CalibrationState(path, "gemini-x")
        assert not resumed.has("semver", 1)

    def test_a_different_model_invalidates_earlier_trials(self, tmp_path: Path):
        """Re-pinning the worker must not silently mix models into one mean."""
        path = tmp_path / "trials.jsonl"
        CalibrationState(path, "gemini-old").record(_trial("semver", 1))

        assert not CalibrationState(path, "gemini-new").has("semver", 1)

    def test_real_failures_are_kept(self, tmp_path: Path):
        path = tmp_path / "trials.jsonl"
        state = CalibrationState(path, "gemini-x")
        state.record(_trial("semver", 1, error="ValueError: agent produced nothing"))

        assert CalibrationState(path, "gemini-x").has("semver", 1)


class TestVerdicts:
    def _calibration(self, rates: list[float]) -> TaskCalibration:
        return TaskCalibration(
            task_id="t",
            trials=[_trial("t", i, pass_fraction=r) for i, r in enumerate(rates, 1)],
        )

    def test_always_solved_is_too_easy(self):
        assert self._calibration([1.0, 1.0, 1.0]).verdict(0.2, 0.8) == "TOO_EASY"

    def test_never_solved_is_too_hard(self):
        assert self._calibration([0.0, 0.0, 0.1]).verdict(0.2, 0.8) == "TOO_HARD"

    def test_middle_band_is_usable(self):
        assert self._calibration([0.3, 0.6, 0.5]).verdict(0.2, 0.8) == "USABLE"

    def test_no_successful_trials_is_an_error_not_a_verdict(self):
        calibration = TaskCalibration(
            task_id="t", trials=[_trial("t", 1, error="boom")]
        )
        assert calibration.verdict(0.2, 0.8) == "ERROR"
