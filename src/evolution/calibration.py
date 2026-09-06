"""Measure empty-memory pass rates to find tasks with a usable gradient.

This is the gate the whole experiment rests on (AGENTS.md §10). A task the
model always solves and a task it never solves both carry zero signal: memory
cannot move a score that is pinned at either end. Only tasks in the middle band
can show a lift.

Every calibration run is a group-B run by construction — empty memory, frozen
prompt — so these numbers double as the generation-0 baseline.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .agent import CodingAgent
from .config import REPO_ROOT, ExperimentConfig
from .evaluator import Evaluator
from .sandbox import Workspace
from .tasks import Task

CALIBRATION_DIR = REPO_ROOT / "results" / "calibration"
STATE_PATH = CALIBRATION_DIR / "trials.jsonl"


@dataclass
class TrialResult:
    task_id: str
    trial: int
    pass_fraction: float
    held_out_pass_fraction: float | None
    turns: int
    stop_reason: str
    total_tokens: int
    thinking_tokens: int
    wall_clock_s: float
    violations: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class TaskCalibration:
    task_id: str
    trials: list[TrialResult]

    @property
    def pass_rates(self) -> list[float]:
        return [t.pass_fraction for t in self.trials if t.error is None]

    @property
    def mean_pass_rate(self) -> float:
        rates = self.pass_rates
        return statistics.fmean(rates) if rates else 0.0

    @property
    def stdev_pass_rate(self) -> float:
        rates = self.pass_rates
        return statistics.stdev(rates) if len(rates) > 1 else 0.0

    @property
    def mean_held_out(self) -> float:
        values = [
            t.held_out_pass_fraction
            for t in self.trials
            if t.error is None and t.held_out_pass_fraction is not None
        ]
        return statistics.fmean(values) if values else 0.0

    @property
    def mean_tokens(self) -> float:
        values = [t.total_tokens for t in self.trials if t.error is None]
        return statistics.fmean(values) if values else 0.0

    @property
    def solved_outright(self) -> int:
        return sum(1 for r in self.pass_rates if r == 1.0)

    def verdict(self, low: float, high: float) -> str:
        if not self.pass_rates:
            return "ERROR"
        mean = self.mean_pass_rate
        if mean < low:
            return "TOO_HARD"
        if mean > high:
            return "TOO_EASY"
        return "USABLE"

    def as_dict(self, low: float, high: float) -> dict:
        return {
            "task_id": self.task_id,
            "verdict": self.verdict(low, high),
            "mean_pass_rate": round(self.mean_pass_rate, 4),
            "stdev_pass_rate": round(self.stdev_pass_rate, 4),
            "mean_held_out_pass_rate": round(self.mean_held_out, 4),
            "solved_outright": self.solved_outright,
            "trials": len(self.trials),
            "mean_tokens": round(self.mean_tokens, 1),
            "per_trial": [asdict(t) for t in self.trials],
        }


BUDGET_MARKERS = ("RequestBudgetExceeded", "PerDay", "RESOURCE_EXHAUSTED")


def _is_budget_error(message: str) -> bool:
    """Quota exhaustion means stop, not retry — the rest of today's trials
    would all fail identically and pollute the record with fake zeros."""
    return any(marker in message for marker in BUDGET_MARKERS)


class CalibrationState:
    """Append-only trial log so calibration can span days and resume.

    Keyed by (model, task, trial): re-pinning the worker model invalidates
    earlier trials rather than silently mixing models into one mean.
    """

    def __init__(self, path: Path | None, model: str) -> None:
        self.path = path
        self.model = model
        self._done: dict[tuple[str, int], TrialResult] = {}
        if path and path.exists():
            self._load()

    @classmethod
    def ephemeral(cls) -> CalibrationState:
        return cls(None, "")

    def _load(self) -> None:
        assert self.path is not None
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("model") != self.model:
                continue
            trial = TrialResult(**{k: v for k, v in record.items() if k != "model"})
            # A trial that failed on quota is not a result; drop it so a resume
            # re-runs it instead of recording a spurious 0.0.
            if trial.error and _is_budget_error(trial.error):
                continue
            self._done[(trial.task_id, trial.trial)] = trial

    def has(self, task_id: str, trial: int) -> bool:
        return (task_id, trial) in self._done

    def record(self, result: TrialResult) -> None:
        self._done[(result.task_id, result.trial)] = result
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"model": self.model, **asdict(result)}) + "\n")

    def completed_count(self) -> int:
        return sum(1 for t in self._done.values() if not t.error)

    def calibrations(self, tasks: list[Task]) -> list[TaskCalibration]:
        out = []
        for task in tasks:
            trials = sorted(
                (t for (task_id, _), t in self._done.items() if task_id == task.id),
                key=lambda t: t.trial,
            )
            out.append(TaskCalibration(task_id=task.id, trials=trials))
        return out


class Calibrator:
    def __init__(
        self,
        client,
        config: ExperimentConfig,
        evaluator: Evaluator,
        *,
        offer_run_tests: bool = True,
    ) -> None:
        self.config = config
        self.evaluator = evaluator
        self.agent = CodingAgent(
            client, config, evaluator=evaluator if offer_run_tests else None
        )

    async def _trial(self, task: Task, trial: int) -> TrialResult:
        try:
            with Workspace.create(scaffold=task.scaffold_dir) as workspace:
                run = await self.agent.run(
                    task,
                    workspace,
                    memory=None,  # empty memory: this IS a group-B run
                    group="B",
                    seed=self.config.seed + trial,
                )
                evaluation = self.evaluator.evaluate(
                    workspace, task, network=self.config.sandbox.network
                )
            return TrialResult(
                task_id=task.id,
                trial=trial,
                pass_fraction=evaluation.pass_fraction,
                held_out_pass_fraction=evaluation.held_out_pass_fraction,
                turns=run.turns,
                stop_reason=run.stop_reason,
                total_tokens=run.usage.total,
                thinking_tokens=run.usage.thinking,
                wall_clock_s=run.wall_clock_s,
                violations=run.violations,
            )
        except Exception as exc:  # noqa: BLE001 - a failed trial must not sink the run
            return TrialResult(
                task_id=task.id,
                trial=trial,
                pass_fraction=0.0,
                held_out_pass_fraction=None,
                turns=0,
                stop_reason="error",
                total_tokens=0,
                thinking_tokens=0,
                wall_clock_s=0.0,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def calibrate(
        self,
        tasks: list[Task],
        trials: int,
        *,
        state: CalibrationState | None = None,
    ) -> tuple[list[TaskCalibration], bool]:
        """Run each (task, trial) pair, resuming past completed ones.

        Returns (calibrations, complete). Runs sequentially and persists after
        every trial: on a constrained quota this spans days, and losing a day's
        work to a crash on the last trial is not acceptable. Sequential also
        keeps the request budget from being overshot by in-flight calls.
        """
        state = state or CalibrationState.ephemeral()
        complete = True

        for task in tasks:
            for trial in range(1, trials + 1):
                if state.has(task.id, trial):
                    continue
                result = await self._trial(task, trial)
                state.record(result)
                if result.error and _is_budget_error(result.error):
                    complete = False
                    return state.calibrations(tasks), complete

        return state.calibrations(tasks), complete


def save_report(
    calibrations: list[TaskCalibration], config: ExperimentConfig
) -> Path:
    low = config.tasks.calibration.min_pass_rate
    high = config.tasks.calibration.max_pass_rate
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = CALIBRATION_DIR / f"{stamp}.json"
    path.write_text(
        json.dumps(
            {
                "generated_at": stamp,
                "worker_model": config.models.worker,
                "worker_temperature": config.models.worker_temperature,
                "band": {"min": low, "max": high},
                "tasks": [c.as_dict(low, high) for c in calibrations],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path
