"""Load and validate config/experiment.yaml.

Every knob in the experiment comes through here. Code should never carry a
default that isn't visible in the YAML.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "experiment.yaml"

PLACEHOLDER_PREFIX = "PLACEHOLDER"


class ModelConfig(BaseModel):
    worker: str
    curator: str
    judge: str
    worker_temperature: float = 0.7
    curator_temperature: float = 0.2
    judge_temperature: float = 0.0
    worker_thinking_budget: int | None = None
    curator_thinking_budget: int | None = None

    def placeholders(self) -> list[str]:
        """Model IDs still left at their placeholder value."""
        return [
            name
            for name, value in (
                ("worker", self.worker),
                ("curator", self.curator),
                ("judge", self.judge),
            )
            if value.startswith(PLACEHOLDER_PREFIX)
        ]


class SplitRule(BaseModel):
    until_generation: int
    group_a: int


class EvolutionConfig(BaseModel):
    generations: int
    agents_per_generation: int
    trials_per_confirmation: int
    compaction_interval: int
    memory_token_cap: int
    group_split_schedule: list[SplitRule]
    min_group_b: int
    # Free-tier mode: one task per generation, rotating, instead of averaging
    # over the whole selection set (which costs N times the requests).
    task_rotation: bool = False

    @model_validator(mode="after")
    def _check_split(self) -> EvolutionConfig:
        total = self.agents_per_generation
        for rule in self.group_split_schedule:
            if not 0 <= rule.group_a <= total:
                raise ValueError(
                    f"group_a={rule.group_a} out of range for "
                    f"agents_per_generation={total}"
                )
            if total - rule.group_a < self.min_group_b:
                raise ValueError(
                    f"split rule until_generation={rule.until_generation} leaves "
                    f"{total - rule.group_a} in group B, below min_group_b="
                    f"{self.min_group_b}. Losing the control is worse than losing "
                    f"exploitation (AGENTS.md §9)."
                )
        if not self.group_split_schedule:
            raise ValueError("group_split_schedule must not be empty")
        return self

    def split_for(self, generation: int) -> tuple[int, int]:
        """Return (n_group_a, n_group_b) for a 1-indexed generation."""
        for rule in sorted(self.group_split_schedule, key=lambda r: r.until_generation):
            if generation <= rule.until_generation:
                return rule.group_a, self.agents_per_generation - rule.group_a
        last = max(self.group_split_schedule, key=lambda r: r.until_generation)
        return last.group_a, self.agents_per_generation - last.group_a


class ScoringConfig(BaseModel):
    lambda_cost: float
    mu_length: float
    cost_normalizer_tokens: int
    length_normalizer_tokens: int


class CalibrationConfig(BaseModel):
    trials: int
    min_pass_rate: float
    max_pass_rate: float


class TasksConfig(BaseModel):
    selection: list[str] = Field(default_factory=list)
    held_out: list[str] = Field(default_factory=list)
    calibration: CalibrationConfig

    @model_validator(mode="after")
    def _no_overlap(self) -> TasksConfig:
        overlap = set(self.selection) & set(self.held_out)
        if overlap:
            raise ValueError(
                f"tasks appear in both selection and held_out: {sorted(overlap)}. "
                "Held-out tasks must never influence selection (AGENTS.md §10)."
            )
        return self


class SandboxConfig(BaseModel):
    backend: Literal["auto", "docker", "local"]
    image: str
    max_turns: int
    wall_clock_s: int
    max_tokens_per_run: int
    network: Literal["deny", "allow"]


class ApiConfig(BaseModel):
    concurrency: int
    max_retries: int
    initial_backoff_s: float
    max_backoff_s: float
    # On the free tier the binding constraint is requests per day, not tokens.
    # null disables the guard (paid tier).
    daily_request_budget: int | None = None


class ExperimentConfig(BaseModel):
    run_id: str
    models: ModelConfig
    evolution: EvolutionConfig
    scoring: ScoringConfig
    tasks: TasksConfig
    sandbox: SandboxConfig
    api: ApiConfig
    seed: int

    # Populated from the environment, never from YAML.
    api_key: str = Field(repr=False, default="")

    @property
    def results_dir(self) -> Path:
        return REPO_ROOT / "results" / self.run_id

    @property
    def memory_dir(self) -> Path:
        return REPO_ROOT / "memory" / "runs" / self.run_id


def _generate_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def load_config(
    path: Path | str | None = None,
    *,
    require_api_key: bool = True,
) -> ExperimentConfig:
    """Load experiment config from YAML plus GEMINI_API_KEY from the environment.

    The API key is read from the environment only. It is never stored in YAML and
    is excluded from reprs so it cannot leak into logs.
    """
    load_dotenv(REPO_ROOT / ".env")

    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not raw.get("run_id"):
        raw["run_id"] = _generate_run_id()

    config = ExperimentConfig.model_validate(raw)

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if require_api_key and not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    config.api_key = api_key
    return config
