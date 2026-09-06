"""Task registry: the build tasks agents attempt, and their test suites."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from .config import REPO_ROOT

TASKS_ROOT = REPO_ROOT / "tasks"
REGISTRY_PATH = TASKS_ROOT / "registry.yaml"


class Task(BaseModel):
    """One build task.

    `expected_tests` and `expected_held_out_tests` are declared here rather than
    counted at runtime on purpose: if an agent's code breaks test collection,
    the observed test count is 0 and pass_fraction would be an undefined 0/0.
    Dividing by the declared count makes a broken solution score 0.0, which is
    the answer we want.
    """

    id: str
    description: str
    entrypoint: str = Field(description="file the agent is expected to produce")
    expected_tests: int
    expected_held_out_tests: int = 0
    runner: str = "pytest"
    timeout_s: int = 120
    requires: list[str] = Field(default_factory=list)

    @property
    def dir(self) -> Path:
        return TASKS_ROOT / self.id

    @property
    def prompt_path(self) -> Path:
        return self.dir / "prompt.md"

    @property
    def tests_dir(self) -> Path:
        return self.dir / "tests"

    @property
    def held_out_dir(self) -> Path:
        return self.dir / "tests_held_out"

    @property
    def scaffold_dir(self) -> Path | None:
        scaffold = self.dir / "scaffold"
        return scaffold if scaffold.exists() else None

    def prompt(self) -> str:
        """The frozen group-B prompt. Never mutate this mid-experiment."""
        return self.prompt_path.read_text(encoding="utf-8")

    def validate_layout(self) -> list[str]:
        """Return a list of layout problems, empty if the task is well-formed."""
        problems: list[str] = []
        if not self.prompt_path.exists():
            problems.append(f"missing prompt: {self.prompt_path}")
        if not self.tests_dir.exists():
            problems.append(f"missing tests dir: {self.tests_dir}")
        if self.expected_held_out_tests and not self.held_out_dir.exists():
            problems.append(f"missing held-out tests dir: {self.held_out_dir}")
        return problems


class TaskRegistry(BaseModel):
    tasks: dict[str, Task]

    @classmethod
    def load(cls, path: Path | None = None) -> TaskRegistry:
        registry_path = path or REGISTRY_PATH
        raw = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
        tasks = {
            task_id: Task.model_validate({"id": task_id, **spec})
            for task_id, spec in (raw.get("tasks") or {}).items()
        }
        return cls(tasks=tasks)

    def get(self, task_id: str) -> Task:
        if task_id not in self.tasks:
            raise KeyError(
                f"unknown task {task_id!r}; registry has {sorted(self.tasks)}"
            )
        return self.tasks[task_id]

    def validate_all(self) -> dict[str, list[str]]:
        return {
            task_id: problems
            for task_id, task in self.tasks.items()
            if (problems := task.validate_layout())
        }
