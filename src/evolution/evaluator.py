"""Score a workspace by running the task's tests against it.

The scoring directory is assembled here, not in the agent's workspace: we copy
the agent's output and the task's tests into a fresh throwaway directory. The
agent therefore never has a filesystem path to the tests (AGENTS.md §4).
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree

from .config import ScoringConfig
from .sandbox import ExecResult, SandboxBackend, Workspace
from .tasks import Task

logger = logging.getLogger(__name__)

REPORT_NAME = "junit-report.xml"


@dataclass
class SuiteResult:
    """Outcome of running one test directory against one workspace."""

    passed: int
    expected: int
    collected: int
    exit_code: int
    timed_out: bool
    stdout: str = ""
    stderr: str = ""
    collection_failed: bool = False

    @property
    def pass_fraction(self) -> float:
        # Divide by the *declared* count so a solution that breaks collection
        # scores 0.0 rather than an undefined 0/0.
        if self.expected <= 0:
            return 0.0
        return max(0.0, min(1.0, self.passed / self.expected))


@dataclass
class EvalResult:
    task_id: str
    visible: SuiteResult
    held_out: SuiteResult | None = None
    guardrail_violations: list[str] = field(default_factory=list)

    @property
    def pass_fraction(self) -> float:
        return self.visible.pass_fraction

    @property
    def held_out_pass_fraction(self) -> float | None:
        return self.held_out.pass_fraction if self.held_out else None

    def as_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "pass_fraction": round(self.pass_fraction, 4),
            "held_out_pass_fraction": (
                round(self.held_out_pass_fraction, 4)
                if self.held_out_pass_fraction is not None
                else None
            ),
            "tests_passed": self.visible.passed,
            "tests_expected": self.visible.expected,
            "collection_failed": self.visible.collection_failed,
            "timed_out": self.visible.timed_out,
            "guardrail_violations": self.guardrail_violations,
        }


def parse_junit_report(path: Path) -> tuple[int, int]:
    """Return (passed, collected) from a pytest --junit-xml report."""
    if not path.exists():
        return 0, 0
    try:
        root = ElementTree.parse(path).getroot()
    except ElementTree.ParseError:
        logger.warning("malformed junit report at %s", path)
        return 0, 0

    suites = (
        list(root.iter("testsuite")) if root.tag == "testsuites" else [root]
    )
    total = failures = errors = skipped = 0
    for suite in suites:
        total += int(suite.get("tests", 0))
        failures += int(suite.get("failures", 0))
        errors += int(suite.get("errors", 0))
        skipped += int(suite.get("skipped", 0))
    passed = max(0, total - failures - errors - skipped)
    return passed, total


class Evaluator:
    def __init__(self, backend: SandboxBackend) -> None:
        self.backend = backend

    def evaluate(
        self,
        workspace: Workspace,
        task: Task,
        *,
        include_held_out: bool = True,
        network: str = "deny",
    ) -> EvalResult:
        visible = self._run_suite(
            workspace, task, task.tests_dir, task.expected_tests, network
        )
        held_out: SuiteResult | None = None
        # Triggered by the directory existing, not by the declared count: the
        # count is what we're trying to discover when validating a new task.
        if include_held_out and task.held_out_dir.exists():
            held_out = self._run_suite(
                workspace,
                task,
                task.held_out_dir,
                task.expected_held_out_tests,
                network,
            )
        return EvalResult(task_id=task.id, visible=visible, held_out=held_out)

    def _run_suite(
        self,
        workspace: Workspace,
        task: Task,
        tests_dir: Path,
        expected: int,
        network: str,
    ) -> SuiteResult:
        if not tests_dir.exists():
            raise FileNotFoundError(f"tests directory not found: {tests_dir}")

        scoring_dir = Path(tempfile.mkdtemp(prefix="lme-score-"))
        try:
            self._assemble(workspace, tests_dir, scoring_dir)
            # The suite must be addressed by the directory it was copied into,
            # which is not always "tests" — the held-out suite has its own name.
            result = self._run_pytest(scoring_dir, tests_dir.name, task, network)
            passed, collected = parse_junit_report(scoring_dir / REPORT_NAME)
            return SuiteResult(
                passed=passed,
                expected=expected,
                collected=collected,
                exit_code=result.exit_code,
                timed_out=result.timed_out,
                stdout=result.stdout[-8000:],
                stderr=result.stderr[-8000:],
                # Fewer tests collected than declared means the suite never
                # materialized — usually the agent broke importability. Note a
                # collection *error* still reports as one collected item, so
                # this cannot be a `collected == 0` check.
                collection_failed=collected < expected,
            )
        finally:
            shutil.rmtree(scoring_dir, ignore_errors=True)

    @staticmethod
    def _assemble(workspace: Workspace, tests_dir: Path, scoring_dir: Path) -> None:
        """Fresh dir = agent output + tests. Tests are copied in last."""
        shutil.copytree(workspace.root, scoring_dir, dirs_exist_ok=True)
        # Drop anything the agent may have named 'tests' so it cannot shadow the
        # real suite; then lay the real tests down.
        agent_tests = scoring_dir / tests_dir.name
        if agent_tests.exists():
            shutil.rmtree(agent_tests, ignore_errors=True)
        shutil.copytree(tests_dir, agent_tests)

        # Our own conftest, written last so it overwrites any the agent
        # produced. Two jobs: make the agent's module importable from the
        # scoring root (pytest only puts the tests dir on sys.path), and deny
        # the agent a conftest hook it could use to fake results.
        (scoring_dir / "conftest.py").write_text(
            "import os\nimport sys\n\n"
            "sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n",
            encoding="utf-8",
        )

    def _run_pytest(
        self, scoring_dir: Path, suite_dir_name: str, task: Task, network: str
    ) -> ExecResult:
        command = [
            "python",
            "-m",
            "pytest",
            suite_dir_name,
            "-p",
            "no:cacheprovider",
            f"--junit-xml={REPORT_NAME}",
            "-q",
        ]
        return self.backend.run(
            command, cwd=scoring_dir, timeout_s=task.timeout_s, network=network
        )


def compute_score(
    pass_fraction: float,
    *,
    total_tokens: int,
    memory_tokens: int,
    config: ScoringConfig,
) -> float:
    """score = pass_fraction − λ·normalized_cost − μ·normalized_memory_length

    The length penalty is not cosmetic: memory grows every generation and
    nothing removes it unless growth costs something (AGENTS.md §6).
    """
    cost_term = config.lambda_cost * (
        total_tokens / max(1, config.cost_normalizer_tokens)
    )
    length_term = config.mu_length * (
        memory_tokens / max(1, config.length_normalizer_tokens)
    )
    return pass_fraction - cost_term - length_term
