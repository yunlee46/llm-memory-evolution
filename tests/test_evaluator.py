"""Phase 1 acceptance: correct solution scores 1.0, broken ones score less.

These run the real evaluator against the real word_freq task via the local
backend, so they exercise assembly, pytest invocation, and report parsing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from evolution.config import DEFAULT_CONFIG_PATH, ScoringConfig
from evolution.evaluator import Evaluator, compute_score, parse_junit_report
from evolution.sandbox import LocalBackend, SandboxError, Workspace
from evolution.tasks import TaskRegistry

FIXTURES = Path(__file__).parent / "fixtures" / "word_freq"


@pytest.fixture(scope="module")
def task():
    return TaskRegistry.load().get("word_freq")


@pytest.fixture(scope="module")
def evaluator():
    return Evaluator(LocalBackend())


def _workspace_from(fixture_name: str) -> Workspace:
    workspace = Workspace.create()
    source = FIXTURES / fixture_name / "word_freq.py"
    workspace.write_file("word_freq.py", source.read_text(encoding="utf-8"))
    return workspace


def test_registry_layout_is_valid():
    problems = TaskRegistry.load().validate_all()
    assert problems == {}, problems


@pytest.mark.parametrize("task_id", sorted(TaskRegistry.load().tasks))
def test_every_task_is_satisfiable_and_honestly_counted(evaluator, task_id):
    """Each reference must score 1.0 with counts matching what pytest collects.

    A suite nobody can pass would look like a hard task during calibration
    rather than a bug, and `expected_tests` is the denominator of every
    pass_fraction, so drift there skews every score in the experiment.
    """
    registry_task = TaskRegistry.load().get(task_id)
    reference = FIXTURES.parent / task_id / "reference"
    assert reference.exists(), f"no reference solution for {task_id}"

    with Workspace.create(scaffold=reference) as workspace:
        result = evaluator.evaluate(workspace, registry_task, network="allow")

    assert result.visible.collected == registry_task.expected_tests
    assert result.visible.passed == result.visible.collected, result.visible.stdout[-2000:]
    assert result.held_out is not None
    assert result.held_out.collected == registry_task.expected_held_out_tests
    assert result.held_out.passed == result.held_out.collected, result.held_out.stdout[-2000:]


def test_reference_solution_scores_one(evaluator, task):
    with _workspace_from("reference") as workspace:
        result = evaluator.evaluate(workspace, task, network="allow")

    assert result.pass_fraction == 1.0
    assert result.held_out_pass_fraction == 1.0
    assert not result.visible.collection_failed


def test_partial_solution_scores_between_zero_and_one(evaluator, task):
    with _workspace_from("partial") as workspace:
        result = evaluator.evaluate(workspace, task, network="allow")

    assert result.visible.passed == 7
    assert result.pass_fraction == pytest.approx(7 / 8)
    assert result.held_out_pass_fraction == pytest.approx(2 / 3)


def test_broken_solution_scores_zero_without_crashing(evaluator, task):
    with _workspace_from("broken") as workspace:
        result = evaluator.evaluate(workspace, task, network="allow")

    assert result.pass_fraction == 0.0
    assert result.visible.collection_failed


def test_empty_workspace_scores_zero(evaluator, task):
    with Workspace.create() as workspace:
        result = evaluator.evaluate(workspace, task, network="allow")

    assert result.pass_fraction == 0.0


def test_agent_cannot_shadow_the_test_suite(evaluator, task):
    """An agent that writes its own passing 'tests/' must not be scored on it."""
    with _workspace_from("broken") as workspace:
        workspace.write_file("tests/test_word_freq.py", "def test_free_pass():\n    assert True\n")
        result = evaluator.evaluate(workspace, task, include_held_out=False, network="allow")

    assert result.pass_fraction == 0.0


def test_agent_conftest_cannot_hijack_collection(evaluator, task):
    """Our conftest is written last, so an agent's hooks never take effect."""
    with _workspace_from("partial") as workspace:
        workspace.write_file(
            "conftest.py",
            "def pytest_collection_modifyitems(items):\n    items.clear()\n",
        )
        result = evaluator.evaluate(workspace, task, include_held_out=False, network="allow")

    assert result.visible.passed == 7


class TestWorkspace:
    def test_paths_escaping_the_workspace_are_refused(self):
        with Workspace.create() as workspace:
            with pytest.raises(SandboxError, match="escapes workspace"):
                workspace.write_file("../escaped.py", "nope")

    def test_workspaces_start_clean(self):
        """Group B inheriting state would corrupt the only baseline."""
        with Workspace.create() as workspace:
            assert workspace.list_files() == []

    def test_cleanup_removes_the_directory(self):
        workspace = Workspace.create()
        root = workspace.root
        workspace.write_file("a.py", "x = 1")
        workspace.cleanup()
        assert not root.exists()


class TestJunitParsing:
    def test_missing_report_is_zero(self, tmp_path: Path):
        assert parse_junit_report(tmp_path / "nope.xml") == (0, 0)

    def test_malformed_report_is_zero(self, tmp_path: Path):
        path = tmp_path / "bad.xml"
        path.write_text("<testsuites", encoding="utf-8")
        assert parse_junit_report(path) == (0, 0)

    def test_skipped_tests_do_not_count_as_passed(self, tmp_path: Path):
        path = tmp_path / "r.xml"
        path.write_text(
            '<testsuites><testsuite tests="10" failures="2" errors="1" '
            'skipped="3"></testsuite></testsuites>',
            encoding="utf-8",
        )
        assert parse_junit_report(path) == (4, 10)


class TestScoring:
    @pytest.fixture
    def scoring(self) -> ScoringConfig:
        raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        return ScoringConfig.model_validate(raw["scoring"])

    def test_free_solution_scores_its_pass_fraction(self, scoring):
        assert compute_score(1.0, total_tokens=0, memory_tokens=0, config=scoring) == 1.0

    def test_token_cost_reduces_score(self, scoring):
        cheap = compute_score(1.0, total_tokens=10_000, memory_tokens=0, config=scoring)
        pricey = compute_score(1.0, total_tokens=200_000, memory_tokens=0, config=scoring)
        assert pricey < cheap < 1.0

    def test_memory_length_reduces_score(self, scoring):
        """Memory grows every generation; growth has to cost something."""
        short = compute_score(1.0, total_tokens=0, memory_tokens=100, config=scoring)
        long = compute_score(1.0, total_tokens=0, memory_tokens=1500, config=scoring)
        assert long < short
