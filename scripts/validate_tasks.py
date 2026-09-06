"""Verify every task is well-formed, satisfiable, and honestly counted.

Two things are checked per task, both of which would silently poison the
experiment if wrong:

1. The reference solution scores 1.0 on both suites. A test suite nobody can
   pass would look like a "hard task" during calibration rather than a bug.
2. `expected_tests` matches what pytest actually collects. That number is the
   denominator of every pass_fraction, so drift skews every score. Counting
   source lines is not good enough — parametrized tests expand at collection.

    uv run python scripts/validate_tasks.py
    uv run python scripts/validate_tasks.py --fix   # rewrite counts in registry
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402
from evolution.evaluator import Evaluator  # noqa: E402
from evolution.sandbox import LocalBackend, Workspace  # noqa: E402
from evolution.tasks import REGISTRY_PATH, Task, TaskRegistry  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def reference_dir(task: Task) -> Path:
    return FIXTURES / task.id / "reference"


def check(task: Task, evaluator: Evaluator) -> tuple[list[str], dict[str, int]]:
    problems = task.validate_layout()
    if problems:
        return problems, {}

    source = reference_dir(task)
    if not source.exists():
        return [f"no reference solution at {source}"], {}

    with Workspace.create(scaffold=source) as workspace:
        result = evaluator.evaluate(workspace, task, network="allow")

    counts = {
        "visible": result.visible.collected,
        "held_out": result.held_out.collected if result.held_out else 0,
    }

    if result.visible.collected != task.expected_tests:
        problems.append(
            f"expected_tests={task.expected_tests} but pytest collected "
            f"{result.visible.collected}"
        )
    held_out_collected = result.held_out.collected if result.held_out else 0
    if held_out_collected != task.expected_held_out_tests:
        problems.append(
            f"expected_held_out_tests={task.expected_held_out_tests} but pytest "
            f"collected {held_out_collected}"
        )

    if result.visible.passed != result.visible.collected:
        problems.append(
            f"reference solution fails its own visible suite: "
            f"{result.visible.passed}/{result.visible.collected} passed\n"
            f"{result.visible.stdout[-1500:]}"
        )
    if result.held_out and result.held_out.passed != result.held_out.collected:
        problems.append(
            f"reference solution fails its own held-out suite: "
            f"{result.held_out.passed}/{result.held_out.collected} passed\n"
            f"{result.held_out.stdout[-1500:]}"
        )
    return problems, counts


def apply_fix(counts: dict[str, dict[str, int]]) -> None:
    raw = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    for task_id, actual in counts.items():
        if task_id in raw["tasks"]:
            raw["tasks"][task_id]["expected_tests"] = actual["visible"]
            raw["tasks"][task_id]["expected_held_out_tests"] = actual["held_out"]
    REGISTRY_PATH.write_text(
        yaml.safe_dump(raw, sort_keys=False, width=88), encoding="utf-8"
    )
    print(f"\nrewrote counts in {REGISTRY_PATH}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fix", action="store_true", help="rewrite expected counts from collection"
    )
    args = parser.parse_args()

    registry = TaskRegistry.load()
    if not registry.tasks:
        print("no tasks registered")
        return 1

    evaluator = Evaluator(LocalBackend())
    failed = False
    all_counts: dict[str, dict[str, int]] = {}

    for task_id, task in sorted(registry.tasks.items()):
        problems, counts = check(task, evaluator)
        if counts:
            all_counts[task_id] = counts
        if problems:
            failed = True
            print(f"FAIL {task_id}")
            for problem in problems:
                print(f"       {problem}")
        else:
            print(
                f"ok   {task_id:<16} reference 1.0 · "
                f"{counts['visible']} visible / {counts['held_out']} held out"
            )

    if args.fix and all_counts:
        apply_fix(all_counts)
        return 0
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
