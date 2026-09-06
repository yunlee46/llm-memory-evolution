"""Phase 3 gate: find which tasks have a usable gradient.

    uv run python scripts/calibrate.py                    # every registered task
    uv run python scripts/calibrate.py --tasks semver,csv_parse
    uv run python scripts/calibrate.py --trials 3 --no-run-tests

Costs real money: tasks x trials agent runs. Reports which tasks land inside the
configured pass-rate band, and prints the `tasks:` block to paste into
config/experiment.yaml.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evolution.calibration import (  # noqa: E402
    STATE_PATH,
    CalibrationState,
    Calibrator,
    save_report,
)
from evolution.config import load_config  # noqa: E402
from evolution.evaluator import Evaluator  # noqa: E402
from evolution.llm import GeminiClient  # noqa: E402
from evolution.sandbox import select_backend  # noqa: E402
from evolution.tasks import TaskRegistry  # noqa: E402


async def main_async(args: argparse.Namespace) -> int:
    config = load_config()
    registry = TaskRegistry.load()

    task_ids = (
        [t.strip() for t in args.tasks.split(",")] if args.tasks else sorted(registry.tasks)
    )
    tasks = [registry.get(task_id) for task_id in task_ids]
    trials = args.trials or config.tasks.calibration.trials
    low = config.tasks.calibration.min_pass_rate
    high = config.tasks.calibration.max_pass_rate

    backend = select_backend(config.sandbox)
    evaluator = Evaluator(backend)
    budget = None if args.budget == 0 else (args.budget or config.api.daily_request_budget)
    client = GeminiClient(config.api_key, config.api, request_budget=budget)
    calibrator = Calibrator(client, config, evaluator, offer_run_tests=args.run_tests)

    state = CalibrationState(None if args.no_resume else STATE_PATH, config.models.worker)
    already = state.completed_count()

    print(
        f"calibrating {len(tasks)} tasks x {trials} trials = {len(tasks) * trials} runs"
        f" ({already} already done)\n"
        f"model={config.models.worker} temp={config.models.worker_temperature} "
        f"run_tests={args.run_tests} backend={backend.name}\n"
        f"band=[{low}, {high}] request_budget={budget or 'unlimited'}\n"
    )

    calibrations, complete = await calibrator.calibrate(tasks, trials, state=state)
    report_path = save_report(calibrations, config)
    print(f"requests used this session: {client.request_count}\n")

    header = f"{'task':<16}{'mean':>7}{'sd':>7}{'held':>7}{'1.0s':>6}{'tokens':>9}  verdict"
    print(header)
    print("-" * len(header))
    for calibration in calibrations:
        print(
            f"{calibration.task_id:<16}"
            f"{calibration.mean_pass_rate:>7.2f}"
            f"{calibration.stdev_pass_rate:>7.2f}"
            f"{calibration.mean_held_out:>7.2f}"
            f"{calibration.solved_outright:>4}/{len(calibration.trials):<2}"
            f"{calibration.mean_tokens:>9,.0f}  {calibration.verdict(low, high)}"
        )

    errors = [t for c in calibrations for t in c.trials if t.error]
    if errors:
        print(f"\n{len(errors)} trial(s) errored:")
        for trial in errors[:5]:
            print(f"  {trial.task_id} #{trial.trial}: {trial.error}")

    usable = [c.task_id for c in calibrations if c.verdict(low, high) == "USABLE"]
    print(f"\ntotal tokens: {sum(t.total_tokens for c in calibrations for t in c.trials):,}")
    print(f"report: {report_path}")

    if not complete:
        done = state.completed_count()
        print(
            f"\nSTOPPED on quota with {done}/{len(tasks) * trials} trials done.\n"
            f"Progress is saved in {STATE_PATH}. Re-run the same command when the\n"
            f"quota resets and it will pick up where it left off. Verdicts above\n"
            f"are provisional until every trial completes."
        )
        return 2

    if not usable:
        print("\nNo task landed in the band. Options: harder tasks, or --no-run-tests")
        return 1

    # Held-out tasks measure generalization, so reserve the ones whose scores
    # are most stable; noisy tasks are better spent on selection.
    ranked = sorted(
        (c for c in calibrations if c.task_id in usable),
        key=lambda c: c.stdev_pass_rate,
    )
    n_held_out = min(3, max(1, len(ranked) // 3))
    held_out = sorted(c.task_id for c in ranked[:n_held_out])
    selection = sorted(c.task_id for c in ranked[n_held_out:])

    print(f"\n{len(usable)} usable. Suggested config/experiment.yaml block:\n")
    print("tasks:")
    print(f"  selection: {selection}")
    print(f"  held_out: {held_out}")
    return 0


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default=None, help="comma-separated task ids")
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument(
        "--no-run-tests",
        dest="run_tests",
        action="store_false",
        help="withhold the run_tests tool — use if tasks are saturating",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="max API requests this session (0 = unlimited; default from config)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore saved trials and start fresh",
    )
    parser.set_defaults(run_tests=True)
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
