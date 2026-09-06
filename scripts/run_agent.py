"""Phase 2 acceptance: one agent solves one task end to end, unaided.

    uv run python scripts/run_agent.py --task word_freq
    uv run python scripts/run_agent.py --task word_freq --memory memory/seed.md

With no --memory this is a group-B run: the frozen task prompt and nothing else.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evolution.agent import CodingAgent, transcript_to_json  # noqa: E402
from evolution.config import load_config  # noqa: E402
from evolution.evaluator import Evaluator, compute_score  # noqa: E402
from evolution.llm import GeminiClient  # noqa: E402
from evolution.sandbox import Workspace, select_backend  # noqa: E402
from evolution.tasks import TaskRegistry  # noqa: E402


async def main_async(args: argparse.Namespace) -> int:
    config = load_config()
    task = TaskRegistry.load().get(args.task)

    memory_text = None
    if args.memory:
        memory_text = Path(args.memory).read_text(encoding="utf-8")

    backend = select_backend(config.sandbox)
    evaluator = Evaluator(backend)
    client = GeminiClient(config.api_key, config.api)
    agent = CodingAgent(client, config, evaluator=evaluator if args.run_tests else None)

    group = "A" if memory_text else "B"
    print(f"task={task.id} group={group} backend={backend.name} model={config.models.worker}")

    with Workspace.create(scaffold=task.scaffold_dir) as workspace:
        result = await agent.run(
            task, workspace, memory=memory_text, group=group, seed=config.seed
        )
        evaluation = evaluator.evaluate(workspace, task, network=config.sandbox.network)
        files = workspace.list_files()

        if args.save_transcript:
            out = Path(args.save_transcript)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(transcript_to_json(result), encoding="utf-8")
            print(f"transcript -> {out}")

    score = compute_score(
        evaluation.pass_fraction,
        total_tokens=result.usage.total,
        memory_tokens=len(memory_text or "") // 4,
        config=config.scoring,
    )

    print("\n--- agent ---")
    print(json.dumps(result.as_dict(), indent=2))
    print(f"files produced: {files}")
    print("\n--- evaluation ---")
    print(json.dumps(evaluation.as_dict(), indent=2))
    print(f"score: {score:.4f}")

    if result.violations:
        print(f"\nguardrail violations: {result.violations}")

    ok = evaluation.pass_fraction > 0
    print(f"\n{'PASS' if ok else 'FAIL'}: agent produced a scoring solution.")
    return 0 if ok else 1


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="word_freq")
    parser.add_argument("--memory", default=None, help="path to a memory file (group A)")
    parser.add_argument("--save-transcript", default=None)
    parser.add_argument(
        "--no-run-tests",
        dest="run_tests",
        action="store_false",
        help="withhold the run_tests tool from the agent",
    )
    parser.set_defaults(run_tests=True)
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
