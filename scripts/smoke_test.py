"""Phase 0 acceptance: one prompt round-trips and token counts are recorded.

Also resolves model pinning empirically:

    python scripts/smoke_test.py --list-models

prints the model IDs this API key can actually reach. Paste the exact strings
into config/experiment.yaml rather than guessing at them (AGENTS.md §3).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evolution.config import load_config  # noqa: E402
from evolution.llm import GeminiClient  # noqa: E402

PROMPT = "Reply with exactly the word: pong"


def cmd_list_models(client: GeminiClient) -> int:
    models = [
        m
        for m in client.list_models()
        if not m["supported_actions"] or "generateContent" in m["supported_actions"]
    ]
    models.sort(key=lambda m: m["name"])
    print(f"{len(models)} models support generateContent:\n")
    for model in models:
        short = model["name"].removeprefix("models/")
        limits = f"in={model['input_token_limit']} out={model['output_token_limit']}"
        print(f"  {short:<45} {limits}")
    print("\nPaste the exact IDs above into config/experiment.yaml -> models.")
    return 0


async def cmd_roundtrip(client: GeminiClient, config) -> int:
    missing = client.validate_model_ids(
        [config.models.worker, config.models.curator, config.models.judge]
    )
    if missing:
        print(f"ERROR: model IDs not reachable with this key: {missing}")
        print("Run with --list-models to see valid IDs.")
        return 1

    response = await client.generate(
        model=config.models.worker,
        contents=PROMPT,
        role="worker",
        temperature=0.0,
        seed=config.seed,
    )

    print(f"model:         {response.model_id}")
    print(f"reply:         {response.text.strip()!r}")
    print(f"finish_reason: {response.finish_reason}")
    print(f"usage:         {json.dumps(response.usage.as_dict())}")
    print(f"ledger:        {json.dumps(client.ledger.as_dict())}")

    if response.usage.total <= 0:
        print("\nFAIL: no token usage recorded — cost analysis depends on this.")
        return 1
    print("\nPASS: prompt round-tripped and token counts recorded.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="list model IDs reachable with this API key, then exit",
    )
    parser.add_argument("--config", default=None, help="path to experiment.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    client = GeminiClient(config.api_key, config.api)

    if args.list_models:
        return cmd_list_models(client)

    placeholders = config.models.placeholders()
    if placeholders:
        print(f"ERROR: model IDs still unset in config: {placeholders}")
        print("Run with --list-models, then paste real IDs into config/experiment.yaml.")
        return 1

    return asyncio.run(cmd_roundtrip(client, config))


if __name__ == "__main__":
    raise SystemExit(main())
