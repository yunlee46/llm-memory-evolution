# llm-memory-evolution

Evolving a persistent memory file for an LLM coding agent — and measuring whether it
actually helps.

Each generation, 10 agents attempt the same build task. **Group A** gets the latest
memory file; **Group B** gets only the frozen task prompt. A curator model reads the
transcripts and edits the memory file into the next generation. Group B is a live
control, so the gap between the groups is the primary result:

```
lift_g = mean(score | group A) − mean(score | group B)
```

The full design, including failure modes and the build plan, is in **[AGENTS.md](AGENTS.md)**.

## Setup

```bash
uv sync --extra dev
cp .env.example .env        # then add your GEMINI_API_KEY
```

Resolve model IDs against your key rather than guessing at them, then paste the exact
strings into `config/experiment.yaml`:

```bash
uv run python scripts/smoke_test.py --list-models
uv run python scripts/smoke_test.py          # Phase 0 acceptance
```

Scoring runs in a container. Build it once:

```bash
docker build -f docker/Dockerfile.runner -t llm-memory-evolution-runner:latest .
```

Without a running Docker daemon the sandbox falls back to a local subprocess backend,
which **cannot enforce network denial** — fine for development, not for a scored run.

## Checks

```bash
uv run pytest                                # harness tests (offline)
uv run python scripts/validate_tasks.py      # declared vs. real test counts
uv run python scripts/run_agent.py --task word_freq   # one live agent run
```

## Status

| Phase | | |
|---|---|---|
| 0 | Scaffolding, config, Gemini client | done |
| 1 | Sandbox, task registry, evaluator | done |
| 2 | Agent loop with tools | done |
| 3 | Task calibration | 7 tasks authored + validated; **runs blocked on daily quota** |
| 4–9 | Memory file, orchestrator, curator, harvest, compaction, analysis | not started |

## Free-tier constraint

The worker model is capped at **20 API requests/day**, so `config/experiment.yaml` is
tuned for that, not for the design defaults. Calibration is resumable — run it once a
day and it picks up where it left off:

```bash
uv run python scripts/calibrate.py --tasks expr_eval,semver,csv_parse,lru_ttl,glob_match
```

It stops cleanly on quota (exit 2) and saves progress to
`results/calibration/trials.jsonl`. See **AGENTS.md §13** for the request arithmetic
and for what this design cannot demonstrate.

## Layout

```
config/experiment.yaml   all knobs; no magic numbers in code
tasks/                   build tasks + their test suites
memory/                  seed memory file and per-run lineages
src/evolution/           the harness
results/<run_id>/        run logs, changelogs — the experiment record
```
