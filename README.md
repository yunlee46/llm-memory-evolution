# llm-memory-evolution

Evolve CLAUDE.md-style instruction files against a real fitness function. The task under study:
**one-shot an MLB win-probability model** whose predictions are backtested against Kalshi's
pre-game contract prices.

## How it works

1. A population of instruction files (`seeds/*.md`) is used as the **system prompt** for a builder
   model (`models.builder`). The user prompt is always `tasks/mlb_kalshi/spec.md`, which asks for a
   single `model.py` exposing `predict(train, test) -> P(home wins)`.
2. Each candidate `model.py` runs in a sandboxed subprocess (network disabled, wall-clock timeout,
   CPU limit) on walk-forward folds of real games. Its probabilities are scored by 19 weighted
   checks: gates (parses, allowed imports, runs, deterministic, valid probabilities), skill
   (log-loss/Brier vs a naive baseline and vs the Kalshi market, calibration), and a fixed
   betting rule (quarter-Kelly against the Kalshi ask, paying the taker fee) that must be
   profitable with a bounded drawdown.
3. Fitness = mean check score over `samples_per_eval` builds, minus a lint penalty that stops the
   instruction file from smuggling in the solution (long code blocks, domain keywords, length).
   Ties break toward fewer output tokens.
4. A different model (`models.mutator`) produces the next generation: elites are copied, the rest
   come from section-level crossover, blind section rewrites, and "informed" whole-file rewrites
   that see a random half of the per-check pass rates.

`00_empty.md` is the control: no system prompt at all.

## Setup

```bash
python3.11+ -m venv .venv && source .venv/bin/activate
pip install -e .                 # add `.[browser]` to keep the old cat_bounce task runnable
cp .env.example .env             # DEEPINFRA_API_KEY=...
python -m evolve.llm --ping
```

## Data

```bash
python -m tasks.mlb_kalshi.fetch_data              # ~10-20 min the first time
python -m tasks.mlb_kalshi.fetch_data --stage build  # rebuild folds from cached raw data
```

Sources: Retrosheet game logs 2005-2025 (outcomes, starters), MLB Stats API (2026 outcomes,
probable pitchers), sportsbookreviewsonline closing moneylines 2010-2021 (optional training
feature), and Kalshi `KXMLBGAME` settled markets with the last hourly candle before first pitch
(April 2025 onward). Output: `tasks/mlb_kalshi/data/games.parquet` plus walk-forward fold files.

**Kalshi data stays on your machine.** Kalshi's data terms forbid redistribution and model
training on archived market data; the `data/` directory is gitignored for that reason.
Retrosheet: "The information used here was obtained free of charge from and is copyrighted by
Retrosheet. Interested parties may contact Retrosheet at www.retrosheet.org."

## Calibration (2026-09-06, all 9 folds)

| artifact | score | note |
|---|---|---|
| `reference/model.py` | 0.72 | tracks the Kalshi mid almost exactly (log-loss 0.6802 vs market 0.6801), so it never finds a 2% edge and places no bets |
| first real DeepSeek build | 0.52 | bets on 80% of games, ROI −3.3% (roughly the fee), log-loss well behind the market |
| `reference/broken.py` | 0.16 | crashes reading the outcome column |

The Kalshi MLB market is efficient against Elo, starter form and team form: every blend that
deviates from the market enough to bet loses about the fee. The seven betting/market checks
(weight 8 of 26) are therefore the open frontier the evolution is pushing on, and 0.72 is the
"honest market-tracking" baseline, not a ceiling.

## Run

```bash
# calibrate: reference ~0.72, broken <= 0.2
python -m evolve.fitness --folds all tasks/mlb_kalshi/reference/model.py tasks/mlb_kalshi/reference/broken.py
python -m evolve.fitness --folds 0 tasks/mlb_kalshi/reference/model.py     # one fold
python -m evolve.run --mock -g 2 -p 4 -s 1        # harness smoke test, no API spend
python -m evolve.builder seeds/09_structured.md   # one real build, scored
python -m evolve.run                              # full run from config.yaml
python -m evolve.run --resume runs/<id>
python -m evolve.fitness --folds holdout runs/<id>/gen_05/<best>/sample_0.py   # untouched holdout
```

## Fitness details

- **Folds.** The Kalshi era (2025-04-16 onward) is cut into calendar-month folds; each generation
  evaluates every candidate on the same `folds_per_eval` folds chosen by `fold_seed + generation`,
  so the evolution cannot overfit one slice. Fold *k*'s train set is every game before it. Games
  from `holdout_start` onward are never used during evolution.
- **Staking.** For each game with both Kalshi asks in `[0.03, 0.97]`: take the side with the larger
  edge `p - ask`; bet only if edge > 2%; stake = min(¼ Kelly, 5%) of a flat 1.0 bankroll; fee =
  0.035 × contracts × ask × (1 − ask). Copying the market yields no bets and no profit.
- **Sandbox.** Subprocess with `socket`/`subprocess`/`os.system` monkey-patched, proxies pointed at
  a dead port, stripped environment, 2 BLAS threads, RLIMIT_CPU, process-group kill on timeout.
  Test files have the outcome columns physically removed. This stops honest mistakes and casual
  cheating, not a determined adversary.

| check | weight | passes when |
|---|---|---|
| syntax_ok, imports_allowed, defines_predict | 1 each | static AST checks |
| runs_ok | 3 | sandbox finishes within `timeout_s` without error |
| no_network, output_shape, probs_valid, deterministic | 1 each | gates |
| not_degenerate | 1 | std(p) > 0.03 |
| logloss_beats_naive / brier_beats_naive | 2 / 1 | beats constant home-win rate by margin |
| logloss_near_market / logloss_beats_market | 2 / 1 | within 0.005 of / better than the Kalshi mid |
| calibration_ece | 2 | 10-bin ECE < 0.05 |
| min_bets | 1 | bets on ≥ 10% of priced games |
| roi_positive / roi_stretch | 2 / 1 | ROI > 0 / > 3% after fees |
| max_drawdown | 1 | worst drawdown < 10% of total turnover |
| no_fold_disaster | 1 | no single fold has ROI below −10% |

## Layout

```
config.yaml              models, evolution params, lint, fitness (folds, staking, thresholds)
evolve/                  harness: run.py loop, builder.py, fitness.py, operators.py, population.py, analyze.py
seeds/                   generation-0 instruction files
tasks/mlb_kalshi/        spec.md, task.yaml, fetch_data.py, sources/, tests/checks.py, tests/sandbox_main.py, reference/
tasks/cat_bounce/        the previous browser task (needs `pip install -e .[browser]`)
runs/<id>/               gen_NN/<ind>/{agent.md, sample_S.py, result.json}, usage.json, report.md
```

## Adding a task

Create `tasks/<name>/` with `spec.md` (the user prompt), `task.yaml` (`artifact: {filename, lang}`,
`mock_reply`, `domain_hint`, `practice_hint`), `tests/checks.py` exposing
`run(ctx, artifact_path, fitness_cfg) -> list[CheckResult]` (plus optional
`setup_worker(fitness_cfg) -> ctx` / `teardown_worker(ctx)`), and `reference/` with a
near-perfect and a broken artifact. Point `task:` in `config.yaml` at it and update
`lint.forbidden_keywords`.
