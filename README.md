# llm-memory-evolution

Evolve CLAUDE.md-style instruction files against a real fitness function. The task under study:
**one-shot an MLB win-probability model** whose predictions are backtested against Kalshi's
pre-game contract prices.

## How it works

1. A population of instruction files (`seeds/*.md`) is used as the **system prompt** for a builder
   model (`models.builder`). The user prompt is always `tasks/mlb_kalshi/spec.md`, which asks for a
   single `model.py` exposing `predict(train, test) -> P(home wins)`.
2. Each candidate `model.py` runs in a sandboxed subprocess (network disabled, wall-clock timeout,
   CPU limit) on walk-forward folds of real games. Its probabilities are scored by 17 weighted
   checks: binary gates (parses, allowed imports, runs, deterministic, valid probabilities) and
   graded quality checks with partial credit: share of the market's log-loss skill captured, a
   paired t-test against the Kalshi price, calibration, and a fixed betting rule (quarter-Kelly
   against the Kalshi ask, paying the taker fee) whose profit is judged by its t-statistic so luck
   on a few hundred bets earns little.
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

Sources: Retrosheet game logs 2005-2025 (outcomes, starters, per-team box-score totals), MLB Stats
API (2026 outcomes, probable pitchers, box scores), sportsbookreviewsonline closing moneylines
2010-2021 (optional training feature), and Kalshi `KXMLBGAME` settled markets with the last hourly
candle before first pitch (April 2025 onward). Output: `tasks/mlb_kalshi/data/games.parquet` plus
walk-forward fold files. Train frames carry sixteen train-only box-score columns (hits, HR, walks,
strikeouts, LOB, errors, pitchers used, earned runs for each side) so a model can build run-differential
or starter-quality features; test frames have every outcome column physically removed.
`--stage outcomes` re-parses the cached game logs and box scores without touching the Kalshi cache.

**Kalshi data stays on your machine.** Kalshi's data terms forbid redistribution and model
training on archived market data; the `data/` directory is gitignored for that reason.
Retrosheet: "The information used here was obtained free of charge from and is copyrighted by
Retrosheet. Interested parties may contact Retrosheet at www.retrosheet.org."

## Calibration (2026-09-06, all 9 folds, graded fitness)

| artifact | score | note |
|---|---|---|
| `reference/model.py` | 0.67 | tracks the Kalshi mid (log-loss 0.6802 vs market 0.6801, captures 99% of the market's skill over naive), never finds a 2% edge, so earns nothing from the six betting weights |
| first-run winner, build 1 (gradient boosting on market + encodings) | 0.76 | captures 73% of market skill, ECE 0.008, bets 54% of games at ROI −1.4% (t = −0.5) |
| first-run winner, build 0 (logistic on rolling win rates) | 0.65 | log-loss worse than the naive rate (t = −5.2 vs market) yet ROI +1.8% on 3008 bets (t = +0.8): the profit check gives that under half credit |
| `reference/broken.py` | 0.16 | crashes reading the outcome column |

The Kalshi MLB market is efficient against Elo, starter form and team form: every blend that
deviates from the market enough to bet loses about the fee. Under the old pass/fail checks the
same two winner builds scored 0.76 and 0.84 on the untouched holdout with their ROI signs flipped
relative to the evaluation folds; under the graded checks both score 0.74. The betting weights
(8 of 26) remain the open frontier, but a model can no longer collect them by luck.

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

| check | weight | credit |
|---|---|---|
| syntax_ok, imports_allowed, defines_predict | 1 each | static AST checks (binary) |
| runs_ok | 3 | sandbox finishes within `timeout_s` without error (binary) |
| no_network, output_shape, probs_valid, deterministic | 1 each | gates (binary) |
| not_degenerate | 1 | std(p) > 0.03 (binary) |
| skill_vs_naive | 3 | share of the market's log-loss improvement over the constant home-win rate that the model captures, on market-priced rows (0 = naive, 1 = market) |
| skill_vs_market | 3 | sigmoid of the paired per-game log-loss t-statistic vs the Kalshi mid: 0.5 = indistinguishable, 0.88 at t = +2 |
| calibration | 2 | 1 − ECE / (2 × `ece_max`): full at 0, half at 0.05 |
| bet_coverage | 1 | bets / (`min_bet_frac` × priced games), capped at 1 |
| roi_tstat | 3 | sigmoid(t − `roi_t_half`) of per-bet profit after fees; 0 below `min_bets_abs` bets |
| drawdown | 1 | 1 − dd / (2 × `max_drawdown`), dd as a fraction of turnover; 0 below `min_bets_abs` bets |
| fold_consistency | 1 | mean over bet folds of (ROI − `fold_disaster`) / −`fold_disaster`, clipped to [0, 1] |

All quality metrics are pooled over the `folds_per_eval` folds of that generation (four by default,
roughly 1,600 games), and a check's `passed` flag means at least half credit. `evolve.fitness`
prints the fractional credit for graded checks and PASS/FAIL for gates.

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
