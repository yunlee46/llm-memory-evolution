# tasks/mlb_kalshi/data

Everything in this directory except this file is generated and **gitignored**. Rebuild with:

    python -m tasks.mlb_kalshi.fetch_data            # downloads + builds (10-20 min first time)
    python -m tasks.mlb_kalshi.fetch_data --stage build   # rebuild parquet/folds from cached raw data

Contents after a build:

- `raw/retrosheet/gl*.zip` - Retrosheet game logs. "The information used here was obtained free of
  charge from and is copyrighted by Retrosheet. Interested parties may contact Retrosheet at
  www.retrosheet.org."
- `raw/statsapi/*.json` - MLB Stats API schedule pages (2026 results, probable pitchers).
- `raw/sbro/*.xlsx` - sportsbookreviewsonline closing lines 2010-2021 (optional, `--no-sbro` to skip).
- `raw/kalshi/*.json` - Kalshi KXMLBGAME settled markets and hourly candlesticks.
  **Kalshi's Data Terms of Use forbid redistributing archived market data or using it to train
  models without written consent.** This cache is for personal backtesting only: never commit it,
  never copy it elsewhere, never bundle it into a dataset.
- `games.parquet` - one row per game with outcomes, closing lines, and the Kalshi pregame snapshot.
- `folds/fold_<k>_{train,test}.parquet`, `folds/holdout_{train,test}.parquet`, `folds.json` - walk-forward
  splits. Test files have the outcome columns physically removed.
