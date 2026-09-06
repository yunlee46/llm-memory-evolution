Write a one-shot win-probability model for Major League Baseball games.

Deliver exactly one file, model.py, that defines:

    def predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray

`train` holds past games with outcomes. `test` holds later games WITHOUT the
outcome columns. Return a 1-D float64 array of length len(test), aligned with
test's row order, giving the probability that the HOME team wins each game.
Every value must be finite and strictly between 0 and 1.

Columns (both frames unless noted; NaN means unknown):
  game_id            str   unique id, e.g. "2025-04-16_NYY_BOS_0"
  date               datetime64  local game date
  season             int
  game_num           int   0 for a single game, 1/2 for a doubleheader
  day_night          str   "D" or "N"
  home, away         str   MLB team abbreviations, e.g. "NYY", "LAD", "CWS"
  park               str   ballpark id
  home_sp, away_sp   str   starting pitcher name ("First Last"); the probable
                           starter for the most recent season, may be empty
  close_home_ml      float sportsbook closing moneyline for the home team in
                           American odds (e.g. -135), 2010-2021 only
  close_away_ml      float same for the away team
  market_home_prob   float vig-free market probability of a home win taken
                           shortly before first pitch (sportsbook close before
                           2025, exchange mid-price from April 2025 onward);
                           NaN when no market existed
  kalshi_home_yes_bid, kalshi_home_yes_ask, kalshi_home_last
                     float in [0, 1]: pre-game bid, ask and last trade for the
                           "home team wins" contract (April 2025 onward, else NaN)
  kalshi_away_yes_bid, kalshi_away_yes_ask, kalshi_away_last
                     float same for the "away team wins" contract
  kalshi_home_volume, kalshi_away_volume
                     float contracts traded before the snapshot
  home_score, away_score, home_win
                     int   TRAIN ONLY. Not present in test.

Rules:
- Rows are in chronological order and every test row is later than every
  train row. Use only information that would be known before first pitch:
  the test outcomes are not provided, and later test rows must not inform
  earlier ones through anything derived from outcomes.
- Allowed imports: the Python standard library, numpy, pandas, scipy and
  scikit-learn. Nothing else is installed. No network access of any kind, no
  reading or writing of files, no subprocesses, no threads.
- predict is called three times in one process on different splits and must
  finish within 60 seconds per call on a laptop CPU limited to 2 threads.
  Train frames have tens of thousands of rows; test frames a few hundred.
- predict must be deterministic: set every random seed explicitly (numpy,
  scikit-learn random_state). Two calls with identical inputs must return
  identical outputs.
- Handle missing values: any column may contain NaN, strings may be empty, and
  the market columns are NaN for most of the training history.
- Predictions are scored by log-loss and calibration against the outcomes and
  by the profit of a fixed rule that buys a contract on the exchange whenever
  your probability exceeds its ask price by a margin, paying the exchange
  fee. Being confident and wrong is expensive; copying the market exactly
  earns nothing.

Respond with the complete model.py inside a single ```python code block and
nothing else.
