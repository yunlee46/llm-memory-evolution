"""Build tasks/mlb_kalshi/data/games.parquet and the walk-forward fold files.

    python -m tasks.mlb_kalshi.fetch_data                # all stages
    python -m tasks.mlb_kalshi.fetch_data --stage build  # rebuild parquet/folds from cached raw data
    python -m tasks.mlb_kalshi.fetch_data --no-sbro      # skip sportsbook odds

Stages: retrosheet (outcomes 2005-2025) -> statsapi (2026 outcomes) -> sbro (closing lines 2010-2021,
optional) -> kalshi (settled KXMLBGAME markets + pregame snapshot) -> build.
Every raw download is cached under data/raw/. Kalshi data must stay local (see data/README.md).
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from evolve.config import ROOT, load_config

from .sources import kalshi, retrosheet, sbro, statsapi

LABELS = ["home_score", "away_score", "home_win"]
META = ["kalshi_ticker_home", "kalshi_ticker_away", "kalshi_snapshot_ts", "first_pitch_ts", "split", "fold",
        "source", "game_pk", "home_sp_id", "away_sp_id"]
FEATURES = ["game_id", "date", "season", "game_num", "day_night", "home", "away", "park", "home_sp", "away_sp",
            "close_home_ml", "close_away_ml", "market_home_prob",
            "kalshi_home_yes_bid", "kalshi_home_yes_ask", "kalshi_home_last",
            "kalshi_away_yes_bid", "kalshi_away_yes_ask", "kalshi_away_last",
            "kalshi_home_volume", "kalshi_away_volume"]


def stage_outcomes(raw: Path, first_season: int, last_retro: int) -> pd.DataFrame:
    rs = retrosheet.load_games(list(range(first_season, last_retro + 1)), raw / "retrosheet")
    print(f"retrosheet: {len(rs)} games {first_season}-{last_retro}")
    api = statsapi.load_games(date(last_retro + 1, 3, 1), date.today(), raw / "statsapi")
    print(f"statsapi: {len(api)} final games from {last_retro + 1}")
    g = pd.concat([rs, api], ignore_index=True)
    g["season"] = g["date"].dt.year.astype("int16")
    g["home_win"] = (g["home_score"] > g["away_score"]).astype("int8")
    g = g.drop_duplicates(["date", "away", "home", "game_num"], keep="first")
    g = g.sort_values(["date", "game_num", "home"]).reset_index(drop=True)
    g["game_id"] = (g["date"].dt.strftime("%Y-%m-%d") + "_" + g["away"] + "_" + g["home"] + "_"
                    + g["game_num"].astype(str))
    return g


def merge_sbro(g: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    if odds.empty:
        g["close_home_ml"] = np.nan
        g["close_away_ml"] = np.nan
        return g
    g = g.copy()
    g["dh_idx"] = g.groupby(["date", "home", "away"]).cumcount()
    m = g.merge(odds[["date", "home", "away", "dh_idx", "close_home_ml", "close_away_ml"]],
                on=["date", "home", "away", "dh_idx"], how="left")
    matched = m["close_home_ml"].notna()
    span = odds["date"].dt.year
    in_span = m["season"].between(span.min(), span.max())
    print(f"sbro: matched {matched.sum()}/{in_span.sum()} games in {span.min()}-{span.max()}")
    return m.drop(columns=["dh_idx"])


def merge_kalshi(g: pd.DataFrame, mk: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    for side in ("home", "away"):
        for col in ("ticker", "yes_bid", "yes_ask", "last", "volume"):
            g[f"kalshi_{side}_{col}"] = np.nan
        g[f"kalshi_{side}_ticker"] = g[f"kalshi_{side}_ticker"].astype(object)
    g["kalshi_snapshot_ts"] = pd.Series(pd.NaT, index=g.index, dtype="datetime64[ns, UTC]")
    g["first_pitch_ts"] = pd.Series(pd.NaT, index=g.index, dtype="datetime64[ns, UTC]")
    if mk.empty:
        return g
    idx = {}
    for i, r in g.iterrows():
        idx.setdefault((r["date"], r["away"], r["home"]), []).append(i)
    unmatched = 0
    matched_games = set()
    # pair the two team markets of each event
    for ev, grp in mk.groupby("event_ticker", sort=False):
        r0 = grp.iloc[0]
        key = (r0["date"], r0["away"], r0["home"])
        cands = idx.get(key, [])
        if not cands:
            unmatched += 1
            continue
        if len(cands) == 1:
            gi = cands[0]
        else:
            # doubleheader: 2026 tickers carry G1/G2; 2025 ones are ordered by first pitch
            gn = r0["game_num"]
            if gn == 0:
                order = sorted(mk[(mk["date"] == key[0]) & (mk["away"] == key[1]) & (mk["home"] == key[2])]
                               .groupby("event_ticker")["first_pitch_ts"].min().items(), key=lambda kv: kv[1])
                gn = [e for e, _ in order].index(ev) + 1
            by_num = {int(g.at[i, "game_num"]): i for i in cands}
            gi = by_num.get(gn)
            if gi is None:
                unmatched += 1
                continue
        if gi in matched_games:
            unmatched += 1
            continue
        matched_games.add(gi)
        g.at[gi, "first_pitch_ts"] = pd.Timestamp(r0["first_pitch_ts"]).tz_convert("UTC") if pd.Timestamp(r0["first_pitch_ts"]).tzinfo else pd.Timestamp(r0["first_pitch_ts"]).tz_localize("UTC")
        for _, m in grp.iterrows():
            side = "home" if m["side"] == m["home"] else "away"
            g.at[gi, f"kalshi_{side}_ticker"] = m["ticker"]
            g.at[gi, f"kalshi_{side}_yes_bid"] = m["yes_bid"]
            g.at[gi, f"kalshi_{side}_yes_ask"] = m["yes_ask"]
            g.at[gi, f"kalshi_{side}_last"] = m["last"]
            g.at[gi, f"kalshi_{side}_volume"] = m["pre_volume"]
            ts = pd.Timestamp(m["snapshot_ts"])
            g.at[gi, "kalshi_snapshot_ts"] = (ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")) if pd.notna(ts) else pd.NaT
    n_events = mk["event_ticker"].nunique()
    both = g["kalshi_home_yes_ask"].notna() & g["kalshi_away_yes_ask"].notna()
    print(f"kalshi: {n_events} events, {len(matched_games)} matched to games, {unmatched} unmatched; "
          f"{both.sum()} games with both asks")
    return g


def add_market_prob(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    p = pd.Series(np.nan, index=g.index)
    # sportsbook devig
    ph = sbro.american_to_prob(g["close_home_ml"])
    pa = sbro.american_to_prob(g["close_away_ml"])
    ok = ~np.isnan(ph) & ~np.isnan(pa)
    p[ok] = (ph / (ph + pa))[ok]
    # kalshi mid devig overrides where present
    mh = (g["kalshi_home_yes_bid"] + g["kalshi_home_yes_ask"]) / 2
    ma = (g["kalshi_away_yes_bid"] + g["kalshi_away_yes_ask"]) / 2
    ok = mh.notna() & ma.notna() & ((mh + ma) > 0)
    p[ok] = (mh / (mh + ma))[ok]
    g["market_home_prob"] = p.astype("float32")
    return g


def assign_folds(g: pd.DataFrame, eval_start: str, holdout_start: str, min_block: int = 250) -> tuple[pd.DataFrame, list[dict]]:
    """Eval era -> calendar-month folds (small partial months merge into their neighbour); holdout untouched."""
    g = g.copy()
    es, hs = pd.Timestamp(eval_start), pd.Timestamp(holdout_start)
    g["split"] = np.where(g["date"] < es, "train", np.where(g["date"] < hs, "eval", "holdout"))
    g["fold"] = -1
    ev = g[g["split"] == "eval"]
    counts = ev.groupby([ev["date"].dt.year, ev["date"].dt.month]).size()
    # month blocks; a small month merges into the next month of the same season (or the previous one)
    blocks: list[list[tuple[int, int]]] = []
    pending: list[tuple[int, int]] = []
    for (y, m), n in counts.items():
        if pending and pending[-1][0] != y:  # season changed: flush whatever was pending
            if blocks and blocks[-1][-1][0] == pending[0][0]:
                blocks[-1].extend(pending)
            else:
                blocks.append(pending)
            pending = []
        pending.append((y, m))
        if sum(counts[ym] for ym in pending) >= min_block:
            blocks.append(pending)
            pending = []
    if pending:
        if blocks and blocks[-1][-1][0] == pending[0][0]:
            blocks[-1].extend(pending)
        else:
            blocks.append(pending)
    folds = []
    for k, b in enumerate(blocks):
        mask = pd.Series(False, index=g.index)
        for y, m in b:
            mask |= (g["split"] == "eval") & (g["date"].dt.year == y) & (g["date"].dt.month == m)
        g.loc[mask, "fold"] = k
        sub = g[mask]
        folds.append({"k": k, "name": "/".join(f"{y}-{m:02d}" for y, m in b),
                      "start": str(sub["date"].min().date()), "end": str(sub["date"].max().date()), "n": int(len(sub))})
    ho = g[g["split"] == "holdout"]
    folds.append({"k": "holdout", "name": "holdout", "start": str(hs.date()),
                  "end": str(ho["date"].max().date()) if len(ho) else None, "n": int(len(ho))})
    return g, folds


def write_folds(g: pd.DataFrame, folds: list[dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    feat = [c for c in FEATURES if c in g.columns]
    for f in folds:
        if f["n"] == 0:
            continue
        start = pd.Timestamp(f["start"])
        train = g[g["date"] < start]
        test = g[g["fold"] == f["k"]] if f["k"] != "holdout" else g[g["split"] == "holdout"]
        train[feat + LABELS].reset_index(drop=True).to_parquet(out / f"fold_{f['k']}_train.parquet", index=False)
        test[feat].reset_index(drop=True).to_parquet(out / f"fold_{f['k']}_test.parquet", index=False)
    (out.parent / "folds.json").write_text(json.dumps(folds, indent=1))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--stage", choices=["all", "build"], default="all",
                    help="'build' rebuilds parquet + folds from the cached intermediate frames")
    ap.add_argument("--no-sbro", action="store_true")
    ap.add_argument("--first-season", type=int, default=2005)
    ap.add_argument("--last-retro", type=int, default=2025, help="last season with Retrosheet game logs")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    fc = cfg["fitness"]
    data = ROOT / fc["data_dir"]
    raw = data / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    if args.stage == "all":
        g = stage_outcomes(raw, args.first_season, args.last_retro)
        g.to_parquet(raw / "outcomes.parquet", index=False)
        odds = pd.DataFrame() if args.no_sbro else sbro.load_odds(list(range(max(2010, args.first_season), 2022)), raw / "sbro")
        odds.to_parquet(raw / "sbro.parquet", index=False)
        mk = kalshi.load_markets(raw / "kalshi")
        mk = kalshi.add_snapshots(mk, raw / "kalshi")
        mk.to_parquet(raw / "kalshi_markets.parquet", index=False)
    else:
        g = pd.read_parquet(raw / "outcomes.parquet")
        odds = pd.read_parquet(raw / "sbro.parquet") if (raw / "sbro.parquet").exists() else pd.DataFrame()
        mk = pd.read_parquet(raw / "kalshi_markets.parquet")

    g = merge_sbro(g, odds)
    g = merge_kalshi(g, mk)
    g = add_market_prob(g)
    g, folds = assign_folds(g, fc["eval_start"], fc["holdout_start"])
    g.to_parquet(data / "games.parquet", index=False)
    write_folds(g, folds, data / "folds")

    print("\nrows per season:")
    print(g.groupby("season").size().to_string())
    print("\nfolds:")
    for f in folds:
        print(f"  {f['k']:>7}  {f['name']:<16} {f['start']} .. {f['end']}  n={f['n']}")
    both = g["kalshi_home_yes_ask"].notna() & g["kalshi_away_yes_ask"].notna()
    ev = g["split"] != "train"
    print(f"\nkalshi-era games: {ev.sum()}, with both asks: {both.sum()} ({both.sum() / max(1, ev.sum()):.1%})")
    print(f"wrote {data / 'games.parquet'} ({len(g)} rows) and {len([f for f in folds if f['n']])} fold pairs")


if __name__ == "__main__":
    main()
