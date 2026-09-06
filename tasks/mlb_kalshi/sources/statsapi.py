"""MLB Stats API schedule -> results, first-pitch timestamps, probable pitchers (2025+, esp. 2026).

Public endpoint, no key: https://statsapi.mlb.com/api/v1/schedule
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from ..team_map import KALSHI, map_codes

URL = ("https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={start}&endDate={end}"
       "&gameType=R&hydrate=probablePitcher,team,linescore,decisions")
BOX_URL = "https://statsapi.mlb.com/api/v1/game/{pk}/boxscore"
# same eight per-team totals as sources/retrosheet.BOX, read from the box-score endpoint
BOX_KEYS = {"hits": ("batting", "hits"), "hr": ("batting", "homeRuns"), "bb": ("batting", "baseOnBalls"),
            "so": ("batting", "strikeOuts"), "er": ("pitching", "earnedRuns"), "err": ("fielding", "errors")}
# (team LOB comes from the schedule linescore; the box-score batting total is a per-batter sum)


def _fetch(start: date, end: date, raw_dir: Path) -> dict:
    raw_dir.mkdir(parents=True, exist_ok=True)
    dst = raw_dir / f"schedule_{start}_{end}.json"
    # never trust a cached window that ends in the future/today (games may not be final yet)
    if dst.exists() and end < date.today() - timedelta(days=1):
        return json.loads(dst.read_text())
    r = requests.get(URL.format(start=start, end=end), timeout=60)
    r.raise_for_status()
    dst.write_text(r.text)
    return r.json()


def _fetch_box(pk: int, raw_dir: Path) -> dict:
    """Per-team box-score totals for one final game (cached forever: a final box score does not change)."""
    box_dir = raw_dir / "boxscore"
    box_dir.mkdir(parents=True, exist_ok=True)
    dst = box_dir / f"{pk}.json"
    if dst.exists():
        data = json.loads(dst.read_text())
    else:
        r = requests.get(BOX_URL.format(pk=pk), timeout=60)
        r.raise_for_status()
        dst.write_text(r.text)
        data = r.json()
    out = {"game_pk": pk}
    for side in ("home", "away"):
        t = data.get("teams", {}).get(side, {})
        stats = t.get("teamStats", {})
        for k, (grp, key) in BOX_KEYS.items():
            v = stats.get(grp, {}).get(key)
            out[f"{side}_{k}"] = float(v) if v is not None else float("nan")
        out[f"{side}_pitchers"] = float(len(t.get("pitchers", []))) or float("nan")
    return out


def load_boxscores(game_pks: list[int], raw_dir: Path, threads: int = 8) -> pd.DataFrame:
    """Box-score totals for every game_pk (a few thousand small requests; the API is not throttled)."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=threads) as ex:
        rows = list(ex.map(lambda pk: _fetch_box(int(pk), raw_dir), game_pks))
    return pd.DataFrame(rows)


def load_games(start: date, end: date, raw_dir: Path) -> pd.DataFrame:
    rows = []
    cur = start
    while cur <= end:
        nxt = min(cur + timedelta(days=30), end)
        data = _fetch(cur, nxt, raw_dir)
        for d in data.get("dates", []):
            for g in d.get("games", []):
                if g.get("status", {}).get("abstractGameState") != "Final":
                    continue
                if g.get("status", {}).get("detailedState", "").startswith("Completed Early"):
                    pass
                h, a = g["teams"]["home"], g["teams"]["away"]
                if "score" not in h or "score" not in a:
                    continue
                rows.append({
                    "date": pd.Timestamp(d["date"]),
                    "game_pk": g["gamePk"],
                    "game_num": (g.get("gameNumber", 1) if g.get("doubleHeader", "N") != "N" else 0),
                    "home_api": h["team"]["abbreviation"], "away_api": a["team"]["abbreviation"],
                    "home_score": int(h["score"]), "away_score": int(a["score"]),
                    "day_night": g.get("dayNight", "")[:1].upper(),
                    "park": str(g.get("venue", {}).get("id", "")),
                    "home_sp": h.get("probablePitcher", {}).get("fullName", ""),
                    "away_sp": a.get("probablePitcher", {}).get("fullName", ""),
                    "first_pitch_ts": pd.Timestamp(g["gameDate"]),
                    "home_lob": float(g.get("linescore", {}).get("teams", {}).get("home", {}).get("leftOnBase", float("nan"))),
                    "away_lob": float(g.get("linescore", {}).get("teams", {}).get("away", {}).get("leftOnBase", float("nan"))),
                })
        cur = nxt + timedelta(days=1)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["home"] = map_codes(df["home_api"], KALSHI, "statsapi")
    df["away"] = map_codes(df["away_api"], KALSHI, "statsapi")
    df["source"] = "statsapi"
    box = load_boxscores(df["game_pk"].tolist(), raw_dir)
    df = df.merge(box, on="game_pk", how="left")
    print(f"statsapi: box scores for {box.dropna().shape[0]}/{len(df)} games")
    return df.drop(columns=["home_api", "away_api"]).sort_values(["date", "first_pitch_ts"]).reset_index(drop=True)
