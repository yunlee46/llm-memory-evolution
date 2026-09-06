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
                })
        cur = nxt + timedelta(days=1)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["home"] = map_codes(df["home_api"], KALSHI, "statsapi")
    df["away"] = map_codes(df["away_api"], KALSHI, "statsapi")
    df["source"] = "statsapi"
    return df.drop(columns=["home_api", "away_api"]).sort_values(["date", "first_pitch_ts"]).reset_index(drop=True)
