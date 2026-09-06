"""sportsbookreviewsonline.com MLB odds archives (2010-2021) -> closing moneylines per game.

Best-effort: the archive format drifts across years. Failures here are logged, not fatal;
close_* columns simply stay NaN for seasons that could not be parsed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ..team_map import SBRO, map_codes

URL = "https://www.sportsbookreviewsonline.com/wp-content/uploads/sportsbookreviewsonline_com_737/mlb-odds-{year}.xlsx"


def download(year: int, raw_dir: Path) -> Path | None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    dst = raw_dir / f"mlb-odds-{year}.xlsx"
    if dst.exists():
        return dst
    r = requests.get(URL.format(year=year), timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    if r.status_code != 200 or not r.content[:2] == b"PK":
        print(f"  sbro {year}: not available (HTTP {r.status_code})")
        return None
    dst.write_bytes(r.content)
    return dst


def american_to_prob(ml: pd.Series) -> pd.Series:
    ml = pd.to_numeric(ml, errors="coerce")
    return np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))


def parse(path: Path, year: int) -> pd.DataFrame:
    df = pd.read_excel(path, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    need = {"Date", "VH", "Team", "Close"}
    if not need <= set(df.columns):
        raise ValueError(f"unexpected columns {list(df.columns)[:12]}")
    df = df[df["VH"].isin(["V", "H"])].reset_index(drop=True)
    d = df["Date"].str.strip().str.zfill(4)
    df["date"] = pd.to_datetime(str(year) + d, format="%Y%m%d", errors="coerce")
    df["team"] = map_codes(df["Team"], SBRO, "sbro")
    df["close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["pitcher"] = df.get("Pitcher", pd.Series([""] * len(df))).fillna("").str.strip()
    v = df[df["VH"] == "V"].reset_index(drop=True)
    h = df[df["VH"] == "H"].reset_index(drop=True)
    n = min(len(v), len(h))
    out = pd.DataFrame({"date": h["date"][:n], "home": h["team"][:n], "away": v["team"][:n],
                        "close_home_ml": h["close"][:n], "close_away_ml": v["close"][:n],
                        "sbro_home_pitcher": h["pitcher"][:n]})
    # doubleheader order within a day: keep appearance order
    out["dh_idx"] = out.groupby(["date", "home", "away"]).cumcount()
    return out.dropna(subset=["date"])


def load_odds(seasons: list[int], raw_dir: Path) -> pd.DataFrame:
    frames = []
    for y in seasons:
        p = download(y, raw_dir)
        if p is None:
            continue
        try:
            frames.append(parse(p, y))
            print(f"  sbro {y}: {len(frames[-1])} games")
        except Exception as e:  # noqa: BLE001
            print(f"  sbro {y}: parse failed ({type(e).__name__}: {e})")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["date", "home", "away", "close_home_ml", "close_away_ml", "sbro_home_pitcher", "dh_idx"])
