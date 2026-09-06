"""Retrosheet game logs -> normalised game rows.

Data: https://www.retrosheet.org/gamelogs/  (free, attribution required - see data/README.md)
Layout: https://www.retrosheet.org/gamelogs/glfields.txt (161 fields, no header)
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

from ..team_map import RETROSHEET, map_codes

URL = "https://www.retrosheet.org/gamelogs/gl{year}.zip"

# 0-indexed positions in the game log (glfields.txt is 1-indexed)
COLS = {
    "date": 0, "game_num": 1, "away_rs": 3, "home_rs": 6, "away_score": 9, "home_score": 10,
    "day_night": 12, "park": 16, "away_sp_id": 101, "away_sp": 102, "home_sp_id": 103, "home_sp": 104,
}


def download(year: int, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    dst = raw_dir / f"gl{year}.zip"
    if not dst.exists():
        r = requests.get(URL.format(year=year), timeout=60)
        r.raise_for_status()
        dst.write_bytes(r.content)
    return dst


def parse(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".txt"))
        raw = pd.read_csv(io.BytesIO(z.read(name)), header=None, dtype=str, keep_default_na=False)
    df = pd.DataFrame({k: raw.iloc[:, i] for k, i in COLS.items()})
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df["game_num"] = df["game_num"].astype(int)
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["home"] = map_codes(df["home_rs"], RETROSHEET, "retrosheet")
    df["away"] = map_codes(df["away_rs"], RETROSHEET, "retrosheet")
    df["day_night"] = df["day_night"].str.upper().str[:1]
    df["home_sp"] = df["home_sp"].str.strip()
    df["away_sp"] = df["away_sp"].str.strip()
    return df[["date", "game_num", "home", "away", "home_score", "away_score", "day_night", "park",
               "home_sp", "away_sp", "home_sp_id", "away_sp_id"]]


def load_games(seasons: list[int], raw_dir: Path) -> pd.DataFrame:
    frames = [parse(download(y, raw_dir)) for y in seasons]
    df = pd.concat(frames, ignore_index=True)
    df["source"] = "retrosheet"
    return df
