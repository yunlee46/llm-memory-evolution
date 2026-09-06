"""Kalshi KXMLBGAME settled markets + pregame candlestick snapshot.

Public read endpoints (no auth): https://docs.kalshi.com/getting_started/historical_data
All responses are cached under data/raw/kalshi/. Kalshi's data terms forbid redistribution:
keep this cache local (it is gitignored).
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from ..team_map import KALSHI

BASE = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = "KXMLBGAME"
_TICKER = re.compile(r"^KXMLBGAME-(\d{2}[A-Z]{3}\d{2})(\d{4})?([A-Z]+?)(G\d)?-([A-Z]+)$")
_MON = {m: i for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}


class Client:
    def __init__(self, raw_dir: Path, pause: float = 0.03):
        self.raw = raw_dir
        self.raw.mkdir(parents=True, exist_ok=True)
        self.s = requests.Session()
        self.pause = pause

    def get(self, path: str, params: dict | None = None, cache: str | None = None) -> dict | None:
        if cache:
            f = self.raw / cache
            if f.exists():
                return json.loads(f.read_text())
        for attempt in range(12):
            try:
                r = self.s.get(BASE + path, params=params, timeout=60)
            except requests.RequestException:
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code == 404:
                data = None
                break
            if r.status_code == 429 or r.status_code >= 500:
                wait = float(r.headers.get("Retry-After", 0) or 0) or min(30, 1.5 * (attempt + 1))
                time.sleep(wait)
                continue
            r.raise_for_status()
            data = r.json()
            break
        else:
            raise RuntimeError(f"kalshi GET {path} failed after retries")
        time.sleep(self.pause)
        if cache:
            (self.raw / cache).write_text(json.dumps(data))
        return data

    def iter_markets(self):
        """Yield every settled KXMLBGAME market: historical tier first, then live tier."""
        for path, params, tag in (("/historical/markets", {"series_ticker": SERIES, "limit": 1000}, "hist"),
                                  ("/markets", {"series_ticker": SERIES, "status": "settled", "limit": 1000}, "live")):
            cursor = None
            page = 0
            while True:
                p = dict(params)
                if cursor:
                    p["cursor"] = cursor
                # live pages change as games settle; only cache historical pages
                cache = f"markets_{tag}_{page:04d}.json" if tag == "hist" else None
                data = self.get(path, p, cache=cache) or {}
                for m in data.get("markets", []):
                    yield m
                cursor = data.get("cursor")
                page += 1
                if not cursor:
                    break

    def candles(self, ticker: str, start_ts: int, end_ts: int, period: int = 60) -> list[dict]:
        params = {"start_ts": start_ts, "end_ts": end_ts, "period_interval": period}
        data = self.get(f"/historical/markets/{ticker}/candlesticks", params, cache=f"candles_{ticker}_{period}.json")
        if not data or not data.get("candlesticks"):
            # post-cutoff markets live on the series endpoint (different key names: *_dollars, volume_fp)
            data = self.get(f"/series/{SERIES}/markets/{ticker}/candlesticks", params, cache=f"live_candles_{ticker}_{period}.json")
        return (data or {}).get("candlesticks", [])


def parse_ticker(ticker: str) -> dict | None:
    m = _TICKER.match(ticker)
    if not m:
        return None
    d, hhmm, teams, g, side = m.groups()
    side = KALSHI.get(side)
    if side is None:
        return None
    # split concatenated AWAY+HOME using the side team we know
    away = home = None
    for code in KALSHI:
        if teams == code + side and KALSHI[code] != side:
            away, home = KALSHI[code], side
        elif teams == side + code and KALSHI[code] != side:
            away, home = side, KALSHI[code]
    # the loop above matches raw codes; also try raw side code
    if away is None:
        raw_side = m.group(5)
        if teams.startswith(raw_side) and teams[len(raw_side):] in KALSHI:
            away, home = side, KALSHI[teams[len(raw_side):]]
        elif teams.endswith(raw_side) and teams[:-len(raw_side)] in KALSHI:
            away, home = KALSHI[teams[:-len(raw_side)]], side
    if away is None or home is None:
        return None
    date = pd.Timestamp(year=2000 + int(d[:2]), month=_MON[d[2:5]], day=int(d[5:7]))
    return {"date": date, "away": away, "home": home, "side": side, "hhmm": hhmm,
            "game_num": int(g[1]) if g else 0}


def load_markets(raw_dir: Path) -> pd.DataFrame:
    """One row per settled team market with parsed game keys and first-pitch estimate."""
    c = Client(raw_dir)
    rows, skipped = [], {"unparsed": 0, "scalar": 0, "novolume": 0}
    for m in c.iter_markets():
        p = parse_ticker(m["ticker"])
        if p is None:
            skipped["unparsed"] += 1
            continue
        if m.get("result") not in ("yes", "no"):
            skipped["scalar"] += 1
            continue
        if float(m.get("volume_fp", 0) or 0) <= 0:
            skipped["novolume"] += 1
            continue
        exp = pd.Timestamp(m["expected_expiration_time"])
        rows.append({**p, "ticker": m["ticker"], "event_ticker": m["event_ticker"], "result": m["result"],
                     "won": 1 if m["result"] == "yes" else 0, "volume": float(m["volume_fp"]),
                     "open_time": pd.Timestamp(m["open_time"]), "first_pitch_ts": exp - pd.Timedelta(hours=3)})
    print(f"kalshi: {len(rows)} usable team markets; skipped {skipped}")
    return pd.DataFrame(rows)


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _ohlc(d: dict | None, key: str) -> float:
    """Candle sub-dict value; historical tier uses `close`, live tier `close_dollars`."""
    d = d or {}
    return _f(d.get(key, d.get(f"{key}_dollars")))


def snapshot(c: Client, ticker: str, first_pitch: pd.Timestamp, hours: int = 6) -> dict:
    """Last hourly candle ending at or before first pitch (within `hours` before it)."""
    fp = int(first_pitch.timestamp())
    cs = c.candles(ticker, fp - hours * 3600, fp + 1, 60)
    cs = [x for x in cs if int(x["end_period_ts"]) <= fp]
    if not cs:
        return {"yes_bid": float("nan"), "yes_ask": float("nan"), "last": float("nan"),
                "pre_volume": float("nan"), "snapshot_ts": pd.NaT}
    x = cs[-1]
    vol = float(np.nansum([_f(y.get("volume", y.get("volume_fp", 0))) for y in cs]))
    return {"yes_bid": _ohlc(x.get("yes_bid"), "close"), "yes_ask": _ohlc(x.get("yes_ask"), "close"),
            "last": _ohlc(x.get("price"), "close"), "pre_volume": vol,
            "snapshot_ts": pd.Timestamp(int(x["end_period_ts"]), unit="s", tz="UTC")}


def add_snapshots(markets: pd.DataFrame, raw_dir: Path, threads: int = 8) -> pd.DataFrame:
    """Fetch the pregame snapshot for every market (cache-first, a few requests in flight at once)."""
    from concurrent.futures import ThreadPoolExecutor

    n = len(markets)
    rows = list(markets.itertuples(index=False))
    done = [0]

    def one(r):
        c = _thread_client(raw_dir)
        out = snapshot(c, r.ticker, r.first_pitch_ts)
        done[0] += 1
        if done[0] % 500 == 0:
            print(f"  kalshi snapshots {done[0]}/{n}", flush=True)
        return out

    with ThreadPoolExecutor(max_workers=threads) as ex:
        out = list(ex.map(one, rows))
    snap = pd.DataFrame(out, index=markets.index)
    return pd.concat([markets, snap], axis=1)


_local = __import__("threading").local()


def _thread_client(raw_dir: Path) -> Client:
    if getattr(_local, "client", None) is None:
        _local.client = Client(raw_dir, pause=0.0)
    return _local.client
