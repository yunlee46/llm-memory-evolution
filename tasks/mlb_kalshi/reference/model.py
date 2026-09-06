"""Reference win-probability model: logistic regression on market logit + Elo + starter/team form."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

K, HFA, REGRESS, BASE = 4.0, 24.0, 1 / 3, 1500.0


def _elo_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Sequential Elo over train (date order); test uses the frozen end-of-train ratings."""
    r: dict[str, float] = {}
    last_season: dict[str, int] = {}
    diffs = np.zeros(len(train))
    for i, (h, a, s, y) in enumerate(zip(train["home"].values, train["away"].values, train["season"].values,
                                          train["home_win"].values)):
        for t in (h, a):
            r.setdefault(t, BASE)
            if last_season.get(t, s) != s:
                r[t] = r[t] + (BASE - r[t]) * REGRESS
            last_season[t] = s
        d = r[h] - r[a] + HFA
        diffs[i] = d
        e = 1.0 / (1.0 + 10 ** (-d / 400.0))
        r[h] += K * (y - e)
        r[a] -= K * (y - e)
    tdiffs = np.array([r.get(h, BASE) - r.get(a, BASE) + HFA for h, a in zip(test["home"].values, test["away"].values)])
    return diffs, tdiffs


def _rolling_rate(keys: np.ndarray, wins: np.ndarray, window: int, prior_n: float) -> tuple[np.ndarray, dict[str, float]]:
    """Per-key rolling win rate over the previous `window` items, Laplace-smoothed; returns in-sample values + final."""
    hist: dict[str, list[float]] = {}
    out = np.zeros(len(keys))
    for i, (k, w) in enumerate(zip(keys, wins)):
        h = hist.setdefault(k, [])
        out[i] = (sum(h[-window:]) + 0.5 * prior_n) / (len(h[-window:]) + prior_n)
        if k:
            h.append(float(w))
    final = {k: (sum(h[-window:]) + 0.5 * prior_n) / (len(h[-window:]) + prior_n) for k, h in hist.items()}
    return out, final


def _features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    tr = train.sort_values("date", kind="stable").reset_index(drop=True)
    y = tr["home_win"].values.astype(float)
    elo_tr, elo_te = _elo_features(tr, test)

    hsp = tr["home_sp"].fillna("").astype(str).values
    asp = tr["away_sp"].fillna("").astype(str).values
    sp_h_tr, sp_final = _rolling_rate(np.concatenate([hsp, asp]), np.concatenate([y, 1 - y]), 30, 10.0)
    # interleave back: first half are home starters, second half away starters (both in date order)
    n = len(tr)
    sp_diff_tr = sp_h_tr[:n] - sp_h_tr[n:]
    sp_diff_te = np.array([sp_final.get(str(h or ""), 0.5) - sp_final.get(str(a or ""), 0.5)
                           for h, a in zip(test["home_sp"].fillna("").values, test["away_sp"].fillna("").values)])

    form_h, form_final = _rolling_rate(tr["home"].values, y, 30, 5.0)
    form_a, _ = _rolling_rate(tr["away"].values, 1 - y, 30, 5.0)
    # recompute team form over combined home/away appearances in date order
    team_keys = np.empty(2 * n, dtype=object)
    team_wins = np.empty(2 * n)
    team_keys[0::2], team_keys[1::2] = tr["home"].values, tr["away"].values
    team_wins[0::2], team_wins[1::2] = y, 1 - y
    tf, tf_final = _rolling_rate(team_keys, team_wins, 30, 5.0)
    form_diff_tr = tf[0::2] - tf[1::2]
    form_diff_te = np.array([tf_final.get(h, 0.5) - tf_final.get(a, 0.5) for h, a in zip(test["home"].values, test["away"].values)])

    def mk(frame, elo, spd, fd):
        pm = pd.to_numeric(frame["market_home_prob"], errors="coerce").values.astype(float)
        has = np.isfinite(pm)
        pmc = np.clip(np.where(has, pm, 0.5), 0.02, 0.98)
        logit = np.log(pmc / (1 - pmc))
        night = (frame["day_night"].fillna("").astype(str).values == "N").astype(float)
        return np.column_stack([logit, has.astype(float), elo / 100.0, spd, fd, night]), has

    Xtr, has_tr = mk(tr, elo_tr, sp_diff_tr, form_diff_tr)
    Xte, has_te = mk(test, elo_te, sp_diff_te, form_diff_te)
    return (Xtr, y, has_tr), (Xte, has_te)


def predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    np.random.seed(0)
    (Xtr, y, has_tr), (Xte, has_te) = _features(train, test)
    # market-aware model on rows with a market; Elo/form-only fallback for the rest
    full = LogisticRegression(C=1.0, max_iter=500, random_state=0).fit(Xtr[has_tr], y[has_tr]) if has_tr.sum() > 100 else None
    cols = [2, 3, 4, 5]
    base = LogisticRegression(C=1.0, max_iter=500, random_state=0).fit(Xtr[:, cols], y)
    p = base.predict_proba(Xte[:, cols])[:, 1]
    if full is not None and has_te.any():
        p[has_te] = full.predict_proba(Xte[has_te])[:, 1]
    return np.clip(p.astype(np.float64), 0.02, 0.98)
