"""Fitness checks for a one-shot MLB win-probability model.py.

Contract used by evolve.fitness:
    setup_worker(fcfg) -> ctx          (once per worker thread: loads labels/prices, resolves folds)
    run(ctx, artifact_path, fcfg) -> list[CheckResult]
    teardown_worker(ctx)

The candidate runs in a subprocess (sandbox_main.py) with the network disabled and a wall-clock
timeout; predictions come back as .npy files and every metric is computed here against
games.parquet, which the candidate never sees.
"""
from __future__ import annotations

import ast
import json
import os
import platform
import random
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from evolve.fitness import CheckResult  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
SANDBOX_MAIN = Path(__file__).with_name("sandbox_main.py")
LABELS = ["home_score", "away_score", "home_win"]
BANNED_MODULES = {"socket", "urllib", "requests", "http", "subprocess", "multiprocessing", "ctypes", "pickle",
                  "threading", "asyncio", "importlib", "shutil", "pathlib", "aiohttp", "httpx"}
ALLOWED_TOP = {"numpy", "pandas", "scipy", "sklearn", "math", "statistics", "itertools", "functools", "collections",
               "typing", "dataclasses", "datetime", "re", "warnings", "random", "operator", "numbers", "abc", "enum",
               "copy", "heapq", "bisect", "string", "decimal", "fractions", "__future__", "time"}


@dataclass
class Ctx:
    data_dir: Path
    games: pd.DataFrame            # indexed by game_id: labels + prices
    folds: list[dict]
    scratch: Path = field(default_factory=lambda: Path(tempfile.mkdtemp(prefix="mlb_eval_")))


# ---------------------------------------------------------------- worker lifecycle

def setup_worker(fcfg: dict[str, Any]) -> Ctx:
    data_dir = ROOT / fcfg["data_dir"]
    games = pd.read_parquet(data_dir / "games.parquet").set_index("game_id")
    folds = json.loads((data_dir / "folds.json").read_text())
    return Ctx(data_dir=data_dir, games=games, folds=folds)


def teardown_worker(ctx: Ctx) -> None:
    shutil.rmtree(ctx.scratch, ignore_errors=True)


def select_folds(fcfg: dict[str, Any], folds: list[dict]) -> list[dict]:
    """Which folds this evaluation uses: explicit `folds` ("all", "holdout", "3,7") or a per-generation rotation."""
    eval_folds = [f for f in folds if f["k"] != "holdout" and f["n"] > 0]
    sel = fcfg.get("folds")
    if sel in (None, "", "auto"):
        gen = int(fcfg.get("generation", 0))
        k = min(int(fcfg.get("folds_per_eval", 2)), len(eval_folds))
        rng = random.Random(int(fcfg.get("fold_seed", 0)) + gen)
        chosen = sorted(rng.sample(range(len(eval_folds)), k))
        return [eval_folds[i] for i in chosen]
    if sel == "all":
        return eval_folds
    if sel == "holdout":
        return [f for f in folds if f["k"] == "holdout"]
    wanted = {s.strip() for s in str(sel).split(",")}
    return [f for f in folds if str(f["k"]) in wanted]


# ---------------------------------------------------------------- static checks

def static_checks(src: str) -> tuple[list[CheckResult], bool]:
    results: list[CheckResult] = []
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        results.append(CheckResult("syntax_ok", 1, False, f"SyntaxError: {e}"))
        results.append(CheckResult("imports_allowed", 1, False, "not earned"))
        results.append(CheckResult("defines_predict", 1, False, "not earned"))
        return results, False
    results.append(CheckResult("syntax_ok", 1, True))

    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                top = a.name.split(".")[0]
                if top in BANNED_MODULES or top not in ALLOWED_TOP:
                    bad.append(a.name)
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if top in BANNED_MODULES or top not in ALLOWED_TOP:
                bad.append(node.module or "?")
        elif isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else "")
            if name in {"open", "system", "popen", "exec", "eval", "__import__"}:
                bad.append(f"call:{name}")
            if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) and fn.value.id in {"pd", "pandas"} \
                    and name.startswith(("read_", "to_")) and name not in {"to_numpy", "to_datetime", "to_numeric", "to_timedelta"}:
                bad.append(f"call:pd.{name}")
    results.append(CheckResult("imports_allowed", 1, not bad, ", ".join(sorted(set(bad)))[:300]))

    has = any(isinstance(n, ast.FunctionDef) and n.name == "predict" and len(n.args.args) >= 2 for n in tree.body)
    results.append(CheckResult("defines_predict", 1, has, "" if has else "no top-level def predict(train, test)"))
    return results, (not bad) and has


# ---------------------------------------------------------------- sandbox

@dataclass
class SandboxResult:
    ok: bool
    detail: str
    preds: dict[str, np.ndarray]
    repeat: np.ndarray | None
    network_attempt: bool
    elapsed: dict[str, float]


def _limits_factory(timeout_s: int, memory_mb: int) -> Callable[[], None]:
    def _limits() -> None:
        cpu = int(timeout_s * 3)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        if platform.system() == "Linux" and memory_mb:
            b = int(memory_mb) * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (b, b))
    return _limits


def run_sandbox(model_path: Path, folds: list[dict], ctx: Ctx, fcfg: dict[str, Any]) -> SandboxResult:
    workdir = Path(tempfile.mkdtemp(prefix="cand_", dir=ctx.scratch))
    shutil.copy(model_path, workdir / "model.py")
    fold_dir = ctx.data_dir / "folds"
    job = {"model": str(workdir / "model.py"), "repeat": str(folds[0]["k"]) if folds else None,
           "folds": [{"k": str(f["k"]), "train": str(fold_dir / f"fold_{f['k']}_train.parquet"),
                      "test": str(fold_dir / f"fold_{f['k']}_test.parquet"), "out": f"preds_{f['k']}.npy"} for f in folds]}
    (workdir / "job.json").write_text(json.dumps(job))
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(workdir), "TMPDIR": str(workdir), "PYTHONHASHSEED": "0",
           "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "VECLIB_MAXIMUM_THREADS": "2",
           "MPLBACKEND": "Agg", "http_proxy": "http://127.0.0.1:9", "https_proxy": "http://127.0.0.1:9", "no_proxy": ""}
    timeout = int(fcfg.get("timeout_s", 240))
    try:
        proc = subprocess.Popen([sys.executable, "-s", "-B", str(SANDBOX_MAIN), "job.json"], cwd=workdir, env=env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True,
                                preexec_fn=_limits_factory(timeout, int(fcfg.get("memory_mb", 0) or 0)))
        try:
            _, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            return SandboxResult(False, f"timeout after {timeout}s", {}, None, (workdir / "network_attempt").exists(), {})
    except Exception as e:  # noqa: BLE001
        return SandboxResult(False, f"sandbox launch failed: {type(e).__name__}: {e}", {}, None, False, {})
    status_path = workdir / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"ok": False, "error": "no status.json"}
    net = (workdir / "network_attempt").exists()
    if not status.get("ok"):
        tail = (status.get("traceback") or err.decode(errors="replace"))[-600:]
        return SandboxResult(False, f"{status.get('error', 'crashed')} | {tail}".strip(), {}, None, net, status.get("elapsed", {}))
    preds: dict[str, np.ndarray] = {}
    for f in job["folds"]:
        p = workdir / f["out"]
        preds[f["k"]] = np.load(p) if p.exists() else np.array([])
    rp = workdir / f"preds_{job['repeat']}_repeat.npy"
    repeat = np.load(rp) if rp.exists() else None
    return SandboxResult(True, "", preds, repeat, net, status.get("elapsed", {}))


# ---------------------------------------------------------------- metrics

def _logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def _pointwise_logloss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _tstat(x: np.ndarray) -> float:
    """mean / standard error; 0 when there is nothing to measure."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return 0.0
    sd = float(np.std(x, ddof=1))
    return float(np.mean(x) / (sd / np.sqrt(len(x)))) if sd > 0 else 0.0


def _sigmoid(t: float) -> float:
    return float(1.0 / (1.0 + np.exp(-t)))


def metrics(p: np.ndarray, y: np.ndarray, p_market: np.ndarray, p_naive: float, skill_floor: float = 0.005) -> dict[str, float]:
    """Pooled accuracy metrics. Skill measures are computed on the rows that have a market price so
    the model, the naive rate and the market are compared on identical games."""
    ok = ~np.isnan(p_market)
    out = {
        "logloss": _logloss(p, y), "brier": _brier(p, y), "ece": _ece(p, y), "pred_std": float(np.std(p)),
        "logloss_naive": _logloss(np.full_like(p, p_naive), y), "brier_naive": _brier(np.full_like(p, p_naive), y),
        "n": int(len(p)), "n_market": int(ok.sum()),
        "logloss_market": float("nan"), "logloss_on_market_rows": float("nan"), "logloss_naive_market_rows": float("nan"),
        "skill_frac": 0.0, "market_t": 0.0,
    }
    if ok.any():
        lm, lp = _pointwise_logloss(p_market[ok], y[ok]), _pointwise_logloss(p[ok], y[ok])
        ln = _pointwise_logloss(np.full(int(ok.sum()), p_naive), y[ok])
        out["logloss_market"], out["logloss_on_market_rows"], out["logloss_naive_market_rows"] = float(lm.mean()), float(lp.mean()), float(ln.mean())
        # share of the market's edge over the naive rate that the model captures (0 = naive, 1 = market)
        out["skill_frac"] = float((ln.mean() - lp.mean()) / max(ln.mean() - lm.mean(), skill_floor))
        # paired per-game comparison with the market: positive = model better
        out["market_t"] = _tstat(lm - lp)
    return out


def backtest(p: np.ndarray, ask_h: np.ndarray, ask_a: np.ndarray, y: np.ndarray, dates: np.ndarray,
             fold_ids: np.ndarray, st: dict[str, Any]) -> dict[str, Any]:
    """Fixed flat-bankroll rule: buy the side whose model edge over the ask exceeds min_edge."""
    lo, hi = st.get("min_price", 0.03), st.get("max_price", 0.97)
    priced = ~np.isnan(ask_h) & ~np.isnan(ask_a) & (ask_h >= lo) & (ask_h <= hi) & (ask_a >= lo) & (ask_a <= hi)
    edge_h, edge_a = p - ask_h, (1 - p) - ask_a
    take_home = edge_h >= edge_a
    edge = np.where(take_home, edge_h, edge_a)
    ask = np.where(take_home, ask_h, ask_a)
    won = np.where(take_home, y, 1 - y)
    prob = np.where(take_home, p, 1 - p)
    bet = priced & (edge > st.get("min_edge", 0.02))
    kelly = np.clip((prob - ask) / np.clip(1 - ask, 1e-9, None), 0, None)
    stake = np.minimum(st.get("kelly_fraction", 0.25) * kelly, st.get("max_stake", 0.05))
    n = np.where(bet, stake / np.clip(ask, 1e-9, None), 0.0)
    fee = st.get("fee_rate", 0.035) * n * ask * (1 - ask)
    pnl = np.where(bet, n * (won - ask) - fee, 0.0)
    order = np.argsort(dates, kind="stable")
    cum = np.cumsum(pnl[order])
    dd = float(np.max(np.maximum.accumulate(np.concatenate([[0.0], cum])) - np.concatenate([[0.0], cum])))
    staked = np.where(bet, stake, 0.0)
    total_stake = float(staked.sum())
    # per-fold ROI (pnl / stake in that fold); drawdown as a fraction of total turnover so it is
    # comparable between a model that bets 50 games and one that bets 700
    by_fold = {str(k): (float(pnl[fold_ids == k].sum() / staked[fold_ids == k].sum()) if staked[fold_ids == k].sum() > 0 else 0.0)
               for k in np.unique(fold_ids)}
    # only folds where the rule actually bet carry a per-fold ROI
    by_fold = {k: v for k, v in by_fold.items() if staked[fold_ids == k].sum() > 0}
    return {"n_bets": int(bet.sum()), "n_priced": int(priced.sum()), "total_stake": total_stake,
            "total_pnl": float(pnl.sum()), "roi": float(pnl.sum() / total_stake) if total_stake > 0 else 0.0,
            "max_drawdown": float(dd / total_stake) if total_stake > 0 else 0.0, "roi_by_fold": by_fold,
            "pnl_t": _tstat(pnl[bet])}


# ---------------------------------------------------------------- run

def _guard(results: list[CheckResult], name: str, weight: float, fn: Callable[[], tuple[bool, str]]) -> bool:
    try:
        ok, detail = fn()
    except Exception as e:  # noqa: BLE001
        ok, detail = False, f"{type(e).__name__}: {e}"
    results.append(CheckResult(name, weight, bool(ok), str(detail)[:300]))
    return bool(ok)


def _graded(results: list[CheckResult], name: str, weight: float, fn: Callable[[], tuple[float, str]]) -> float:
    """A check with partial credit: fn returns (score in [0, 1], detail). `passed` means at least half credit."""
    try:
        score, detail = fn()
        score = float(np.clip(score, 0.0, 1.0)) if np.isfinite(score) else 0.0
    except Exception as e:  # noqa: BLE001
        score, detail = 0.0, f"{type(e).__name__}: {e}"
    results.append(CheckResult(name, weight, score >= 0.5, str(detail)[:300], score=score))
    return score


def _not_earned(results: list[CheckResult], names: list[tuple[str, float]]) -> None:
    for n, w in names:
        results.append(CheckResult(n, w, False, "not earned"))


QUALITY = [("not_degenerate", 1), ("skill_vs_naive", 3), ("skill_vs_market", 3), ("calibration", 2),
           ("bet_coverage", 1), ("roi_tstat", 3), ("drawdown", 1), ("fold_consistency", 1)]


def run(ctx: Ctx, artifact_path: Path, fcfg: dict[str, Any]) -> list[CheckResult]:
    results: list[CheckResult] = []
    src = Path(artifact_path).read_text()
    static, ok = static_checks(src)
    results.extend(static)
    gates = [("runs_ok", 3), ("no_network", 1), ("output_shape", 1), ("probs_valid", 1), ("deterministic", 1)]
    if not ok:
        _not_earned(results, gates + QUALITY)
        return results

    folds = select_folds(fcfg, ctx.folds)
    sb = run_sandbox(Path(artifact_path), folds, ctx, fcfg)
    results.append(CheckResult("runs_ok", 3, sb.ok, sb.detail if not sb.ok else f"elapsed {sb.elapsed}"))
    results.append(CheckResult("no_network", 1, not sb.network_attempt, "network call attempted" if sb.network_attempt else ""))
    if not sb.ok:
        _not_earned(results, gates[2:] + QUALITY)
        return results

    # assemble predictions with labels/prices from games.parquet
    fold_dir = ctx.data_dir / "folds"
    parts = []
    shape_ok, shape_detail = True, ""
    for f in folds:
        test = pd.read_parquet(fold_dir / f"fold_{f['k']}_test.parquet", columns=["game_id", "date"])
        p = sb.preds.get(str(f["k"]), np.array([]))
        if p.ndim != 1 or len(p) != len(test) or not np.issubdtype(p.dtype, np.floating):
            shape_ok, shape_detail = False, f"fold {f['k']}: got shape {p.shape} dtype {p.dtype}, want ({len(test)},) float"
            break
        g = ctx.games.loc[test["game_id"].values]
        parts.append(pd.DataFrame({
            "p": p, "y": g["home_win"].values.astype(float), "pm": g["market_home_prob"].values.astype(float),
            "ask_h": g["kalshi_home_yes_ask"].values.astype(float), "ask_a": g["kalshi_away_yes_ask"].values.astype(float),
            "date": g["date"].values, "fold": str(f["k"]),
        }))
    results.append(CheckResult("output_shape", 1, shape_ok, shape_detail))
    if not shape_ok:
        _not_earned(results, gates[3:] + QUALITY)
        return results
    df = pd.concat(parts, ignore_index=True)
    p = df["p"].values

    valid = bool(np.all(np.isfinite(p)) and np.all(p > 0) and np.all(p < 1))
    results.append(CheckResult("probs_valid", 1, valid, "" if valid else f"min={np.nanmin(p):.3g} max={np.nanmax(p):.3g} nan={int(np.isnan(p).sum())}"))
    if not valid:
        _not_earned(results, gates[4:] + QUALITY)
        return results

    first = sb.preds[str(folds[0]["k"])]
    det = sb.repeat is not None and len(sb.repeat) == len(first) and bool(np.max(np.abs(sb.repeat - first)) < 1e-9)
    results.append(CheckResult("deterministic", 1, det, "" if det else "repeat call differed"))

    # ---- quality (graded: partial credit in [0, 1], pooled over the selected folds)
    th = fcfg.get("thresholds", {})
    st = fcfg.get("staking", {})
    train_rate = float(ctx.games.loc[ctx.games["date"] < pd.Timestamp(folds[0]["start"]), "home_win"].mean())
    m = metrics(p, df["y"].values, df["pm"].values, train_rate, th.get("skill_floor", 0.005))
    bt = backtest(p, df["ask_h"].values, df["ask_a"].values, df["y"].values, df["date"].values, df["fold"].values, st)
    ece_max, dd_max, disaster = th.get("ece_max", 0.05), th.get("max_drawdown", 0.10), th.get("fold_disaster", -0.10)
    min_bets = int(th.get("min_bets_abs", 20))
    enough = bt["n_bets"] >= min_bets
    too_few = f"bets={bt['n_bets']} < {min_bets}: no evidence"

    _guard(results, "not_degenerate", 1, lambda: (m["pred_std"] > th.get("min_pred_std", 0.03), f"std={m['pred_std']:.3f}"))
    # share of the market's log-loss edge over the constant home-win rate that the model captures
    _graded(results, "skill_vs_naive", 3, lambda: (m["skill_frac"],
            f"ll={m['logloss_on_market_rows']:.4f} naive={m['logloss_naive_market_rows']:.4f} market={m['logloss_market']:.4f} "
            f"captured={m['skill_frac']:+.2f} n={m['n_market']}"))
    # paired t-test of per-game log-loss vs the market: 0.5 = indistinguishable, ~0.88 at t=+2, ~0.12 at t=-2
    _graded(results, "skill_vs_market", 3, lambda: (_sigmoid(m["market_t"]),
            f"ll={m['logloss_on_market_rows']:.4f} market={m['logloss_market']:.4f} t={m['market_t']:+.2f}"))
    # full credit at ECE 0, half at ece_max, none at 2 x ece_max
    _graded(results, "calibration", 2, lambda: (1.0 - m["ece"] / (2 * ece_max), f"ece={m['ece']:.4f}"))
    # betting rule: linear credit up to min_bet_frac of priced games
    _graded(results, "bet_coverage", 1, lambda: (bt["n_bets"] / max(1.0, th.get("min_bet_frac", 0.10) * bt["n_priced"]),
            f"bets={bt['n_bets']} priced={bt['n_priced']}"))
    # t-statistic of per-bet profit after fees, half credit at t = roi_t_half: luck on a few hundred bets
    # (|t| < 1) earns less than half, a consistent edge (t > 2) earns most of it
    _graded(results, "roi_tstat", 3, lambda: ((_sigmoid(bt["pnl_t"] - th.get("roi_t_half", 1.0)), f"roi={bt['roi']:+.4f} pnl={bt['total_pnl']:+.4f} "
            f"stake={bt['total_stake']:.3f} t={bt['pnl_t']:+.2f} n={bt['n_bets']}") if enough else (0.0, too_few)))
    # full credit at no drawdown, half at max_drawdown, none at 2 x max_drawdown (fractions of turnover)
    _graded(results, "drawdown", 1, lambda: ((1.0 - bt["max_drawdown"] / (2 * dd_max), f"dd/turnover={bt['max_drawdown']:.4f}")
            if enough else (0.0, too_few)))
    # per-fold ROI: full credit at >= 0, none at fold_disaster, averaged over folds that were bet
    _graded(results, "fold_consistency", 1, lambda: ((float(np.mean([np.clip((v - disaster) / -disaster, 0, 1) for v in bt["roi_by_fold"].values()]))
            if bt["roi_by_fold"] else 0.0, " ".join(f"{k}:{v:+.3f}" for k, v in bt["roi_by_fold"].items())) if enough else (0.0, too_few)))
    return results
