"""Fitness = weighted check pass rate on a built artifact, minus MD lint penalties.

Task folders are pluggable: tasks/<name>/tests/checks.py must expose
    run(ctx, artifact_path: Path, fitness_cfg) -> list[CheckResult]
and may expose setup_worker(fitness_cfg) -> ctx / teardown_worker(ctx) for per-thread resources
(a browser, loaded datasets, ...).
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import ROOT, load_config


@dataclass
class CheckResult:
    name: str
    weight: float
    passed: bool
    detail: str = ""


@dataclass
class EvalResult:
    score: float
    checks: list[CheckResult] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"score": self.score, "error": self.error, "checks": [asdict(c) for c in self.checks]}


def load_checks_module(task_dir: Path):
    path = task_dir / "tests" / "checks.py"
    spec = importlib.util.spec_from_file_location(f"checks_{task_dir.name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def score_checks(checks: list[CheckResult]) -> float:
    total = sum(c.weight for c in checks)
    if total == 0:
        return 0.0
    return round(sum(c.weight for c in checks if c.passed) / total, 4)


def _worker_ctx(mod, fcfg):
    return mod.setup_worker(fcfg) if hasattr(mod, "setup_worker") else None


def _worker_close(mod, ctx):
    if ctx is not None and hasattr(mod, "teardown_worker"):
        mod.teardown_worker(ctx)


def evaluate_artifact(path: Path, cfg: dict[str, Any], ctx=None, mod=None) -> EvalResult:
    """Evaluate one built artifact. Creates (and tears down) its own worker context unless one is passed in."""
    task_dir = ROOT / cfg["task"]
    mod = mod or load_checks_module(task_dir)
    fcfg = cfg.get("fitness", {})
    own = ctx is None
    try:
        if own:
            ctx = _worker_ctx(mod, fcfg)
        try:
            checks = mod.run(ctx, Path(path).resolve(), fcfg)
            return EvalResult(score=score_checks(checks), checks=checks)
        finally:
            if own:
                _worker_close(mod, ctx)
    except Exception as e:  # a crashing artifact is a fitness of 0, not a harness failure
        return EvalResult(score=0.0, error=f"{type(e).__name__}: {e}")


evaluate_html = evaluate_artifact  # backwards-compatible name


def evaluate_many(paths: list[Path], cfg: dict[str, Any], workers: int | None = None) -> list[EvalResult]:
    """Evaluate several artifacts in parallel; each worker thread owns one task context."""
    workers = workers or cfg.get("fitness", {}).get("workers", 3)
    if not paths:
        return []
    task_dir = ROOT / cfg["task"]
    mod = load_checks_module(task_dir)
    fcfg = cfg.get("fitness", {})

    def worker(chunk: list[Path]) -> list[EvalResult]:
        out = []
        ctx = _worker_ctx(mod, fcfg)
        try:
            for path in chunk:
                out.append(evaluate_artifact(path, cfg, ctx=ctx, mod=mod))
        finally:
            _worker_close(mod, ctx)
        return out

    n = min(workers, len(paths))
    chunks = [paths[i::n] for i in range(n)]
    with ThreadPoolExecutor(max_workers=n) as ex:
        chunk_results = list(ex.map(worker, chunks))
    # re-interleave to original order
    results: list[EvalResult] = [None] * len(paths)  # type: ignore[list-item]
    for i, chunk in enumerate(chunk_results):
        for j, r in enumerate(chunk):
            results[i + j * n] = r
    return results


# ---------------------------------------------------------------- MD lint

_FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.S)


def md_lint(md: str, cfg: dict[str, Any]) -> tuple[float, list[str]]:
    """Penalise MD files that try to smuggle the solution in. Returns (penalty, reasons)."""
    lc = cfg.get("lint", {})
    penalty = 0.0
    reasons: list[str] = []

    long_blocks = [b for b in _FENCE.findall(md) if len(b.strip().splitlines()) > lc.get("max_code_block_lines", 8)]
    if long_blocks:
        p = min(len(long_blocks) * lc.get("long_code_block_penalty", 0.15), lc.get("long_code_block_cap", 0.45))
        penalty += p
        reasons.append(f"{len(long_blocks)} code block(s) over {lc.get('max_code_block_lines', 8)} lines (-{p:.2f})")

    words = re.findall(r"[a-zA-Z']+", md.lower())
    hits = sorted({w for w in words if w in set(lc.get("forbidden_keywords", []))})
    if hits:
        penalty += lc.get("keyword_penalty", 0.2)
        reasons.append(f"task keywords {hits} (-{lc.get('keyword_penalty', 0.2):.2f})")

    if len(words) > lc.get("max_words", 1500):
        penalty += lc.get("length_penalty", 0.1)
        reasons.append(f"{len(words)} words > {lc.get('max_words', 1500)} (-{lc.get('length_penalty', 0.1):.2f})")

    return round(penalty, 4), reasons


def main(argv: list[str]) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Score one or more built artifacts (model.py / index.html)")
    ap.add_argument("artifact", nargs="+")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--folds", default=None, help="mlb task: all | holdout | comma list of fold ids (default: rotation for --generation)")
    ap.add_argument("--generation", type=int, default=0, help="which generation's fold rotation to use")
    ap.add_argument("--headed", action="store_true", help="browser tasks: show the browser")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    fc = cfg.setdefault("fitness", {})
    fc["generation"] = args.generation
    if args.folds:
        fc["folds"] = args.folds
    if args.headed:
        fc["headless"] = False
    for h in args.artifact:
        r = evaluate_artifact(Path(h), cfg)
        print(f"\n{h}: score={r.score}" + (f"  ERROR {r.error}" if r.error else ""))
        for c in r.checks:
            print(f"  {'PASS' if c.passed else 'FAIL'} {c.name:<24} w={c.weight:<3} {c.detail}")


if __name__ == "__main__":
    main(sys.argv[1:])
