"""Main evolutionary loop.

    python -m evolve.run                       # full run from config.yaml
    python -m evolve.run --mock -g 2 -p 4      # exercise the harness with stub LLM (no spend)
    python -m evolve.run --resume runs/<id>    # continue after an interruption
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from .builder import build, load_spec
from .config import ROOT, load_config
from .fitness import evaluate_many, md_lint
from .llm import LLM
from .population import Individual, next_generation, ranked

console = Console()


def load_seeds(cfg: dict[str, Any], n: int) -> list[Individual]:
    seeds_dir = ROOT / cfg["seeds_dir"]
    files = sorted(seeds_dir.glob("*.md"))
    if not files:
        raise SystemExit(f"no seed .md files in {seeds_dir}")
    files = files[:n]
    pop = [Individual(id=f"g00_i{i:02d}", md=f.read_text(), generation=0, operator=f"seed:{f.stem}")
           for i, f in enumerate(files)]
    while len(pop) < n:  # duplicate seeds if fewer than population
        src = pop[len(pop) % len(files)]
        pop.append(Individual(id=f"g00_i{len(pop):02d}", md=src.md, generation=0, operator=src.operator))
    return pop


async def evaluate_population(pop: list[Individual], gen_dir: Path, llm: LLM, cfg: dict[str, Any], gen: int = 0) -> None:
    spec = load_spec(cfg)
    n_samples = cfg["evolution"]["samples_per_eval"]
    ext = Path(cfg["artifact"]["filename"]).suffix

    async def build_one(ind: Individual, s: int) -> Path:
        d = gen_dir / ind.id
        d.mkdir(parents=True, exist_ok=True)
        out = d / f"sample_{s}{ext}"
        meta = d / f"sample_{s}.meta.json"
        if out.exists() and meta.exists():
            return out
        code, raw, ct = await build(llm, cfg, ind.md, spec)
        out.write_text(code)
        (d / f"sample_{s}.reply.txt").write_text(raw)
        meta.write_text(json.dumps({"completion_tokens": ct}))
        return out

    for ind in pop:
        (gen_dir / ind.id).mkdir(parents=True, exist_ok=True)
        (gen_dir / ind.id / "agent.md").write_text(ind.md)

    jobs = [(ind, s) for ind in pop for s in range(n_samples)]
    console.print(f"  building {len(jobs)} artifacts with {cfg['models']['builder']} ...")
    t0 = time.time()
    paths = await asyncio.gather(*(build_one(ind, s) for ind, s in jobs))
    console.print(f"  built in {time.time() - t0:.0f}s; evaluating ...")
    t0 = time.time()
    ecfg = {**cfg, "fitness": {**cfg.get("fitness", {}), "generation": gen}}
    results = await asyncio.to_thread(evaluate_many, list(paths), ecfg)
    console.print(f"  evaluated in {time.time() - t0:.0f}s")

    by_ind: dict[str, list] = {ind.id: [] for ind in pop}
    for (ind, s), r in zip(jobs, results):
        by_ind[ind.id].append(r)
    for ind in pop:
        rs = by_ind[ind.id]
        toks = []
        for s in range(n_samples):
            meta = gen_dir / ind.id / f"sample_{s}.meta.json"
            if meta.exists():
                toks.append(json.loads(meta.read_text()).get("completion_tokens", 0))
        ind.mean_completion_tokens = round(sum(toks) / len(toks), 1) if toks else 0.0
        ind.samples = [dict(r.to_dict(), completion_tokens=(toks[i] if i < len(toks) else None)) for i, r in enumerate(rs)]
        ind.test_score = round(sum(r.score for r in rs) / len(rs), 4)
        names = {c.name for r in rs for c in r.checks}
        # mean credit per check across samples (graded checks contribute partial credit)
        ind.check_rates = {n: round(sum(c.value for r in rs for c in r.checks if c.name == n) / len(rs), 3)
                           for n in sorted(names)}
        ind.lint_penalty, ind.lint_reasons = md_lint(ind.md, cfg)
        ind.fitness = round(max(0.0, ind.test_score - ind.lint_penalty), 4)
        (gen_dir / ind.id / "result.json").write_text(json.dumps(ind.to_dict(), indent=1))


def print_generation(gen: int, pop: list[Individual]) -> None:
    t = Table(title=f"Generation {gen}")
    for col in ("id", "fitness", "tests", "lint", "out tok", "operator", "parents", "words"):
        t.add_column(col)
    for ind in ranked(pop):
        t.add_row(ind.id, f"{ind.fitness:.3f}", f"{ind.test_score:.3f}", f"-{ind.lint_penalty:.2f}" if ind.lint_penalty else "",
                  f"{ind.mean_completion_tokens:.0f}", ind.operator[:40], ",".join(ind.parents), str(len(ind.md.split())))
    console.print(t)


def save_generation(gen_dir: Path, pop: list[Individual]) -> None:
    (gen_dir / "generation.json").write_text(json.dumps([i.to_dict() for i in pop], indent=1))


def load_generation(gen_dir: Path) -> list[Individual]:
    return [Individual.from_dict(d) for d in json.loads((gen_dir / "generation.json").read_text())]


async def main_async(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    ev = cfg["evolution"]
    if args.population:
        ev["population"] = args.population
        # shrink offspring mix proportionally, keep it summing correctly
        target = ev["population"] - ev["elites"]
        keys = list(ev["offspring_mix"])
        ev["offspring_mix"] = {k: 0 for k in keys}
        for i in range(target):
            ev["offspring_mix"][keys[i % len(keys)]] += 1
    if args.generations:
        ev["generations"] = args.generations
    if args.samples:
        ev["samples_per_eval"] = args.samples

    llm = LLM(cfg, mock=args.mock)
    rng = random.Random(args.seed)

    if args.resume:
        run_dir = Path(args.resume)
        done = sorted(p for p in run_dir.glob("gen_*") if (p / "generation.json").exists())
        if not done:
            raise SystemExit("nothing to resume")
        start_gen = int(done[-1].name.split("_")[1]) + 1
        pop = load_generation(done[-1])
        console.print(f"resuming {run_dir} from generation {start_gen}")
    else:
        run_dir = ROOT / "runs" / (datetime.now().strftime("%Y%m%d_%H%M%S") + ("_mock" if args.mock else ""))
        run_dir.mkdir(parents=True)
        (run_dir / "config.used.yaml").write_text(Path(cfg["_path"]).read_text())
        pop = load_seeds(cfg, ev["population"])
        start_gen = 0
        console.print(f"run dir: {run_dir}")

    for gen in range(start_gen, ev["generations"]):
        console.rule(f"generation {gen}")
        if gen > 0:  # pop holds the evaluated previous generation (fresh or loaded on resume)
            pop = await next_generation(pop, gen, llm, cfg, rng)
        gen_dir = run_dir / f"gen_{gen:02d}"
        await evaluate_population(pop, gen_dir, llm, cfg, gen)
        save_generation(gen_dir, pop)
        print_generation(gen, pop)
        (run_dir / "usage.json").write_text(json.dumps(llm.usage.to_dict(), indent=1))

    from .analyze import write_report

    report = write_report(run_dir)
    console.print(f"\nreport: {report}")
    console.print(f"usage: {json.dumps(llm.usage.to_dict()['by_model'])}")


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--mock", action="store_true", help="stub LLM: builder returns the reference page")
    ap.add_argument("--resume", metavar="RUN_DIR")
    ap.add_argument("-g", "--generations", type=int)
    ap.add_argument("-p", "--population", type=int)
    ap.add_argument("-s", "--samples", type=int, help="samples per eval")
    ap.add_argument("--seed", type=int, default=None, help="rng seed for selection/crossover")
    asyncio.run(main_async(ap.parse_args(argv)))


if __name__ == "__main__":
    main(sys.argv[1:])
