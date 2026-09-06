"""Individuals, selection and generation stepping."""
from __future__ import annotations

import asyncio
import random
from dataclasses import asdict, dataclass, field
from typing import Any

from . import operators
from .llm import LLM


@dataclass
class Individual:
    id: str
    md: str
    generation: int
    parents: list[str] = field(default_factory=list)
    operator: str = "seed"
    fitness: float | None = None
    test_score: float | None = None
    lint_penalty: float = 0.0
    lint_reasons: list[str] = field(default_factory=list)
    check_rates: dict[str, float] = field(default_factory=dict)
    samples: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Individual":
        return cls(**d)


def tournament(pop: list[Individual], k: int, rng: random.Random) -> Individual:
    contestants = rng.sample(pop, min(k, len(pop)))
    return max(contestants, key=lambda i: i.fitness or 0.0)


def ranked(pop: list[Individual]) -> list[Individual]:
    return sorted(pop, key=lambda i: (i.fitness or 0.0, i.test_score or 0.0), reverse=True)


async def next_generation(pop: list[Individual], gen: int, llm: LLM, cfg: dict[str, Any],
                          rng: random.Random) -> list[Individual]:
    ev = cfg["evolution"]
    n, n_elite, k = ev["population"], ev["elites"], ev["tournament_k"]
    mix = dict(ev["offspring_mix"])
    if sum(mix.values()) != n - n_elite:
        raise SystemExit(f"offspring_mix sums to {sum(mix.values())}, expected population - elites = {n - n_elite}")

    order = ranked(pop)
    children: list[Individual] = []
    for e in order[:n_elite]:
        children.append(Individual(id="", md=e.md, generation=gen, parents=[e.id], operator="elite"))

    async def make_crossover() -> Individual:
        a, b = tournament(pop, k, rng), tournament(pop, k, rng)
        tries = 0
        while b.id == a.id and tries < 5 and len(pop) > 1:
            b = tournament(pop, k, rng)
            tries += 1
        md = operators.crossover(a.md, b.md, rng)
        md, touched = await operators.section_mutate(llm, cfg, md, ev["section_mutation_rate"], rng)
        op = "crossover" + (f"+mut[{','.join(touched)}]" if touched else "")
        return Individual(id="", md=md, generation=gen, parents=[a.id, b.id], operator=op)

    async def make_informed() -> Individual:
        p = tournament(pop, k, rng)
        md, op = await operators.mutate_informed(llm, cfg, p.md, p.check_rates, rng)
        return Individual(id="", md=md, generation=gen, parents=[p.id], operator=op)

    async def make_blind() -> Individual:
        p = tournament(pop, k, rng)
        md, op = await operators.mutate_blind(llm, cfg, p.md, rng)
        return Individual(id="", md=md, generation=gen, parents=[p.id], operator=op)

    jobs = ([make_crossover() for _ in range(mix.get("crossover", 0))]
            + [make_informed() for _ in range(mix.get("informed", 0))]
            + [make_blind() for _ in range(mix.get("blind", 0))])
    children += await asyncio.gather(*jobs)
    for i, c in enumerate(children):
        c.id = f"g{gen:02d}_i{i:02d}"
    return children
