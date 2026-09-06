"""Summarise a run: fitness per generation, per-check pass rates, surviving sections, best MD."""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .markdown import split_sections


def load_run(run_dir: Path) -> list[list[dict]]:
    gens = []
    for gd in sorted(run_dir.glob("gen_*")):
        f = gd / "generation.json"
        if f.exists():
            gens.append(json.loads(f.read_text()))
    return gens


def write_report(run_dir: Path) -> Path:
    gens = load_run(run_dir)
    out: list[str] = [f"# Evolution report: `{run_dir.name}`\n"]
    if not gens:
        out.append("no completed generations")
        (run_dir / "report.md").write_text("\n".join(out))
        return run_dir / "report.md"

    out.append("## Fitness per generation\n")
    out.append("| gen | mean | max | min | mean out tokens | best id | best out tokens | best operator |")
    out.append("|---|---|---|---|---|---|---|---|")
    for g, pop in enumerate(gens):
        fs = [i["fitness"] for i in pop]
        tk = [i.get("mean_completion_tokens", 0) for i in pop]
        best = max(pop, key=lambda i: (i["fitness"], -i.get("mean_completion_tokens", 0)))
        out.append(f"| {g} | {sum(fs)/len(fs):.3f} | {max(fs):.3f} | {min(fs):.3f} | {sum(tk)/len(tk):.0f} | "
                   f"{best['id']} | {best.get('mean_completion_tokens', 0):.0f} | {best['operator'][:40]} |")

    out.append("\n## Per-check mean score (population mean) per generation\n")
    names = sorted({n for pop in gens for i in pop for n in i["check_rates"]})
    out.append("| check | " + " | ".join(f"g{g}" for g in range(len(gens))) + " |")
    out.append("|---|" + "---|" * len(gens))
    for n in names:
        row = []
        for pop in gens:
            vals = [i["check_rates"].get(n, 0.0) for i in pop]
            row.append(f"{sum(vals)/len(vals):.2f}")
        out.append(f"| {n} | " + " | ".join(row) + " |")

    out.append("\n## Operator effectiveness (child fitness minus mean parent fitness)\n")
    by_id = {i["id"]: i for pop in gens for i in pop}
    deltas: dict[str, list[float]] = defaultdict(list)
    for pop in gens[1:]:
        for i in pop:
            ps = [by_id[p]["fitness"] for p in i["parents"] if p in by_id]
            if ps:
                deltas[i["operator"].split("[")[0].split("+")[0]].append(i["fitness"] - sum(ps) / len(ps))
    out.append("| operator | n | mean delta | best delta |")
    out.append("|---|---|---|---|")
    for op, ds in sorted(deltas.items()):
        out.append(f"| {op} | {len(ds)} | {sum(ds)/len(ds):+.3f} | {max(ds):+.3f} |")

    out.append("\n## Section headings in the final generation (count, mean fitness of carriers)\n")
    last = gens[-1]
    carriers: dict[str, list[float]] = defaultdict(list)
    for i in last:
        for s in split_sections(i["md"]):
            carriers[s.title or "(preamble)"].append(i["fitness"])
    out.append("| heading | carriers | mean fitness |")
    out.append("|---|---|---|")
    for h, fs in sorted(carriers.items(), key=lambda kv: (-len(kv[1]), -sum(kv[1]) / len(kv[1]))):
        out.append(f"| {h[:50]} | {len(fs)} | {sum(fs)/len(fs):.3f} |")

    out.append("\n## Seed lineage of the final generation\n")
    seed_ops = {i["id"]: i["operator"] for i in gens[0]}

    def roots(iid: str, seen=None) -> set[str]:
        seen = seen or set()
        i = by_id.get(iid)
        if not i or not i["parents"]:
            return {seed_ops.get(iid, iid)}
        return set().union(*(roots(p) for p in i["parents"]))

    cnt = Counter()
    for i in last:
        for r in roots(i["id"]):
            cnt[r] += 1
    for r, c in cnt.most_common():
        out.append(f"- {r}: ancestor of {c}/{len(last)} final individuals")

    best_overall = max((i for pop in gens for i in pop),
                       key=lambda i: (i["fitness"], -i.get("mean_completion_tokens", 0), i["generation"]))
    out.append(f"\n## Best individual overall: `{best_overall['id']}` fitness {best_overall['fitness']:.3f} "
               f"(tests {best_overall['test_score']:.3f}, lint -{best_overall['lint_penalty']:.2f}, "
               f"mean output tokens {best_overall.get('mean_completion_tokens', 0):.0f})\n")
    if best_overall["lint_reasons"]:
        out.append("Lint: " + "; ".join(best_overall["lint_reasons"]) + "\n")
    out.append("```markdown\n" + best_overall["md"].strip() + "\n```\n")

    baseline = next((i for i in gens[0] if "empty" in i["operator"]), None)
    if baseline:
        out.append(f"\nEmpty-MD control (generation 0): fitness {baseline['fitness']:.3f}\n")

    (run_dir / "report.md").write_text("\n".join(out))
    return run_dir / "report.md"


if __name__ == "__main__":
    print(write_report(Path(sys.argv[1])).read_text())
