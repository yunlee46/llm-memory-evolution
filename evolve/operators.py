"""Variation operators over markdown instruction files.

crossover        deterministic section-level splice of two parents (no LLM)
mutate_blind     mutator model rewrites one random section with NO knowledge of results
mutate_informed  mutator model rewrites the whole file given a random SUBSET of check results
section_mutate   each section independently has `rate` chance of a blind rewrite
"""
from __future__ import annotations

import random
import re
from typing import Any

from .llm import LLM
from .markdown import Section, join_sections, split_sections

PREAMBLE = (
    "You are improving a general-purpose instruction file that will be given, verbatim, as the system prompt "
    "to a coding model before it receives a programming task. The file must stay task-agnostic: it must not "
    "contain task-specific code, no code block may exceed {max_lines} lines, and it must never mention "
    "{forbidden}. Keep the whole file under {max_words} words. Write in markdown with headings. "
    "Return only the markdown between the markers ---BEGIN--- and ---END---, nothing else."
)


def _preamble(cfg: dict[str, Any]) -> str:
    lc = cfg.get("lint", {})
    return PREAMBLE.format(
        max_lines=lc.get("max_code_block_lines", 8),
        forbidden=", ".join(lc.get("forbidden_keywords", [])),
        max_words=lc.get("max_words", 1500),
    )


def _clean(reply: str) -> str:
    """Pull the markdown out of the mutator reply, tolerating missing markers and outer fences."""
    reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.S)
    if "---BEGIN---" in reply:
        reply = reply.split("---BEGIN---", 1)[1]
    if "---END---" in reply:
        reply = reply.split("---END---", 1)[0]
    reply = reply.strip()
    m = re.fullmatch(r"```(?:markdown|md)?\s*\n(.*)\n```", reply, re.S)
    if m:
        reply = m.group(1).strip()
    return reply + "\n"


# ------------------------------------------------------------------ crossover

def crossover(a: str, b: str, rng: random.Random) -> str:
    sa, sb = split_sections(a), split_sections(b)
    ka = {s.key: s for s in sa}
    kb = {s.key: s for s in sb}
    child: list[Section] = []
    for s in sa:
        if s.key in kb:
            child.append(rng.choice([s, kb[s.key]]))
        elif rng.random() < 0.5:
            child.append(s)
    for s in sb:
        if s.key not in ka and rng.random() < 0.5:
            child.append(s)
    if not child:  # never return an empty file
        child = [rng.choice(sa + sb)]
    return join_sections(child)


# ------------------------------------------------------------------ mutations

async def _rewrite_section(llm: LLM, cfg: dict[str, Any], md: str, sec: Section, rng: random.Random) -> str:
    mc = cfg["models"]
    label = f'the section titled "{sec.title}"' if sec.title else "the untitled opening section"
    prompt = (
        f"{_preamble(cfg)}\n\nHere is the current file for context:\n\n---BEGIN---\n{md}\n---END---\n\n"
        f"Rewrite ONLY {label}. Change its approach or emphasis, not just its wording: you may make it "
        f"stricter, looser, more concrete, more structured, shorter or longer. Keep the same heading line. "
        f"Return only that one section (heading plus body) between ---BEGIN--- and ---END---."
    )
    reply = await llm.chat(mc["mutator"], [{"role": "user", "content": prompt}],
                           temperature=mc.get("mutator_temperature_blind", 1.0), max_tokens=3000)
    new_secs = split_sections(_clean(reply))
    if not new_secs:
        return md
    new = new_secs[0]
    if sec.level:
        new.title, new.level = sec.title, sec.level  # keep heading stable so crossover keys still match
        new.body = "\n".join(s.render() for s in new_secs).split("\n", 1)[1] if len(new_secs) > 1 else new.body
    else:
        new = Section("", 0, "\n".join(s.render() for s in new_secs))
    secs = split_sections(md)
    idx = next((i for i, s in enumerate(secs) if s.key == sec.key and s.level == sec.level), None)
    if idx is None:
        return md
    secs[idx] = new
    return join_sections(secs)


async def _add_section(llm: LLM, cfg: dict[str, Any], md: str) -> str:
    mc = cfg["models"]
    prompt = (
        f"{_preamble(cfg)}\n\nHere is the current file:\n\n---BEGIN---\n{md}\n---END---\n\n"
        "Write ONE new section (a level-2 heading plus body, under 150 words) covering a useful practice "
        "this file is missing. Return only the new section between ---BEGIN--- and ---END---."
    )
    reply = await llm.chat(mc["mutator"], [{"role": "user", "content": prompt}],
                           temperature=mc.get("mutator_temperature_blind", 1.0), max_tokens=1500)
    new = _clean(reply).strip()
    return md.rstrip() + "\n\n" + new + "\n"


async def mutate_blind(llm: LLM, cfg: dict[str, Any], md: str, rng: random.Random) -> tuple[str, str]:
    """Returns (child_md, description of what was done)."""
    secs = split_sections(md)
    r = rng.random()
    if r < 0.1 and len(secs) > 1:
        victim = rng.choice(secs)
        secs.remove(victim)
        return join_sections(secs), f"blind:delete[{victim.title or 'preamble'}]"
    if r < 0.2 or not md.strip():
        return await _add_section(llm, cfg, md), "blind:add"
    sec = rng.choice(secs)
    return await _rewrite_section(llm, cfg, md, sec, rng), f"blind:rewrite[{sec.title or 'preamble'}]"


async def section_mutate(llm: LLM, cfg: dict[str, Any], md: str, rate: float, rng: random.Random) -> tuple[str, list[str]]:
    touched = []
    for sec in list(split_sections(md)):
        if rng.random() < rate:
            md = await _rewrite_section(llm, cfg, md, sec, rng)
            touched.append(sec.title or "preamble")
    return md, touched


async def mutate_informed(llm: LLM, cfg: dict[str, Any], md: str, check_rates: dict[str, float],
                          rng: random.Random) -> tuple[str, str]:
    """Rewrite the whole file, shown a random subset of per-check pass rates (partial context)."""
    mc = cfg["models"]
    frac = cfg["evolution"].get("informed_context_fraction", 0.5)
    names = sorted(check_rates)
    k = max(1, round(len(names) * frac))
    shown = sorted(rng.sample(names, k)) if names else []
    lines = "\n".join(f"- {n}: passed {check_rates[n]:.0%} of attempts" for n in shown) or "- (no results available)"
    prompt = (
        f"{_preamble(cfg)}\n\nA coding model was given the instruction file below as its system prompt and "
        f"then asked to build a small browser app. Each build was scored by automated behavioural checks. "
        f"A partial view of the results (check name and pass rate) is:\n\n{lines}\n\n"
        f"Current file:\n\n---BEGIN---\n{md}\n---END---\n\n"
        "Rewrite the whole file so that a model following it is more likely to pass checks like these, "
        "using only general engineering practices (planning, output discipline, defensive coding, "
        "verifying requirements, DOM/event/animation habits). Do not name the checks or describe the app. "
        "Return the full file between ---BEGIN--- and ---END---."
    )
    reply = await llm.chat(mc["mutator"], [{"role": "user", "content": prompt}],
                           temperature=mc.get("mutator_temperature_informed", 0.7), max_tokens=4000)
    return _clean(reply), f"informed[{len(shown)}/{len(names)} checks shown]"
