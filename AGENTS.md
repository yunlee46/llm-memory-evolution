# AGENTS.md — Build Instructions

Instructions for a coding agent (or human) building this repository.

> **Naming note.** This repo *evolves* markdown memory files. Do not confuse them with
> this file. `AGENTS.md` = build instructions for the repo. The evolved artifacts live in
> `memory/runs/<run_id>/gen_XX.md` and are always called **memory files**, never "AGENTS.md".

---

## 1. What we are building

A test harness that evolves a **persistent memory file** for an LLM coding agent, and
measures whether that memory actually helps.

Each generation, 10 agents attempt the same build task in parallel:

- **Group A (memory)** — receives the latest memory file plus the task prompt.
- **Group B (fresh)** — receives *only* the frozen task prompt. No memory.

All 10 runs are scored. A **curator** model reads the transcripts and edits the memory
file into the next generation. Group B doubles as a live control: the gap between the two
groups is the primary result.

The memory file carries two distinct kinds of content (see §5), because "write tests
first" and "the bounce damping should be ~0.8" generalize very differently and must be
measured separately.

### Research question

For a given model, which memory contents measurably improve coding-agent performance —
and does the benefit come from durable practices or from cached task knowledge?

### Primary metric

```
lift_g = mean(score | group A, generation g) − mean(score | group B, generation g)
```

`lift_g` must be reported both with and without Layer 2 (§5). If `lift_g` is not clearly
positive and growing by generation 5, stop and debug — do not keep burning budget.

### Design decisions (flip these in config if you disagree)

| Decision | Choice | Why |
|---|---|---|
| Lineage | **Single** memory lineage, refined per generation | Cheaper, simpler; 10 runs/gen give a usable estimate. Two-lineage variant in §12. |
| Layer 2 content | **Procedural description only — no verbatim code blocks** | Verbatim solutions saturate fitness by ~gen 4 and kill the gradient. |
| Provider | Google Gemini API | See §3. |

---

## 2. Repository layout

Build exactly this structure.

```
llm-memory-evolution/
├── AGENTS.md                    # this file
├── README.md
├── pyproject.toml
├── .env.example
├── config/
│   └── experiment.yaml          # all knobs; no magic numbers in code
├── memory/
│   ├── seed.md                  # generation 0, hand-written
│   └── runs/<run_id>/gen_00.md … gen_NN.md
├── tasks/
│   ├── registry.yaml
│   └── <task_id>/
│       ├── prompt.md            # FROZEN. Group B's entire input.
│       ├── tests/               # read-only at runtime
│       └── scaffold/            # optional starting files
├── src/evolution/
│   ├── config.py                # load + validate experiment.yaml
│   ├── llm.py                   # Gemini wrapper: retries, backoff, token accounting
│   ├── agent.py                 # the coding agent loop (tool use)
│   ├── sandbox.py               # isolated per-run workspace
│   ├── evaluator.py             # run tests → score
│   ├── memory_file.py           # parse / render / edit the two-layer memory file
│   ├── curator.py               # generation n transcripts → generation n+1 edits
│   ├── harvest.py               # promote group-B discoveries into memory
│   ├── orchestrator.py          # generation loop, group split, confirmation runs
│   └── analysis.py              # lift curve, replay, ablation
├── results/<run_id>/
│   ├── runs.jsonl
│   ├── generations.jsonl
│   └── changelog.md
└── tests/                       # tests for THIS harness, not for the tasks
```

---

## 3. Gemini API

Use the current unified Google Gen AI SDK (`google-genai`, `from google import genai`).
Key is `GEMINI_API_KEY`, loaded from `.env` — never hardcoded, never logged.

**Do not take model IDs from this file on faith.** Look up the current IDs and pricing in
Google's docs and put the exact strings in `config/experiment.yaml`. Pinning the precise
model ID is a reproducibility requirement, not a formality — record it in every run row.

Three roles, deliberately different tiers:

| Role | Tier | Temperature | Notes |
|---|---|---|---|
| `worker_model` | Flash-class | > 0 (e.g. 0.7) | 10 parallel runs/gen; this is where the budget goes. Needs variance for diversity. |
| `curator_model` | Pro-class | low (e.g. 0.2) | One call per generation. Reasoning quality matters more than cost. |
| `judge_model` | Pro-class | 0 | Optional, for qualitative transcript diffing during harvest. |

Since everything is now one vendor, the tier split *is* the cross-model diversity
mechanism that a second provider would otherwise supply.

Implementation requirements for `llm.py`:

- **Concurrency + 429 handling.** 10 workers fire simultaneously. Implement a semaphore
  bounded by `config.concurrency` and exponential backoff with jitter on rate-limit and
  5xx responses. Assume you *will* hit limits.
- **Token accounting.** Record `usage_metadata` (prompt / candidates / total, and thinking
  tokens if the model reports them) on every call. Cost analysis depends on this.
- **Structured output for the curator.** Use a response schema so curator edits come back
  as validated JSON (§7), not prose you have to parse.
- **Thinking budget**, if the chosen model exposes one, belongs in config — it materially
  changes both cost and behavior, so it must be pinned per run.
- Retries must be **idempotent-safe**: a retried worker run gets a fresh sandbox, and
  partial runs are never scored.

---

## 4. Sandbox and guardrails

Agent-generated code is executed. Run every worker in an isolated container workspace.

Non-negotiable guardrails — the harness enforces these, not the prompt:

1. `tasks/<id>/tests/` is **mounted read-only**. An agent that edits tests cannot.
2. Every run starts from a **clean workspace**. Group B especially: no leftover artifacts,
   no shared cache. A contaminated "fresh" group destroys your only baseline.
3. **Hidden held-out tests** are run at scoring time and never exposed to the agent.
4. Hard caps on wall-clock, max agent turns, and total tokens per run.
5. Network egress denied by default, except an allowlist (package registry if the task
   needs it).
6. Log every guardrail violation attempt to `runs.jsonl`. These are selected-for
   behaviors, not bugs — the catalogue of attempted hacks is a real finding.

Agent tools (`agent.py`) should be minimal and auditable: `read_file`, `write_file`,
`list_files`, `run_tests`, `finish`.

---

## 5. Memory file format

Two layers, explicitly separated, with **stable IDs**. IDs are what make attribution
possible — never renumber them, and carry them across generations.

```markdown
# Agent Memory — Generation 07

<!-- run_id: 2026-09-06-a | parent: gen_06 | created: 2026-09-06T14:02Z -->

## Layer 1 — Practices
General working rules. Should transfer to unseen tasks.

### P-003
Write the failing test before the implementation.

### P-011
Prefer the standard library; do not add a dependency without a stated reason.

## Layer 2 — Task Knowledge
How the working solution is built. Procedural description only.
NO verbatim code blocks — describe the approach, not the answer.

### K-014
Drive animation from `requestAnimationFrame`, not a fixed-interval timer; the tests
assert frame-rate independence.

### K-022
Collision response needs a damping factor slightly below 1.0 or the objects gain energy
and the stability test fails.
```

`memory_file.py` must provide: parse → structured object, render → markdown, and
section-level ops (add / delete / revise / reorder / regroup) addressed **by ID**.

**Held-out evaluation strips Layer 2.** That is the whole reason for the split — it lets
you separate "good instructions" from "has the answer key."

---

## 6. Scoring

```
score = mean over tasks of pass_fraction
      − λ · normalized_token_cost
      − μ · normalized_memory_length
```

`λ` and `μ` live in config. The length penalty is not cosmetic: memory grows every
generation and nothing removes it unless you make growth cost something.

Also record, per run: variance across trials, turns to completion, wall-clock, and
guardrail violations. "Which memory reduces variance" is a worthwhile secondary finding —
a file that makes the agent reliably competent may beat one that is occasionally brilliant.

---

## 7. The curator

Once per generation. Input: current memory file, all 10 run records with scores and group
labels, and transcripts (truncated to fit).

**Output must be a list of edits, never a full rewrite.** Full rewrites cause silent drift
and make it impossible to attribute a score change to any specific edit. Schema:

```json
{
  "edits": [
    {
      "op": "ADD | DELETE | REVISE | REORDER | COMPACT",
      "layer": 1,
      "target_id": "P-011",
      "content": "…",
      "rationale": "one line — which run(s) motivated this"
    }
  ]
}
```

Apply edits mechanically in `memory_file.py`, append them to `results/<run_id>/changelog.md`,
and write `gen_NN+1.md`. With a single lineage the changelog is your only source of
per-edit attribution — treat it as a primary artifact.

**Compaction** runs every `compaction_interval` generations (default 3): rewrite shorter
while preserving what earns fitness. What repeatedly survives compaction is your durable
knowledge, and is worth reporting on its own.

---

## 8. Harvest (group B → memory)

When a group-B run beats the group-A mean:

1. Diff its approach against the group-A approach (a `judge_model` call, or structured
   transcript comparison).
2. **Confirm before harvesting: re-run the candidate 2–3× with the same seed policy.**
   This is mandatory. Picking the best of 10 noisy runs systematically selects *lucky*
   runs, and unconfirmed harvest writes noise into memory as if it were knowledge — which
   then persists and compounds across generations.
3. Only if it holds up, pass it to the curator as a candidate edit.

Without harvest, group B is 30% of your budget spent on runs you discard.

---

## 9. Configuration

`config/experiment.yaml` holds everything. No magic numbers in code.

| Key | Default | Notes |
|---|---|---|
| `generations` | 15 | Six points is not a curve. Generations are the signal axis. |
| `agents_per_generation` | 10 | Fixed. |
| `group_split` | schedule below | Fraction in group A. |
| `trials_per_confirmation` | 3 | §8. |
| `tasks` | 5 selected + 3 held out | §10. |
| `compaction_interval` | 3 | §7. |
| `memory_token_cap` | 1500 | Hard cap. |
| `lambda_cost` / `mu_length` | tune | §6. |
| `concurrency` | 5 | Tune against actual rate limits. |
| `worker_model` / `curator_model` | pin exact IDs | §3. |
| `seed` | int | Common random numbers across groups. |
| `api.daily_request_budget` | null | Request guard; null = paid tier. §13. |

> **The shipped `config/experiment.yaml` does not use these defaults.** It is tuned
> for a 20-request/day free tier — see §13 for the derivation and for what that
> costs the result.

Group split schedule — memory is bad early and good late, so a fixed ratio wastes budget
at both ends:

| Generations | Group A | Group B |
|---|---|---|
| 1–3 | 3 | 7 |
| 4–8 | 6 | 4 |
| 9+ | 8 | 2 |

Never let group B fall below 2. Losing the control is worse than losing exploitation.

Both groups must see the **same task and same seed** within a generation.

---

## 10. Task calibration — do this before anything else

Build the task suite first and calibrate it. This is cheap and it de-risks everything
downstream.

1. Write `tasks/<id>/prompt.md` for ~8 small build tasks. The cat-bounce clone
   (`https://cat-bounce.com/`) is the seed task; browser behavior is checked with
   Playwright/Selenium.
2. Run each with an **empty memory file**, 5 times.
3. Keep only tasks whose pass rate lands in the **20–80%** band. Tasks at 0% or 100%
   provide no gradient and only burn budget.
4. Reserve 3 calibrated tasks as **held-out**. They are never used for selection — only
   for the generalization measurement in §11.
5. **Freeze `prompt.md` verbatim** for the entire experiment. If group B's prompt drifts,
   `lift_g` becomes meaningless.

---

## 11. Analysis (`analysis.py`)

Build these as first-class outputs, not notebook afterthoughts:

- **Lift curve** — `lift_g` per generation, with and without Layer 2.
- **Replay** — at the end, re-run *every* saved `gen_XX.md` against a fixed task set in
  one batch. Scores collected across generations are confounded by drifting conditions;
  the replay curve is measured under identical conditions and is the one you publish.
- **Held-out generalization** — Layer 1 only, on the 3 reserved tasks.
- **Ablation** — drop each section from the best memory file one at a time and re-score.
  This is the causal check on the changelog's correlational story.
- **Length vs. fitness** — find where added memory stops paying for itself.
- **Hack catalogue** — every logged guardrail violation, grouped by kind.

### Logging schema

`runs.jsonl`, one row per agent run:

```
run_id, generation, group, agent_idx, memory_version_hash, task_id, seed,
worker_model_id, temperature, pass_fraction, held_out_pass_fraction, score,
prompt_tokens, output_tokens, thinking_tokens, turns, wall_clock_s,
memory_token_count, guardrail_violations[], transcript_path, harvested
```

`generations.jsonl`, one row per generation: `generation, lift, lift_layer1_only,
mean_a, mean_b, var_a, var_b, memory_token_count, n_edits, curator_tokens`.

Log from day one. You will want to re-analyze without re-running anything.

---

## 12. Build order

Each phase has an acceptance criterion. Do not advance until it passes.

| # | Phase | Done when |
|---|---|---|
| 0 | Scaffolding, config loader, `.env`, Gemini smoke test | One prompt round-trips; token counts recorded |
| 1 | Sandbox + one task + evaluator | A hand-written correct solution scores 1.0; a broken one scores < 1.0 |
| 2 | Agent loop with tools | A single agent solves the task end-to-end unaided |
| 3 | **Task calibration (§10)** | ≥5 tasks land in the 20–80% band; 3 held out; prompts frozen |
| 4 | `memory_file.py` | Round-trip parse→render is lossless; ID-addressed edits work |
| 5 | Orchestrator, one generation, both groups | 10 parallel runs complete; `lift_1` computed and logged |
| 6 | Curator + changelog | Generation 2 memory differs from generation 1 by a validated, logged edit list |
| 7 | Harvest + confirmation runs | A group-B win propagates into memory only after confirming |
| 8 | Compaction | Memory stays under `memory_token_cap` across 6+ generations |
| 9 | Analysis + replay | Full lift curve and replay curve render from `results/` alone |

Phase 3 is the real gate. If the tasks aren't calibrated, every number after it is noise.

### Optional upgrade: two competing lineages

If you want genuine *selection* rather than sequential hill-climbing: run lineages A and B
(4 agents each) diverging from a common ancestor, plus 2 fresh agents. Every 3
generations the loser adopts the winner's memory and re-diverges. Same 10 agents, same
cost, stronger claim — at the price of sample size per lineage. Implement only after
phase 9 works single-lineage.

---

## 13. Operating under the free tier

The API key is on the free tier: **20 requests per day** for the worker model.
Requests, not tokens or dollars, are the binding constraint, and every parameter in
`config/experiment.yaml` is derived from that number.

```
requests per agent run  ~= max_turns + 1          (one request per turn)
requests per generation  = agents_per_generation * requests per run

max_turns=3  ->  ~4 req/run  ->  40 req/generation  ->  2 days
6 generations                -> 240 requests        -> ~12 days
```

Consequences, all deliberate:

| Parameter | Full design | Free tier | Why |
|---|---|---|---|
| `generations` | 15 | 6 | ~2 days each |
| `max_turns` | 25 | 3 | Directly sets requests/run |
| `agents_per_generation` | 10 | 10 | **Unchanged** — per-generation precision matters more than generation count when lift is noisy |
| tasks per generation | all of `selection` | 1, rotating | Averaging over 5 tasks costs 5x the requests |
| `calibration.trials` | 5 | 3 | Floor for a usable mean |

Two mechanisms exist only because of this constraint:

- **Request budgeting** (`api.daily_request_budget`) stops cleanly *before* the quota
  turns into a wall of 429s mid-generation. Retries count against it, because they
  are real requests.
- **Resumable state** (`results/calibration/trials.jsonl`) lets a run span days. A
  trial that failed on quota is discarded rather than recorded, because quota
  exhaustion is an *absence of data*, not a score of zero — persisting it would
  silently drag the mean down. Re-pinning the worker model invalidates earlier
  trials rather than mixing models into one mean.

### What this design cannot show

State these limits in any writeup rather than discovering them in review:

- **Underpowered.** With ~5 runs per group per generation and pass-rate variance in
  the 0.2–0.8 band, the confidence interval on `lift_g` is wide. A small positive
  lift will not be distinguishable from sampling noise. Report the interval, not
  just the mean.
- **One task per generation** means `lift_g` confounds memory quality with task
  difficulty. Only the end-of-run replay (§11), which scores every memory version
  against a fixed task set in one batch, gives a clean curve.
- **`max_turns=3` is a different agent.** Findings are about memory for a
  near-single-shot agent, not a long-horizon one. Practices that only pay off over
  many turns cannot appear.
- **6 generations is a short lineage.** Compaction fires at most twice, so
  "what survives repeated compaction" will be weak evidence.

Set `daily_request_budget: null`, restore the §9 defaults, and re-run calibration if
the quota situation changes. Nothing in the harness assumes free-tier numbers.

## 14. Known failure modes

Design against these; they are expected, not hypothetical.

| Failure | Symptom | Mitigation |
|---|---|---|
| Fitness saturation | Pass rate pins at 100% by ~gen 4 | No verbatim code in Layer 2; rotate task variants |
| Winner's curse | Memory fills with noise dressed as knowledge | Confirmation runs before harvest (§8) |
| Bloat | Memory grows monotonically | Length penalty + compaction + hard cap |
| Reward hacking | Memory learns to edit or bypass tests | Read-only tests, hidden held-out tests, violation logging |
| Fake-fresh | Group B silently benefits from prior state | Clean workspace per run; verify by inspection |
| Confounded curve | Generation trend mixes with condition drift | End-of-run replay (§11) |
| Premature convergence | Group A stops improving, group B never harvested | Adaptive split; floor of 2 on group B |

---

## 15. Prior work to position against

Skim before committing to the design so you extend rather than re-derive: **PromptBreeder**,
**EvoPrompt**, **GEPA**, **DSPy** optimizers, **Reflexion**, **Voyager**'s skill library.
Verify details at the source — these are pointers, not citations.

The distinct angle here: those largely optimize prompts for a *single call*. This project
evolves a **persistent memory document for a multi-turn agent**, splits **durable practices
from cached task knowledge**, and carries a **built-in live control group**. Lead with that.
