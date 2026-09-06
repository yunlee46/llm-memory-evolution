# llm-memory-evolution

Evolve CLAUDE.md-style instruction files to find which practices make a specific
coding model write better code. The model under study is DeepSeek V4 Flash via
DeepInfra. Fitness is how well a single-shot build of a cat-bounce.com clone,
generated with the instruction file as system prompt, passes a fixed Playwright
test suite.

## How it works

1. **Population**: 10 markdown files (`seeds/` for generation 0, deliberately
   diverse styles including an empty control).
2. **Evaluation**: each file becomes the system prompt for one chat call to the
   builder model; the user message is `tasks/cat_bounce/spec.md`. The reply's
   `index.html` is scored by 12 behavioural Playwright checks (gravity, floor
   bounce, drag and throw with momentum, "make it rain", resize recolour, no
   network, no console errors, accessibility). Each file is built 3 times and
   fitness is the mean score, minus lint penalties for files that try to
   smuggle the solution in (long code blocks, task keywords, over 1500 words).
3. **Next generation**: top 2 carried over unchanged, then 4 children by
   section-level crossover (with a 10% per-section blind rewrite), 2 by an
   *informed* rewrite that sees a random half of the parent's check results,
   and 2 by a *blind* rewrite of one section with no results context. All
   rewrites are done by a different model (Qwen3 by default) so the target
   model is not writing instructions for itself.
4. Repeat for 6 generations. `runs/<id>/report.md` summarises what survived.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -e . && .venv/bin/python -m playwright install chromium
cp .env.example .env            # add DEEPINFRA_API_KEY
.venv/bin/python -m evolve.llm --list | grep -i -E 'deepseek|qwen'   # confirm model ids in config.yaml
.venv/bin/python -m evolve.llm --ping                                # one tiny call per model
```

## Run

```bash
# calibrate the test suite (reference should score 1.0, broken well under 0.3)
python -m evolve.fitness tasks/cat_bounce/reference/index.html tasks/cat_bounce/reference/broken.html

# dry run of the whole loop with a stub LLM, no API spend
python -m evolve.run --mock -g 2 -p 4 -s 1

# one real build from one seed, scored
python -m evolve.builder seeds/09_structured.md

# the real thing (10 x 6 x 3 = 180 builder calls plus ~30 mutator calls)
python -m evolve.run
python -m evolve.run --resume runs/<id>      # after an interruption
python -m evolve.analyze runs/<id>           # regenerate report.md
```

## Layout

```
config.yaml                 population, generations, models, operator mix, lint thresholds
evolve/llm.py               DeepInfra client (OpenAI-compatible), usage accounting, mock mode
evolve/builder.py           MD + spec -> index.html
evolve/fitness.py           Playwright evaluation + MD lint
evolve/markdown.py          heading-based section split/join
evolve/operators.py         crossover, blind and informed mutation
evolve/population.py        selection and generation stepping
evolve/run.py               CLI loop with resume
evolve/analyze.py           report.md
tasks/cat_bounce/           spec.md, tests/checks.py, reference solutions
seeds/                      generation-0 instruction files
runs/                       per-run artefacts (gitignored): agent.md, sample_N.html, result.json
```

## Adding a task

Create `tasks/<name>/spec.md` and `tasks/<name>/tests/checks.py` exposing
`run(browser, url, cfg) -> list[CheckResult]`, then point `task:` in
`config.yaml` at it.
