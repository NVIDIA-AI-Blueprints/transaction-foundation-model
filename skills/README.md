# Loom skill-pack (v0.1)

Claude Code skills that drive Loom conversationally. Loom is a general-purpose,
domain-neutral automated ML engine (the metric is the spec); these skills are the
human-facing front door to it — they plan, gate on cost/data, invoke the `loom`
CLI, and narrate results. They do not reimplement any engine logic; they call the
same `loom run` entrypoint a human would.

## Skills

| Skill | What it does | Reach for it when |
| --- | --- | --- |
| [`loom-eda`](loom-eda/SKILL.md) | Quick, **read-only** data profile — shape, dtypes, missingness, target balance, leakage smells — and suggested goal/metric phrasing. | You point at a dataset and ask "what's in here?" / "is this ready for Loom?" |
| [`loom-optimize`](loom-optimize/SKILL.md) | Metric-is-the-spec entry → plan → **approval gate (cost/data)** → invoke `loom run` → narrate best metric + leaderboard. | You want Loom to optimize solution code against a measurable metric. |

## Typical flow

1. **`loom-eda`** — profile the data, confirm the target, get suggested goal and
   metric sentences. Read-only; spends no budget.
2. **`loom-optimize`** — pin the data/goal/metric, propose a run plan
   (providers, budget, models), **stop at the approval gate** (cost shape + data
   scope + the exact command), then run `loom run` and summarize the best
   solution, the leaderboard, and the artifact paths.

## Interface (v0.1)

Both skills are plain Claude-Code `SKILL.md` files: YAML frontmatter
(`name` + `description`) followed by markdown instructions. They shell out to the
project's CLI:

```bash
loom run --data DIR --goal STR --metric STR [--steps N] [--mlops metaflow|local] [--search aide]
```

Providers are selected by name (search brain default `aide`; MLOps muscle default
`metaflow`, with `local` as a Metaflow-free dev path). See the repo
[`README.md`](../README.md) and [`docs/architecture.md`](../docs/architecture.md)
for the engine and provider model.

## Conventions

- **Domain-neutral.** No customer-, vertical-, or pricing-specific content — that
  strategy lives elsewhere, never in this repo.
- **Cost/data is gated.** A run spends model tokens and compute and reads real
  data, so `loom-optimize` requires explicit user approval before it invokes the
  CLI. Prefer the cheap path first (`--mlops local`, small `--steps`).
- **Secrets via env only.** Keys/endpoints come from `.env`/environment; skills
  never print or pass key material.
