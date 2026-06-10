# Experimentation

You've learned the system ([Learning Path](../03-learning-path/README.md)), you have a backlog of ideas ([Research](../05-research/02-improvement-ideas.md)) and new data sources ([Data](../04-data/README.md)). What turns that into *research* rather than a pile of half-remembered notebook runs is **discipline**: every change run as a tracked, comparable, reproducible experiment.

Our tool for that discipline is **[Loom](https://github.com/ZKAI-Network/loom)** (ZKAI-Network) — an agentic CLI that orchestrates the full data-science lifecycle with versioned runs, machine-checkable gates, and cost controls.

| Page | Contents |
|------|----------|
| [01 — The Loom Workflow](01-loom-workflow.md) | What Loom is, its concepts (verbs, flows, providers, gates), and exactly how we use it to run this repo's experiment lifecycle |

## The problem Loom solves, in one paragraph

An experiment on this repo touches five artifacts (corpus → config → checkpoint → embeddings → metrics) across hours of GPU time. Done by hand in notebooks, the failure modes are predictable: nobody remembers which corpus produced which checkpoint; a "win" turns out to be a leaky feature; two results aren't comparable because preprocessing drifted between them; an 8-GPU run launches with a typo'd config. Loom's design counters each: **lineage** (every artifact is a versioned run with recorded parents), **gates** (leakage checks block feature stages; failed validation blocks deployment; GPU cost is quoted and confirmed *before* launch), and **a declared metric as the spec** (you state the goal and the measurement up front; the search optimizes against it).

## The principles (whether or not you use the tool)

Even running by hand, these are the house rules for any experiment on this model:

1. **One question per run.** "T1 + longer context + new data" answers nothing when it wins.
2. **The metric is declared before the run.** Headline: test-split Average Precision (plus the [multi-task battery](../05-research/02-improvement-ideas.md#e1--build-the-multi-task-behavioral-benchmark) as it lands).
3. **Baselines are sacred.** Every claim is *vs* the unmodified pipeline on identical splits.
4. **Lineage or it didn't happen.** Corpus hash, tokenizer config, YAML, checkpoint path, eval split — recorded together.
5. **Negative results get written down** — in the [research KB](https://github.com/ZKAI-Network/research), so the next person doesn't re-run them.
