# Backlog: Executable Phase Specs

This folder holds the **step-by-step execution specs** for the near-term phases of the
[Dataset & Training Roadmap](../05-research/03-dataset-and-training-roadmap.md). The roadmap decides
*what and why*; these pages are *how* — concrete steps, files to create, commands, and a hard
**advance gate** per phase.

> **Relationship to the rest of the docs.** The roadmap lives in [Research](../05-research/03-dataset-and-training-roadmap.md) because it sequences the [improvement-ideas backlog](../05-research/02-improvement-ideas.md). *This* folder is operational: an engineer should be able to open a phase page and start working. Every step cites the [universal recipe](../04-data/08-from-raw-data-to-training-run.md) (the generic how-to), the [Level 400 contracts](../03-learning-path/level-400-design-contracts-and-extensions.md) it touches, and the [Loom](../06-experimentation/01-loom-workflow.md) verbs that run it.

## Pages

| Phase | Page | Goal | Gate to advance |
|---|---|---|---|
| **0** | [phase-0-guardrails.md](phase-0-guardrails.md) | Guardrails before any GPU spend | Golden tests pass; vocab integer asserted; leak-free split; baselines in hand |
| **1** | [phase-1-first-production-run.md](phase-1-first-production-run.md) | Crypto next-trade FM — the first production run (D5) | Beat the baseline panel on the wallet-disjoint split; training sanity gates green |

*Phases 1F (fiat first run), 2 (multi-chain/protocol), 3 (infra + LoRA) get their own pages as they
come up; Phase 4 (D3 transfer) is deferred — see the [roadmap](../05-research/03-dataset-and-training-roadmap.md#3-the-phased-plan).*

## How to use a phase page

1. **Read the gate first.** It is the definition of done. If you can't state how you'll check it, stop.
2. **Work the steps in order.** Each step is *what · why · how · done-when*. The "why" cites the contract or literature so you can't shortcut it without knowing the cost.
3. **Run it as one Loom `--experiment`.** One hypothesis, one ID (e.g. `tfm-d5-dex-nexttrade`); baseline runs live in the same experiment so the report carries its own control.
4. **Don't skip the cheap guardrails.** Phase 0 is graded S (hours) precisely so that the expensive phases can't silently lie to you.

## The gating principle (why these are sequenced, not parallel)

Each phase's advance gate is a **real stop**. The roadmap's core bet is that *credibility is cheaper than
compute*: a leaky split or a miscounted vocabulary produces a confident-but-wrong green light that costs
far more (a customer demo built on a phantom result) than the day it takes to prevent. So:

- **No GPU spend before Phase 0's golden tests pass.**
- **No corpus growth (more chains, more protocols) before Phase 1 beats its clean baselines.**
- **No parameter scale-up before measured non-redundant token supply justifies it** ([A2](../05-research/02-improvement-ideas.md#a2--widthdepth-scaling-sweep-with-data-scaling) D/N math).
- **No multi-billion-token run before the [I1](../05-research/02-improvement-ideas.md#i1--streaming-corpus-loading) streaming loader exists.**
