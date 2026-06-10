# Research

This section connects the blueprint you just learned to the **research frontier**: what the field has demonstrated (2021–2026), where this repo's design sits in that landscape, and a prioritized menu of concrete improvements — each tied to specific files in this repo and specific evidence in the literature.

| Page | Contents |
|------|----------|
| [01 — Literature Review](01-literature-review.md) | The transaction-foundation-model landscape: production systems (Visa, Revolut, Nubank…), architecture & objective findings, blockchain-native models, datasets & benchmarks, open problems |
| [02 — Improvement Ideas](02-improvement-ideas.md) | A ranked backlog of experiments for *this* repo: tokenization upgrades, objective variants, scaling, evaluation hardening, graph augmentation — with effort estimates and literature backing |

## Where this material comes from

Our literature synthesis is maintained in a dedicated knowledge base: **[ZKAI-Network/research](https://github.com/ZKAI-Network/research)**. It contains the full reviews, ~75 collected papers (PDF), per-source verification notes, and strategy documents. The pages here distill the parts relevant to this model; when you need depth, follow the pointers:

- `wiki/transaction-intelligence.md` — the core survey of foundation transaction models (FTMs)
- `wiki/fm-finance.md` — the broader finance-FM landscape (language, time-series, event-sequence, on-chain)
- `raw/transaction-intelligence/foundation-transaction-models.md` — the long-form research brief (universal schema, architecture proposal, 12-week PoC roadmap, 72 references)
- `wiki/credit-scoring.md` — alternative-data credit scoring (a key downstream application)

## Epistemic conventions

The research KB tags every claim, and we preserve those tags here:

- **`[checked]`** — verified against the primary source (paper section/table read directly)
- **`[inferred]`** — reasonable synthesis or estimate, not directly measured
- **`[unverified]`** — reported by a credible secondary source; primary not independently parsed

Treat `[unverified]` numbers as directional. If you verify or refute one while working, update the research KB — that's the loop.

## How research happens here, practically

1. **Pick an idea** from [the backlog](02-improvement-ideas.md) (or add one — with a citation or a falsifiable hypothesis).
2. **Read the contracts** ([Level 400](../03-learning-path/level-400-design-contracts-and-extensions.md)) the idea touches; most ideas renegotiate at least one (usually the vocabulary).
3. **Run it as a tracked experiment** — datasets, configs, metrics, and verdicts managed [with Loom](../06-experimentation/01-loom-workflow.md), so results are comparable and reproducible.
4. **Evaluate beyond a single number** — the [multi-task evaluation framework](02-improvement-ideas.md#e1--build-the-multi-task-behavioral-benchmark) is the goal-state; AP on one fraud split is the minimum.
5. **Write back** — findings (positive *or* negative) go to the research KB; durable how-tos go into these docs.
