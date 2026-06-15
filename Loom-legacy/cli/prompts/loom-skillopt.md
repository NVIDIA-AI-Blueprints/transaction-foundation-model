---
description: Optimize a /loom-* SKILL.md against the learnings corpus (the self-improvement moat verb; gated). Proposes a sidecar by default; the in-place overwrite is OFF unless --apply and only when the gate PROMOTED; never auto-fired.
argument-hint: <the /loom-* verb to optimize, e.g. 'loom-eda'; --candidate PATH or --propose; --apply only to overwrite in place>
---

# /loom-skillopt — optimize a skill from the corpus (gated; never ship a worse skill)

Optimize a `/loom-*` `SKILL.md` against Loom's own accumulated learnings corpus —
the **self-improvement / moat** verb. It does not touch data or models; it improves
the **other skills**. Its centerpiece is a machine-checkable, **never-worse
promotion GATE** (the parallel of deploy's exit gate): a candidate is promoted
**only if** it satisfies every HARD acceptance-contract constraint **and** beats the
incumbent's total score by a margin. A contract violator or a regression can never
win, no matter how well it reads. The corpus is the spec.

Optimize: $@

## 1. Intake — pin the verb + the candidate source (refuse if ambiguous)
- **Verb (required)** — which `/loom-*` skill to optimize (e.g. `loom-eda`),
  resolved to `skills/<verb>/SKILL.md`. **Refuse without a real verb.**
- **Candidate source (at most one)** — `candidate PATH` (a proposed SKILL.md to
  score) **or** `propose` (the optional LLM proposer, a no-op when no model is
  configured). With neither, the run just **scores + reports** the incumbent
  against the contract + corpus (a safe read-only health read). Never pass both.
- **`apply` (default OFF)** — overwrite the shipped `SKILL.md` in place. Default is
  a proposed sidecar (`skills/<verb>/SKILL.candidate.md`). Do NOT set `apply` until
  the user explicitly confirms after seeing the gate VERDICT + the diff.

## 2. Plan — propose = workspace-write; --apply = always gate, never auto-fire
Two-tiered: **propose (default) = workspace-write** (scores deterministically,
writes only a sidecar). **`apply` = expensive/mutate, always gate** — overwriting a
shipped skill is the consequential act, so it **always gates** and is **never
model-auto-invoked**. It NEVER auto-promotes: even with `apply`, the overwrite
happens only when the gate PROMOTED. Show the plan and **stop**: "I'll capture
`<verb>`'s `owned_by=general` corpus, score the incumbent + candidate on the
7-point contract (hard) + corpus coverage (soft), and gate; if it PROMOTES I'll
propose a sidecar — **no in-place overwrite** unless you confirm `apply`."

## 3. Run — call the `loom_skillopt` tool
Call `loom_skillopt` with `verb` (+ `candidate` / `propose`). Default = score, gate,
PROPOSE a sidecar (no in-place write). Only with explicit user confirmation, and
only when the gate PROMOTED, set `apply: true`. Never re-implement the scorer/gate
in a prompt — the decision is plain-Python and machine-checkable. The harness will
require a confirmation before this irreversible verb runs.

## 4. Verify — the gate IS the verifier
Read the **corpus digest** (rollouts, success rate, verdict histogram, recurring
failure modes), the **gate VERDICT** (`PROMOTE` / `KEEP` / `KEEP_ALL_DISQUALIFIED` /
`KEEP_NO_CANDIDATE`), and the **incumbent + candidate scores** (`hard_ok` / `soft` /
`total`, with any `hard_misses`). A `PROMOTE` prints a unified diff — confirm it is
intended before any `apply`. A `hard_ok=False` candidate is DISQUALIFIED; a tie or
regression KEEPS the incumbent.

## 5. Deliver — narrate the VERDICT, propose-or-apply
- Lead with the **VERDICT** and why. Make crystal clear whether this was a proposed
  sidecar (default) or a real in-place apply.
- Hand back the VERDICT + scores + diff, and — when promoted — the sidecar path (or
  confirmation that `SKILL.md` was overwritten under `apply`). **Never deploy a
  worse skill:** a `KEEP` leaves the incumbent untouched.
- **Next step:** if `KEEP_ALL_DISQUALIFIED`, point at the exact `hard_misses` to
  fix; if a proposed `PROMOTE` and the user is satisfied with the diff, offer the
  explicit `apply`; after an apply, suggest re-running the affected verb.
