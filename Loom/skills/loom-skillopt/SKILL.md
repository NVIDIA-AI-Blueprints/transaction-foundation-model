---
name: loom-skillopt
description: Optimize a /loom-* SKILL.md against Loom's own learnings corpus — the self-improvement MOAT verb (the one meta verb that improves the OTHER skills). HiveMind captures the verb's trace corpus (learnings/rollouts.jsonl, owned_by=general only — the IP boundary), then SkillOpt's deterministic scorer grades the incumbent + any candidate on the 7-point acceptance contract (HARD) + corpus failure-mode coverage (SOFT) and applies a never-worse promotion GATE (the parallel of loom-deploy's exit gate). SAFE BY DEFAULT — it PROPOSES a sidecar candidate + the gate VERDICT + a diff; the real in-place SKILL.md overwrite is behind --apply and ONLY when the gate PROMOTED. NEVER deploy a worse skill. Use when the user says "optimize the eda skill", "improve a loom skill from the corpus", "score this candidate SKILL.md", "run skillopt". NEVER auto-fire — editing the shipped skill library is a deliberate, human-fired act.
when_to_use: "improve/optimize a /loom-* skill from accumulated usage traces, score a candidate SKILL.md against the contract + corpus, gate a skill edit (never ship a worse skill), run the self-improvement / moat loop"
when_not_to_use: "to deploy a validated MODEL, use loom-deploy (skillopt optimizes the SKILL.md text, not a model); to read what passed/failed in the field, use loom-ops; to assemble a run write-up, use loom-report."
argument-hint: "<the /loom-* verb to optimize, e.g. 'loom-eda'; --candidate PATH or --propose; --apply only to overwrite in place>"
disable-model-invocation: true
---

# loom-skillopt

Optimize a `/loom-*` `SKILL.md` against Loom's own accumulated learnings corpus — the
**self-improvement / moat** verb (design-spec §5; build order §6 #5). This is the one
**ops/meta** verb: it does not touch data or models, it improves the **other skills**.
Each `/loom-*` command **IS a `SKILL.md`** — the *trainable artifact* — so optimizing a
skill is the text-space (zero-GPU) inner loop of the flywheel: `usage → trace corpus →
HiveMind capture → SkillOpt gate → a better skill (never a worse one) → more usage`.

It is a **planned, gated run through Loom's interface** (the `loom` CLI), never a loose
"rewrite this prompt" — because its centerpiece is a **machine-checkable, never-worse
promotion GATE** that mirrors `loom-deploy`'s exit gate: a candidate SKILL.md is promoted
**only if** it satisfies every HARD acceptance-contract constraint **and** beats the
incumbent's total score by a margin. A contract violator (e.g. one that stopped recording
a learnings row, or started naming a backend) or a regression **can never win**, no matter
how well it reads. Stay domain-neutral — the corpus is the spec; reflect back only what the
traces and the contract actually say.

## When to use

- The user says "optimize the eda skill", "improve a `/loom-*` skill from the corpus",
  "score this candidate SKILL.md", "run skillopt", or "is this candidate skill better?".
- You have a candidate rewrite of a skill and want a **machine-checkable** verdict on
  whether it is safe to ship (it satisfies the 7-point contract and beats the incumbent).

## When NOT to use

- To **deploy a validated model** to serving — use **`loom-deploy`** (skillopt optimizes a
  `SKILL.md` *text*, not a model; the two gates are parallel but distinct).
- To **read what passed/failed** in the field (run health, leaderboard, drift) — use
  **`loom-ops`** (skillopt *consumes* that corpus; it does not monitor).
- To **assemble a run write-up** for a teammate — use **`loom-report`** / **`loom-collab`**.

## 1. Intake — pin the verb + the candidate source (refuse if ambiguous)

Pin the inputs in the user's own terms and write them back for confirmation:

- **Verb (required)** — which `/loom-*` skill to optimize (e.g. `loom-eda`), resolved to
  `skills/<verb>/SKILL.md` (the incumbent / trainable artifact). **Refuse to start** without
  a real verb whose `SKILL.md` exists.
- **Candidate source (at most one)** — `--candidate PATH` (a proposed SKILL.md text to score,
  the deterministic, LLM-free source) **or** `--propose` (the OPTIONAL pluggable LLM proposer,
  a clearly-marked no-op when no model is configured). With **neither**, the run just
  **scores + reports** the incumbent against the contract + the captured corpus (a safe,
  read-only health read of the skill). Never pass both.
- **`--apply` (default OFF)** — whether to overwrite the shipped `SKILL.md` **in place**.
  Default is a **proposed sidecar** (`skills/<verb>/SKILL.candidate.md`); do not pass
  `--apply` until the user explicitly confirms after seeing the gate VERDICT + the diff.

## 2. Plan — show the plan + tier (workspace-write to propose; EXPENSIVE/MUTATE + always-gate to --apply)

Skillopt is **two-tiered** (see `CONVENTIONS.md` §1):

- **Propose (default) = workspace-write** — it reads the corpus + the incumbent, scores
  deterministically, and writes only a **sidecar** candidate in the workspace. Light/auto.
- **`--apply` = expensive/mutate, ALWAYS gate** — overwriting a *shipped* skill in the
  library is the consequential act, so it **always gates** and is **never model-auto-invoked**
  (`disable-model-invocation: true` in the frontmatter — the model proposes, only the user
  fires). It **NEVER auto-promotes**: even with `--apply`, the in-place overwrite happens
  **only when the gate PROMOTED** a candidate.

Show the plan and **stop at the gate**: "I'll capture `<verb>`'s `owned_by=general` corpus,
score the incumbent + candidate on the 7-point contract (hard) + corpus coverage (soft), and
gate; if it PROMOTES I'll propose a sidecar — **no in-place overwrite** unless you confirm
`--apply`." Name the exact verb and candidate source. **Do not run `--apply` until the user
confirms after seeing the VERDICT + diff.**

## 3. Run — call Loom's INTERFACE (the `loom` CLI), never re-implement the scorer

Speak only Loom's interface — shell out to the `loom` CLI, which captures the corpus
(HiveMind) and runs the pure scorer + gate (SkillOpt) through `loom skillopt`. **Never
re-implement the contract scorer or the gate in a prompt** — the whole point is that the
decision is plain-Python and machine-checkable, not a model talking itself into a promotion.
**Never touch raw S3**; the learnings corpus is read through Loom's own learnings store.

```bash
loom skillopt --verb loom-eda                          # score + report the incumbent + corpus digest (read-only health read)
loom skillopt --verb loom-eda --candidate ./cand.md    # default: score, gate, PROPOSE a sidecar (NO in-place write)
loom skillopt --verb loom-eda --candidate ./cand.md --apply   # overwrite SKILL.md IN PLACE — ONLY if the gate PROMOTED, and only after the user confirms
loom skillopt --verb loom-eda --propose                # OPTIONAL LLM proposer (no-op stub when no model is configured)
```

- The capture is filtered to the **`owned_by=general` IP boundary** — tenant-tagged rows are
  **excluded** from the cross-tenant moat (only general-method skill edits promote to the
  shared library). The corpus is read through Loom's learnings store; no datastore touch.
- The MLOps default is **Metaflow** and the search default is **AIDE**, both swappable by
  config — but skillopt itself is text-space and needs **no LLM** (the scorer + gate are
  deterministic); `--propose` is the *optional* model seam.
- Secrets/endpoints come from the **environment** only; the audit row persists no skill text
  and no secrets.

## 4. Verify — the gate IS the verifier; assert the scores + the diff

- The command prints the **corpus digest** (rollouts, success rate, verdict histogram, the
  recurring **failure modes**), then the **gate VERDICT** (`PROMOTE` / `KEEP` /
  `KEEP_ALL_DISQUALIFIED` / `KEEP_NO_CANDIDATE`) and the **incumbent + candidate scores**
  (`hard_ok` / `soft` / `total`, with any `hard_misses`).
- A `PROMOTE` prints a **unified diff** of the winner vs. the incumbent — confirm the change
  is what was intended before any `--apply`. A candidate with `hard_ok=False` is
  **DISQUALIFIED** (excluded before selection); a tie or regression **KEEPS** the incumbent.
- **Assert lineage.** Each corpus row skillopt scores against carries its source verb's
  **run pathspec + `@card`** reference (the mandated artifact the lifecycle verbs emit), so a
  promotion is grounded in real, lineage-tagged usage — not vibes. The proposed
  `SKILL.candidate.md` is itself the versioned candidate artifact.
- **Large output → cap.** The diff is capped inline (~20k chars); the audit is a small derived
  dict (verb · verdict · scores), **never** raw skill text or secrets.

## 5. Deliver — narrate the VERDICT, propose-or-apply, append a learnings row

- **Narrate the gate:** lead with the **VERDICT** and *why* — for a `KEEP`, whether it was a
  tie/regression (`KEEP`), a disqualified candidate (`KEEP_ALL_DISQUALIFIED`, with the
  `hard_misses`), or no candidate (`KEEP_NO_CANDIDATE`); for a `PROMOTE`, the score margin and
  the failure modes the winner newly covers. Make crystal clear whether this was a **proposed
  sidecar** (the default) or a **real in-place apply**.
- **Hand back the artifact:** the printed VERDICT + scores + diff, and — when promoted — the
  **sidecar path** (`skills/<verb>/SKILL.candidate.md`) for the default, or the confirmation
  that `SKILL.md` was overwritten in place for `--apply`. **Never deploy a worse skill:** a
  `KEEP` leaves the incumbent untouched, and even `--apply` refuses to write unless the gate
  PROMOTED.
- **Learnings:** the run appends one `command="skillopt"` row to the flywheel corpus
  (`learnings/rollouts.jsonl`) — verb · promoted · verdict · the incumbent/candidate scores ·
  the apply/propose flags — sanitized, `owned_by=general`, **no skill text, no secrets**. The
  CLI does this; do not hand-write the row. This closes the loop: the optimizer's own decisions
  are corpus fuel for the next round.
- **Next step:** if `KEEP_ALL_DISQUALIFIED`, point at the exact `hard_misses` to fix in the
  candidate; if a proposed `PROMOTE` and the user is satisfied with the diff, offer the
  explicit `--apply`; after an apply, suggest re-running the affected verb so the corpus
  reflects the improved skill.

## Composition — machine-checkable promotion gate (executable self-test)

- **Consumes:** the verb's `learnings/rollouts.jsonl` corpus (the `owned_by=general` slice via
  HiveMind capture) + the incumbent `skills/<verb>/SKILL.md` + an optional candidate text.
- **Exit gate:** the pure `optimize_skill(...)` scores the incumbent + every candidate with the
  deterministic `ContractCorpusScorer` (the **7-point acceptance contract** = the HARD
  constraints; corpus failure-mode coverage + convention adherence = the SOFT score) and
  promotes the **best hard-valid** candidate **only if** `best.total > incumbent.total + margin`
  — in plain Python, never a prompt the model can talk past. It **fails closed**: a contract
  violation DISQUALIFIES a candidate before selection (it can never be promoted even with a
  higher soft score), and a tie/regression KEEPS the incumbent.
- **Self-test (ships with the verb):** the gate has executable self-tests that feed a
  **known-bad candidate** (violates a hard contract constraint, or regresses) and assert it is
  **BLOCKED / never promoted** — `tests/test_skillopt.py::test_gate_never_promotes_contract_violator`
  and `::test_gate_never_promotes_regressing_candidate` (plus
  `::test_gate_promotes_best_valid_candidate` for selection and
  `::test_cli_apply_writes_only_when_promoted` for the gated `--apply`). This mirrors
  `loom-deploy`'s BLOCK-on-sub-threshold self-test. "Guards failing open" is exactly the
  failure mode these guard against — a never-worse gate without a test does not count.

---

## Acceptance test (the bar before this verb joins the pack)

1. **Speaks only the interface** — shells out to `loom skillopt` (HiveMind capture + the
   SkillOpt scorer/gate behind the `loom` CLI), never re-implements the contract scorer in a
   prompt, never touches raw S3; the corpus is read via Loom's learnings store.
2. **Output is a versioned artifact** — the gate VERDICT + scores + a unified diff, and a
   proposed `SKILL.candidate.md` sidecar (or an in-place edit under `--apply`), not a chat
   transcript.
3. **Approval tier is correct** — workspace-write to PROPOSE, **expensive/mutate + always-gate**
   to `--apply` (overwriting a shipped skill); `disable-model-invocation: true` so the model
   never auto-fires it, and it NEVER auto-promotes (the gate must PROMOTE *and* `--apply` must be
   passed).
4. **Writes a learnings row** — the run appends a sanitized `command="skillopt"` row to
   `learnings/rollouts.jsonl` every run (`owned_by=general`, no skill text/secrets).
5. **Exit gate has a self-test** — the never-promote-a-worse-skill gate is covered by the
   `tests/test_skillopt.py` tests above (contract violator + regression BLOCKED; selection;
   apply-only-when-promoted).
6. **Single free-text arg** — one `/loom-*` verb, plus the `--candidate`/`--propose` source and
   the explicit `--apply` safety flag.
7. **Dual-invocation** — user-typed only by design (`/loom-skillopt`); never model-auto-loaded
   (`disable-model-invocation: true`) because editing the shipped skill library is a deliberate,
   human-fired act.
