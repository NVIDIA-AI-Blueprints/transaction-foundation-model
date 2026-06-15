---
description: Gate on a validate VERDICT==PASS and produce a deploy PLAN -> run + @card. IRREVERSIBLE/EXTERNAL — the real apply is OFF unless --apply, and is never auto-fired.
argument-hint: <an upstream loom-validate run pathspec (or a --solution run); --apply only to really deploy>
---

# /loom-deploy — promote a validated solution (IRREVERSIBLE / EXTERNAL — always gate)

Promote a validated solution to a deployment target — the **irreversible /
external** verb. Its centerpiece is a machine-checkable exit gate: deploy
**asserts the upstream `loom-validate` `VERDICT == PASS`** (no leakage, sealed
holdout clears any floor) **before** it will deploy; a sub-threshold / `REVIEW` /
`FAIL` / leaky validation **BLOCKS** it. The real external apply is **OFF by
default**: the default produces a deployment **PLAN** + a staged manifest with no
external mutation; only `apply` performs the real action, and only when the gate
ALLOWED.

Deploy: $@

## 1. Intake — pin the upstream validate run (refuse without one)
- **Validate run (required)** — the upstream `loom_validate` run pathspec (e.g.
  `ValidateFlow/12`) whose `VERDICT` gates this deploy, via `validate`. **Refuse
  without one** — you cannot deploy what was never validated. **Gate-assert
  first:** read that validate run's `details.VERDICT`; if it is not `PASS`, stop
  and tell the user what to fix — do not propose a deploy.
- **`apply` (default OFF)** — the real external action. Do NOT set it until the
  user explicitly confirms after seeing the plan and the gate decision.

## 2. Plan — always gate; never auto-fire
This is the irreversible/external tier: it **always gates** and is **never
model-auto-invoked**. The model proposes; only the user fires. Show the plan and
**stop**: "I'll gate on `<validate-run>`'s `VERDICT == PASS`; if it ALLOWS, produce
a deployment PLAN + a staged registry manifest to `<env-driven target>` — **no
external mutation** unless you confirm `apply`." Name the upstream run and the
resolved target (env `LOOM_DEPLOY_TARGET`, never a hardcoded customer). Do not set
`apply` until the user confirms.

## 3. Run — call the `loom_deploy` tool
Call `loom_deploy` with `validate` (the default = PLAN + staged register, no
external mutation). Only with explicit user confirmation set `apply: true`. The
harness will require an interactive confirmation before any irreversible verb runs.

## 4. Verify — the gate IS the verifier
Read the **GATE decision** in `details` — `ALLOW` or `BLOCK` (with reasons). A
`BLOCKED` status means nothing was registered, staged, or applied. The manifest's
lineage points back to exactly what was validated.

## 5. Deliver — narrate, return run + GATE decision
- Lead with the **GATE decision** and, when blocked, the exact reasons (verdict not
  PASS / leakage / no holdout / below floor); then the what-would-deploy and the
  lineage. Make crystal clear whether this was a **staged PLAN** (default) or a
  **real apply**.
- Hand back the run + `@card` + the typed plan with the `GATE` decision and the
  top-level `VERDICT` (`STAGED` / `DEPLOYED` / `BLOCKED`).
- **Next step:** if `BLOCKED`, point back at `/loom-validate`; if `STAGED` and the
  user is satisfied, offer the explicit `apply`; after a real apply, offer
  `/loom-ops` to monitor it.
