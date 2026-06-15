---
name: loom-deploy
description: Promote a validated solution through Loom — IRREVERSIBLE / EXTERNAL. The centerpiece is the cross-verb exit gate — deploy ASSERTS the upstream loom-validate VERDICT==PASS (no leakage, holdout clears the floor) before it will deploy; a sub-threshold / REVIEW / FAIL / leaky validation BLOCKS it. The real external apply is OFF by default — the default run produces a deployment PLAN + a staged registry manifest, no external mutation; only --apply (with an ALLOWED gate) performs the real action. Use when the user says "ship this", "deploy the model", "promote to serving", "register this model". NEVER auto-fire — the model proposes; only the user fires.
when_to_use: "ship/promote a validated model, deploy to serving, register a model, schedule a validated flow, produce a deployment plan"
when_not_to_use: "to check whether a candidate is good enough first, use loom-validate (deploy asserts its VERDICT); to monitor a deployed flow afterwards, use loom-ops; to share a run with a teammate, use loom-collab."
argument-hint: "<an upstream loom-validate run pathspec (or a solution run); --apply only to really deploy>"
disable-model-invocation: true
---

# loom-deploy

Promote a validated solution to a deployment target — the **irreversible / external**
verb. This is a **planned, always-gated run through Loom's MLOps interface**, never a
loose `mlflow register` call, because its centerpiece is a **machine-checkable exit
gate**: deploy **asserts the upstream `loom-validate` `VERDICT == PASS`** (no leakage,
sealed holdout clears any declared floor) **in plain Python** before it will deploy —
a sub-threshold / `REVIEW` / `FAIL` / leaky validation **BLOCKS** the deploy and the
gate refuses rather than fails open. The real external apply is **OFF by default**:
the default run produces a deployment **PLAN** + a staged registry **manifest** (model
ref + lineage + the validate metric) with **no external mutation**; only `--apply`
(and only when the gate ALLOWED) performs the real, env/config-driven action. Stay
domain-neutral — the deployment target is an env/config-driven string, **never** a
hardcoded customer/vertical.

## When to use

- The user has a `loom-validate` run that **PASSED** and asks to "ship this", "deploy
  the model", "promote to serving", or "register this model".
- They want a **deployment plan / staged manifest** to review before the real apply.

## When NOT to use

- To *check whether a candidate is good enough* first — run **`loom-validate`**
  (deploy asserts its `VERDICT`; deploy is not a substitute for validating).
- To *monitor* a deployed flow afterwards — use **`loom-ops`**.
- To *share* a run with a teammate — use **`loom-collab`**.

## 1. Intake — pin the upstream validate run (refuse without one)

Pin the inputs in the user's own terms and write them back for confirmation:

- **Validate run (required)** — the upstream **`loom-validate` run pathspec** (e.g.
  `ValidateFlow/12`) whose `VERDICT` gates this deploy, via `--validate`. A
  `--solution` run may be promoted instead, but it is still read for a `report`
  artifact and the gate **BLOCKS** if it carries no PASS validate report. **Refuse to
  start without one** — you cannot deploy what was never validated.
- **`--apply` (default OFF)** — whether to perform the real external action. Default
  is a PLAN + staged register only; do not pass `--apply` until the user explicitly
  confirms after seeing the plan and the gate decision.

## 2. Plan — show the plan + tier (irreversible/external, ALWAYS gate)

Deploy is the **irreversible / external tier** of the approval matrix (see
`CONVENTIONS.md`): it **always gates** and is **never model-auto-invoked**
(`disable-model-invocation: true` in the frontmatter — the model proposes, only the
user fires). Show the plan and **stop at the gate**: "I'll gate on
`<validate-run>`'s `VERDICT == PASS`; if it ALLOWS, produce a deployment PLAN + a
staged registry manifest to `<env-driven target>` — **no external mutation** unless
you confirm `--apply`." Name the exact upstream run, the resolved target (env/config
`LOOM_DEPLOY_TARGET`, never a hardcoded customer), and that the default is a dry-run.
**Do not run `--apply` until the user confirms after seeing the plan + gate
decision.**

## 3. Run — call Loom's MLOps INTERFACE (the `loom` CLI), never the backend

Speak only Loom's interface — shell out to the `loom` CLI, which resolves the MLOps
provider by name (default **Metaflow**, swappable by config) and runs the deploy flow
through the interface's `run_flow` seam. **Never call Metaflow or AIDE directly, and
never touch raw S3** — the upstream validate report is read only through the Client
API; the datastore is the interface's opaque concern.

```bash
loom deploy --validate <RUN>            # default: PLAN + staged register, NO external mutation
loom deploy --validate <RUN> --apply    # real external action — ONLY after the user confirms
```

- The work executes as a **Metaflow run**; the upstream validate `report` is read via
  the Client API. The default is a dry-run / staged register; `--apply` performs the
  real action **only when the gate ALLOWED**.
- Lifecycle flows need the **metaflow** MLOps provider — the `local` dev provider
  cannot run them (it will say so, pointing at `--mlops metaflow`).
- The deployment target + sink come from the **environment** only (e.g.
  `LOOM_DEPLOY_TARGET` / `LOOM_DEPLOY_DIR`) — never a hardcoded customer, never on the
  command line.

## 4. Verify — the gate IS the verifier; assert lineage

- The command returns a **run pathspec** and the **`@card` reference**; confirm it
  reported success and read the **GATE decision** — `ALLOW` or `BLOCK` (with the
  blocking reasons). The plan/manifest is a small *derived* dict (model ref +
  lineage + validate metric) — no raw rows.
- The manifest's lineage points back to **exactly what was validated** (the validate
  run, its `dataset_ref`, target, the sealed-holdout metric, the commit). A `BLOCKED`
  status means nothing was registered, staged or applied.

## 5. Deliver — narrate the @card, return run + summary + GATE decision, append a learnings row

- **Narrate the `@card`:** lead with the **GATE decision** (`ALLOW`/`BLOCK`) and,
  when blocked, the exact reasons (verdict not PASS / leakage present / no holdout /
  below the floor); then the **what-would-deploy** (the validate metric the manifest
  trusts) and the **lineage**. Make crystal clear whether this was a **staged PLAN**
  (the default) or a **real apply**.
- **Hand back the mandated artifact:** the versioned **Metaflow run + `@card`** plus
  the typed plan the CLI prints, with both the `GATE` decision and the top-level
  `VERDICT` (`STAGED` / `DEPLOYED` / `BLOCKED`).
- **Learnings:** the run appends one `command="deploy"` row to the flywheel corpus
  (`learnings/rollouts.jsonl`) — source run · apply flag · gate decision · upstream
  verdict · deploy status · run + card pathspecs — sanitized, no secrets. The CLI
  does this; do not hand-write the row.
- **Next step:** if `BLOCKED`, point back at `loom-validate` to fix the leak / raise
  the holdout *before* any deploy; if `STAGED` and the user is satisfied, offer the
  explicit `--apply`; after a real apply, offer `loom-ops` to monitor it.

## Composition — machine-checkable exit gate (executable self-test)

- **Consumes:** an upstream `loom-validate` run pathspec via `--validate` (the
  `validate → deploy` composition edge), or a `--solution` run read for a validate
  report.
- **Exit gate:** the pure `deploy_gate(validate_summary)` asserts `VERDICT == PASS`
  (no leakage, holdout present and clearing the optional floor) and returns
  `allow/block` **in plain Python** — never a prompt the model can talk past. It
  **fails closed**: anything but a clean PASS BLOCKS, and even `--apply` cannot run
  the real action unless the gate ALLOWED.
- **Self-test (ships with the verb):** the gate has an executable self-test that
  feeds a **known-bad (sub-threshold) validate summary** and asserts the deploy is
  BLOCKED — `tests/test_deploy.py::test_deploy_gate_blocks_subthreshold_validate`
  (plus `::test_deploy_gate_blocks_missing_report`,
  `::test_deploy_gate_blocks_pass_with_leakage`,
  `::test_deploy_gate_blocks_below_holdout_floor`), and
  `::test_deploy_gate_allows_clean_pass` asserts a clean PASS ALLOWS. The
  `build_deploy_plan` tests further assert the real apply happens **only** when
  allowed **and** `--apply` is set
  (`::test_deploy_plan_applied_only_when_allowed_and_apply`,
  `::test_deploy_plan_staged_by_default`). "Guards failing open" is exactly the
  failure mode these guard against.

---

## Acceptance test (the bar before this verb joins the pack)

1. **Speaks only the interface** — shells out to `loom deploy` (the MLOps interface,
   provider-by-name), never Metaflow/AIDE directly, never raw S3; the upstream
   validate report is read via the Client API.
2. **Output is a versioned run + `@card`** — a deployment PLAN / manifest, not a chat
   transcript or a loose register call.
3. **Approval tier is correct** — irreversible/external tier: **always gates**, the
   real apply is behind `--apply` and OFF by default, and the skill sets
   `disable-model-invocation: true` so the model never auto-fires it.
4. **Writes a learnings row** — the run appends a sanitized `command="deploy"` row to
   `learnings/rollouts.jsonl` (the CLI does this every run).
5. **Exit gate has a self-test** — the BLOCK-on-sub-threshold and ALLOW-on-PASS gate
   is covered by the `tests/test_deploy.py` tests above (and apply-only-when-allowed).
6. **Single free-text arg** — one upstream validate (or solution) run pathspec, plus
   the explicit `--apply` safety flag.
7. **Dual-invocation** — user-typed only by design (`/loom-deploy`); never
   model-auto-loaded (`disable-model-invocation: true`) because the action is
   irreversible/external.
