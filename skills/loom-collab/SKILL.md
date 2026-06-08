---
name: loom-collab
description: Assemble a sanitized, shareable bundle of a run through Loom — its report/model-card payload + a lineage manifest (pathspecs + fingerprints + commit) — as a versioned Metaflow run + @card, then narrate it. The off-box SEND is OFF by default — the default run builds the bundle only, no data leaves the box; only --send pushes to an env/config-driven sink (LOOM_COLLAB_WEBHOOK / LOOM_COLLAB_OUTBOX, never a hardcoded target). Use when the user says "share this run", "hand this off to a teammate", "package this for review", "send the report". Sending data off-box is IRREVERSIBLE/EXTERNAL — NEVER auto-fire the send; the model proposes, only the user fires.
when_to_use: "share/hand off a run or report, package a run for review, build a shareable bundle with lineage, send a model-card to a teammate"
when_not_to_use: "to ship/promote a model to serving, use loom-deploy; to monitor run health or drift, use loom-ops; to assemble the report content itself, use loom-report first."
argument-hint: "<a run pathspec (or an --experiment ID); --send only to really push off-box>"
disable-model-invocation: true
---

# loom-collab

Package a run into a **sanitized, shareable bundle** — its report / model-card plus a
**lineage manifest** (pathspecs + fingerprints + commit) — so a teammate can review
exactly what was done and trace every claim. This is a **planned, gated run through
Loom's MLOps interface**: it reads the run's report via the Client API, assembles +
**sanitizes** the bundle (secret-looking keys redacted, long strings truncated, raw
objects reduced to a repr — **no secrets, no raw rows ride off-box**), and emits a
versioned Metaflow run + an `@card`. Building the bundle is workspace-write; the
actual **off-box SEND is OFF by default** — the default run **builds only** (no data
leaves the box) and only `--send` pushes to an **env/config-driven** sink. Stay
domain-neutral — the sink is `LOOM_COLLAB_WEBHOOK` / `LOOM_COLLAB_OUTBOX`, **never** a
hardcoded customer/vertical target.

## When to use

- The user asks to "share this run", "hand this off to a teammate", "package this for
  review", or "send the report".
- They want a **lineage-grounded bundle** of a run they can hand off.

## When NOT to use

- To *ship / promote* a model to serving — use **`loom-deploy`**.
- To *monitor* run health or drift — use **`loom-ops`**.
- To *assemble the report content itself* (runs + metrics + lineage into a model-
  card) — run **`loom-report`** first, then bundle that run.

## 1. Intake — pin the source run (refuse without one)

Pin the inputs in the user's own terms and write them back for confirmation:

- **Run (required)** — the **run pathspec** whose report/card to bundle (e.g.
  `ValidateFlow/12` or a `loom-report` run), via `--run`; or an `--experiment ID` to
  bundle the experiment's report. Take a pathspec, **never a raw S3 URI or a loose
  local file**. **Refuse to start without a source.**
- **`--send` (default OFF)** — whether to push the bundle off-box. Default builds the
  bundle only; do not pass `--send` until the user explicitly confirms after seeing
  the bundle preview and the would-send target.

## 2. Plan — show the plan + tier (build = workspace-write; SEND = irreversible/external)

The build is **workspace-write** (a bundle as a run + `@card`), but the **off-box
SEND is the irreversible / external tier** (data leaving the perimeter) — so the
send **always gates** and is **never model-auto-invoked** (`disable-model-invocation:
true` in the frontmatter — the model proposes, only the user fires). Show the plan
and **stop at the gate for any send**: "I'll assemble a **sanitized** shareable
bundle of `<run>` (report/card + lineage manifest) — **build-only, nothing leaves the
box** unless you confirm `--send` to `<env-driven sink>`." Name the exact run, the
resolved sink (env `LOOM_COLLAB_WEBHOOK` / `LOOM_COLLAB_OUTBOX`, never a hardcoded
target), and that everything is sanitized. **Do not run `--send` until the user
confirms after seeing the bundle preview + the would-send target.**

## 3. Run — call Loom's MLOps INTERFACE (the `loom` CLI), never the backend

Speak only Loom's interface — shell out to the `loom` CLI, which resolves the MLOps
provider by name (default **Metaflow**, swappable by config) and runs the collab flow
through the interface's `run_flow` seam. **Never call Metaflow or AIDE directly, and
never touch raw S3** — the run's report is read only through the Client API; the
datastore is the interface's opaque concern.

```bash
loom collab --run <PATHSPEC>           # default: build the sanitized bundle only, NO send
loom collab --run <PATHSPEC> --send    # push off-box to the env/config sink — ONLY after the user confirms
```

- The work executes as a **Metaflow run**; the source run's `report`/`summary` +
  `@card` reference are read via the Client API. The default is build-only; `--send`
  pushes to the env/config-driven sink only.
- Lifecycle flows need the **metaflow** MLOps provider — the `local` dev provider
  cannot run them (it will say so, pointing at `--mlops metaflow`).
- The send sink comes from the **environment** only (`LOOM_COLLAB_WEBHOOK` /
  `LOOM_COLLAB_OUTBOX`) — never a hardcoded customer, never on the command line.

## 4. Verify — assert lineage + sanitization

- The command returns a **run pathspec** and the **`@card` reference**; confirm it
  reported success and read the **would-send target** + whether it was **sent**. The
  bundle payload is the sanitized report — no secrets, no raw rows.
- The **lineage manifest** carries the source ref + card path + a content
  **fingerprint** + commit, so the bundle traces back to exactly the run it was built
  from.

## 5. Deliver — narrate the @card, return run + summary, append a learnings row

- **Narrate the `@card`:** walk the user through the **bundle preview** (the
  sanitized top-level fields of the report), the **lineage manifest** (source ref,
  card path, fingerprint, commit), and make crystal clear whether this was
  **build-only** (the default) or a **real send** (and to which env-driven target).
- **Hand back the mandated artifact:** the versioned **Metaflow run + `@card`** plus
  the typed bundle the CLI prints, with its `VERDICT` (`BUILT` / `SEND_REQUESTED` /
  `SENT`).
- **Learnings:** the run appends one `command="collab"` row to the flywheel corpus
  (`learnings/rollouts.jsonl`) — source ref · send flag · whether it sent · bundle
  fingerprint · run + card pathspecs — sanitized, never the payload, raw rows, or
  secrets. The CLI does this; do not hand-write the row.
- **Next step:** if build-only and the user is satisfied with the preview, offer the
  explicit `--send`; otherwise hand back the run for them to share by pathspec.

## Composition — machine-checkable exit gate (executable self-test)

- **Consumes:** a run pathspec via `--run` (the `report → collab` composition edge —
  a `loom-report`/`loom-validate` run), or an `--experiment` id.
- **Exit gate:** the off-box send is **OFF by default** and is the gated, irreversible
  action; the pure `build_bundle` records the send *intent* and the would-send sink
  but **never performs the send itself** (`sent` is always `False` from the builder),
  and everything is run through `sanitize_bundle` first so no secret/raw-row rides
  off-box.
- **Self-test (ships with the verb):** the send-off-by-default + sanitization gate has
  executable self-tests — `tests/test_collab.py::test_build_bundle_send_off_by_default`
  asserts the default is build-only (`sent=False`, verdict `BUILT`),
  `::test_build_bundle_send_requested_records_intent_not_sent` asserts even a requested
  send is not performed by the pure builder, and
  `::test_sanitize_redacts_secret_keys` / `::test_sanitize_truncates_long_strings` /
  `::test_sanitize_reduces_non_json_objects_to_repr` assert no secret/long-blob/raw
  object survives into the bundle. "A bundle quietly carrying a secret off-box" is the
  failure mode these guard against.

---

## Acceptance test (the bar before this verb joins the pack)

1. **Speaks only the interface** — shells out to `loom collab` (the MLOps interface,
   provider-by-name), never Metaflow/AIDE directly, never raw S3; the run's report is
   read via the Client API.
2. **Output is a versioned run + `@card`** — a sanitized shareable bundle, not a chat
   transcript or a loose file dump.
3. **Approval tier is correct** — build = workspace-write; the off-box send is
   irreversible/external: behind `--send`, OFF by default, gated, and the skill sets
   `disable-model-invocation: true` so the model never auto-fires the send.
4. **Writes a learnings row** — the run appends a sanitized `command="collab"` row to
   `learnings/rollouts.jsonl` (the CLI does this every run).
5. **Exit gate has a self-test** — send-off-by-default + sanitization is covered by the
   `tests/test_collab.py` tests above.
6. **Single free-text arg** — one run pathspec (or an `--experiment`), plus the
   explicit `--send` safety flag.
7. **Dual-invocation** — user-typed only by design (`/loom-collab`); never
   model-auto-loaded (`disable-model-invocation: true`) because the send is
   irreversible/external (data off-box).
