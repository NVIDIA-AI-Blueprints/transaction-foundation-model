---
description: Assemble a sanitized shareable bundle of a run -> run + @card. The off-box SEND is OFF unless --send (IRREVERSIBLE/EXTERNAL); never auto-fired.
argument-hint: <a run pathspec (or an --experiment ID); --send only to really push off-box>
---

# /loom-collab — package a run for handoff (build-only by default; SEND always gates)

Package a run into a **sanitized, shareable bundle** — its report / model-card plus
a **lineage manifest** (pathspecs + fingerprints + commit) — so a teammate can
review exactly what was done and trace every claim. The bundle is sanitized
(secret-looking keys redacted, long strings truncated, raw objects reduced to a
repr — no secrets, no raw rows ride off-box). Building the bundle is workspace-write;
the **off-box SEND is OFF by default** — the default builds only (nothing leaves the
box) and only `send` pushes to an env/config-driven sink.

Package: $@

## 1. Intake — pin the source run (refuse without one)
- **Run (required)** — the run pathspec whose report/card to bundle (e.g. a
  `loom_report` or `loom_validate` run), via `run`; or an `experiment` id. Take a
  pathspec, never a raw S3 URI / loose file. **Refuse without a source.** (If the
  report content itself does not exist yet, run `/loom-report` first.)
- **`send` (default OFF)** — push the bundle off-box. Do NOT set it until the user
  explicitly confirms after seeing the bundle preview and the would-send target.

## 2. Plan — build = workspace-write; SEND = irreversible/external, always gate
The build is workspace-write, but the off-box **send** is the irreversible/external
tier (data leaving the perimeter) — so the send **always gates** and is **never
model-auto-invoked**. Show the plan and **stop for any send**: "I'll assemble a
**sanitized** shareable bundle of `<run>` (report/card + lineage manifest) —
**build-only, nothing leaves the box** unless you confirm `send` to `<env-driven
sink>`." Name the run, the resolved sink (env `LOOM_COLLAB_WEBHOOK` /
`LOOM_COLLAB_OUTBOX`, never a hardcoded target), and that everything is sanitized.

## 3. Run — call the `loom_collab` tool
Call `loom_collab` with `run` (or `experiment`). Default = build the sanitized
bundle only, no send. Only with explicit user confirmation set `send: true`. The
harness will require a confirmation before this irreversible verb runs.

## 4. Verify — assert lineage + sanitization
Confirm `status` and read the **would-send target** + whether it was **sent**. The
bundle payload is the sanitized report — no secrets, no raw rows. The lineage
manifest carries the source ref + card path + a content fingerprint + commit.

## 5. Deliver — narrate, return run + summary
- Walk through the **bundle preview** (sanitized top-level fields), the **lineage
  manifest**, and make crystal clear whether this was **build-only** (default) or a
  **real send** (and to which env-driven target).
- Hand back the run + `@card` + the typed bundle with its `VERDICT` (`BUILT` /
  `SEND_REQUESTED` / `SENT`).
- **Next step:** if build-only and the user is satisfied with the preview, offer the
  explicit `send`; otherwise hand back the run for them to share by pathspec.
