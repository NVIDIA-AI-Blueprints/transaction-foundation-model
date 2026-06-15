---
description: Diagnose the local Loom + Metaflow datastore stack (read-only) — the first thing to run on any setup/datastore failure.
argument-hint: <no args — diagnoses the local stack>
---

# /loom-doctor — diagnose the Loom + Metaflow stack (read-only)

Check the local Loom + Metaflow datastore stack. Read-only: it never prompts or
mutates. Run this first whenever a verb reports a setup-or-bad-args failure
(exit 2) or "datastore unreachable".

## Run — call the `loom_doctor` tool
Call `loom_doctor`. It checks the venv + `import loom`, `import metaflow`, the
datastore env vars, a TCP probe to the S3 endpoint, and a Client-API smoke
(counting ingested data objects — zero is fine). It exits 0 iff no check FAILs.

## Read the result and act
- `details.VERDICT == PASS` (exit 0) — the stack is healthy; proceed with the
  lifecycle verbs.
- `details.VERDICT == FAIL` (exit 1) — read `details.summary` / `error` for the
  failing check and its `fix:` line, and relay it. Common fixes: the minio
  port-forward is not running; the env was not sourced; the datastore was never
  stood up. If the datastore is unset/unreachable, point the user at
  `/loom-setup-metaflow` to stand up the local-dev stack.

A FAIL here is a domain outcome to act on, not a crash — read the verdict and tell
the user the exact next step. Do not retry lifecycle verbs blindly against a FAIL.
