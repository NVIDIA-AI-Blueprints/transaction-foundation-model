---
description: Stand up the local-dev Metaflow datastore (minikube + minio) so the lifecycle verbs have a backend, then verify with loom doctor. EXPENSIVE/MUTATE — always gate; local-dev + reversible.
argument-hint: "<confirm — stand up the local-dev Metaflow-on-minikube datastore>"
---

# /loom-setup-metaflow — stand up the local-dev datastore (EXPENSIVE/MUTATE — always gate)

Stand up the local-dev Metaflow datastore the lifecycle verbs need — a minikube
cluster running **minio** (an S3-compatible store) with the Metaflow profile
pointed at it — so Loom's MLOps interface has a real backend. This is the one verb
that is *not* a lifecycle run and not a `loom_<verb>` tool: it drives the shipped,
idempotent installer script via bash, then verifies with the `loom_doctor` tool.
Everything is local-dev and reversible (`minikube delete`). To only *diagnose* the
stack, run `/loom-doctor` instead.

## 1. Intake — confirm the setup
Confirm: the **local-dev Metaflow datastore** (minikube + in-cluster minio, S3
datastore on minio, local metadata) — no cloud, no GPU, entirely on this machine in
the minikube `loom` namespace, reversible via `minikube delete`. The auto-install
path is macOS / Homebrew; on Linux the script refuses to auto-install and names the
tool to install by hand. If the user wants a *cloud* datastore, this is the wrong
tool — say so.

## 2. Plan — show EXACTLY what installs/starts, then ALWAYS gate
This is the expensive/mutate tier: it installs software and starts a local cluster,
so it **always gates** — present the exact mutations and do not proceed until the
user confirms:
- **Tools it may install (only what is missing, macOS `brew`):** colima + docker,
  minikube, kubectl, aws-cli. Already-present tools are left alone.
- **Cluster it starts:** `minikube start --driver=docker`; namespace `loom`;
  `kubectl apply` of the minio manifest; the `metaflow` bucket once if absent.
- **Files it writes:** the gitignored `.env.metaflow` (local-dev minio creds only —
  no secrets).
- **What it does NOT do:** never pushes off-box, never deploys, never edits a remote
  registry.

State: "This installs the missing local tools above and starts a minikube cluster
on this machine — all local-dev, reversible with `minikube delete`. Proceed?" Stop
here until the user confirms.

## 3. Run — drive the shipped installer via bash
Once confirmed, run the verified idempotent recipe via the shipped script (never
hand-rolled cluster commands):

```bash
bash scripts/setup_metaflow_minikube.sh
```

It is `set -euo pipefail`, idempotent, safe to re-run. Where it needs judgment
(missing prerequisite on a non-macOS box; missing Homebrew; a step that fails),
surface the script's own diagnostic line, help interpret it, and re-run the
idempotent script after the fix. Never silently retry; never invent cluster
commands the script does not run. Secrets come from the environment only.

## 4. Verify — run loom doctor; it MUST end PASS
Do the two manual steps the script prints — keep the minio port-forward alive in a
separate terminal, then `source .env.metaflow` — and verify by calling the
`loom_doctor` tool. The exit gate is `loom_doctor` reading `VERDICT == PASS`. If any
check is FAIL, read its `fix:` line back and act on it (commonly: the port-forward
is not running, or the env was not sourced). Do not declare setup done until doctor
reads PASS. Then call `loom_datasets` to smoke the Client API (an empty list on a
fresh datastore is fine).

## 5. Deliver — hand back the env file + the next step
Narrate what was installed vs already-present, that the cluster is up, and that
`loom_doctor` reads PASS. Hand back the sourceable `.env.metaflow` and the live
port-forward command (remind the user it must stay running). Point to `/loom-ingest`
to register a source, then the lifecycle verbs. Mention `minikube delete` is the
one-line teardown.
