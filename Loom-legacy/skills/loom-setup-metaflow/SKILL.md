---
name: loom-setup-metaflow
description: Stand up Loom's LOCAL-DEV Metaflow datastore on minikube + minio (an S3-compatible store) so the MLOps interface has a backend, then verify it with `loom doctor`. EXPENSIVE / MUTATE — it installs software (colima/docker, minikube, kubectl, aws-cli via brew on macOS) and starts a local cluster, so it ALWAYS gates and shows exactly what it will install/start before touching the machine. Local-dev + fully reversible (`minikube delete`). Use when the user says "set up Loom", "set up the Metaflow datastore", "stand up minikube/minio", "I need a datastore for the lifecycle verbs", or `loom doctor` reports the datastore is unreachable. Drives scripts/setup_metaflow_minikube.sh.
when_to_use: "set up the local Metaflow datastore, stand up minikube + minio for Loom, give the lifecycle verbs a backend, fix an unreachable datastore reported by loom doctor"
when_not_to_use: "to only diagnose the stack (no install), run `loom doctor` directly (read-only, never prompts); to register a source as a data object once the datastore is up, use loom-connect; to run the lifecycle once it is up, use the lifecycle verbs."
argument-hint: "<confirm: stand up the local-dev Metaflow-on-minikube datastore>"
---

# loom-setup-metaflow

Stand up the **local-dev Metaflow datastore** the lifecycle verbs need — a minikube
cluster running **minio** as an S3-compatible object store, with the Metaflow profile
pointed at it — so Loom's MLOps interface (default **Metaflow**) has a real backend to
read/write through the Client API. This is a **planned, always-gated, idempotent
installer**, not a loose `brew install` spree: it detects each prerequisite before
installing, applies the cluster objects re-runnably, writes the sourceable env file,
and ends by running the read-only `loom doctor` until it reads **PASS**. It is the one
thing that is *not* a lifecycle run — it provisions the substrate every other verb
assumes. Everything it does is **local-dev and reversible** (`minikube delete` tears
the whole cluster down); the only credentials it touches are the **local-dev minio
creds already in [`manifests/minio.yaml`](manifests/minio.yaml)** (`minioadmin` /
`minioadmin123`) — not secrets.

## When to use

- The user asks to "set up Loom", "set up the Metaflow datastore", "stand up
  minikube/minio", or says the lifecycle verbs have no datastore.
- `loom doctor` reports the datastore env is unset or the endpoint is **unreachable**
  and the user wants the local stack stood up to fix it.

## When NOT to use

- To **only diagnose** the stack (no install, no cluster) — run **`loom doctor`**
  directly; it is read-only and never prompts.
- To **register a source** as a data object once the datastore is up — hand off to
  **`loom-connect`** (`loom ingest` / `loom datasets`).
- To **run the lifecycle** once the datastore is up — hand off to `loom-eda` /
  `loom-features` / `loom-validate` / etc.

## 1. Intake — confirm the setup (local-dev Metaflow-on-minikube)

Pin what is being stood up and write it back for confirmation:

- **What** — the **local-dev Metaflow datastore**: a minikube cluster + an in-cluster
  **minio** S3-compatible store, with the Metaflow profile pointed at it (S3 datastore
  on minio, local metadata). This is the verified laptop recipe — **no cloud, no GPU**.
- **Where** — entirely on this machine, in the minikube `loom` namespace. Reversible
  with `minikube delete`.
- **Platform** — the auto-install path is **macOS / Homebrew**. On Linux the script
  refuses to auto-install and tells the user which tool to install by hand; confirm the
  platform before proceeding.

If the user actually wants a *cloud* datastore (a shared/team Metaflow profile), this
verb is the wrong tool — say so; it provisions the **local-dev** stack only.

## 2. Plan — show EXACTLY what will be installed/started, then ALWAYS gate

Setup is the **expensive / mutate tier** of the approval matrix (see
`CONVENTIONS.md`): it **installs software and starts a local cluster**, so it
**always gates** — present the exact mutations and do not proceed until the user
confirms. It is **not** `disable-model-invocation` (the work is local-dev and fully
reversible via `minikube delete`), but the gate is mandatory. Show the plan before
running anything:

- **Tools it may install (only what is missing, detected first; macOS `brew`):**
  `colima` + `docker` (the docker runtime on macOS), `minikube`, `kubectl`, `aws`
  (aws-cli). Each is detected before install — already-present tools are left alone.
- **Cluster it starts:** `minikube start --driver=docker` (only if not already
  running); namespace `loom`; `kubectl apply` of
  [`manifests/minio.yaml`](manifests/minio.yaml) (the minio Deployment + Service);
  the `metaflow` datastore bucket created once if absent.
- **Files it writes:** the gitignored `.env.metaflow` at the repo root (7 exports,
  local-dev minio creds only — no secrets).
- **What it does NOT do:** it never pushes off-box, never deploys, never edits a remote
  registry. The only creds are the local-dev minio creds already in the manifest.

State the cost/data shape plainly: "This installs the missing local tools above and
starts a minikube cluster on this machine — all local-dev, reversible with
`minikube delete`. Proceed?" **Stop here until the user confirms.**

## 3. Run — drive the installer; adapt to the user's machine

Once gated, run the verified, idempotent recipe via the shipped script — never
hand-rolled cluster commands:

```bash
bash scripts/setup_metaflow_minikube.sh
```

The script is `set -euo pipefail`, **idempotent, and safe to re-run**. It echoes each
step (`==> ...`): detect-before-install each prerequisite, start colima + minikube
only if down, create the `loom` namespace, `kubectl apply` the minio manifest,
`kubectl wait` for minio ready, create the `metaflow` bucket if absent (via a throwaway
in-cluster `minio/mc` pod; it echoes an aws-cli fallback if that does not confirm), and
write the 7 exports to `.env.metaflow`.

**Where the script needs judgment, the LLM assists** — this is the gated, LLM-assisted
part of the installer:

- **Missing prerequisite on a non-macOS box:** the script refuses auto-install and
  names the tool — suggest the platform's package-manager install (`apt`/`dnf`/etc.),
  let the user run it, then re-run the script (idempotent).
- **Missing Homebrew on macOS:** point the user at https://brew.sh, then re-run.
- **A step fails** (e.g. minio not ready within the timeout, the in-cluster bucket
  setup did not confirm): surface the script's own diagnostic line
  (`kubectl -n loom get pods,events`, or the echoed aws-cli `s3 mb` fallback once the
  port-forward is up), help interpret it, and re-run the (idempotent) script after the
  fix. Never silently retry; never invent cluster commands the script does not run.

Secrets/endpoints come from the **environment only** — the local-dev minio creds live
in the manifest + the generated `.env.metaflow`; never echo other key material into the
transcript.

## 4. Verify — run `loom doctor`; it MUST end PASS

After the script completes, do the two manual steps it prints and verify:

```bash
# In a SEPARATE terminal, keep the port-forwards alive (datastore + metadata + UI):
kubectl port-forward -n loom svc/minio 9000:9000 9001:9001 &
kubectl port-forward -n loom svc/metaflow-metadata 8080:8080 &
kubectl port-forward -n loom svc/metaflow-ui 3000:3000 &

# Then source the env and run the read-only doctor (must end VERDICT: PASS):
source .env.metaflow && loom doctor
```

- `loom doctor` is **read-only** (it never prompts/mutates) — it checks the venv +
  `import loom`, `import metaflow`, the datastore env vars, a TCP socket probe to
  `METAFLOW_S3_ENDPOINT_URL`, the **metadata service** (`/ping` on `METAFLOW_SERVICE_URL`
  when `METAFLOW_DEFAULT_METADATA=service` — so runs register to the service, not local
  `~/.metaflow` files), and a Client-API smoke counting ingested data objects (zero is
  fine). It exits 0 iff no check FAILs.
- **The exit gate is `loom doctor` reading PASS.** If any line is FAIL, read its
  `fix:` line back to the user and act on it (commonly: a port-forward is not running,
  so the endpoint or metadata-service probe FAILs → start the port-forward and re-run;
  or the env was not sourced → `source .env.metaflow`). Do not declare the setup done
  until the VERDICT line reads `PASS`.

### Local dashboards (hand these to the user)

All three are reached through the port-forwards above — keep them running:

| Dashboard | URL | What it shows |
|---|---|---|
| **Metaflow UI** | http://localhost:3000 | the visual dashboard — runs, DAGs, timelines, artifacts (the SPA's `/api` is proxied to the UI backend in-cluster, so this one port-forward is all the browser needs) |
| **minio console** | http://localhost:9001 | browse the S3 datastore — login `minioadmin` / `minioadmin123` (local-dev only) |
| **Metaflow metadata service** | http://localhost:8080 | the metadata API Loom registers/reads runs through (`/ping`, `/flows`, …) |

`loom report` / `loom datasets` / `loom viz` give Loom's own run views from the CLI
(no port-forward needed). The first time the UI is set up, the installer builds the
`metaflow-ui` image from source — a one-time native build; on later runs it is reused.

Then smoke the datastore through the Client API:

```bash
loom datasets   # lists ingested data objects (an empty list on a fresh datastore is fine)
```

## 5. Deliver — narrate, hand back the env file + the next step

- **Narrate** what was installed vs already-present, that the cluster is up in the
  `loom` namespace, and that `loom doctor` reads **PASS**.
- **Hand back the deliverable:** the sourceable **`.env.metaflow`** (source it in any
  shell that runs a datastore verb), the live port-forward commands, and the **local
  dashboards** (Metaflow UI http://localhost:3000, minio console http://localhost:9001,
  metadata service http://localhost:8080 — see the table above). Remind the user the
  port-forwards must stay running for the datastore/UI to be reachable.
- **Next step:** the datastore is ready — point to **`loom-connect`** (`loom ingest`)
  to register a source as a data object, then the lifecycle verbs (`loom-eda`,
  `loom-features`, `loom-validate`, …). Mention the teardown (`minikube delete`) is the
  one-line reverse.
- This verb **provisions infra**, so it does not itself emit a Metaflow run / `@card` /
  learnings row (there is no flow to run yet); the deliverable is the verified
  datastore. The very next lifecycle run is the first to write to the flywheel corpus.

## Composition — machine-checkable exit gate (executable self-test)

- **Consumes:** nothing upstream — this is the front-door provisioning step.
- **Exit gate:** `loom doctor`'s **VERDICT** is the machine-checkable gate. A correctly
  set-up stack (full datastore env + a reachable endpoint) must read **`VERDICT: PASS`
  (exit 0)**; a broken stack (datastore env unset) must read **`VERDICT: FAIL`
  (exit 1)**. The downstream lifecycle verbs assume a PASS here.
- **Self-test:** the doctor gate ships executable self-tests in
  `tests/test_doctor.py` — the env-check function PASSes on a full stub env and FAILs
  (with the source-the-env `fix:` string) when vars are missing, the endpoint probe
  FAILs on an unreachable host:port, and the `loom doctor` CLI exits 0 on an all-set
  stub and 1 when the datastore env is unset (the two exit-gate self-tests:
  full-env-reachable ⇒ PASS/exit 0; env-unset ⇒ FAIL/exit 1).

---

## Acceptance test (the bar before this verb joins the pack)

1. **Speaks only the interface** — it provisions the **datastore** the MLOps interface
   owns and verifies it through `loom doctor` (the read-only CLI, which checks
   reachability via a TCP socket probe + the Metaflow Client API). It never calls
   Metaflow/AIDE directly in code and never touches raw S3 (`loom doctor` has no
   boto3 / no `s3://` literal; the only S3-protocol talker is in-cluster minio).
2. **Output is the verified datastore** — the deliverable is `.env.metaflow` + a
   `loom doctor` PASS, not a chat transcript or a hand-rolled cluster.
3. **Approval tier is correct** — expensive / mutate; it **always gates** and shows
   exactly what it installs/starts. NOT `disable-model-invocation` (local-dev +
   reversible via `minikube delete`), but never proceeds without confirmation.
4. **Writes a learnings row** — N/A: this verb provisions infra and runs no flow, so
   there is no rollout to record; the first lifecycle run after setup writes the row.
5. **Exit gate has a self-test** — the `loom doctor` PASS/FAIL gate is covered by the
   `tests/test_doctor.py` self-tests above (full env ⇒ PASS/exit 0; env unset ⇒
   FAIL/exit 1).
6. **Single free-text arg** — one confirmation noun (stand up the local-dev datastore).
7. **Dual-invocation** — works user-typed (`/loom-setup-metaflow`) and
   model-auto-loaded on the `description` / `when_to_use` match; it always gates before
   installing/starting anything, so auto-loading still cannot mutate without confirm.
