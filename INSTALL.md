# Installing Loom (macOS)

This is the canonical install playbook. The easy path is **`./install.sh`** — it
hands these steps to **Claude** or **Codex** when either is installed (the assistant
adapts to your machine and fixes errors as it goes) and otherwise runs them as a
script. You can also follow them by hand below.

Loom is three pieces:

- the **`loom` command** (Node) — the agentic CLI you talk to,
- the **engine** (Python) it drives — the lifecycle verbs,
- a **local datastore** (minikube + minio) the lifecycle verbs read/write through.

## Prerequisites

- **Node ≥ 22.19** and **Python 3.12** — `brew install node python@3.12`.
- The datastore stack (colima, minikube, kubectl, awscli) is **detect-before-installed** by the datastore step — you don't need it beforehand.

## Steps (from the repo root)

**1 — Engine (Python).**

```bash
python3.12 -m venv .venv && source .venv/bin/activate && pip install -e .
```

> **Confirm this completed:** `python -c "import metaflow"` must succeed. `pip install -e .`
> pulls a heavy ML stack (AIDE); if it errors, fix that error and re-run — a partial
> install leaves the lifecycle verbs broken even though `import loom` still works
> from the repo dir.

Then point `loom` at this interpreter (**required** — add the export to your shell profile):

```bash
export LOOM_PYTHON="$(pwd)/.venv/bin/python"
```

**2 — The `loom` command (Node).**

```bash
( cd cli && npm install && npm run build && npm link )
```

`loom --help` should print the branded help + verb list.

**3 — Local datastore.** The port-forward must stay running while you use the lifecycle verbs.

```bash
bash scripts/setup_metaflow_minikube.sh
kubectl port-forward -n loom svc/minio 9000:9000 9001:9001 &
source .env.metaflow
```

**4 — Verify.**

```bash
"$LOOM_PYTHON" -m loom doctor      # must end: VERDICT: PASS
```

**5 — (optional) Model key** — only the natural-language turns need one:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."   # or run `loom`, then /login
```

## Troubleshooting

- **doctor: `import metaflow` FAIL, or "loom importable but NOT pip-installed"** — step 1's
  `pip install -e .` didn't complete. Re-run it from the repo root with the venv active and
  read the **first** error (often a build failure in the heavy AIDE stack). `pip install metaflow`
  unblocks just that check.
- **doctor: datastore unreachable** — the step-3 `kubectl port-forward` isn't running. Start it
  (in its own shell) and re-`source .env.metaflow`.
- **`loom: command not found`** — re-run step 2's `npm link`, or make sure your npm global bin is on `PATH`.

## Uninstall

```bash
./uninstall.sh        # removes the loom command, venv, datastore, and .env.metaflow (gates each step)
```
