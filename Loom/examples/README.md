# Loom examples -- per-use-case walkthroughs that double as a regression eval bed

Each example here is a self-contained, **self-checking** walkthrough of one Loom
use case. It generates deterministic synthetic data, ingests it, runs a sequence
of verbs through the real engine (Metaflow datastore), and **asserts the
outcomes inline**. Read top-to-bottom they are a tour of the lifecycle; run as a
suite they are a regression eval bed -- `tests/test_examples.py` replays every
`run.sh` and fails on any drift.

All data is **domain-neutral** (generic classification / sequence / drift
scenarios -- no customer, vertical, or PII content), deterministic, seeded, and
generated offline (no downloads).

## The six use cases

| # | Example | What it walks through | Key verbs (keyless) |
| --- | --- | --- | --- |
| 01 | [`01-tabular-classification`](01-tabular-classification/) | The core lifecycle: ingest a labeled tabular dataset, profile it, build features, validate a baseline. | `ingest` · `datasets` · `eda` · `features` · `validate` |
| 02 | [`02-leakage-detection`](02-leakage-detection/) | Data readiness + the composition gate: a planted leak is flagged by `eda`, dropped by `features --from`, and `validate` comes back clean. | `ingest` · `eda` · `features --from` · `validate` |
| 03 | [`03-sequence-embeddings`](03-sequence-embeddings/) | The model-builder seam (CPU): build a backbone from event sequences; show the GPU gate refuses cleanly without a target. | `ingest` · `train` (local + nemo gate) |
| 04 | [`04-validate-and-gated-deploy`](04-validate-and-gated-deploy/) | The cross-verb exit gate + safe-by-default: a `validate` VERDICT gates `deploy` (PLAN only, `--apply` off). | `ingest` · `validate` · `deploy --validate` |
| 05 | [`05-ops-and-drift`](05-ops-and-drift/) | Monitoring: run-health counts for a flow + a drift check of a shifted dataset vs a reference. | `ingest` · `ops --flow` · `ops --dataset --reference` |
| 06 | [`06-telemetry-distillation`](06-telemetry-distillation/) | The moat capture: keyless verbs generate rollouts; `telemetry` summarizes + exports a general-only, redacted SFT corpus. | `eda` · `validate` · `telemetry status` · `telemetry export` |

The [`_template/`](_template/) directory is the canonical layout every example
copies: `README.md` (the walkthrough), `make_data.py` (the deterministic
generator), and `run.sh` (the self-checking recipe with the `_assert_json`
helper).

## How to run

The examples talk to a real Metaflow datastore, so source the cluster env first
(minio on `localhost:9000` + local metadata):

```bash
source /tmp/loom-cluster-env.sh
bash examples/01-tabular-classification/run.sh
```

Run them all (the same thing the test harness does):

```bash
source /tmp/loom-cluster-env.sh
for d in examples/[0-9]*/; do bash "$d/run.sh"; done
```

`run.sh` is `set -euo pipefail` and exits **nonzero on any regression**. Each
example ingests under a **uniquely-named dataset** (an `ex-<dir>-<pid>-<rand>`
tag), so concurrent and repeat runs never collide. The default verb entrypoint
is `/Users/anub/Work/Loom/.venv/bin/python -m loom` (overridable via `$LOOM`),
so no `PATH` / console-script is required.

## The eval / regression story

* **Each `run.sh` self-asserts.** It parses every verb's `--json` envelope with
  the venv python (no `jq` dependency) and asserts the stable contract fields
  (`status`, `VERDICT`, `pathspec`, `summary[...]`, `gate[...]`). A changed shape
  or a regressed outcome exits nonzero.
* **`tests/test_examples.py` replays them.** It discovers `examples/*/run.sh`,
  runs one parametrized test per example, and asserts exit 0. It **skips cleanly**
  (module-level) when the engine / cluster is unreachable -- the doctor-style
  check is "can `loom datasets --json` return `status == ok`?" -- so the suite
  stays green on a box without a datastore.
* **The Python engine + verbs are unchanged.** These examples are content +
  one harness test; they pin the `--json` contract as a black-box, so they catch
  regressions without coupling to internals.

## Keyless vs key-gated

Everything asserted here is **keyless** -- it runs without a model key. The
keyless lifecycle verbs are: `ingest`, `datasets`, `eda`, `features`,
`validate`, `viz`, `report`, `ops`, `train` (local CPU adapter), `deploy` (plan
only), `collab` (build only), `telemetry`, `doctor`.

The **LLM / key-gated** steps -- `loom run`, `loom optimize`, the pipeline's
optimize stage, and the agentic natural-language flow (`loom "..."`) -- appear in
each walkthrough's **"Ask Loom"** prose, marked _needs a model key_, but are
**never** in the asserted `run.sh`. That keeps the eval bed runnable in CI with
no secrets while still documenting the full product UX. For the model-builder
example, `train` uses `LOOM_MODEL_BUILDER_PROVIDER=local` (the torch-free CPU
adapter) so it builds for real without a GPU.
