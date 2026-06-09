# Loom tutorials — teach the full DS lifecycle, one use case at a time

Loom is an **agentic CLI for the whole data-science lifecycle** — ingest, profile,
engineer, validate, build, ship, and monitor — not an "automated ML engine." Each
tutorial here takes a single, common DS job and walks it end-to-end through the
real engine, explaining what every verb does, what to look for in its output, and
where the (key-gated) "make it better" step would slot in.

These are **teaching** documents: prose-first, beginner-friendly, copy-pasteable.
For the terse, assert-heavy counterpart, see [`examples/`](../examples/) — the
per-use-case regression **eval bed** that `tests/test_examples.py` replays on every
change. The split is simple: **tutorials teach, examples assert.** (The `run.sh`
in each tutorial is itself self-checking, so a tutorial doubles as a smoke test —
but its job is to walk you through the *why*, not to be the test suite.)

All tutorials are **keyless**: they talk to your live local Metaflow + minio
datastore with no API key and no cost. Data is synthesized inline (deterministic,
seeded, no downloads). Each ingests under a unique dataset name so repeat and
concurrent runs never collide.

## The tutorials

| # | Tutorial | What it teaches | Verbs (keyless) |
| --- | --- | --- | --- |
| 01 | [`01-classification`](./01-classification/README.md) | Your first model: the core lifecycle for binary tabular classification — ingest → eda/leakage scan → leakage-aware features → CV + sealed-holdout validate → model-card report. | `ingest` · `datasets` · `eda` · `features` · `validate` · `report` |
| 02 | [`02-regression`](./02-regression/README.md) | Predicting a number: the same lifecycle on a **continuous** target, showing regression auto-detection (`task_type=regression`, `metric=rmse`, lower-is-better) vs. classification. | `ingest` · `datasets` · `eda` · `features` · `validate` · `report` |
| 03 | [`03-leakage-safe-features`](./03-leakage-safe-features/README.md) | Engineering features without leaking: `eda` surfaces planted leaks via `leakage_flags`, `features --from` the eda run drops them, and `validate` gives an honest verdict (~0.86 instead of a fake ~1.0). | `ingest` · `eda` · `features` · `validate` |
| 04 | [`04-trustworthy-validation`](./04-trustworthy-validation/README.md) | Validation you can trust + a model card: `validate --sensitive` runs CV + sealed holdout + calibration/Brier + per-slice fairness + leakage checks + a VERDICT, then `report` builds the card. | `ingest` · `validate` · `report` |
| 05 | [`05-build-a-backbone`](./05-build-a-backbone/README.md) | Building a model backbone with no GPU: the keyless model-builder (`train`) seam — build a CPU backbone at probe budget (STATUS BUILT + sha256 fingerprint), and watch the GPU path refuse cleanly (`REFUSED_NO_GPU_TARGET`). | `ingest` · `train` |
| 06 | [`06-monitoring-drift`](./06-monitoring-drift/README.md) | Monitoring data drift in production: the read-only monitoring tier — `ops --flow` for run health and `ops --dataset --reference` to compare a shifted batch against a reference. | `ingest` · `validate` · `ops` |

## How to run

The keyless verbs talk to your **already-running** local Metaflow + minio
datastore, so bring that environment up first and confirm it's healthy:

```bash
source /tmp/loom-cluster-env.sh
loom doctor            # read-only health check — must end "VERDICT: PASS"
```

Once `doctor` reports **PASS**, run any tutorial's tested companion script from the
repo root:

```bash
bash tutorials/01-classification/run.sh
```

Each `run.sh` is **self-checking**: every verb runs with `--json` and its outcome
is asserted inline, so the script **exits nonzero on any regression** (and exits 0
when everything is green). The tutorials verified PASS against the live engine, and
the scripts are idempotent — safe to re-run.

## One thing the scripts deliberately skip

The tutorials are end-to-end except for the **modeling / `optimize` step (the AIDE
search)** that some of them narrate as "the next step that makes it better." That
agentic, model-using path — like the NL turn and `run`/`optimize` — needs an **LLM
API key** and costs money, so the keyless scripts never invoke it. Everything you
run here is free and offline; the README prose just shows you where the key-gated
step would slot in.
