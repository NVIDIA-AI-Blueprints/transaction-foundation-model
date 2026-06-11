# 07 · Interactive GPU notebooks: `loom notebook`

> ⚠️ **Unlike tutorials 01–06, this one is NOT keyless and NOT free.** It launches a
> real **GPU on Modal** (default: an H100) and **costs money** while it runs. You
> need a Modal account + token. The `--dry-run` step below is free; the live launch
> bills until you stop it.

## What you'll build

A **GPU-backed JupyterLab**, running in the **NeMo container** on an on-demand
Modal GPU, opened in your browser from a plain laptop — no local GPU, no Linux
host, no Docker. Your Loom datastore comes with it, so you can read the data
objects you've ingested straight from the notebook via the Metaflow Client API.

This is the answer to "I'm a data scientist on a fresh Mac and the repo's notebooks
need a Linux/NVIDIA GPU host." **Loom doesn't host notebooks — it *launches* them.**
`loom notebook` reuses the exact Modal + NeMo seam Loom already uses for remote
training; it's a remote-compute launcher, not a notebook IDE.

## Setup

You need three things:

```bash
# 1. The Modal client + a token (one-time). This is the only paid dependency.
pip install modal
modal token set        # or: modal setup   (opens a browser to authenticate)

# 2. Your local datastore up + sourced, so the notebook can read your data objects.
source /tmp/loom-cluster-env.sh      # or: source .env.metaflow
loom doctor                          # should be VERDICT: PASS

# 3. Something to read in the notebook — ingest a dataset if you haven't:
loom ingest --source <a.csv-or-dir> --name my-first-dataset
```

`loom notebook` forwards the datastore env (the `METAFLOW_*` / `AWS_*` vars `loom
doctor` checks) into the remote container, so whatever `loom datasets` lists on
your laptop is reachable inside the notebook.

## Step by step

### 1. Preview what it will launch — **free** (`--dry-run`)

Always look before you spend. `--dry-run` builds the full submission and prints it
**without** launching anything or touching Modal:

```bash
loom notebook --dry-run
```

```
loom notebook -- would launch (dry run):
  app:       loom-nemo-pretrain
  gpu:       H100
  image:     nvcr.io/nvidia/nemo:24.07
  port:      8888 (forwarded to your laptop)
  timeout:   4h
  datastore: forwarded

  Drop --dry-run to launch (needs `modal` + a Modal token).
```

### 2. Launch the notebook — **this spends** (real GPU)

```bash
loom notebook
```

What happens:
- Loom declares a Modal app + a GPU function over the NeMo container and starts
  JupyterLab inside it. **The first launch pulls the ~30 GB NeMo image** — expect a
  few minutes (the GPU is billing during the pull; subsequent launches reuse the
  cached image).
- A public URL prints in your terminal — open it:

  ```
  Open your GPU notebook:  https://....modal.host/lab?token=...
  ```
- **Keep this terminal open** — it's the control plane; the session lives as long
  as it runs.

### 3. Use it — confirm the GPU and read your data

Inside the notebook, confirm you're on the GPU:

```python
!nvidia-smi          # should show an H100
```

Read a Loom data object through the Metaflow Client API (the datastore env was
forwarded, so this just works):

```python
from metaflow import Flow
run = Flow("IngestDataset").latest_run        # your most recent `loom ingest`
print(run.id, run.data)                        # the ingested object's artifacts
```

From here it's an ordinary GPU JupyterLab — open the repo's `.ipynb` files, run
NeMo, train, explore.

### 4. Stop it — **stops the billing**

Press **Ctrl-C** in the terminal running `loom notebook`. That ends the remote
session and the GPU burst disappears (Modal stops billing). Don't leave it idling.

## Options

| Flag / env | Effect |
|---|---|
| `--gpu modal://my-app` | run under a named Modal app (default `modal`) |
| `--no-datastore` | don't forward the datastore env (no Client API access in the notebook) |
| `--dry-run` | print the plan, don't launch (free) |
| `LOOM_NEMO_IMAGE=...` | use a different container (e.g. a lighter image with Jupyter) |
| `LOOM_NOTEBOOK_TIMEOUT=600` | wall-clock ceiling in seconds (default 4h) — a safety net so a forgotten session self-terminates |

## Validate everything works (the checklist)

Use this to confirm the path end-to-end on your machine:

1. `loom notebook --dry-run` prints the plan (no Modal needed). ✅ free
2. `loom notebook` prints a `https://….modal.host/...` URL within a few minutes.
3. The URL opens JupyterLab in your browser.
4. `!nvidia-smi` in a cell shows an H100.
5. `Flow("IngestDataset").latest_run` returns your ingested dataset.
6. Ctrl-C ends it; `modal app list` shows nothing running afterward.

The **mechanism** (Modal auth, GPU allocation, JupyterLab + the public tunnel) is
already verified against real Modal; step 2's first run is what exercises the exact
H100 + NeMo image on your account.

## Troubleshooting

- **`the 'modal' package is not installed`** → `pip install modal` (into the same
  environment Loom's engine runs in), then `modal token set`.
- **Auth error / no token** → `modal token set` (or `modal setup`).
- **`gpu target 'X' is not a Modal target`** → `loom notebook` launches on Modal;
  use `--gpu modal` (or `modal://<app>`). Other targets aren't supported yet.
- **The notebook can't see your datasets** → you launched with `--no-datastore`, or
  the datastore env wasn't sourced on the laptop before launching (`source
  .env.metaflow && loom doctor` first). The forwarded env is a snapshot of your
  shell's at launch time.
- **First launch is slow** → the one-time ~30 GB NeMo image pull. It's cached after.

## See also

- [`05-build-a-backbone`](../05-build-a-backbone/README.md) — the *keyless* CPU
  model-builder, and the same Modal GPU seam used for **training** (vs. this
  interactive notebook use).
- The architecture note on the shared Modal launch seam: `docs/architecture.md`
  (ModelBuilderProvider → GPU launch targets).
