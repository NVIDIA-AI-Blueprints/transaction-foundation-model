# Environment Setup

**Goal:** get the five notebooks running, with the pretrained checkpoint available.
**Time:** ~20 minutes (plus dataset download).

The stack is GPU-native end to end. You will run everything inside NVIDIA's **NeMo Framework container**, which ships with CUDA, PyTorch, RAPIDS (cuDF/cuML), and NeMo AutoModel pre-installed and version-matched — this is the single biggest setup time-saver.

> 🧠 **New concept?** If "NeMo", "RAPIDS", or "NGC container" mean nothing to you yet, skim [The GPU Stack primer](../02-concepts/06-gpu-stack.md) first.

## Prerequisites

| Component | Requirement | Why |
|-----------|-------------|-----|
| GPU | 1× NVIDIA A100 (80 GB) or H100 | Notebooks 01–02 run on smaller GPUs; pretraining and 24M-row preprocessing want big memory |
| System RAM | 32 GB | TabFormer CSV is ~2.2 GB; intermediate frames are larger |
| OS | Ubuntu 22.04+ (or any Docker host) | Container runtime |
| Software | Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) | `--gpus all` support |
| Disk | ~15 GB free | Dataset + corpus + checkpoints |

No A100 at hand? You can still read everything, run notebook 01–02 on a smaller GPU (e.g. 24 GB) with reduced sample sizes, and use the shipped checkpoint for notebooks 04–05. Cloud GPU instances (one A100 for an afternoon) are the pragmatic path for full runs.

## Working from macOS + Conductor

If you are working from a Mac, do **not** try to run this stack locally. Docker Desktop on macOS does not expose an NVIDIA CUDA GPU to `docker run --gpus all`, and Apple Silicon runs Linux/arm64 containers while this notebook stack expects NVIDIA CUDA on Linux/x86_64.

Use the repo's GCP notebook runtime instead:

```bash
gcloud auth login
gcloud auth application-default login
scripts/gcp-gpu-up.sh
scripts/gcp-sync-workspace.sh
scripts/gcp-marimo.sh
```

That workflow keeps Conductor and your editor local, creates or updates a GCP A100 VM, syncs the current workspace to `/mnt/tfm/workspace`, starts the NeMo container remotely, and forwards marimo back to your browser at `localhost:8080`.

The GCP VM also installs Google's Ops Agent and grants its service account Cloud Logging/Monitoring write roles. This is what makes GPU utilization appear in the GCP Console monitoring tab; metric data can take a few minutes to show up after the VM boots or the agent is installed.

The current GCP default is `a2-highgpu-1g` (A100 40GB) because that quota is available in the project. Treat it as the working bootstrap target; full-size preprocessing or longer training may still require A100 80GB/H100 quota or reduced notebook sample sizes.

After marimo is open, keep development source-of-truth in Conductor. Edit locally, run `scripts/gcp-sync-workspace.sh`, then rerun cells in marimo. The Conductor Run button starts the remote tunnel, but it does not sync local changes first.

Notebooks 04 and 05 require the shipped Git LFS checkpoint. In the Conductor + GCP workflow, pull it on your Mac and sync it to the VM model mount:

```bash
brew install git-lfs
git lfs install
git lfs pull --include='models/decoder-foundation-model/**' --exclude=''
scripts/gcp-sync-models.sh
```

The full runbook is [`infra/gcp-notebook/README.md`](../../infra/gcp-notebook/README.md). It documents the Terraform resources, persistent disk layout, Conductor integration, and each helper script.

## Step 1 — Launch the NeMo container

From your clone of this repo:

```bash
docker run --gpus all --rm -it \
  -v $(pwd):/workspace \
  --shm-size=8g \
  -p 8080:8080 \
  --ulimit memlock=-1 \
  nvcr.io/nvidia/nemo:25.09.01
```

Flag-by-flag, because each one prevents a real failure mode:

- `--gpus all` — exposes the GPU(s); without it, `torch.cuda.is_available()` is `False` and cuDF won't import.
- `-v $(pwd):/workspace` — bind-mounts the repo; your edits and data persist outside the container.
- `--shm-size=8g` — PyTorch DataLoader workers exchange tensors via shared memory; the 64 MB default causes cryptic crashes.
- `-p 8080:8080` — publishes marimo to your browser.
- `--ulimit memlock=-1` — removes the locked-memory cap some CUDA operations need.

> **Remote machine?** Add SSH port forwarding so marimo reaches your laptop: `ssh -L 8080:localhost:8080 user@host`.

## Step 2 — Pull the pretrained checkpoint (Git LFS)

The ~56 MB pretrained checkpoint in [`models/decoder-foundation-model/`](../../models/decoder-foundation-model) is stored with **Git LFS** — the repo only contains small pointer files until you pull. Notebooks 04 and 05 *require* it.

Inside the container:

```bash
cd /workspace
git config --global --add safe.directory /workspace
apt-get update && apt-get install -y git-lfs
git lfs install
git lfs pull
```

> **Why `safe.directory`?** The repo is bind-mounted from the host, so the file owner inside the container doesn't match the container user. Git refuses to operate on "dubious ownership" directories unless told otherwise — without this line, `git lfs pull` fails.

Verify it worked — the safetensors file should be ~56 MB, not a few hundred bytes:

```bash
ls -lh models/decoder-foundation-model/
```

## Step 3 — Install dependencies and start marimo

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv pip install --python "$(which python)" -r requirements.txt
marimo edit --headless --host 0.0.0.0 --port 8080
```

Open the printed marimo URL, or `http://localhost:8080`. Dependencies are installed once from `requirements.txt`; the notebooks no longer run inline package-install cells.

## Step 4 — Get the dataset

Notebook 01's first code cell downloads the **TabFormer** credit-card dataset (~2.2 GB `transactions.tgz` from IBM Box) and extracts it to:

```
data/
└── TabFormer/
    └── raw/
        └── card_transaction.v1.csv
```

If the automated download fails, fetch `transactions.tgz` manually from the [IBM Box link](https://ibm.ent.box.com/v/tabformer-data/folder/130747715605) (see [the TabFormer page](../04-data/01-tabformer.md) for dataset details) and place it at `data/TabFormer/transactions.tgz`, then re-run the cell.

## Step 5 — Run the notebooks in order

| Run | Produces | Needed by |
|-----|----------|-----------|
| `01_dataset_baseline.py` | `data/TabFormer/temporal_split/{train,val,test}.parquet` + eval subsets | 02, 04, 05 |
| `02_seq_preproc_tokenization.py` | `data/decoder_corpus/{train,val,test}_corpus.txt` | 03 |
| `03_foundation_model_training.py` *(optional)* | `models/decoder-demo/` (30-step demo checkpoint) | — (educational) |
| `04_inference_embedding_extraction.py` | `data/embeddings/*.npy` | 05 |
| `05_xgboost_fraud_detection.py` | Final comparison results | — |

**Important:** notebook 03 is a 2-minute, 30-step *demo* of the training pipeline — its output goes to `models/decoder-demo/` and is **not** what notebooks 04–05 use. They load the LFS checkpoint at `models/decoder-foundation-model/` (trained ~3,000 steps on 8× A100). You can skip 03 entirely and still complete the workflow.

## Training for real (beyond the demo)

When you want to launch actual pretraining — after generating the corpus in notebook 02:

```bash
# Multi-GPU (recommended)
torchrun --nproc-per-node=8 scripts/train_decoder_model.py \
    -c configs/pretrain_financial_decoder.yaml \
    --dataset.data_path data/decoder_corpus/train_corpus.txt \
    --validation_dataset.data_path data/decoder_corpus/val_corpus.txt

# Single GPU (testing)
python scripts/train_decoder_model.py \
    -c configs/pretrain_financial_decoder.yaml \
    --dataset.data_path data/decoder_corpus/train_corpus.txt
```

Note the `--dataset.data_path` style: any key in [`configs/pretrain_financial_decoder.yaml`](../../configs/pretrain_financial_decoder.yaml) can be overridden from the command line. Raise `step_scheduler.max_steps` (the default `30` is the demo setting) for a real run. Use `torchrun` directly rather than the `automodel` CLI for multi-GPU — the CLI misparses `--nproc-per-node`.

## Troubleshooting quick reference

| Symptom | Cause | Fix |
|---------|-------|-----|
| `git lfs pull` fails with "dubious ownership" | Bind-mount ownership mismatch | `git config --global --add safe.directory /workspace` |
| Notebook 04/05 errors loading model / shape mismatch | LFS pointer files instead of real weights | `git lfs pull`, verify file sizes |
| Notebook 04 says checkpoint not found under `/workspace/models` on GCP | VM model mount was not populated | Pull LFS locally, then run `scripts/gcp-sync-models.sh` |
| DataLoader workers crash | Shared memory too small | Relaunch container with `--shm-size=8g` |
| `cudf` import error | Running outside the container / no GPU | Use the NeMo container on a GPU host |
| OOM during notebook 01 | 24M rows on a small GPU | Reduce sample sizes in the notebook, or use a bigger GPU |

Next: [Level 100 — The Big Picture](../03-learning-path/level-100-the-big-picture.md).
