# GCP GPU marimo Runtime

This directory defines the repeatable GCP runtime for working from macOS and Conductor while executing GPU marimo notebooks on Compute Engine.

The first runtime image is NVIDIA NeMo because the current notebooks need CUDA, RAPIDS, PyTorch, and NeMo preinstalled. The VM and scripts are image-agnostic: change `NOTEBOOK_IMAGE` when the project moves to a different FM or recommendation-system stack.

## What Runs Where

| Layer | Runs on | Purpose |
|-------|---------|---------|
| Conductor + editor | macOS | Workspaces, branches, code edits, local orchestration |
| Terraform + `gcloud` | macOS | Create and manage the GCP VM, disk, bucket, IAM, and metadata |
| GPU host | GCP Compute Engine | NVIDIA driver, Docker, NVIDIA Container Toolkit, Ops Agent, persistent disk |
| Notebook runtime | Docker container on GCP | marimo, CUDA libraries, RAPIDS, PyTorch, NeMo or replacement image |
| Observability | GCP Cloud Monitoring/Logging | CPU, memory, disk, and GPU metrics from the Ops Agent |
| Browser | macOS | Opens marimo through an SSH local port forward |

Your Mac does not run CUDA. It controls a Linux/NVIDIA host and forwards the notebook UI back to `localhost`.

## Quick Start

From a Conductor workspace:

```bash
gcloud auth login
gcloud auth application-default login
scripts/gcp-gpu-up.sh
scripts/gcp-sync-workspace.sh
scripts/gcp-marimo.sh
```

Open the URL printed by `scripts/gcp-marimo.sh`.

Stop the VM when you are done:

```bash
scripts/gcp-gpu-down.sh
```

## Day-to-Day Development Loop

Once marimo is visible through the tunnel, treat Conductor as the source of truth and marimo as the GPU execution surface.

Use this loop for normal code changes:

1. Edit code, configs, docs, and notebooks locally in the Conductor workspace.
2. Ask agents to make repo changes in Conductor, not inside the remote marimo container.
3. Sync the current workspace to the GPU VM:
   ```bash
   scripts/gcp-sync-workspace.sh
   ```
4. In marimo, rerun the affected cells.
5. If you changed Python modules under `src/`, restart the marimo session before rerunning dependent cells.
6. Keep generated outputs on the VM under `/workspace/data`, `/workspace/models`, and `/workspace/artifacts`.
7. Commit or open a PR from Conductor after validation.

Important: avoid editing repo-tracked source files directly inside marimo. The current sync direction is local-to-remote only, so remote edits can drift or be overwritten by the next `scripts/gcp-sync-workspace.sh`. If you intentionally edit a notebook in marimo, copy that change back into the Conductor workspace before treating it as committed work.

The Conductor Run button starts or reconnects the remote marimo tunnel through `scripts/gcp-marimo.sh`. It does **not** sync local changes first. After any local edit, run:

```bash
scripts/gcp-sync-workspace.sh
```

before rerunning notebook cells.

For agent work, use one Conductor workspace per coherent change:

1. Let the agent edit locally.
2. Sync to the GPU VM.
3. Validate in marimo.
4. Feed notebook results or errors back to the agent.
5. Have the agent update code, tests, and docs locally.
6. Sync and rerun until the change is ready.

The default setup has one shared remote workspace at `/mnt/tfm/workspace`, so avoid syncing multiple Conductor workspaces to the same VM at the same time. For truly parallel GPU work, use a different `GCP_INSTANCE`, `GCP_ZONE`, or `REMOTE_WORKSPACE` per active experiment.

## Defaults

- Project: `level-mark-437714-b1`
- Zone: `us-central1-f`
- VM: `a2-highgpu-1g` by default, which is a fixed-GPU A2 machine with one A100 40GB GPU
- Persistent disk: 1 TB mounted at `/mnt/tfm`
- Remote workspace: `/mnt/tfm/workspace`
- Data/artifacts: `/mnt/tfm/data`, `/mnt/tfm/models`, `/mnt/tfm/artifacts`
- Bucket: `$GCP_BUCKET`

Use `GCP_MACHINE_TYPE=a2-ultragpu-1g` for A100 80GB only after `NVIDIA_A100_80GB_GPUS` quota is approved. Use `GCP_MACHINE_TYPE=g2-standard-24` for a lower-cost 2x L4 exploratory box.

The A100 40GB default is enough to validate the environment, run reduced exploratory notebooks, and start recommendation-system baseline work. Full-size preprocessing or longer FM training may still need A100 80GB/H100 quota or smaller notebook sample sizes.

If GCP returns a zone stockout, move `GCP_ZONE` to one of the zones suggested in the error and rerun `scripts/gcp-gpu-up.sh`.

## Files and Configuration

### `.env.gcp`

`.env.gcp` is a local, gitignored file copied from `.env.gcp.example`. All helper scripts source it before applying defaults. Explicit shell environment overrides take precedence, so `MARIMO_PORT=18888 scripts/gcp-marimo.sh` temporarily overrides the local file.

Current local defaults:

```bash
GCP_PROJECT=level-mark-437714-b1
GCP_REGION=us-central1
GCP_ZONE=us-central1-f
GCP_MACHINE_TYPE=a2-highgpu-1g
GCP_INSTANCE=tfm-gpu-notebook
GCP_BUCKET=level-mark-437714-b1-tfm-gpu-artifacts
NOTEBOOK_IMAGE=nvcr.io/nvidia/nemo:25.09.01
GPU_TEST_IMAGE=nvidia/cuda:12.4.1-base-ubuntu22.04
MARIMO_PORT=8080
```

Change only `.env.gcp` for local machine preferences. Commit defaults only when they should apply to everyone.

### Conductor

[`.conductor/settings.toml`](../../.conductor/settings.toml) wires this workflow into Conductor:

- `file_include_globs = ".env.gcp\n"` copies your local GCP settings into new workspaces.
- `setup = "scripts/gcp-setup-check.sh || true"` reports missing local tools or stale auth without blocking workspace creation.
- `run = "scripts/gcp-marimo.sh"` makes the Conductor Run button start the remote notebook/tunnel workflow.
- `run_mode = "nonconcurrent"` avoids multiple workspaces fighting over one marimo port and one remote VM.

## Terraform Resources

`scripts/gcp-gpu-up.sh` writes an ignored `terraform.auto.tfvars.json` from `.env.gcp`, then runs Terraform in this directory.

Terraform manages:

- required APIs: Compute Engine, IAM, Storage, BigQuery, Cloud Logging, Cloud Monitoring;
- service account: `tfm-gpu-notebook-sa`;
- IAM grants for the service account:
  - BigQuery job user;
  - BigQuery data viewer;
  - BigQuery read-session user;
  - Logs Writer for Ops Agent log ingestion;
  - Monitoring Metric Writer for Ops Agent metric ingestion;
  - object admin on the artifact bucket;
- GCS bucket for datasets, model artifacts, checkpoints, and notebook outputs;
- persistent disk `tfm-gpu-notebook-data`, mounted at `/mnt/tfm`;
- Compute Engine VM `tfm-gpu-notebook`;
- VM startup script metadata.

The boot disk is disposable. The data disk and bucket are the durable state.

## Remote Filesystem Layout

| Path | Meaning |
|------|---------|
| `/mnt/tfm/workspace` | Synced copy of the current Conductor workspace |
| `/mnt/tfm/data` | Durable data mount exposed to containers as `/workspace/data` |
| `/mnt/tfm/models` | Durable model/checkpoint mount exposed as `/workspace/models` |
| `/mnt/tfm/artifacts` | Durable run artifacts exposed as `/workspace/artifacts` |
| `/opt/tfm/bootstrap.done` | Marker written when VM bootstrap completed |
| `/opt/tfm/notebook-image` | The image recorded by Terraform metadata |

## Script Reference

### `scripts/gcp-lib.sh`

Shared shell helpers used by all GCP scripts.

It:

- finds the repo root;
- sources `.env.gcp` if present;
- applies default values;
- validates required commands;
- builds and runs the `gcloud compute ssh` command;
- waits for SSH readiness after VM starts;
- maps common GPU machine types to quota metrics;
- checks GPU quota before Terraform starts creating resources.

The quota preflight catches errors like `NVIDIA_A100_80GB_GPUS` being `0` before Terraform spends time creating or modifying resources.

### `scripts/gcp-setup-check.sh`

Local readiness check. It does not create cloud resources.

It verifies:

- `gcloud`, `terraform`, and `jq` exist locally;
- `GCP_PROJECT` is set;
- an active `gcloud` account exists;
- `gcloud` can list enabled services, which also catches stale auth tokens.

Conductor runs this as a non-blocking setup check, so stale GCP auth does not prevent opening a workspace.

### `scripts/gcp-gpu-up.sh`

Creates, updates, or starts the GCP runtime.

It:

1. loads `.env.gcp`;
2. checks `gcloud`, `terraform`, `jq`, and `GCP_PROJECT`;
3. confirms the chosen machine type exists in `GCP_ZONE`;
4. checks regional GPU quota for the chosen machine type;
5. writes `infra/gcp-notebook/terraform.auto.tfvars.json`;
6. runs `terraform init`;
7. runs `terraform apply -auto-approve`;
8. starts the VM if Terraform found it already exists but it is currently stopped;
9. waits until SSH on the VM is ready before returning.

It is safe to rerun. Terraform converges the real GCP resources to the repo configuration.

Observed setup decisions:

- `a2-ultragpu-1g` failed because `NVIDIA_A100_80GB_GPUS` quota was `0` in `us-central1`.
- `a2-highgpu-1g` passed quota but `us-central1-a` was stocked out.
- `us-central1-f` created the A100 40GB VM successfully.

### `scripts/gcp-sync-workspace.sh`

Copies the current Conductor workspace to the VM.

It:

1. creates `/mnt/tfm/workspace`, `/mnt/tfm/data`, `/mnt/tfm/models`, and `/mnt/tfm/artifacts`;
2. waits until SSH on the VM is ready;
3. changes ownership of `/mnt/tfm/workspace` to the SSH user;
4. streams a tar archive over `gcloud compute ssh`;
5. suppresses macOS extended attributes;
6. extracts without preserving local owners, permissions, or timestamps.

It excludes:

- `.git`;
- `.context`;
- `.env.gcp`;
- large generated data and model-output directories;
- temporary notebook rerun outputs.

This is a workspace copy, not a Git remote. Rerun it after local edits you want available inside marimo.

`scripts/gcp-sync-workspace.sh` also attempts `scripts/gcp-sync-models.sh`, which copies the resolved Git LFS checkpoint from local `models/decoder-foundation-model/` into the VM's durable `/mnt/tfm/models/decoder-foundation-model/` mount. If the local checkpoint is still a 133-byte Git LFS pointer, workspace sync still completes and model sync prints the exact `git lfs pull` command to run locally.

### `scripts/gcp-sync-models.sh`

Copies the shipped decoder checkpoint into the VM model mount used by notebooks 04 and 05.

It:

1. checks `models/decoder-foundation-model/model-00001-of-00001.safetensors` exists locally;
2. fails if that file is still a Git LFS pointer;
3. waits until SSH on the VM is ready;
4. creates `/mnt/tfm/models/decoder-foundation-model`;
5. copies the model files into that directory.

This script exists because the marimo container mounts `/mnt/tfm/models` at `/workspace/models`. That durable mount intentionally hides the synced repo's `models/` directory, so the checkpoint must be copied into the VM model mount instead of relying on `git lfs pull` inside the container.

### `scripts/gcp-marimo.sh`

Starts the notebook container on the VM and opens the local tunnel.

It:

1. checks Docker exists on the VM;
2. waits until SSH on the VM is ready;
3. waits for `/opt/tfm/bootstrap.done` so host startup work has finished before the GPU container starts;
4. checks `nvidia-smi` exists on the VM;
5. verifies Docker can run a short-lived CUDA container with GPU access;
6. removes any existing `tfm-marimo` container;
7. pulls `NOTEBOOK_IMAGE`;
8. starts the container with:
   - `--gpus all`;
   - explicit `/dev/nvidia*` and `/dev/nvidia-caps/*` device mounts;
   - `--shm-size=8g`;
   - `--ulimit memlock=-1`;
   - `/mnt/tfm/workspace` mounted as `/workspace`;
   - durable `data`, `models`, and `artifacts` mounts;
9. warns if the pretrained checkpoint is missing from `/mnt/tfm/models/decoder-foundation-model`;
10. starts marimo on port `8080` inside the container and binds it to `127.0.0.1:$MARIMO_PORT` on the VM;
11. verifies the running marimo container can initialize NVML with `nvidia-smi`;
12. opens an SSH local port forward from `localhost:$MARIMO_PORT` to the VM.

The final SSH process intentionally stays open. Closing it closes the browser tunnel, but the remote container continues running.

### `scripts/gcp-jupyter.sh`

Deprecated compatibility wrapper. It prints a warning and delegates to `scripts/gcp-marimo.sh` so old Conductor muscle memory starts the new marimo workflow.

### `scripts/gcp-gpu-down.sh`

Stops the VM:

```bash
scripts/gcp-gpu-down.sh
```

It does not delete:

- the persistent disk;
- the GCS bucket;
- Terraform state;
- synced workspace files.

Use this whenever you are done with GPU work.

## VM Bootstrap

The Terraform startup script at [`startup.sh`](startup.sh) prepares the VM host.

It:

- schedules an automatic OS shutdown after `auto_shutdown_minutes` to cap runaway GPU cost;
- installs base packages: Docker, Git, Git LFS, `jq`, `rsync`, Python, curl, GPG;
- formats and mounts the persistent disk if needed;
- creates `/mnt/tfm/{workspace,data,models,artifacts,tmp}`;
- installs NVIDIA drivers through Google's GPU installer if `nvidia-smi` is absent;
- installs NVIDIA Container Toolkit;
- configures Docker's NVIDIA runtime;
- installs and starts Google's Ops Agent for Cloud Monitoring and Logging;
- removes stale `/opt/tfm/bootstrap.done` at the start of each boot;
- writes `/opt/tfm/bootstrap.done` when host bootstrap has completed.

The Ops Agent uses its built-in Linux `hostmetrics` receiver. On GPU VMs with an installed NVIDIA driver, that receiver emits NVIDIA NVML metrics under `agent.googleapis.com/gpu/*`, which is what the GCP Console monitoring tab uses for GPU utilization.

The current VM was repaired in place after the first bootstrap attempt exposed a non-interactive GPG issue. The startup script now uses `gpg --batch --yes --dearmor`, so future VMs should bootstrap cleanly.

## Validation Commands

Host GPU:

```bash
gcloud compute ssh tfm-gpu-notebook \
  --zone us-central1-f \
  --project level-mark-437714-b1 \
  --command 'nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader'
```

Container GPU:

```bash
gcloud compute ssh tfm-gpu-notebook \
  --zone us-central1-f \
  --project level-mark-437714-b1 \
  --command 'sudo docker run --gpus all --rm nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi'
```

Workspace:

```bash
gcloud compute ssh tfm-gpu-notebook \
  --zone us-central1-f \
  --project level-mark-437714-b1 \
  --command 'cd /mnt/tfm/workspace && ls README.md scripts/gcp-marimo.sh'
```

Ops Agent:

```bash
gcloud compute ssh tfm-gpu-notebook \
  --zone us-central1-f \
  --project level-mark-437714-b1 \
  --command 'systemctl is-active google-cloud-ops-agent && dpkg-query --show google-cloud-ops-agent'
```

GPU metric descriptors:

```bash
curl -fsS -H "Authorization: Bearer $(gcloud auth print-access-token)" --get \
  --data-urlencode 'filter=metric.type = starts_with("agent.googleapis.com/gpu")' \
  'https://monitoring.googleapis.com/v3/projects/level-mark-437714-b1/metricDescriptors' \
  | jq -r '.metricDescriptors[].type'
```

Metric data can take a few minutes to appear after the agent starts. The per-process GPU metrics only appear while a process is actively using the GPU.

Latest GPU utilization sample:

```bash
INSTANCE_ID=$(gcloud compute instances describe tfm-gpu-notebook \
  --zone us-central1-f \
  --project level-mark-437714-b1 \
  --format='value(id)')
START=$(date -u -v-20M +%Y-%m-%dT%H:%M:%SZ)
END=$(date -u +%Y-%m-%dT%H:%M:%SZ)
FILTER="metric.type = \"agent.googleapis.com/gpu/utilization\" AND resource.type = \"gce_instance\" AND resource.labels.instance_id = \"$INSTANCE_ID\""

curl -fsS -H "Authorization: Bearer $(gcloud auth print-access-token)" --get \
  --data-urlencode "filter=$FILTER" \
  --data-urlencode "interval.startTime=$START" \
  --data-urlencode "interval.endTime=$END" \
  'https://monitoring.googleapis.com/v3/projects/level-mark-437714-b1/timeSeries' \
  | jq -r '(.timeSeries // [])[] | "points=" + ((.points // []) | length | tostring) + " latest=" + (.points[0].value.doubleValue // .points[0].value.int64Value // "n/a" | tostring) + " at " + .points[0].interval.endTime'
```

Terraform:

```bash
terraform -chdir=infra/gcp-notebook plan -detailed-exitcode
```

Exit code `0` means the deployed infra matches the repo config. Exit code `2` means Terraform found pending changes.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `NVIDIA_A100_80GB_GPUS exceeded. Limit: 0.0` | Project has no A100 80GB quota in the region | Use `a2-highgpu-1g`, or request A100 80GB quota before using `a2-ultragpu-1g` |
| `state: STOCKOUT` | The selected zone currently lacks capacity | Move `GCP_ZONE` to a zone suggested by the error, then rerun `scripts/gcp-gpu-up.sh` |
| `gpg: cannot open '/dev/tty'` in startup logs | GPG tried to prompt inside a non-interactive startup script | Fixed in `startup.sh` with `gpg --batch --yes --dearmor` |
| `tar: Ignoring unknown extended header keyword LIBARCHIVE...` | macOS extended attributes in tar stream | Fixed in sync script with `COPYFILE_DISABLE=1` and `--no-xattrs` |
| `tar: .: Cannot change mode` | Remote workspace directory was root-owned | Fixed in sync script by chowning `/mnt/tfm/workspace` to the SSH user before extracting |
| `gcloud.auth.print-access-token` says reauthentication failed | Local `gcloud` user credentials expired while the workspace was idle | Run `gcloud auth login`, then rerun `scripts/gcp-gpu-up.sh` |
| Terraform reports expired Application Default Credentials | Local ADC credentials expired; Terraform uses ADC separately from the active `gcloud` account | Run `gcloud auth application-default login`, then rerun `scripts/gcp-gpu-up.sh` |
| `failed to connect to backend` or `Failed to connect to port 22` right after VM start | GCP reported the VM as running before SSH was ready, or the local environment forced IAP unnecessarily | Rerun the helper script after a minute. The scripts now wait for SSH readiness and only use IAP when `GCP_TUNNEL_THROUGH_IAP=1` |
| Browser cannot reach marimo | SSH tunnel is closed or port differs | Keep `scripts/gcp-marimo.sh` running and open the printed URL |
| `docker run --gpus all` fails | NVIDIA Container Toolkit not installed/configured | Check `/opt/tfm/bootstrap.done`, `docker info`, and `nvidia-container-toolkit` package status |
| Notebook 04 says `Decoder checkpoint not found at /workspace/models/decoder-foundation-model` | The VM model mount is missing the resolved Git LFS checkpoint | Run `git lfs pull --include='models/decoder-foundation-model/**' --exclude=''` locally, then run `scripts/gcp-sync-models.sh` |
| GCP Console does not show GPU utilization | Ops Agent missing, stopped, missing IAM, or no active GPU work yet | Rerun `scripts/gcp-gpu-up.sh`, verify `systemctl is-active google-cloud-ops-agent`, then wait a few minutes while a notebook or `nvidia-smi` workload uses the GPU |
| Notebook errors with `CUDA_ERROR_NO_DEVICE`, `cudaErrorDevicesUnavailable`, or container `nvidia-smi` says `Failed to initialize NVML: Unknown Error` | The running marimo container lost GPU device access after host service changes, usually because it was started before VM bootstrap fully finished | Rerun `scripts/gcp-marimo.sh` to delete and recreate only the marimo container, then restart the notebook kernel and rerun cells |

## Safety

marimo is bound to `127.0.0.1` on the VM and reached through SSH port forwarding. The helper scripts do not expose port 8080 publicly.

Stop the VM when you are done:

```bash
scripts/gcp-gpu-down.sh
```

The persistent disk and GCS bucket remain intact.

The VM also schedules a shutdown after 12 hours by default. This is a cost guard, not a replacement for stopping the VM when you finish.
