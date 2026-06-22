> **Status:** structurally verified, NOT yet run on real hardware. **Last updated:** 2026-06-17

# GPU-RUNBOOK — launching a real TFM decoder pretrain on the single GCP GPU VM

This is the operator runbook for ARCHITECTURE.md §5.1 / §10 step 7: drive a **real**
NeMo decoder-CLM pretrain on the one shared GCP GPU VM, through Loom's
`nemo` model-builder + `gcp-vm` executor.

> [!IMPORTANT]
> **This step's code is structurally verified but has NOT yet been run on real
> hardware.** Everything below passes a full structural test suite with **zero GPU
> spend** (133 passed / 1 unrelated skip; dry-run executor constructs every
> gcloud/docker/torchrun command without firing a subprocess). Going live requires
> the actual VM + real money + GPU time, and exercises two paths that are
> *unverified against a real run* (called out inline): (1) the wrapping launcher's
> per-step hook attachment point inside NeMo's `run_train_validation_loop`, and
> (2) the `fetch_progress` JSONL pull-back from the VM. **Do one short 2-step smoke
> run first** (see step 7) before trusting budget telemetry or kicking off a long run.

---

## 0. What runs where (the mechanism, one paragraph)

The `loom` control plane stays CPU-only — it imports **zero** of
nemo/torch/transformers/cudf. It renders a NeMo YAML (a copy of the committed
`configs/pretrain_financial_decoder.yaml` with exactly 3-4 overrides), builds a
`torchrun` argv over the adapter's **own** wrapping launcher
(`loom/adapters/_nemo_train_entry.py`), and submits a command sequence to the
single VM via `gcloud compute ssh`:

1. `scripts/gcp-gpu-up.sh` — terraform-apply / start the VM (`tfm-gpu-notebook`).
2. `scripts/gcp-sync-workspace.sh` — tar the repo over ssh to `$REMOTE_WORKSPACE`
   (so `src/clm_data.py`, the launcher, and the rendered YAML are present).
3. `docker run -d --gpus all --name loom-nemo-train <NeMo image> sleep infinity`
   (PID-1-alive pattern from `.github/workflows/ci.yml`), wrapped in `gcloud ssh`.
4. `docker exec -w /workspace loom-nemo-train timeout <N> bash -lc 'cd /workspace
   && torchrun --nproc-per-node=<gpus> loom/adapters/_nemo_train_entry.py -c
   <rendered.yaml> --dataset.data_path … --step_scheduler.max_steps …'`

The launcher wraps the SAME 4-line recipe as `scripts/train_decoder_model.py`
(`parse_args_and_load_config` → `TrainFinetuneRecipeForNextTokenPrediction` →
`.setup()` → `.run_train_validation_loop()`) and, around it, writes a structured
**JSONL progress log** (`step, loss, lr, tokens`) — the net-new progress source
Loom tails. The team's `scripts/train_decoder_model.py`, `configs/*.yaml`, and
`src/*` are **never touched**: the adapter ships its own launcher and renders its
own YAML.

The binding budget envelope is enforced two ways: a `timeout` wrapper around the
`docker exec`, and the orchestrator hard-kill (`loom`'s `executor.kill()` →
`scripts/gcp-gpu-down.sh`, which stops the VM and ends all spend).

---

## 1. One-time local prerequisites

```bash
# From the TFM repo root: /Users/anub/Work/transaction-foundation-model
gcloud auth login                      # user creds: ssh / start / stop / describe
gcloud auth application-default login  # ADC: terraform apply (VM/disk/bucket/IAM)
bash scripts/gcp-setup-check.sh        # confirms gcloud + terraform + jq + quota
```

VM identity comes from env (NOT tfvars — `gcp-gpu-up.sh` writes tfvars *from* env).
Defaults live in `.env.gcp.example`; copy to a gitignored `.env.gcp` to override.
Live defaults:

| Var | Default | Meaning |
|---|---|---|
| `GCP_PROJECT` | `level-mark-437714-b1` | project |
| `GCP_ZONE` | `us-central1-f` | zone |
| `GCP_INSTANCE` | `tfm-gpu-notebook` | the VM name |
| `GCP_MACHINE_TYPE` | `a2-highgpu-1g` (1× A100 40GB) | machine / GPU count |
| `GCP_BUCKET` | `level-mark-437714-b1-tfm-gpu-artifacts` | checkpoint bucket |
| `NOTEBOOK_IMAGE` | `nvcr.io/nvidia/nemo:25.09.01` | the NeMo container image |
| `REMOTE_WORKSPACE` | `/mnt/tfm/workspace` | synced repo → mounted `/workspace` |

The `loom` venv is at `Loom/.venv`. Install once:
`/Users/anub/Work/transaction-foundation-model/Loom/.venv/bin/python -m pip install -e '.[dev]'`.

---

## 2. Build the Corpus (CPU, no GPU)

```bash
cd /Users/anub/Work/transaction-foundation-model/Loom
export LOOM_WORKSPACE=$PWD/.loom            # store + run-dir root
PY=$LOOM_WORKSPACE/../.venv/bin/python      # or just Loom/.venv/bin/python

# ingest the raw events, then compile the tokenizer → a Corpus
$PY -m loom ingest   --in <data.csv> --name tfm --entity wallet --event trade
$PY -m loom tokenize --in IngestDataset/1 --preset financial --context-len 4096
# → Corpus/1   (carries vocab_size, signature, tensor_contract clm/input_ids+labels/-100)
```

The corpus's `vocab_size` becomes `model.config.vocab_size`; its `context_len`
drives the token-count in the cost plan. The tensor-contract handshake
(`clm/input_ids+labels/-100`) is what the `nemo` builder's `supports()` checks.

---

## 3. PLAN first — always (no spend, no VM)

```bash
$PY -m loom pretrain --in Corpus/1 \
    --model-builder nemo \
    --gpu-target a2-highgpu-1g \   # a real machine type → real $/hour in the plan
    --nproc-per-node 1 \           # = GPUs on the machine type (a2-highgpu-1g = 1)
    --max-steps 3000 \
    --json
```

You get `status=PLAN` with a **derived** cost (`cost_plan.derived=true`):
`usd = 6·N·D FLOPs / (gpus · ~150 TFLOP/s · MFU) · $/gpu-hour`, where `N` is the
param count off the rendered LlamaConfig and `D = max_steps · global_batch_size ·
seq_len`. The PLAN also mints a **single-use `confirm_token`** scoped to the plan
hash, and STOPS. Nothing launched.

- **No `--gpu-target` (and no `LOOM_GPU_TARGET`)** → `REFUSED_NO_GPU_TARGET`. The
  `nemo` builder requires a machine type; the CPU-rehearsal `local` builder does not.
- **`--execution local` under `--model-builder nemo`** → `REFUSED_NO_GPU_TARGET`
  too (the in-process executor can't reach a GPU). `nemo` defaults to `gcp-vm`.
- **Derived `usd` > `--max-usd`** → `REFUSED_SPEND_CAP` (raise the cap or shrink
  `--max-steps`/the arch).
- **An agent (driver=agent) cannot launch** → `REFUSED_AGENT_CANNOT_LAUNCH`; the
  agent surfaces the PLAN + token, a human confirms.

Read the plan. Confirm the dollar figure, the GPU machine type, `params`, and
`tokens` are what you expect before spending.

---

## 4. Bring up the VM + sync the workspace

These are real GCP operations (cost starts when the VM is RUNNING):

```bash
cd /Users/anub/Work/transaction-foundation-model
bash scripts/gcp-gpu-up.sh          # terraform converge + start; waits for ssh
bash scripts/gcp-sync-workspace.sh  # tar the repo → /mnt/tfm/workspace on the VM
```

`loom` also constructs these two as the first two commands of its submit sequence;
running them by hand here lets you watch the VM come up and verify the sync
landed before you authorize spend. (The VM auto-shuts-down after 12h as a cost
guard, independent of `loom`.)

---

## 5. The gated launch (real spend)

```bash
cd /Users/anub/Work/transaction-foundation-model/Loom
export LOOM_GCP_LIVE=1     # OPT-IN to live execution (absence ⇒ dry-run, no spend)
# do NOT also set LOOM_DRY_RUN — it forces dry-run and wins.

$PY -m loom pretrain --in Corpus/1 \
    --model-builder nemo --gpu-target a2-highgpu-1g --nproc-per-node 1 \
    --max-steps 3000 \
    --max-usd 50 \              # the BINDING spend cap the orchestrator kills at
    --max-wall-clock-min 240 \  # the BINDING wall-clock cap → `timeout` + auto-kill
    --launch \
    --confirm \                 # the typed confirm, required above the $-threshold
    --confirm-token <TOKEN-FROM-THE-PLAN>
```

Gate semantics (verified):
- Above the `$5` threshold (`LOOM_CONFIRM_USD_THRESHOLD`), a launch needs **BOTH**
  a valid `--confirm-token` AND `--confirm`. A launch missing `--confirm` returns a
  fresh PLAN and **does NOT burn the token** — so the documented "relaunch with the
  token AND `--confirm`" actually works (the same token survives).
- At/below the threshold, a valid token alone launches.
- The token is single-use and plan-hash-scoped: any change to corpus/arch/budget
  invalidates it; you must re-PLAN.

On confirm, `loom` runs the up → sync → container-start → `docker exec torchrun`
sequence on the VM and returns a `Checkpoint` object once the run is terminal.

---

## 6. What to watch during the run

**The JSONL progress feed** is the source of truth (NOT the recipe's stdout, which
prints only a config header). The launcher writes it on the VM at
`/mnt/tfm/artifacts/<run>/progress.jsonl` (in-container `/workspace/artifacts/…`).
Loom's handle pulls it back and turns each record into a `ProgressEvent`.

Watch these signals:

1. **The step-0 canary.** First record is `phase=warmup`,
   `note="loss≈ln(vocab)=…"`, `loss ≈ ln(vocab_size)`. For a from-scratch CLM the
   real step-1 loss should start near this value. If step 0 is wildly off, the
   vocab/data wiring is wrong — kill it.
2. **Loss descending** across `phase=train` records (`step, loss, lr, tokens`).
3. **`usd_spent`** on each event — derived from `wall_clock_min × ($/gpu-hour ×
   gpus)/60` — climbing toward the envelope. `usd_envelope` is the binding cap.
4. **The envelope burn / hard-kill.** If the run trips the budget, status becomes
   `stopped_at_budget`; `loom` records verdict `INCOMPLETE` + a `resume_token` and
   fires `executor.kill()` → `gcp-gpu-down.sh`. A clean run is `succeeded`/`PASS`.

Tail it directly if you want a raw view:

```bash
gcloud compute ssh tfm-gpu-notebook --zone us-central1-f --project level-mark-437714-b1 \
  --command 'tail -n +1 -F /mnt/tfm/artifacts/<run>/progress.jsonl'
```

> [!WARNING]
> **Unverified against a real run.** The per-step hook attachment in the launcher
> (`_attach_step_hook` / `_wrap_step_logger`) targets *plausible* NeMo-AutoModel
> surfaces that are not pinned anywhere in this repo. The guaranteed-correct floor
> is the checkpoint-dir + step-counter **poller** (branch c), which always runs.
> On the **first** smoke run, confirm the per-step `loss`/`lr` records actually
> appear (branch a/b attached) — if only `note="poller"` records show up, the fine
> hook didn't attach and per-step loss telemetry is coarse until the hook name is
> pinned. Likewise the control-plane `fetch_progress` pull-back is exercised in
> tests with a fake hook; confirm the live `gcloud ssh cat` fetch populates the
> read path on the first run, or the feed will look empty.

---

## 7. First-run smoke test (do this before any long run)

```bash
export LOOM_GCP_LIVE=1
$PY -m loom pretrain --in Corpus/1 --model-builder nemo \
    --gpu-target a2-highgpu-1g --nproc-per-node 1 \
    --max-steps 2 --max-usd 5 --max-wall-clock-min 30 \
    --launch --confirm --confirm-token <TOKEN>
```

A 2-step run validates the full path end to end for a few dollars: the container
starts, `torchrun` runs the recipe inside it, the JSONL appears with the step-0
canary + per-step records, and a consolidated-safetensors checkpoint lands. Verify
the per-step hook attached (step 6 warning) before trusting budget telemetry on a
long run.

---

## 8. The checkpoint (PASS → durable uri)

On `succeeded`, `loom` writes a `Checkpoint` DataObject:
- `fmt = "hf-safetensors-consolidated"` — loadable by vanilla
  `AutoModelForCausalLM.from_pretrained`, zero NeMo dependency downstream.
- `uri` is a **durable** location, never a tempdir: `gs://<GCP_BUCKET>/loom-checkpoints/<run>/…`
  when the bucket is set, else `vm://tfm-gpu-notebook/mnt/tfm/…`.
- `signatures.representation_signature` ECHOES the corpus signature (the §3/§7
  pairing invariant embed/evaluate assert before any forward pass).

Push the VM checkpoint to GCS so it survives the VM stop:

```bash
cd /Users/anub/Work/transaction-foundation-model
bash scripts/gcp-sync-models.sh   # VM /mnt/tfm/models/… → gs://…-tfm-gpu-artifacts/…
```

---

## 9. Tear down — STOP SPENDING

```bash
cd /Users/anub/Work/transaction-foundation-model
bash scripts/gcp-gpu-down.sh      # gcloud compute instances stop — disk + bucket survive
```

This is also `loom`'s `executor.kill()` target (the binding-envelope hard-kill).
Always run it when done; the 12h auto-shutdown is a backstop, not a substitute.
To stop just the run but keep the VM up: `gcloud compute ssh tfm-gpu-notebook
--zone us-central1-f --command 'sudo docker rm -f loom-nemo-train'`.

---

## Quick reference

| Action | Command |
|---|---|
| Auth | `gcloud auth login` + `gcloud auth application-default login` |
| Readiness | `bash scripts/gcp-setup-check.sh` |
| Build corpus | `loom ingest …` → `loom tokenize …` |
| Plan (no spend) | `loom pretrain --in Corpus/1 --model-builder nemo --gpu-target <type> --json` |
| Bring up VM | `bash scripts/gcp-gpu-up.sh` + `bash scripts/gcp-sync-workspace.sh` |
| Launch (live) | `LOOM_GCP_LIVE=1 loom pretrain … --launch --confirm --confirm-token <T>` |
| Watch | `tail -F /mnt/tfm/artifacts/<run>/progress.jsonl` (step-0 canary, loss, usd_spent) |
| Save ckpt | `bash scripts/gcp-sync-models.sh` |
| Tear down | `bash scripts/gcp-gpu-down.sh` |

**Dry-run by default.** With no `LOOM_GCP_LIVE`, the executor constructs every
command and runs **none** of them — that is how all verification happens with zero
spend. Live execution is an explicit `LOOM_GCP_LIVE=1` opt-in only.
