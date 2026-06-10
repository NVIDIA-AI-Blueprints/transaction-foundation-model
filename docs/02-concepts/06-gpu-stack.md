# Primer 6: The GPU Stack — NeMo, RAPIDS, and Friends

**You know:** HuggingFace is "the place models come from"; you've `pip install`ed PyTorch.
**You'll learn:** what each NVIDIA component in this repo actually does, why the repo uses it, and the minimal mental model for working with it.

The honest framing: this repo contains ~zero lines of training-loop or GPU-management code. That's not laziness — it's the design. Each infrastructure concern is delegated to a tool that specializes in it. Here's the cast.

## The map

```
┌─ NeMo Framework container (nvcr.io/nvidia/nemo:25.09.01) ──────────┐
│   everything below pre-installed & version-matched                 │
│                                                                    │
│   Data prep (notebooks 01–02)        Training (notebook 03)        │
│   ┌────────────────────────┐         ┌──────────────────────────┐  │
│   │ RAPIDS                 │ corpus  │ NeMo AutoModel           │  │
│   │  cuDF   (GPU pandas)   │ ──────► │  recipe + YAML config    │  │
│   │  cuML   (GPU sklearn)  │         │  FSDP2 distribution      │  │
│   │  CuPy   (GPU numpy)    │         │  via torchrun            │  │
│   └────────────────────────┘         └───────────┬──────────────┘  │
│                                                  │ HF checkpoint   │
│   Downstream (notebooks 04–05)                   ▼                 │
│   ┌──────────────────────────────────────────────────────────┐    │
│   │ HuggingFace Transformers (load model, hidden states)      │    │
│   │ XGBoost (gpu hist), cuML UMAP, PyTorch                    │    │
│   └──────────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────────┘
         model checkpoint shipped via Git LFS
```

## RAPIDS: pandas/sklearn/numpy, GPU edition

**What it is:** a suite of GPU-native data libraries with deliberately familiar APIs — **cuDF** mirrors pandas, **cuML** mirrors scikit-learn, **CuPy** mirrors NumPy.

**Why it's here:** notebook 01 parses a 24.4M-row CSV and does groupbys/sorts over it; the tokenizer hashes and maps 19.5M rows. On CPU pandas this is coffee-break territory per cell; on cuDF it's seconds. The tokenizer pipeline is written *against* cuDF — see the `import cudf` at the top of [`src/tokenizer/financial_pipeline.py`](../../src/tokenizer/financial_pipeline.py).

**Mental model:** read `cudf.DataFrame` as `pandas.DataFrame` and you'll follow 95% of the code. Differences that matter here:

- Data lives in GPU memory; `.to_pandas()` moves it to CPU (you'll see this at corpus-writing and plotting boundaries — and in [`pipeline.encode`](../../src/tokenizer/pipeline.py), which maps token strings to IDs via pandas on the host).
- `.hash_values()` (used for merchant hashing) and `.str.*` ops run on-GPU.
- cuML supplies the GPU UMAP in notebook 04 and the optional quantile/kmeans amount binning in [`src/tokenizer/numerical.py`](../../src/tokenizer/numerical.py).
- The pipeline even fits its tokenizer steps on parallel **CUDA streams** ([`pipeline._fit_parallel`](../../src/tokenizer/pipeline.py)) — overlapping independent GPU work, a nicety you get for free by reading the code.

## NeMo AutoModel: training without writing a training loop

**What it is:** NVIDIA **NeMo** is a large framework for building/training generative-AI models. **NeMo AutoModel** is its HuggingFace-native sub-library: point it at any HF-compatible model (a hub ID *or* a from-scratch config) and it handles the training loop, distributed execution, checkpointing, and recovery — driven by a single YAML file.

**Why it's here:** it deletes the hardest engineering. The repo supplies only domain pieces — tokenizer, dataset class, YAML — and NeMo runs the show. The entire launcher ([`scripts/train_decoder_model.py`](../../scripts/train_decoder_model.py)) reduces to:

```python
cfg = parse_args_and_load_config()
recipe = TrainFinetuneRecipeForNextTokenPrediction(cfg)
recipe.setup()
recipe.run_train_validation_loop()
```

**The YAML is the real program.** Three idioms to read it fluently ([full config](../../configs/pretrain_financial_decoder.yaml)):

1. **`_target_` = "instantiate this."** Dotted path → class/function, siblings → kwargs:
   ```yaml
   optimizer:
     _target_: torch.optim.AdamW
     lr: 0.0002
   ```
2. **File-path targets** let configs reference *your local code* without packaging:
   ```yaml
   dataset:
     _target_: src/clm_data.py:build_financial_clm_dataset
     data_path: null            # supplied on the CLI
     seq_length: 4096
   ```
3. **CLI dot-overrides** beat editing files (and make experiments scriptable):
   ```bash
   python scripts/train_decoder_model.py -c configs/pretrain_financial_decoder.yaml \
       --dataset.data_path data/decoder_corpus/train_corpus.txt \
       --step_scheduler.max_steps 3000
   ```

**Checkpoint interop, the underrated feature:** with `save_consolidated: true`, NeMo writes standard **HuggingFace format** (`config.json` + `model.safetensors`). Downstream code needs zero NeMo — notebook 04 loads with vanilla `AutoModelForCausalLM.from_pretrained(...)`. Train with NVIDIA tooling, serve with anything.

## FSDP2 + torchrun: the scaling story

**torchrun** launches N copies of the script (one per GPU) and wires them into a process group:

```bash
torchrun --nproc-per-node=8 scripts/train_decoder_model.py -c configs/... 
```

**FSDP2** (PyTorch Fully Sharded Data Parallel, v2) is the strategy NeMo uses to coordinate those copies — sharding model/optimizer state across GPUs and each rank processing different batches. Configured, not coded:

```yaml
distributed:
  _target_: nemo_automodel.components.distributed.fsdp2.FSDP2Manager
  dp_size: none        # infer data-parallel size from world size
  tp_size: 1           # tensor parallelism off (29M params doesn't need it)
  cp_size: 1           # context parallelism off
```

At 29M parameters FSDP2 is overkill — the point is that **the config is already multi-node-shaped**. Scale the model 100× and you change these numbers, not your code. (The same `torchrun` line is how the shipped checkpoint was trained on 8× A100.)

One gotcha preserved in the config comments: use `torchrun` directly for multi-GPU — the `automodel` CLI wrapper misparses `--nproc-per-node`.

## The NGC container: the version-matching service

CUDA / PyTorch / RAPIDS / NeMo each have mutual version constraints; resolving them by hand is a lost afternoon (RAPIDS pip installs are especially unforgiving). The **NGC** (NVIDIA GPU Cloud) container `nvcr.io/nvidia/nemo:25.09.01` ships the whole matrix pre-resolved. That's the entire reason [setup](../01-getting-started/02-environment-setup.md) says "run in the container" — it's not dogma, it's dependency hygiene. The few extras (xgboost, seaborn, plotly…) are `%pip install`ed per-notebook.

## Git LFS: why the checkpoint needs a special pull

Git stores every version of every file forever — terrible for 56 MB binary weights. **Git LFS** keeps a tiny pointer file in git and the real bytes in sidecar storage, fetched by `git lfs pull`. Forgetting the pull is the #1 setup failure: you'll have a 134-byte "safetensors" file and a confusing load error in notebook 04. Check with `ls -lh models/decoder-foundation-model/`.

## Cheat sheet

| Tool | One-liner | You interact via |
|---|---|---|
| cuDF / cuML / CuPy | pandas / sklearn / numpy on GPU | same APIs, `.to_pandas()` at boundaries |
| NeMo AutoModel | YAML-driven trainer for HF-compatible models | the YAML + CLI overrides |
| FSDP2 | shards training state across GPUs | `distributed:` block (already set) |
| torchrun | spawns the per-GPU processes | `--nproc-per-node=N` |
| NGC container | pre-matched CUDA/PyTorch/RAPIDS/NeMo | `docker run nvcr.io/nvidia/nemo:25.09.01` |
| HF Transformers | loads the trained checkpoint anywhere | `AutoModelForCausalLM.from_pretrained` |
| Git LFS | big files outside git history | `git lfs pull` |

**Next:** you have all the primitives. Head to the [Learning Path](../03-learning-path/README.md) to assemble them into the full system.
