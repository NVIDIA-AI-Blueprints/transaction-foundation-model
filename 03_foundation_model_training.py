import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <!--
    SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
    SPDX-License-Identifier: Apache-2.0

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    -->
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Notebook 03: Foundation Model Training

    We train a **decoder foundation model** on the tokenized transaction sequences from notebook 02 using **causal language modeling** (next-token prediction). At each position, the model predicts the next token from all preceding tokens, learning transaction patterns without any fraud labels.

    ```
    Input:  <bos>  AMT_3  MERCH_1498  CAT_RETAIL  MCC_5912  HOUR_05  DOW_0  ...
    Target:        AMT_3  MERCH_1498  CAT_RETAIL  MCC_5912  HOUR_05  DOW_0  MONTH_05  ...
    ```

    Every token position provides a training signal, so through millions of predictions the model builds rich representations of spending patterns, merchant relationships, and temporal behaviors.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model Configuration

    | Parameter | Value | Notes |
    |-----------|-------|-------|
    | **Architecture** | Decoder-only Transformer | Causal (autoregressive) attention |
    | **Hidden Size** | 512 | Transformer hidden dimension |
    | **Layers** | 8 | Transformer depth |
    | **Attention Heads** | 8 query / 2 KV | Grouped-Query Attention (GQA) |
    | **FFN Size** | 1408 | SwiGLU activation |
    | **Context Window** | 8192 | RoPE positional encoding |
    | **Parameters** | **\~29M** | Compact embedding table |
    | **Vocab Size** | \~6,252 | Matches `FinancialTokenizerPipeline` (merchant_hash_size=2,000) |
    | **Training Data** | \~263M tokens | \~64K sequences from 19.5M transactions |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training Pipeline

    Training uses `scripts/train_decoder_model.py` with **NeMo AutoModel** and FSDP2 for distributed training. Configuration is defined in `configs/pretrain_financial_decoder.yaml`; custom tokenizer and dataset are integrated via the `_target_` syntax. Scaling from 1 GPU to multi-node requires only a `torchrun` wrapper -- no code changes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Custom Components

    We only provide two domain-specific components -- everything else (FSDP2, optimizer, checkpointing, mixed precision) is handled by NeMo AutoModel:

    1. **`FinancialTabularTokenizer`** -- custom tokenization for transaction data
    2. **`FinancialCLMDataset`** -- next-token prediction data loading (referenced in YAML via `_target_`)

    Checkpoints are saved in HuggingFace format (`save_consolidated: true`), enabling downstream use with `AutoModelForCausalLM.from_pretrained()`.
    """)
    return


@app.cell
def _():
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")
    warnings.filterwarnings("ignore", category=SyntaxWarning, module="liger_kernel")
    warnings.filterwarnings("ignore", message="The '.*' attribute with value.*was provided")

    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(".").resolve()
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))

    DATA_DIR = PROJECT_ROOT / "data"
    CORPUS_DIR = DATA_DIR / "decoder_corpus"
    CORPUS_PATH = CORPUS_DIR / "train_corpus.txt"
    VAL_CORPUS_PATH = CORPUS_DIR / "val_corpus.txt"

    YAML_CONFIG = PROJECT_ROOT / "configs/pretrain_financial_decoder.yaml"
    TRAIN_SCRIPT = PROJECT_ROOT / "scripts/train_decoder_model.py"

    PRETRAINED_MODEL = PROJECT_ROOT / "models/decoder-foundation-model"

    # Verify prerequisites from notebook 02
    assert CORPUS_PATH.exists(), f"Training corpus not found: {CORPUS_PATH}  (run notebook 02 first)"
    assert VAL_CORPUS_PATH.exists(), f"Validation corpus not found: {VAL_CORPUS_PATH}  (run notebook 02 first)"
    assert YAML_CONFIG.exists(), f"YAML config not found: {YAML_CONFIG}"
    assert TRAIN_SCRIPT.exists(), f"Training script not found: {TRAIN_SCRIPT}"

    # Pre-trained checkpoint (used by notebook 04 for inference)
    if PRETRAINED_MODEL.exists() and (PRETRAINED_MODEL / "config.json").exists():
        print(f"Pre-trained decoder checkpoint available for notebook 04:")
        print(f"  {PRETRAINED_MODEL}\n")

    print(f"Training corpus : {CORPUS_PATH}")
    print(f"Validation      : {VAL_CORPUS_PATH} ({'exists' if VAL_CORPUS_PATH.exists() else 'NOT FOUND'})")
    print(f"YAML config     : {YAML_CONFIG}")
    print(f"Pre-trained     : {PRETRAINED_MODEL}")
    return (
        CORPUS_PATH,
        PRETRAINED_MODEL,
        PROJECT_ROOT,
        TRAIN_SCRIPT,
        VAL_CORPUS_PATH,
        YAML_CONFIG,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Launch Training

    Training is configured through `configs/pretrain_financial_decoder.yaml` (model architecture, AdamW + cosine annealing optimizer, FSDP2 distribution, consolidated HF checkpoints). The cell below runs a **30-step demo** on 1 GPU (\~2 min). The pre-trained model used in notebook 04 was trained for \~3,000 steps on 8x A100.

    The random baseline loss is `ln(~6,252) ~ 8.7`; loss should drop to \~5–6 within 30 steps.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### YAML Configuration
    """)
    return


@app.cell
def _(YAML_CONFIG):
    print(f"=== {YAML_CONFIG} ===\n")
    with open(YAML_CONFIG) as f:
        print(f.read())
    return


@app.cell
def _(
    CORPUS_PATH,
    PRETRAINED_MODEL,
    PROJECT_ROOT,
    TRAIN_SCRIPT,
    VAL_CORPUS_PATH,
    YAML_CONFIG,
):
    import subprocess

    # Fresh training demo: 30 steps from scratch on 1 GPU.
    # The loss will drop rapidly from ~8.7 (random baseline for ~6K vocab) demonstrating
    # that the model is learning transaction patterns. The full pre-trained checkpoint
    # is available for notebook 04.
    DEMO_CHECKPOINT_DIR = PROJECT_ROOT / "models" / "decoder-demo" / "checkpoints"
    DEMO_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    PYTHON = "/opt/venv/bin/python"

    cmd = [
        PYTHON, str(TRAIN_SCRIPT),
        "-c", str(YAML_CONFIG),
        "--dataset.data_path", str(CORPUS_PATH),
        "--validation_dataset.data_path", str(VAL_CORPUS_PATH),
        "--step_scheduler.max_steps", "30",
        "--step_scheduler.global_batch_size", "16",
        "--step_scheduler.local_batch_size", "16",
        "--lr_scheduler.lr_warmup_steps", "10",
        "--step_scheduler.val_every_steps", "15",
        "--step_scheduler.ckpt_every_steps", "15",
        "--checkpoint.checkpoint_dir", str(DEMO_CHECKPOINT_DIR),
    ]

    print("Launching training (30 steps, 1 GPU):")
    print(" ".join(cmd))
    print()

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"\nTraining exited with code {result.returncode}")
    else:
        print("\nTraining complete!")
        print(f"\nThe full pre-trained model (trained on 8x A100) is available at:")
        print(f"  {PRETRAINED_MODEL}")

    # Show the torchrun command for full distributed training
    print("\n" + "=" * 70)
    print("For full distributed training (8 GPUs):")
    print("=" * 70)
    print(f"\n  torchrun --nproc-per-node=8 {TRAIN_SCRIPT} \\")
    print(f"      -c {YAML_CONFIG} \\")
    print(f"      --dataset.data_path {CORPUS_PATH} \\")
    print(f"      --validation_dataset.data_path {VAL_CORPUS_PATH}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Checkpoints

    With `save_consolidated: true`, checkpoints are in HuggingFace format (`config.json` + `safetensors`), loadable via `AutoModelForCausalLM.from_pretrained()`. The pre-trained checkpoint at `models/decoder-foundation-model/` is ready for inference in notebook 04.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    We trained a \~29M parameter decoder foundation model on \~64K tokenized transaction sequences using NeMo AutoModel. The model learns to predict the next transaction token from context through self-supervised learning, without any fraud labels. The full pre-trained checkpoint (\~3,000 steps on 8x A100) is available for the next step.

    Continue to [04_inference_embedding_extraction.py](./04_inference_embedding_extraction.py).
    """)
    return


if __name__ == "__main__":
    app.run()
