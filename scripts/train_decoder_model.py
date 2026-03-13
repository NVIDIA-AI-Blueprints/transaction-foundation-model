"""
Financial Foundation Model Pre-training — Decoder-Only (NeMo AutoModel)

Pretrains a decoder causal language model on financial transaction
sequences using NeMo AutoModel's TrainFinetuneRecipeForNextTokenPrediction.

Custom components:
  - FinancialTabularTokenizer: Domain-specific tokenizer (12 tokens per txn)
  - FinancialCLMDataset: Next-token prediction data loading (no MLM masking)

Launch methods
--------------
Multi-GPU (recommended):
    torchrun --nproc-per-node=8 scripts/train_decoder_model.py \
        -c configs/pretrain_financial_decoder.yaml \
        --dataset.data_path data/decoder_corpus/train_corpus.txt \
        --validation_dataset.data_path data/decoder_corpus/val_corpus.txt

Single GPU (testing):
    python scripts/train_decoder_model.py \
        -c configs/pretrain_financial_decoder.yaml \
        --dataset.data_path data/decoder_corpus/train_corpus.txt
"""

import sys
from pathlib import Path

BLUEPRINT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BLUEPRINT_ROOT))


def main():
    from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
    from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

    cfg = parse_args_and_load_config()

    model_cfg = cfg.model.config
    target = getattr(model_cfg, '_target_', None)
    if target is not None:
        arch = target.__name__ if isinstance(target, type) else str(target).rsplit('.', 1)[-1]
    else:
        arch = "unknown"

    print("\n" + "=" * 60)
    print("Financial Foundation Model — Decoder-Only Pretraining")
    print("=" * 60)
    print(f"  Architecture: {arch}")
    for key in ['hidden_size', 'n_embd']:
        try:
            print(f"  Hidden size:  {getattr(model_cfg, key)}")
            break
        except (AttributeError, KeyError):
            continue
    for key in ['num_hidden_layers', 'n_layer']:
        try:
            print(f"  Layers:       {getattr(model_cfg, key)}")
            break
        except (AttributeError, KeyError):
            continue
    print(f"  Vocab size:   {model_cfg.vocab_size}")
    print(f"  Max steps:    {cfg.step_scheduler.max_steps}")
    print(f"  Batch size:   {cfg.step_scheduler.global_batch_size} global, "
          f"{cfg.step_scheduler.local_batch_size} local")
    print("=" * 60 + "\n")

    recipe = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    recipe.setup()
    recipe.run_train_validation_loop()


if __name__ == "__main__":
    main()
