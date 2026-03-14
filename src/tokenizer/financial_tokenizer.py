# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tokenizer interface for decoder-only model training and inference.

Wraps FinancialTokenizerPipeline in the same API that FinancialCLMDataset
and train_decoder_model.py expect: encode(), decode(), vocab_size,
and special-token ID attributes.

Uses <bos>/<eos>/<sep>/<pad>/<unk> special tokens for causal language
modeling (no mask token needed).
"""

from typing import Dict, List, Optional

from .financial_pipeline import FinancialTokenizerPipeline


class FinancialTabularTokenizer:
    """Tokenizer for decoder-only causal LM over financial transactions.

    Constructor signature and all public attributes are preserved so that
    training scripts, data modules, and embedding extraction scripts work
    without changes.
    """

    def __init__(
        self,
        merchant_hash_size: int = 2000,
        category_hierarchy: bool = True,
        amount_bins: Optional[List[float]] = None,
        temporal_encoding: bool = True,
        special_tokens: Optional[Dict[str, str]] = None,
        amount_strategy: str = "fixed",
        include_time_delta: bool = False,
        **kwargs,
    ):
        self.merchant_hash_size = merchant_hash_size
        self.category_hierarchy = category_hierarchy
        self.temporal_encoding = temporal_encoding
        self.amount_bins = amount_bins or [0, 10, 50, 100, 500, 1000, 5000, float("inf")]

        self._pipeline = FinancialTokenizerPipeline(
            merchant_hash_size=merchant_hash_size,
            amount_strategy=amount_strategy,
            include_time_delta=include_time_delta,
        )
        self._build_vocab_from_pipeline()

        self.special_tokens = {
            "pad": "<pad>",
            "bos": "<bos>",
            "eos": "<eos>",
            "sep": "<sep>",
            "unk": "<unk>",
        }
        self.pad_token_id = self.vocab.get("<pad>", 0)
        self.bos_token_id = self.vocab.get("<bos>", 1)
        self.eos_token_id = self.vocab.get("<eos>", 2)
        self.sep_token_id = self.vocab.get("<sep>", 3)
        self.unk_token_id = self.vocab.get("<unk>", 4)
        self.special_token_ids = {
            self.pad_token_id,
            self.bos_token_id,
            self.eos_token_id,
            self.sep_token_id,
            self.unk_token_id,
        }

    # ------------------------------------------------------------------
    # Internal: build vocab by running build_vocab on every step
    # ------------------------------------------------------------------

    def _build_vocab_from_pipeline(self) -> None:
        """Build the global vocabulary by calling build_vocab() on each step
        with dummy data (no real data needed for fixed-vocab tokenizers)."""
        pipeline = self._pipeline

        current_offset = pipeline.num_special_tokens
        for tok_id in pipeline.tokenizer_order:
            tokenizer = pipeline.steps[tok_id]
            tokenizer.build_vocab()
            vs = tokenizer.vocab_size
            pipeline.vocab_sizes[tok_id] = vs

            if tokenizer._idx_to_token and isinstance(
                next(iter(tokenizer._idx_to_token.values())), dict
            ):
                for sub in tokenizer._idx_to_token:
                    sub_size = len(tokenizer._idx_to_token[sub])
                    pipeline.vocab_offset[f"{tok_id}.{sub}"] = current_offset
                    current_offset += sub_size
            else:
                pipeline.vocab_offset[tok_id] = current_offset
                current_offset += vs

        pipeline.global_vocab_size = current_offset
        pipeline.is_fitted = True
        pipeline._build_global_vocab()

        self.vocab = dict(pipeline.vocab)
        self.id_to_token = dict(pipeline.id_to_token)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab_size(self) -> int:
        return self.vocab_size

    def tokenize(self, text: str) -> List[str]:
        return text.split()

    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        tokens = self.tokenize(text)
        if max_length:
            tokens = tokens[:max_length]
            while len(tokens) < max_length:
                tokens.append("<pad>")
        unk = self.unk_token_id
        return [self.vocab.get(t, unk) for t in tokens]

    def decode(self, token_ids: List[int]) -> str:
        tokens = []
        for tid in token_ids:
            tok = self.id_to_token.get(tid)
            if tok and tok != "<pad>":
                tokens.append(tok)
        return " ".join(tokens)

