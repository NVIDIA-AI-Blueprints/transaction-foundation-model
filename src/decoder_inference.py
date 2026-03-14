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
Decoder-Only Model Inference Wrapper

Embedding extraction from HuggingFace causal LM models (GPT-2, Llama, Mistral, etc.).

Embedding strategies:
- Last-token pooling: Use the hidden state at the last non-padding position.
  Standard approach for decoder-only embedding models (NV-Embed-v2, E5-Mistral).
- Mean pooling: Average hidden states over all non-padding positions.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import torch
import numpy as np

logger = logging.getLogger(__name__)


class HuggingFaceDecoderInference:
    """
    Fast inference wrapper for any HuggingFace causal LM (GPT-2, Llama, etc.).

    Extracts embeddings from the last hidden layer using last-token
    or mean pooling.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        tokenizer,
        device: Optional[str] = None,
        pooling: str = "last_token",
        use_flash_attention: bool = False,
    ):
        """
        Args:
            model_path: Path to HuggingFace model directory or checkpoint.
            tokenizer: Tokenizer instance with pad_token_id attribute.
            device: Device to load model on (defaults to cuda if available).
            pooling: Embedding pooling strategy: "last_token" or "mean".
            use_flash_attention: Enable Flash Attention 2 if available.
        """
        from transformers import AutoModelForCausalLM

        self.model_path = Path(model_path)
        self.tokenizer = tokenizer
        self.pooling = pooling

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Initializing HuggingFaceDecoderInference on {self.device}")

        load_kwargs = {}
        if use_flash_attention:
            load_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Using Flash Attention 2")

        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path), **load_kwargs
        )
        self.model.to(self.device)
        self.model.eval()

        self._embedding_dim = self.model.config.hidden_size
        logger.info(
            f"Model loaded: {self.model.config.num_hidden_layers} layers, "
            f"{self._embedding_dim} hidden dim, pooling={pooling}"
        )

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def _get_hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run forward pass and return last-layer hidden states."""
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        hidden_states = outputs.hidden_states[-1]
        return hidden_states, attention_mask

    def _pool_embeddings(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply pooling strategy to extract fixed-size embeddings."""
        if self.pooling == "last_token":
            seq_lengths = attention_mask.sum(dim=1) - 1
            seq_lengths = seq_lengths.clamp(min=0)
            batch_indices = torch.arange(
                hidden_states.size(0), device=hidden_states.device
            )
            embeddings = hidden_states[batch_indices, seq_lengths, :]

        elif self.pooling == "mean":
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
            count = mask_expanded.sum(dim=1).clamp(min=1)
            embeddings = sum_hidden / count

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

        return embeddings

    @torch.no_grad()
    def extract_embeddings(
        self,
        input_ids: torch.Tensor,
        return_numpy: bool = True,
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Extract embeddings from input sequences.

        Args:
            input_ids: Token IDs tensor of shape (batch, seq_len).
            return_numpy: If True, return numpy array; else return tensor.

        Returns:
            Embeddings of shape (batch, embedding_dim).
        """
        input_ids = input_ids.to(self.device)
        hidden_states, attention_mask = self._get_hidden_states(input_ids)
        embeddings = self._pool_embeddings(hidden_states, attention_mask)

        if return_numpy:
            return embeddings.float().cpu().numpy()
        return embeddings

    @torch.no_grad()
    def extract_embeddings_batched(
        self,
        padded_ids: np.ndarray,
        batch_size: int = 4096,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Extract embeddings for a large dataset in batches.

        Optimized for throughput with:
        - Pinned memory for async transfers
        - GPU accumulation before CPU transfer

        Args:
            padded_ids: Pre-padded token IDs array of shape (n_samples, seq_len).
            batch_size: Batch size for inference.
            show_progress: Whether to show progress bar.

        Returns:
            Embeddings array of shape (n_samples, embedding_dim).
        """
        from tqdm import tqdm

        n_samples = len(padded_ids)
        embed_dim = self.embedding_dim

        input_tensor = torch.from_numpy(padded_ids).pin_memory()

        gpu_embeddings = torch.empty(
            (n_samples, embed_dim),
            dtype=torch.float32,
            device=self.device,
        )

        n_batches = (n_samples + batch_size - 1) // batch_size
        iterator = range(0, n_samples, batch_size)

        if show_progress:
            iterator = tqdm(
                iterator,
                desc=f"Extracting embeddings ({self.pooling})",
                total=n_batches,
            )

        for i in iterator:
            batch_end = min(i + batch_size, n_samples)
            batch_ids = input_tensor[i:batch_end].to(self.device, non_blocking=True)

            hidden_states, attention_mask = self._get_hidden_states(batch_ids)
            batch_embeddings = self._pool_embeddings(hidden_states, attention_mask)

            gpu_embeddings[i:batch_end] = batch_embeddings.float()

        torch.cuda.synchronize()
        embeddings = gpu_embeddings.cpu().numpy()

        del gpu_embeddings
        torch.cuda.empty_cache()

        return embeddings
