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
Abstract base class for modular tokenizer steps.

Each tokenizer handles one or more DataFrame columns, converting raw values
into token strings (e.g. "AMT_3", "MERCH_1181").  Configuration lives in
__init__; data flows only through build_vocab() and tokenize().
"""

from abc import ABC, abstractmethod
import cudf


class BaseTokenizer(ABC):
    """
    Abstract base for all tokenizer steps in the pipeline.

    Subclasses must implement:
        build_vocab()  — create the idx-to-token mapping
        tokenize()     — map column values to token strings
    """

    def __init__(self):
        self._vocab: dict = None
        self._idx_to_token = None

    # ------------------------------------------------------------------
    # Vocabulary properties
    # ------------------------------------------------------------------

    @property
    def vocab(self) -> dict:
        """token_string -> local_index mapping (built lazily from idx_to_token)."""
        if self._vocab is None and self._idx_to_token is not None:
            if isinstance(next(iter(self._idx_to_token.values())), dict):
                self._vocab = {
                    outer_key: {v: k for k, v in inner_dict.items()}
                    for outer_key, inner_dict in self._idx_to_token.items()
                }
            else:
                self._vocab = dict(
                    zip(self._idx_to_token.values(), self._idx_to_token.keys())
                )
        return self._vocab if self._vocab is not None else {}

    @vocab.setter
    def vocab(self, value: dict) -> None:
        self._vocab = value

    @property
    def vocab_size(self) -> int:
        if self._idx_to_token is None:
            return 0
        if isinstance(next(iter(self._idx_to_token.values()), None), dict):
            return sum(len(v) for v in self._idx_to_token.values())
        return len(self._idx_to_token)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def build_vocab(self) -> None:
        """Build the idx_to_token mapping from configuration (no data needed
        for fixed-vocab tokenizers) or from data passed to fit()."""

    @abstractmethod
    def tokenize(self, column_data) -> cudf.Series:
        """Map column values to token strings."""

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @classmethod
    def from_state(cls, state: dict) -> "BaseTokenizer":
        instance = cls(**state["init_params"])
        instance._set_vocab_state(state["vocab_state"])
        instance._set_fitted_state(state["fitted_state"])
        return instance

    def get_state(self) -> dict:
        return {
            "class": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "init_params": self._get_init_params(),
            "vocab_state": self._get_vocab_state(),
            "fitted_state": self._get_fitted_state(),
        }

    def _get_init_params(self) -> dict:
        return {}

    def _get_vocab_state(self) -> dict:
        if self._idx_to_token is None:
            return {"_idx_to_token": None}
        if isinstance(self._idx_to_token, dict):
            converted = {
                int(k) if hasattr(k, "item") else k: v
                for k, v in self._idx_to_token.items()
            }
            return {"_idx_to_token": converted}
        return {"_idx_to_token": self._idx_to_token}

    def _get_fitted_state(self) -> dict:
        return {}

    def _set_vocab_state(self, state: dict) -> None:
        self._idx_to_token = state.get("_idx_to_token")

    def _set_fitted_state(self, state: dict) -> None:
        pass
