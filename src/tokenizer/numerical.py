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
Data-driven numerical binning tokenizer (GPU-accelerated).

Uses cuML KBinsDiscretizer to learn bin boundaries from data, then maps
continuous values to ordinal bin tokens like "AMT_0" .. "AMT_9".

Data is passed only to build_vocab() (fit) and tokenize() (transform),
never stored in the constructor.
"""

import cudf
import cupy as cp
from cuml.preprocessing import KBinsDiscretizer

from .base import BaseTokenizer


class NumericalTokenizerOptBin(BaseTokenizer):
    """Quantile / uniform / k-means binning for continuous columns."""

    def __init__(
        self,
        special_token: str = "AMT",
        num_bins: int = 10,
        strategy: str = "quantile",
        stream: cp.cuda.Stream = None,
    ):
        super().__init__()
        self.special_token = special_token
        self.num_bins = num_bins
        self.strategy = strategy
        self.stream = stream
        self._vocab_built = False
        self.builder = KBinsDiscretizer(
            n_bins=self.num_bins, encode="ordinal", strategy=self.strategy
        )

    def build_vocab(self, column_data=None) -> None:
        """Build vocab and fit the discretizer on *column_data*."""
        self._idx_to_token = {
            i: f"{self.special_token}_{i}" for i in range(self.num_bins)
        }
        if column_data is not None:
            if isinstance(column_data, cudf.Series):
                column_data = column_data.to_frame()
            if self.stream:
                with self.stream:
                    self.builder.fit(column_data)
            else:
                self.builder.fit(column_data)
        self._vocab_built = True

    def tokenize(self, column_data) -> cudf.Series:
        if isinstance(column_data, cudf.Series):
            column_data = cudf.DataFrame(column_data)
        if self.stream:
            with self.stream:
                bins = self.builder.transform(column_data)
        else:
            bins = self.builder.transform(column_data)
        if isinstance(bins, cudf.DataFrame):
            bins = bins.iloc[:, 0]
        return bins.astype("int32").map(self._idx_to_token)

    def __repr__(self) -> str:
        status = "built" if self._vocab_built else "not built"
        return (
            f"NumericalTokenizerOptBin(token={self.special_token}, "
            f"bins={self.num_bins}, strategy={self.strategy}, {status})"
        )

    # -- serialization -----------------------------------------------------

    def _get_init_params(self) -> dict:
        return {
            "special_token": self.special_token,
            "num_bins": self.num_bins,
            "strategy": self.strategy,
            "stream": None,
        }

    def _get_fitted_state(self) -> dict:
        return {
            "builder": self.builder if self._vocab_built else None,
            "_vocab_built": self._vocab_built,
        }

    def _set_fitted_state(self, state: dict) -> None:
        if state.get("builder") is not None:
            self.builder = state["builder"]
        self._vocab_built = state.get("_vocab_built", False)
