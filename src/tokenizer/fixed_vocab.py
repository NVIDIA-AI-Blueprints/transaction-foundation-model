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
Fixed-vocabulary tokenizer for bounded integer fields.

Maps integer values from a known range to token strings like "HOUR_09",
"ZIP3_100", "CUST_42".  No data-driven fitting — the vocabulary is fully
determined by (prefix, min_val, max_val, pad_width).

Used for: HOUR (0-23), DOW (0-6), MONTH (1-12), CARD (0-9),
          ZIP3 (0-999), CUST (0-2999).
"""

import cudf
import cupy as cp

from .base import BaseTokenizer


class FixedVocabTokenizer(BaseTokenizer):
    """Token = ``{prefix}_{value:0{pad_width}d}`` for every integer in range."""

    def __init__(
        self,
        prefix: str,
        min_val: int = 0,
        max_val: int = 23,
        pad_width: int = 0,
        stream: cp.cuda.Stream = None,
    ):
        super().__init__()
        self.prefix = prefix
        self.min_val = min_val
        self.max_val = max_val
        self.pad_width = pad_width
        self.stream = stream
        self._vocab_built = False

    def build_vocab(self, column_data=None) -> None:
        self._idx_to_token = {
            i: f"{self.prefix}_{i:0{self.pad_width}d}"
            for i in range(self.min_val, self.max_val + 1)
        }
        self._vocab_built = True

    def tokenize(self, column_data) -> cudf.Series:
        """*column_data* must be an integer cudf.Series within [min_val, max_val]."""
        int_vals = column_data.astype("int32").clip(self.min_val, self.max_val)
        return int_vals.map(self._idx_to_token)

    def __repr__(self) -> str:
        status = "built" if self._vocab_built else "not built"
        return (
            f"FixedVocabTokenizer(prefix={self.prefix}, "
            f"range=[{self.min_val},{self.max_val}], {status})"
        )

    # -- serialization -----------------------------------------------------

    def _get_init_params(self) -> dict:
        return {
            "prefix": self.prefix,
            "min_val": self.min_val,
            "max_val": self.max_val,
            "pad_width": self.pad_width,
            "stream": None,
        }

    def _get_fitted_state(self) -> dict:
        return {"_vocab_built": self._vocab_built}

    def _set_fitted_state(self, state: dict) -> None:
        self._vocab_built = state.get("_vocab_built", False)
