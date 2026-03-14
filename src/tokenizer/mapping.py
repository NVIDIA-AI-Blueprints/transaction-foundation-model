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
Mapping-based categorical tokenizer.

Maps discrete values to token strings via a predefined dictionary.
Supports three modes to cover different field types:

  * **direct** — 1:1 value→label dict  (STATE, CHIP)
  * **passthrough** — value is its own label, prefix is added  (MCC)
  * **range** — integer ranges map to labels  (industry CAT from MCC)

Data is never stored in the constructor.
"""

from typing import Dict, List, Optional, Tuple

import cudf
import cupy as cp
import numpy as np

from .base import BaseTokenizer


class MappingTokenizer(BaseTokenizer):
    """Categorical tokenizer driven by a static mapping."""

    def __init__(
        self,
        prefix: str,
        mapping: Optional[Dict] = None,
        values: Optional[List] = None,
        ranges: Optional[List[Tuple[int, int, str]]] = None,
        default: str = "UNK",
        stream: cp.cuda.Stream = None,
    ):
        """
        Parameters
        ----------
        prefix : str
            Token prefix, e.g. "CHIP", "STATE", "MCC", "CAT".
        mapping : dict, optional
            Explicit value→label mapping, e.g. {"Swipe Transaction": "SWIPE"}.
        values : list, optional
            If provided (and mapping is None), each value maps to itself
            (passthrough mode): {"CA": "CA", "NY": "NY", ...}.
        ranges : list of (lo, hi, label), optional
            Integer-range mode for industry categories:
            [(0, 1499, "AGRICULTURAL"), (1500, 2999, "CONTRACTED"), ...].
        default : str
            Fallback label when a value is not found in the mapping.
        """
        super().__init__()
        self.prefix = prefix
        self.default = default
        self.stream = stream
        self._vocab_built = False

        self._mapping_cfg = mapping
        self._values_cfg = values
        self._ranges_cfg = ranges

        if ranges is not None:
            self._mode = "range"
            self._range_lookup = ranges
            self._direct_map = None
        elif mapping is not None:
            self._mode = "direct"
            self._direct_map = mapping
            self._range_lookup = None
        elif values is not None:
            self._mode = "passthrough"
            self._direct_map = {v: str(v) for v in values}
            self._range_lookup = None
        else:
            self._mode = "passthrough"
            self._direct_map = {}
            self._range_lookup = None

    def build_vocab(self, column_data=None) -> None:
        if self._mode == "range":
            labels = sorted({lbl for _, _, lbl in self._range_lookup})
            labels.append(self.default)
            self._idx_to_token = {
                i: f"{self.prefix}_{lbl}" for i, lbl in enumerate(labels)
            }
        elif self._direct_map:
            labels = sorted(set(self._direct_map.values()))
            labels.append(self.default)
            labels = list(dict.fromkeys(labels))  # dedupe preserving order
            self._idx_to_token = {
                i: f"{self.prefix}_{lbl}" for i, lbl in enumerate(labels)
            }
        elif self._mode == "passthrough" and column_data is not None:
            if hasattr(column_data, "to_pandas"):
                unique_vals = sorted(column_data.to_pandas().dropna().unique(), key=str)
            else:
                unique_vals = sorted(column_data.dropna().unique(), key=str)
            labels = [str(v) for v in unique_vals]
            labels.append(self.default)
            labels = list(dict.fromkeys(labels))
            self._direct_map = {v: v for v in labels}
            self._idx_to_token = {
                i: f"{self.prefix}_{lbl}" for i, lbl in enumerate(labels)
            }
        else:
            self._idx_to_token = {0: f"{self.prefix}_{self.default}"}

        self._vocab_built = True

    def tokenize(self, column_data) -> cudf.Series:
        if self._mode == "range":
            return self._tokenize_range(column_data)
        return self._tokenize_direct(column_data)

    def _tokenize_direct(self, column_data) -> cudf.Series:
        """Map values through the direct dict, prepend prefix."""
        s = column_data.astype(str) if not hasattr(column_data, 'str') or column_data.dtype != 'object' else column_data
        host = s.to_pandas()
        if self._direct_map:
            mapped = host.map(self._direct_map).fillna(self.default)
        else:
            mapped = host.fillna(self.default)
        result = self.prefix + "_" + mapped.astype(str)
        return cudf.Series(result.values, index=column_data.index)

    def _tokenize_range(self, column_data) -> cudf.Series:
        """Map integer values via range lookup (vectorized with numpy)."""
        vals = column_data.values.get() if hasattr(column_data.values, 'get') else column_data.values
        vals = np.asarray(vals, dtype=np.int64)

        result = np.full(len(vals), f"{self.prefix}_{self.default}", dtype=object)
        for lo, hi, label in self._range_lookup:
            mask = (vals >= lo) & (vals <= hi)
            result[mask] = f"{self.prefix}_{label}"
        return cudf.Series(result, index=column_data.index)

    def __repr__(self) -> str:
        status = "built" if self._vocab_built else "not built"
        return (
            f"MappingTokenizer(prefix={self.prefix}, "
            f"mode={self._mode}, {status})"
        )

    # -- serialization -----------------------------------------------------

    def _get_init_params(self) -> dict:
        return {
            "prefix": self.prefix,
            "mapping": self._mapping_cfg,
            "values": self._values_cfg,
            "ranges": self._ranges_cfg,
            "default": self.default,
            "stream": None,
        }

    def _get_fitted_state(self) -> dict:
        return {"_vocab_built": self._vocab_built}

    def _set_fitted_state(self, state: dict) -> None:
        self._vocab_built = state.get("_vocab_built", False)
