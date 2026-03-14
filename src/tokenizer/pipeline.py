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
Modular tokenizer pipeline for causal language modeling.

Orchestrates multiple BaseTokenizer steps, manages a global vocabulary with
per-step offsets, and provides corpus generation / encoding utilities.

Special tokens: <bos> / <eos> / <sep> / <pad> / <unk>
Default chunk_size: ~315 transactions for a 4096-token context window.
Sequence format: "<bos> txn1 <sep> txn2 <sep> ... txnN <eos>"
"""

from typing import Dict, List, Optional, Union

import cudf
import cupy as cp
import numpy as np

from .base import BaseTokenizer


class TokenizerPipeline:
    """Manages an ordered sequence of BaseTokenizer steps with global vocab offsets."""

    SPECIAL_TOKENS = {
        "PAD": "<pad>",
        "BOS": "<bos>",
        "EOS": "<eos>",
        "SEP": "<sep>",
        "UNK": "<unk>",
    }

    def __init__(
        self,
        special_tokens: Optional[Dict[str, str]] = None,
        use_streams: bool = True,
        stream_threshold: int = 5,
    ):
        self.steps: Dict[str, BaseTokenizer] = {}
        self.vocab_sizes: Dict[str, int] = {}
        self.vocab_offset: Dict[str, int] = {}
        self.column_specs: Dict[str, List[str]] = {}
        self.tokenizer_order: List[str] = []
        self.stream_threshold = stream_threshold

        self.special_tokens = special_tokens or dict(self.SPECIAL_TOKENS)
        self.special_token_ids = {
            tok: idx for idx, tok in enumerate(self.special_tokens.values())
        }
        self.num_special_tokens = len(self.special_tokens)
        self.global_vocab_size = self.num_special_tokens

        self.use_streams = use_streams
        self.is_fitted = False

        self._vocab: Optional[Dict[str, int]] = None
        self._id_to_token: Optional[Dict[int, str]] = None

    # ------------------------------------------------------------------
    # Pipeline construction
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_key(col):
        return list(col) if isinstance(col, list) else [col]

    def add_step(
        self, column_name: Union[list, str], tokenizer: BaseTokenizer
    ) -> "TokenizerPipeline":
        col = self._normalize_key(column_name)
        tok_id = "_".join(col)
        self.steps[tok_id] = tokenizer
        self.column_specs[tok_id] = col
        self.tokenizer_order.append(tok_id)
        return self

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, df: cudf.DataFrame) -> "TokenizerPipeline":
        if self.use_streams and len(self.steps) >= self.stream_threshold:
            self._fit_parallel(df)
        else:
            self._fit_sequential(df)
        self._build_global_vocab()
        self.is_fitted = True
        return self

    def _fit_sequential(self, df: cudf.DataFrame) -> None:
        current_offset = self.num_special_tokens
        for tok_id in self.tokenizer_order:
            columns = self.column_specs[tok_id]
            missing = set(columns) - set(df.columns)
            if missing:
                raise ValueError(f"Columns {missing} not found in DataFrame")

            tokenizer = self.steps[tok_id]
            col_data = df[columns] if len(columns) > 1 else df[columns[0]]
            tokenizer.build_vocab(col_data)

            vs = tokenizer.vocab_size
            self.vocab_sizes[tok_id] = vs

            if tokenizer._idx_to_token and isinstance(
                next(iter(tokenizer._idx_to_token.values())), dict
            ):
                for sub in tokenizer._idx_to_token:
                    sub_size = len(tokenizer._idx_to_token[sub])
                    self.vocab_offset[f"{tok_id}.{sub}"] = current_offset
                    current_offset += sub_size
            else:
                self.vocab_offset[tok_id] = current_offset
                current_offset += vs

        self.global_vocab_size = current_offset

    def _fit_parallel(self, df: cudf.DataFrame) -> None:
        streams = [
            cp.cuda.Stream(non_blocking=True) for _ in self.tokenizer_order
        ]
        for stream, tok_id in zip(streams, self.tokenizer_order):
            columns = self.column_specs[tok_id]
            missing = set(columns) - set(df.columns)
            if missing:
                raise ValueError(f"Columns {missing} not found in DataFrame")

            tokenizer = self.steps[tok_id]
            col_data = df[columns] if len(columns) > 1 else df[columns[0]]
            original_stream = getattr(tokenizer, "stream", None)
            if hasattr(tokenizer, "stream"):
                tokenizer.stream = stream
            with stream:
                tokenizer.build_vocab(col_data)
            if hasattr(tokenizer, "stream"):
                tokenizer.stream = original_stream

        cp.cuda.Stream.null.synchronize()

        current_offset = self.num_special_tokens
        for tok_id in self.tokenizer_order:
            tokenizer = self.steps[tok_id]
            vs = tokenizer.vocab_size
            self.vocab_sizes[tok_id] = vs

            if tokenizer._idx_to_token and isinstance(
                next(iter(tokenizer._idx_to_token.values())), dict
            ):
                for sub in tokenizer._idx_to_token:
                    sub_size = len(tokenizer._idx_to_token[sub])
                    self.vocab_offset[f"{tok_id}.{sub}"] = current_offset
                    current_offset += sub_size
            else:
                self.vocab_offset[tok_id] = current_offset
                current_offset += vs

        self.global_vocab_size = current_offset

    # ------------------------------------------------------------------
    # Global vocab helpers
    # ------------------------------------------------------------------

    def _build_global_vocab(self) -> None:
        vocab: Dict[str, int] = {}
        id_to_token: Dict[int, str] = {}

        for tok_str, idx in self.special_token_ids.items():
            vocab[tok_str] = idx
            id_to_token[idx] = tok_str

        for tok_id in self.tokenizer_order:
            tokenizer = self.steps[tok_id]
            if tokenizer._idx_to_token is None:
                continue

            if isinstance(next(iter(tokenizer._idx_to_token.values()), None), dict):
                for sub_key, sub_dict in tokenizer._idx_to_token.items():
                    offset = self.vocab_offset.get(f"{tok_id}.{sub_key}", 0)
                    for local_idx, token_str in sub_dict.items():
                        gid = int(local_idx) + offset
                        vocab[token_str] = gid
                        id_to_token[gid] = token_str
            else:
                offset = self.vocab_offset.get(tok_id, 0)
                for local_idx, token_str in tokenizer._idx_to_token.items():
                    gid = int(local_idx) + offset
                    vocab[token_str] = gid
                    id_to_token[gid] = token_str

        self._vocab = vocab
        self._id_to_token = id_to_token

    @property
    def vocab(self) -> Dict[str, int]:
        if self._vocab is None:
            return {}
        return self._vocab

    @property
    def id_to_token(self) -> Dict[int, str]:
        if self._id_to_token is None:
            return {}
        return self._id_to_token

    # ------------------------------------------------------------------
    # Transform  ->  DataFrame of token strings
    # ------------------------------------------------------------------

    def transform(self, df: cudf.DataFrame) -> cudf.DataFrame:
        if not self.is_fitted:
            raise ValueError("Must call fit() before transform()")

        local_results = {}
        if self.use_streams and len(self.steps) >= self.stream_threshold:
            streams = [
                cp.cuda.Stream(non_blocking=True) for _ in self.tokenizer_order
            ]
            for stream, tok_id in zip(streams, self.tokenizer_order):
                columns = self.column_specs[tok_id]
                col_data = df[columns] if len(columns) > 1 else df[columns[0]]
                tokenizer = self.steps[tok_id]
                if hasattr(tokenizer, "stream"):
                    original = tokenizer.stream
                    tokenizer.stream = stream
                with stream:
                    local_results[tok_id] = tokenizer.tokenize(col_data)
                if hasattr(tokenizer, "stream"):
                    tokenizer.stream = original
            cp.cuda.Stream.null.synchronize()
        else:
            for tok_id in self.tokenizer_order:
                columns = self.column_specs[tok_id]
                col_data = df[columns] if len(columns) > 1 else df[columns[0]]
                local_results[tok_id] = self.steps[tok_id].tokenize(col_data)

        parts = []
        for key, val in local_results.items():
            if isinstance(val, cudf.Series):
                parts.append(val.to_frame(name=key))
            elif isinstance(val, cudf.DataFrame):
                parts.append(val)
            else:
                parts.append(cudf.DataFrame({key: val}))

        return cudf.concat(parts, axis=1)

    def fit_transform(self, df: cudf.DataFrame) -> cudf.DataFrame:
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------
    # Encode  ->  padded integer-ID numpy array (causal LM format)
    # ------------------------------------------------------------------

    def encode(
        self,
        token_df: cudf.DataFrame,
        max_length: int = 4096,
        add_special: bool = True,
    ) -> np.ndarray:
        """Convert a token-string DataFrame to a padded int64 array.

        Each row becomes: <bos> col1 col2 ... colN <eos> <pad> ...
        Returns shape (n_rows, max_length).
        """
        vocab = self._vocab
        pad_id = self.special_token_ids["<pad>"]
        bos_id = self.special_token_ids["<bos>"]
        eos_id = self.special_token_ids["<eos>"]
        unk_id = self.special_token_ids["<unk>"]

        n = len(token_df)
        padded = np.full((n, max_length), pad_id, dtype=np.int64)

        col_offset = 0
        if add_special:
            padded[:, 0] = bos_id
            col_offset = 1

        for col_name in token_df.columns:
            if col_offset >= max_length:
                break
            host_col = token_df[col_name].to_pandas()
            ids = host_col.map(vocab).fillna(unk_id).astype(np.int64).values
            padded[:, col_offset] = ids
            col_offset += 1

        if add_special and col_offset < max_length:
            padded[:, col_offset] = eos_id

        return padded

    # ------------------------------------------------------------------
    # Corpus generation  ->  list of "<bos> t1 <sep> t2 ..." lines
    # ------------------------------------------------------------------

    def to_corpus_lines(
        self,
        token_df: cudf.DataFrame,
        df_meta: cudf.DataFrame,
        group_cols: List[str],
        chunk_size: int = 315,
    ) -> List[str]:
        """Assemble token strings into text corpus lines for causal LM training.

        Parameters
        ----------
        token_df : cudf.DataFrame
            Output of transform() -- one column per token field.
        df_meta : cudf.DataFrame
            Original (preprocessed) DataFrame with *group_cols* for grouping.
        group_cols : list of str
            Columns to group transactions by (e.g. ["user", "card"]).
        chunk_size : int
            Max transactions per sequence (315 fits ~4096 tokens with
            12 tokens/txn + separators).
        """
        token_cols = list(token_df.columns)
        txn_text = token_df[token_cols[0]].str.cat(
            [token_df[c] for c in token_cols[1:]], sep=" "
        )

        work = df_meta[group_cols].copy()
        work["_txn_text"] = txn_text

        work["_seq_id"] = work.groupby(group_cols).cumcount()
        work["_chunk_id"] = (work["_seq_id"] // chunk_size).astype("int32")

        grouped = work.groupby(group_cols + ["_chunk_id"])["_txn_text"].agg(list)
        if hasattr(grouped, "to_pandas"):
            grouped = grouped.to_pandas()

        def _fmt(txn_list):
            return "<bos> " + " <sep> ".join(txn_list) + " <eos>"

        return grouped.map(_fmt).tolist()

