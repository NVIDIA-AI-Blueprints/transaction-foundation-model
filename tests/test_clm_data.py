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

"""CPU-only regression tests for the label construction in ``src/clm_data.py``.

NeMo AutoModel's ``MaskedCrossEntropy`` scores ``logits[i]`` against
``labels[i]`` without shifting, so the dataset must emit
``labels[i] = input_ids[i + 1]``. ``<pad>`` must never be a label target.
"""

import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.clm_data import FinancialCLMDataset, load_corpus_and_tokenize  # noqa: E402

PAD, BOS, EOS = 0, 1, 2


def test_labels_are_next_token_not_copy_of_input():
    tokens = [BOS, 10, 11, 12, EOS]
    ex = FinancialCLMDataset([tokens], seq_length=5, pad_token_id=PAD)[0]
    assert ex["input_ids"].tolist() == tokens
    # labels[i] == input_ids[i + 1]; last position has no target.
    assert ex["labels"].tolist() == [10, 11, 12, EOS, -100]
    assert not torch.equal(ex["labels"], ex["input_ids"])


def test_padded_input_does_not_score_pad():
    # Same shape as load_corpus_and_tokenize output: encode() already padded.
    tokens = [BOS, 10, 11, 12, EOS] + [PAD] * 11
    ex = FinancialCLMDataset([tokens], seq_length=16, pad_token_id=PAD)[0]
    assert ex["labels"].tolist() == [10, 11, 12, EOS] + [-100] * 12
    assert (ex["labels"] == PAD).sum() == 0
    assert (ex["labels"] != -100).sum() == 4  # == number of real next-token targets


def test_unpadded_short_sequence_is_padded_by_getitem():
    ex = FinancialCLMDataset([[BOS, 10, EOS]], seq_length=6, pad_token_id=PAD)[0]
    assert ex["input_ids"].tolist() == [BOS, 10, EOS, PAD, PAD, PAD]
    assert ex["labels"].tolist() == [10, EOS, -100, -100, -100, -100]


def test_full_length_sequence_keeps_all_but_last_target():
    tokens = [BOS, 10, 11, 12, 13, EOS]
    ex = FinancialCLMDataset([tokens], seq_length=6, pad_token_id=PAD)[0]
    assert ex["labels"].tolist() == [10, 11, 12, 13, EOS, -100]


def test_overlong_sequence_is_truncated_then_shifted():
    tokens = [BOS, 10, 11, 12, 13, 14, EOS]
    ex = FinancialCLMDataset([tokens], seq_length=4, pad_token_id=PAD)[0]
    assert ex["input_ids"].tolist() == [BOS, 10, 11, 12]
    assert ex["labels"].tolist() == [10, 11, 12, -100]


@pytest.mark.parametrize("tokens", [[], [BOS]])
def test_degenerate_sequences_have_no_targets(tokens):
    ex = FinancialCLMDataset([tokens], seq_length=4, pad_token_id=PAD)[0]
    assert (ex["labels"] == -100).all()


def test_unshifted_loss_is_what_nemo_would_score():
    """MaskedCrossEntropy-style scoring (no internal shift) rewards next-token logits.

    Logits that put all mass on input_ids[i + 1] must give ~0 loss with the
    dataset's labels; logits that copy input_ids[i] must not.
    """
    vocab = 16
    tokens = [BOS, 10, 11, 12, EOS] + [PAD] * 3
    ex = FinancialCLMDataset([tokens], seq_length=8, pad_token_id=PAD)[0]
    ids, labels = ex["input_ids"], ex["labels"]

    next_logits = torch.full((8, vocab), -50.0)
    next_logits[torch.arange(7), ids[1:]] = 50.0  # predicts the next token
    copy_logits = torch.full((8, vocab), -50.0)
    copy_logits[torch.arange(8), ids] = 50.0  # copies the current token

    # reduction="sum" and ignore_index=-100, as in MaskedCrossEntropy.
    n = (labels != -100).sum()
    next_loss = F.cross_entropy(next_logits, labels, reduction="sum") / n
    copy_loss = F.cross_entropy(copy_logits, labels, reduction="sum") / n
    assert next_loss < 1e-3
    assert copy_loss > 10


def test_load_corpus_and_tokenize_does_not_label_padding(tmp_path, monkeypatch):
    """End to end through load_corpus_and_tokenize with a stand-in tokenizer.

    The real tokenizer needs cuDF; the stand-in reproduces the one behaviour that
    matters here: ``encode(text, max_length=n)`` right-pads with <pad> to n.
    """
    vocab = {"<pad>": PAD, "<bos>": BOS, "<eos>": EOS, "A": 10, "B": 11, "C": 12}

    class FakeTokenizer:
        pad_token_id = PAD

        def __init__(self, **kwargs):
            pass

        def encode(self, text, max_length=None):
            toks = text.split()[:max_length]
            toks += ["<pad>"] * (max_length - len(toks))
            return [vocab[t] for t in toks]

    mod = types.ModuleType("tokenizer.financial_tokenizer")
    mod.FinancialTabularTokenizer = FakeTokenizer
    monkeypatch.setitem(sys.modules, "tokenizer", types.ModuleType("tokenizer"))
    monkeypatch.setitem(sys.modules, "tokenizer.financial_tokenizer", mod)

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("<bos> A B C <eos>\n")
    ds = load_corpus_and_tokenize(corpus, seq_length=16)
    assert len(ds.sequences[0]) == 16  # pre-padded, as in the real pipeline

    ex = ds[0]
    assert ex["labels"].tolist() == [10, 11, 12, EOS] + [-100] * 12
    assert (ex["labels"] == PAD).sum() == 0
