"""Loom tokenizer engine — the contract compiler (DESIGN.md item #1).

The public API lives in :mod:`loom.engine.api`. Importing the engine package
re-exports the locked names so callers can ``from loom.engine import compile_spec,
financial_spec, TokenizerSpec``.
"""

from __future__ import annotations

from .api import (  # noqa: F401
    AmountStrategy,
    ChunkGrammar,
    CompiledTokenizer,
    ContractReport,
    FieldStep,
    FixedVocab,
    Hash,
    MappingDirect,
    MappingPassthrough,
    MappingRange,
    SPECIAL_TOKENS,
    Strategy,
    TimeDelta,
    TokenizerSpec,
    chain_spec,
    compile_spec,
    financial_spec,
)
from .contracts import to_corpus_lines  # noqa: F401
from .spec import (  # noqa: F401
    corpus_lines,
    materialize_corpus_lines,
    spec_from_field_map,
)
from .streaming import (  # noqa: F401
    CHUNK_ROWS,
    ColStat,
    StreamingStats,
    stream_stats,
)

__all__ = [
    "AmountStrategy",
    "ChunkGrammar",
    "CompiledTokenizer",
    "ContractReport",
    "FieldStep",
    "FixedVocab",
    "Hash",
    "MappingDirect",
    "MappingPassthrough",
    "MappingRange",
    "SPECIAL_TOKENS",
    "Strategy",
    "TimeDelta",
    "TokenizerSpec",
    "chain_spec",
    "compile_spec",
    "financial_spec",
    "to_corpus_lines",
    "corpus_lines",
    "materialize_corpus_lines",
    "spec_from_field_map",
    "CHUNK_ROWS",
    "ColStat",
    "StreamingStats",
    "stream_stats",
]
