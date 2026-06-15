"""LOCKED public API of the Loom tokenizer engine (the contract compiler).

This module declares the precise, complete types the ``tokenize`` verb and the
contract checks (C1/C2/C3) are implemented against. Everything here is a typed
STUB raising ``NotImplementedError`` until the Implement agent fills it in — the
*signatures* are the contract; do not change them without re-coordinating.

Design grounding (DESIGN.md §0, §2.3, §7.2; the build brief's reference spec):

  * A :class:`TokenizerSpec` is an ordered list of :class:`FieldStep`, each
    carrying a :class:`Strategy` that determines how many contiguous ids that
    field contributes AFTER the 5 special tokens.
  * ``compile_spec`` derives the vocabulary by laying out each step's id block at
    a running offset using **0-based local indices** (``id = offset +
    local_index``) — THIS IS THE FIX for the reference bug where FixedVocab keyed
    ids by raw value (min_val=1 for MONTH ⇒ MONTH_12 collided with CARD_0). The
    compiler MUST assert injectivity (C1): blocks are disjoint, dense ``0..
    vocab_size-1``, and every ``(step, value)`` maps to a unique id.
  * It derives ``vocab_size``, ``vocab_hash``, ``tokens_per_txn`` and
    ``chunk_size = context_len // (tokens_per_txn + 1)`` (C3 grammar).
  * Determinism (C2): the vocab is built from config alone. A fitted strategy
    (``AmountStrategy.QUANTILE`` / ``KMEANS``) is a fitted-artifact and MUST be
    flagged (and its state persisted) — it is not allowed on the default path.

The ``financial`` preset MUST compile to ``vocab_size == 6251`` (or 6283 with
time-delta); the ``chain`` preset's vocab is DERIVED (do not hardcode).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union

from ..types import Diagnostic

# ---------------------------------------------------------------------------
# Special tokens — ids 0-4, in this exact order (build brief, verbatim).
# These occupy the first 5 ids; every field block is offset AFTER them.
# ---------------------------------------------------------------------------

SPECIAL_TOKENS: tuple[str, ...] = ("<pad>", "<bos>", "<eos>", "<sep>", "<unk>")


# ---------------------------------------------------------------------------
# Strategies — the discriminated union a FieldStep carries. Each strategy fully
# determines its contiguous id block (count + the ordered local token strings).
# Params mirror the reference tokenizers (src/tokenizer/*.py) but compile on CPU.
# ---------------------------------------------------------------------------


class AmountStrategy(str, Enum):
    """How the amount field is binned. ``FIXED`` is config-only (C2 deterministic,
    the default path). ``QUANTILE``/``KMEANS`` are FITTED artifacts — flagged by
    C2, state persisted into the Corpus (DESIGN.md §7.2 C2)."""

    FIXED = "fixed"
    QUANTILE = "quantile"
    KMEANS = "kmeans"


@dataclass(frozen=True)
class FixedVocab:
    """A bounded integer field ``prefix_{value:0{pad_width}d}`` for every value in
    ``[min_val, max_val]``. Contributes ``max_val - min_val + 1`` tokens with
    **0-based local indices** (the bug fix: ids do NOT key on raw value)."""

    prefix: str
    min_val: int
    max_val: int
    pad_width: int = 0

    def count(self) -> int:
        from . import strategies
        return strategies.count(self)

    def tokens(self) -> list[str]:
        """Ordered local token strings (index 0..count-1). E.g. MONTH (1..12,
        pad 2) → ['MONTH_01', ..., 'MONTH_12'] at local indices 0..11."""
        from . import strategies
        return strategies.build_vocab(self)


@dataclass(frozen=True)
class Hash:
    """A high-cardinality string field hashed into ``buckets`` buckets:
    ``prefix_0 .. prefix_{buckets-1}``. The CPU hash MUST be a stable/deterministic
    hash (e.g. hashlib-based), NOT Python's salted ``hash()`` (build brief)."""

    prefix: str
    buckets: int

    def count(self) -> int:
        from . import strategies
        return strategies.count(self)

    def tokens(self) -> list[str]:
        from . import strategies
        return strategies.build_vocab(self)


@dataclass(frozen=True)
class MappingRange:
    """Integer ranges → labels (e.g. industry CAT from MCC). Vocab = sorted unique
    labels, then ``default`` appended (deduped) — reference ``MappingTokenizer``
    range mode. Tokens are ``prefix_{label}``."""

    prefix: str
    ranges: tuple[tuple[int, int, str], ...]
    default: str = "GENERAL"

    def count(self) -> int:
        from . import strategies
        return strategies.count(self)

    def tokens(self) -> list[str]:
        from . import strategies
        return strategies.build_vocab(self)


@dataclass(frozen=True)
class MappingDirect:
    """A 1:1 value→label dict (e.g. CHIP). Vocab = sorted unique mapping VALUES,
    then ``default`` appended (deduped) — reference direct mode."""

    prefix: str
    mapping: dict[str, str]
    default: str = "UNK"

    def count(self) -> int:
        from . import strategies
        return strategies.count(self)

    def tokens(self) -> list[str]:
        from . import strategies
        return strategies.build_vocab(self)


@dataclass(frozen=True)
class MappingPassthrough:
    """Value is its own label (e.g. MCC, STATE). Vocab = sorted ``str(v)`` over
    ``values``, then ``default`` appended (deduped) — reference passthrough mode."""

    prefix: str
    values: tuple[str, ...]
    default: str

    def count(self) -> int:
        from . import strategies
        return strategies.count(self)

    def tokens(self) -> list[str]:
        from . import strategies
        return strategies.build_vocab(self)


@dataclass(frozen=True)
class TimeDelta:
    """Log-spaced time-delta bins (the T1 token family). Contributes ``num_bins``
    tokens ``special_token_0 .. special_token_{num_bins-1}``. Config-only ⇒ C2
    deterministic (DESIGN.md §3.1)."""

    special_token: str = "TDIF"
    num_bins: int = 32
    max_years: float = 10.0

    def count(self) -> int:
        from . import strategies
        return strategies.count(self)

    def tokens(self) -> list[str]:
        from . import strategies
        return strategies.build_vocab(self)


# The discriminated strategy union a FieldStep carries.
Strategy = Union[
    FixedVocab,
    Hash,
    MappingRange,
    MappingDirect,
    MappingPassthrough,
    TimeDelta,
]


# ---------------------------------------------------------------------------
# Field step and spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldStep:
    """One field in the tokenizer, contributing one contiguous id block.

    ``name`` is the logical step name (e.g. ``"amt"``, ``"merch"``, ``"month"``)
    used by ``--drop-step``/``--reorder-step`` (DESIGN.md §3.2, §7.2a). ``source``
    is the raw dataframe column the CPU preprocess reads. ``strategy`` determines
    the id block. ``fitted`` marks a C2 fitted-artifact step (amount quantile/
    kmeans)."""

    name: str
    source: str
    strategy: Strategy
    fitted: bool = False

    def count(self) -> int:
        """Number of contiguous ids this step contributes."""
        return self.strategy.count()


@dataclass(frozen=True)
class TokenizerSpec:
    """A declarative tokenizer spec — an ordered list of :class:`FieldStep`.

    ``preset`` records the originating factory (``"financial"`` / ``"chain"``).
    ``tokens_per_txn`` equals ``len(steps)`` (one field token per step per txn);
    it is derived, not stored redundantly. ``entity``/``event`` carry the grouping
    semantics (e.g. entity=wallet, event=trade) used for sequence assembly and the
    C6 sort key (DESIGN.md chain preset)."""

    steps: tuple[FieldStep, ...]
    preset: str = ""
    entity: Optional[str] = None
    event: Optional[str] = None
    amount_strategy: AmountStrategy = AmountStrategy.FIXED

    def tokens_per_txn(self) -> int:
        """Number of field tokens per transaction = number of steps."""
        return len(self.steps)

    def step_names(self) -> list[str]:
        return [s.name for s in self.steps]


# ---------------------------------------------------------------------------
# Compiled artifacts and the contract report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkGrammar:
    """The C3 corpus grammar, derived and announced (DESIGN.md §0 C3).

    A corpus line is ``<bos> txn (<sep> txn)* <eos>``; each txn is the
    space-joined field tokens (``tokens_per_txn`` of them). ``chunk_size`` is
    ``context_len // (tokens_per_txn + 1)`` — the ``+1`` accounts for the
    per-transaction ``<sep>``."""

    context_len: int
    tokens_per_txn: int
    chunk_size: int
    bos: str = "<bos>"
    eos: str = "<eos>"
    sep: str = "<sep>"


@dataclass
class ContractReport:
    """The compile-time contract verdict, surfaced as named diffs not stack traces.

    ``diagnostics`` carries C1/C2/C3 findings as :class:`~loom.types.Diagnostic`
    cards. ``passed`` is True only when C1 (injective + dense) and C3 (grammar)
    hold and no C2 error fired. On a C1 collision the compiler refuses to produce
    a writable ``CompiledTokenizer`` (DESIGN.md §7.2a)."""

    diagnostics: list[Diagnostic] = field(default_factory=list)
    passed: bool = True

    # Convenience flags the verb layer reads to shape the result envelope.
    injective: bool = True          # C1: blocks disjoint, no collisions
    dense: bool = True              # C1: ids are dense 0..vocab_size-1
    deterministic: bool = True      # C2: no un-persisted fitted artifact
    has_fitted_artifact: bool = False  # C2: quantile/kmeans amount strategy present

    def add(self, diag: Diagnostic) -> None:
        from ..types import Severity
        self.diagnostics.append(diag)
        if diag.severity is Severity.ERROR:
            self.passed = False


@dataclass(frozen=True)
class CompiledTokenizer:
    """The compiled vocabulary + derived grammar — the payload of a Corpus.

    ``vocab`` maps every token string → its unique contiguous id (specials at
    0-4, then each step's block at its running offset). ``id_to_token`` is the
    inverse. ``vocab_hash`` is a deterministic hash over the ordered vocab (the
    C1 signature; changes ⇒ retrain required). ``report`` carries the contract
    verdict. ``spec`` is the source spec. ``fitted_state`` persists any C2 fitted
    artifact (amount quantile/kmeans binner) so determinism is recoverable."""

    vocab: dict[str, int]
    id_to_token: dict[int, str]
    vocab_size: int
    vocab_hash: str
    tokens_per_txn: int
    chunk_size: int
    context_len: int
    grammar: ChunkGrammar
    report: ContractReport
    spec: TokenizerSpec
    fitted_state: Optional[dict] = None


# ---------------------------------------------------------------------------
# The compiler + preset factories — LOCKED signatures, stubbed bodies.
# ---------------------------------------------------------------------------


def compile_spec(spec: TokenizerSpec, context_len: int = 4096) -> CompiledTokenizer:
    """Compile a :class:`TokenizerSpec` to a :class:`CompiledTokenizer`.

    Lays out specials (ids 0-4) then each step's contiguous block at a running
    offset using 0-based local indices; runs C1 (injective + dense), C2
    (determinism / fitted-artifact), C3 (grammar / chunk derivation); derives
    ``vocab_size``, ``vocab_hash``, ``tokens_per_txn``, ``chunk_size``.

    On a C1 collision (overlapping blocks) the returned ``report.passed`` is
    False and the verb layer refuses to write a Corpus with a named C1 diagnostic
    (the MONTH_12/CARD_0 case). Raises nothing for contract violations — they are
    reported, not thrown.
    """
    from .contracts import compile_spec as _compile_spec
    return _compile_spec(spec, context_len=context_len)


def financial_spec(
    *,
    merchant_hash_size: int = 2000,
    amount_strategy: AmountStrategy = AmountStrategy.FIXED,
    include_time_delta: bool = False,
    drop_steps: tuple[str, ...] = (),
) -> TokenizerSpec:
    """Build the reference ``financial`` (TabFormer) spec.

    The 12 fields in order: AMT, MERCH, CAT, MCC, HOUR, DOW, MONTH, CARD, CHIP,
    ZIP3, STATE, CUST → ``vocab_size == 6251`` when compiled. With
    ``include_time_delta=True`` a 13th ``TDIF`` field (32 log-bins) is appended →
    vocab 6283, tokens_per_txn 13, chunk_size 292 (at context_len 4096).
    ``drop_steps`` removes named steps (e.g. ``("cust",)`` for T2 → vocab 3251).
    """
    from .spec import financial_spec as _financial_spec
    return _financial_spec(
        merchant_hash_size=merchant_hash_size,
        amount_strategy=amount_strategy,
        include_time_delta=include_time_delta,
        drop_steps=drop_steps,
    )


def chain_spec(
    *,
    item_hash_size: int = 5000,
    size_bins: int = 8,
    include_identity_token: bool = False,
    drop_steps: tuple[str, ...] = (),
) -> TokenizerSpec:
    """Build the Phase-0 ``chain`` (DEX next-trade) spec.

    Fields: venue (FixedVocab, 3), side (MappingDirect BUY/SELL + default, 2+),
    item (Hash ~5000), size_usd (log-bins ~8), inter-trade gap (TimeDelta 32, T1
    non-negotiable), hour (24), dow (7). NO wallet-identity token by default (T2);
    ``include_identity_token`` would add it (kept off). entity=wallet, event=trade;
    sort by [wallet, timestamp] (C6). ``tokens_per_event == 7`` → chunk_size
    ``4096 // 8 == 512``; vocab is DERIVED (~5081 illustrative — compute it).
    """
    from .spec import chain_spec as _chain_spec
    return _chain_spec(
        item_hash_size=item_hash_size,
        size_bins=size_bins,
        include_identity_token=include_identity_token,
        drop_steps=drop_steps,
    )
