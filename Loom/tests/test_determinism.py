"""C2 (determinism): the vocabulary is built from config alone. Compiling the same
spec twice yields an identical ``vocab_hash`` (and identical vocab). A fitted
amount strategy (QUANTILE / KMEANS) is a fitted artifact and MUST be flagged by C2
(and its state persisted), because it is not derivable from config alone."""

from __future__ import annotations

from loom.engine import AmountStrategy, compile_spec, financial_spec
from loom.types import Severity

from .golden_helpers import compiled_financial, diagnostics_for, require_engine


def test_same_spec_compiles_to_identical_vocab_hash():
    a = compiled_financial()
    b = compiled_financial()
    assert a.vocab_hash == b.vocab_hash, "compile is not deterministic"
    assert a.vocab == b.vocab
    assert a.vocab_size == b.vocab_size
    assert a.report.deterministic


def test_vocab_hash_changes_when_the_spec_changes():
    """A different spec (time-delta on) must yield a different signature — the
    hash is a real function of the vocab, not a constant."""
    base = compiled_financial()
    td = compiled_financial(include_time_delta=True)
    assert base.vocab_hash != td.vocab_hash


def test_vocab_hash_is_stable_string():
    ct = compiled_financial()
    assert isinstance(ct.vocab_hash, str) and ct.vocab_hash, "vocab_hash must be a non-empty string"


def test_quantile_amount_strategy_is_flagged_as_fitted_by_c2():
    """QUANTILE binning is a FITTED artifact → C2 flags it (has_fitted_artifact)
    and emits a named C2 diagnostic; it is not allowed on the silent default path.

    The binner STATE is fit from data and persisted into the Corpus at write time
    (DESIGN.md §7.2 C2: ``get_state()``), so on the data-free ``compile_spec`` it is
    legitimately ``None`` — but the FLAG and the C2 warning are config-only and MUST
    fire here. ``deterministic`` is False because an un-persisted fitted artifact is
    present at compile time."""
    ct = require_engine(
        lambda: compile_spec(financial_spec(amount_strategy=AmountStrategy.QUANTILE))
    )
    assert ct.report.has_fitted_artifact, "QUANTILE must set has_fitted_artifact"
    c2 = diagnostics_for(ct.report, "C2")
    assert c2, "QUANTILE must emit a named C2 diagnostic"
    # The C2 card warns of the fitted-artifact burden (not silent).
    blob = " ".join((d.message or "") + " " + (d.fix or "") for d in c2)
    assert any(
        kw in blob.lower() for kw in ("fitted", "quantile", "persist", "determinism")
    ), f"C2 card must explain the fitted-artifact burden: {blob!r}"


def test_fixed_amount_strategy_is_clean_under_c2():
    """The default FIXED strategy is config-only → no fitted artifact, C2 clean."""
    ct = compiled_financial()  # default amount_strategy == FIXED
    assert ct.report.has_fitted_artifact is False
    assert ct.report.deterministic is True
    c2_errors = [
        d for d in diagnostics_for(ct.report, "C2") if d.severity is Severity.ERROR
    ]
    assert not c2_errors, f"FIXED path must be C2-clean, got {c2_errors}"
    assert ct.fitted_state in (None, {})
