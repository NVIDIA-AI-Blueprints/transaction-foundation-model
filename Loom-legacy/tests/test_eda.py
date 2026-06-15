"""Tests for the EDA verb: the pure profiling logic + CLI arg-parsing.

The profiling *logic* is factored out of :class:`flows.eda.EdaFlow` into the
module-level pure function :func:`flows.eda.profile_dataframe`, so it is
unit-testable on a small in-memory DataFrame with **no Metaflow involved**. These
tests pin:

* schema / dtypes / nrows / missingness;
* target resolution (declared, inferred-from-test, inferred-by-name) + class
  balance;
* the simple leakage flags (near-perfect predictor, duplicate-of-target) and that
  an ID-like column does NOT false-positive.

``pandas`` is required for the profiling tests (``importorskip``). The CLI
arg-parse tests are pure-Python (no pandas/Metaflow): they only exercise the
argparse wiring for ``loom eda`` / ``loom datasets``.
"""

from __future__ import annotations

import pytest

from loom.cli import _build_parser


# ---------------------------------------------------------------------------
# Pure profiling logic (no Metaflow).
# ---------------------------------------------------------------------------

pd = pytest.importorskip("pandas")


def test_profile_basic_schema_dtypes_nrows() -> None:
    """Schema, dtypes, nrows/ncols, and columns reflect the input frame."""
    from flows.eda import profile_dataframe

    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [1.0, 2.0, 3.0, 4.0],
            "c": ["x", "y", "x", "z"],
            "target": [0, 1, 0, 1],
        }
    )
    p = profile_dataframe(df, target="target")

    assert p["nrows"] == 4
    assert p["ncols"] == 4
    assert p["columns"] == ["a", "b", "c", "target"]
    assert p["dtypes"]["a"].startswith("int")
    assert p["dtypes"]["b"].startswith("float")
    assert p["dtypes"]["c"] == "object"
    # Numeric describe present for numeric cols, absent for the object col.
    assert "a" in p["numeric_describe"]
    assert "c" not in p["numeric_describe"]
    assert p["numeric_describe"]["a"]["min"] == 1.0
    assert p["numeric_describe"]["a"]["max"] == 4.0


def test_profile_missingness_percentage() -> None:
    """Missingness is reported as a per-column percentage of rows."""
    from flows.eda import profile_dataframe

    df = pd.DataFrame(
        {
            "a": [1, None, None, None],  # 75% missing
            "b": [1, 2, 3, 4],  # 0% missing
            "target": [0, 1, 0, 1],
        }
    )
    p = profile_dataframe(df, target="target")
    assert p["missingness"]["a"] == pytest.approx(75.0)
    assert p["missingness"]["b"] == pytest.approx(0.0)


def test_profile_declared_target_balance() -> None:
    """A declared target reports class balance and is not marked inferred."""
    from flows.eda import profile_dataframe

    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6],
            "label": ["a", "a", "a", "b", "b", "c"],
        }
    )
    p = profile_dataframe(df, target="label")
    assert p["target"] == "label"
    assert p["target_inferred"] is False
    assert p["target_balance"] == {"a": 3, "b": 2, "c": 1}


def test_profile_infers_target_from_train_test_asymmetry() -> None:
    """A single column present in train but absent from test is inferred target."""
    from flows.eda import profile_dataframe

    train = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "target": [0, 1, 0]})
    test = pd.DataFrame({"x": [7, 8], "y": [9, 10]})
    p = profile_dataframe(train, test=test)
    assert p["target"] == "target"
    assert p["target_inferred"] is True


def test_profile_infers_target_by_name_when_no_test() -> None:
    """With no test split, a literal ``target``/``label`` column is inferred."""
    from flows.eda import profile_dataframe

    df = pd.DataFrame({"x": [1, 2, 3], "label": [0, 1, 0]})
    p = profile_dataframe(df)
    assert p["target"] == "label"
    assert p["target_inferred"] is True


def test_profile_no_target_when_none_identifiable() -> None:
    """When no target can be resolved, ``target`` is None and balance is None."""
    from flows.eda import profile_dataframe

    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    p = profile_dataframe(df)
    assert p["target"] is None
    assert p["target_balance"] is None
    # No target -> no leakage flags possible.
    assert p["leakage"] is False
    assert p["leakage_flags"] == []


def test_profile_flags_near_perfect_predictor() -> None:
    """A numeric feature ~perfectly correlated with the target is flagged leaky."""
    import numpy as np

    from flows.eda import profile_dataframe

    rng = np.random.default_rng(0)
    n = 200
    target = rng.integers(0, 2, n)
    df = pd.DataFrame(
        {
            "noise": rng.normal(size=n),
            "leaky": target * 100.0 + rng.normal(scale=1e-3, size=n),
            "target": target,
        }
    )
    p = profile_dataframe(df, target="target")
    assert p["leakage"] is True
    kinds = {(f["column"], f["kind"]) for f in p["leakage_flags"]}
    assert ("leaky", "near_perfect_predictor") in kinds
    # The pure-noise column is NOT flagged.
    assert all(f["column"] != "noise" for f in p["leakage_flags"])


def test_profile_flags_duplicate_of_target() -> None:
    """A low-cardinality feature that determines the target is flagged leaky."""
    import numpy as np

    from flows.eda import profile_dataframe

    rng = np.random.default_rng(1)
    n = 120
    target = rng.integers(0, 2, n)
    df = pd.DataFrame(
        {
            "dup": pd.Series(target).map({0: "no", 1: "yes"}),
            "target": target,
        }
    )
    p = profile_dataframe(df, target="target")
    kinds = {(f["column"], f["kind"]) for f in p["leakage_flags"]}
    assert ("dup", "duplicate_of_target") in kinds


def test_profile_id_column_not_flagged_as_duplicate() -> None:
    """An ID-like column (cardinality ~= nrows) is NOT a duplicate-of-target FP."""
    import numpy as np

    from flows.eda import profile_dataframe

    rng = np.random.default_rng(2)
    n = 100
    df = pd.DataFrame(
        {
            "id": range(n),  # unique per row -> ID-like, must not flag
            "feat": rng.normal(size=n),
            "target": rng.integers(0, 2, n),
        }
    )
    p = profile_dataframe(df, target="target")
    assert all(f["column"] != "id" for f in p["leakage_flags"])


def test_profile_is_json_able() -> None:
    """The whole profile dict round-trips through JSON (suitable for a summary)."""
    import json

    from flows.eda import profile_dataframe

    df = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0], "y": [3.0, 2.0, 1.0], "target": [0, 1, 0]}
    )
    p = profile_dataframe(df, target="target")
    # Must not raise; round-trips to an equal structure.
    assert json.loads(json.dumps(p)) == p


def test_profile_top_correlations_present_and_sorted() -> None:
    """Top correlations are returned strongest-first among numeric columns."""
    import numpy as np

    from flows.eda import profile_dataframe

    rng = np.random.default_rng(3)
    n = 200
    base = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "a": base,
            "b": base * 2.0 + rng.normal(scale=1e-6, size=n),  # ~perfect with a
            "c": rng.normal(size=n),  # independent
        }
    )
    p = profile_dataframe(df)  # no target -> still computes correlations
    corrs = p["top_correlations"]
    assert corrs, "expected at least one correlation pair"
    # Sorted by descending |corr|.
    abs_vals = [abs(row[2]) for row in corrs]
    assert abs_vals == sorted(abs_vals, reverse=True)
    # The (a, b) pair (near-perfect) is the strongest.
    top_pair = {corrs[0][0], corrs[0][1]}
    assert top_pair == {"a", "b"}


# ---------------------------------------------------------------------------
# CLI arg-parsing for the new subcommands (pure-Python, no pandas/Metaflow).
# ---------------------------------------------------------------------------


def test_cli_eda_parses_dataset_and_target() -> None:
    """`loom eda --dataset ... --target ...` parses into the eda handler."""
    from loom.cli import _cmd_eda

    parser = _build_parser()
    args = parser.parse_args(
        ["eda", "--dataset", "IngestDataset/123", "--target", "label"]
    )
    assert args.command == "eda"
    assert args.dataset == "IngestDataset/123"
    assert args.target == "label"
    assert args.func is _cmd_eda


def test_cli_eda_target_optional() -> None:
    """`--target` is optional and defaults to None."""
    parser = _build_parser()
    args = parser.parse_args(["eda", "--dataset", "IngestDataset/9"])
    assert args.dataset == "IngestDataset/9"
    assert args.target is None


def test_cli_eda_requires_dataset() -> None:
    """`loom eda` without --dataset is a parse error (required argument)."""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["eda"])


def test_cli_datasets_parses() -> None:
    """`loom datasets` parses into the datasets handler."""
    from loom.cli import _cmd_datasets

    parser = _build_parser()
    args = parser.parse_args(["datasets"])
    assert args.command == "datasets"
    assert args.func is _cmd_datasets


def test_cli_run_and_ingest_still_parse() -> None:
    """Existing subcommands remain intact alongside the new ones."""
    parser = _build_parser()
    run_args = parser.parse_args(
        ["run", "--dataset", "IngestDataset/1", "--goal", "g", "--metric", "m"]
    )
    assert run_args.command == "run"
    ingest_args = parser.parse_args(["ingest", "--source", "/tmp/data"])
    assert ingest_args.command == "ingest"
