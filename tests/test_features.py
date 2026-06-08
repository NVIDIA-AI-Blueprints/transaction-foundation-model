"""Tests for the features verb: the pure feature-build logic + CLI arg-parsing.

The feature-building *logic* is factored out of :class:`flows.features.FeaturesFlow`
into the module-level pure function :func:`flows.features.build_features` (and its
helpers), so it is unit-testable on a small in-memory DataFrame with **no Metaflow
involved**. These tests pin:

* numeric scaling (z-score) + the new column naming;
* low-cardinality one-hot vs. high-cardinality frequency categorical encoding;
* leakage-column dropping (the ``eda -> features`` composition: ``drop_columns``);
* target preservation (a declared/inferred target is never engineered);
* the IngestDataset-shaped schema + a stable content fingerprint;
* the ``minimal`` recipe (scaling + encoding only, no interactions/datetime/agg);
* the result being JSON-able for a RunResult summary.

``pandas`` + ``numpy`` are required for the build tests (``importorskip``). The CLI
arg-parse tests are pure-Python (no pandas/Metaflow): they only exercise the
argparse wiring for ``loom features``.
"""

from __future__ import annotations

import json

import pytest

from loom.cli import _build_parser


# ---------------------------------------------------------------------------
# Pure feature-build logic (no Metaflow).
# ---------------------------------------------------------------------------

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")


def _frame(n: int = 60, seed: int = 0) -> "pd.DataFrame":
    """A small mixed-dtype frame: numerics, a low-card cat, and a binary target."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(loc=5.0, scale=2.0, size=n),
            "color": rng.choice(["red", "green", "blue"], size=n),
            "target": rng.integers(0, 2, size=n),
        }
    )


def test_build_features_scales_numeric_columns() -> None:
    """Each non-target numeric column gets a standardized ``<col>__z`` feature."""
    from flows.features import build_features

    result = build_features(_frame(), target="target")
    train = result["train"]
    assert "x1__z" in train.columns
    assert "x2__z" in train.columns
    # The z-scored column has ~zero mean and ~unit std on train.
    assert abs(float(train["x1__z"].mean())) < 1e-6
    assert abs(float(train["x1__z"].std()) - 1.0) < 0.2
    # The target is preserved untouched and never z-scored.
    assert "target__z" not in train.columns
    assert "target" in train.columns
    assert result["target"] == "target"


def test_build_features_one_hot_encodes_low_cardinality() -> None:
    """A low-cardinality categorical is one-hot encoded into ``<col>__<value>``."""
    from flows.features import build_features

    result = build_features(_frame(), target="target")
    cols = set(result["train"].columns)
    assert {"color__red", "color__green", "color__blue"} <= cols
    assert result["n_features_after"] > result["n_features_before"]
    assert any(c.startswith("color__") for c in result["added_features"])


def test_build_features_frequency_encodes_high_cardinality() -> None:
    """A high-cardinality categorical is frequency-encoded into ``<col>__freq``."""
    from flows.features import build_features

    n = 60
    df = pd.DataFrame(
        {
            "x": np.arange(n, dtype=float),
            # 30 distinct values -> above the one-hot cardinality cap.
            "id": [f"u{i % 30}" for i in range(n)],
            "target": np.tile([0, 1], n // 2),
        }
    )
    result = build_features(df, target="target")
    cols = set(result["train"].columns)
    assert "id__freq" in cols
    # No per-value one-hot explosion for the high-cardinality column.
    assert not any(c.startswith("id__u") for c in cols)


def test_build_features_drops_leakage_columns() -> None:
    """``drop_columns`` (EDA-flagged leakage) are removed before building (eda->features)."""
    from flows.features import build_features

    df = _frame()
    df["leaky"] = df["target"].to_numpy(dtype=float)
    result = build_features(df, target="target", drop_columns=["leaky"])
    cols = set(result["train"].columns)
    assert "leaky" not in cols
    assert "leaky__z" not in cols  # never engineered from a dropped column
    assert result["dropped_columns"] == ["leaky"]


def test_build_features_minimal_recipe_skips_interactions_and_datetime() -> None:
    """The ``minimal`` recipe applies scaling + encoding only (no interactions/agg)."""
    from flows.features import build_features

    full = build_features(_frame(), target="target", recipe="full")
    minimal = build_features(_frame(), target="target", recipe="minimal")
    full_cols = set(full["train"].columns)
    minimal_cols = set(minimal["train"].columns)
    # Interaction columns (``a__x__b``) appear only in the full recipe.
    assert any("__x__" in c for c in full_cols)
    assert not any("__x__" in c for c in minimal_cols)
    assert minimal["recipe"] == "minimal"


def test_build_features_schema_is_ingest_shaped() -> None:
    """The engineered schema matches the IngestDataset shape (columns/dtypes/nrows/target)."""
    from flows.features import build_features

    result = build_features(_frame(n=40), target="target")
    schema = result["schema"]
    assert set(schema) >= {"columns", "dtypes", "nrows", "target"}
    assert schema["nrows"] == 40
    assert schema["target"] == "target"
    assert schema["columns"] == [str(c) for c in result["train"].columns]


def test_build_features_fingerprint_is_stable_and_content_addressed() -> None:
    """The same build on the same input yields the same fingerprint (content-addressed)."""
    from flows.features import build_features, fingerprint_frame

    r1 = build_features(_frame(seed=1), target="target")
    r2 = build_features(_frame(seed=1), target="target")
    fp1 = fingerprint_frame(r1["train"], r1["schema"])
    fp2 = fingerprint_frame(r2["train"], r2["schema"])
    assert fp1 == fp2
    assert fp1.startswith("sha256:")


def test_build_features_empty_frame_raises() -> None:
    """An empty train frame is refused (nothing to engineer)."""
    from flows.features import build_features

    with pytest.raises(ValueError):
        build_features(pd.DataFrame(), target="target")


def test_build_features_result_is_json_able() -> None:
    """The summary-shaped fields round-trip through JSON (suitable for a RunResult summary)."""
    from flows.features import build_features

    result = build_features(_frame(), target="target")
    summary = {
        "target": result["target"],
        "recipe": result["recipe"],
        "added_features": result["added_features"],
        "dropped_columns": result["dropped_columns"],
        "n_features_before": result["n_features_before"],
        "n_features_after": result["n_features_after"],
        "null_stats": result["null_stats"],
    }
    assert json.loads(json.dumps(summary)) == summary


# ---------------------------------------------------------------------------
# CLI arg-parsing (pure-Python, no pandas/Metaflow).
# ---------------------------------------------------------------------------


def test_cli_features_parses_all_flags() -> None:
    """`loom features` parses dataset/target/--from/recipe into the handler."""
    from loom.cli import _cmd_features

    parser = _build_parser()
    args = parser.parse_args(
        [
            "features",
            "--dataset",
            "IngestDataset/123",
            "--target",
            "y",
            "--from",
            "EdaFlow/7",
            "--recipe",
            "minimal",
        ]
    )
    assert args.command == "features"
    assert args.dataset == "IngestDataset/123"
    assert args.target == "y"
    assert args.from_eda == "EdaFlow/7"
    assert args.recipe == "minimal"
    assert args.func is _cmd_features


def test_cli_features_optional_flags_default_none() -> None:
    """target / --from / recipe default to None."""
    parser = _build_parser()
    args = parser.parse_args(["features", "--dataset", "IngestDataset/9"])
    assert args.target is None
    assert args.from_eda is None
    assert args.recipe is None


def test_cli_features_requires_dataset() -> None:
    """`loom features` without --dataset is a parse error (required argument)."""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["features", "--target", "y"])
