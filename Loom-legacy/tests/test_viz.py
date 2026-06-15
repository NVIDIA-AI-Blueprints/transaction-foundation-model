"""Tests for the viz verb: the pure plotting logic + CLI arg-parsing.

The plotting *logic* is factored out of :class:`flows.viz.VizFlow` into the
module-level pure functions :func:`flows.viz.plot_dataframe` (data-object input) and
:func:`flows.viz.plot_run_metrics` (run input), so they are unit-testable on a small
in-memory DataFrame / leaderboard with **no Metaflow involved**. Each returns the
matplotlib ``Figure`` objects + a JSON-able descriptor, and (with ``save_dir``) also
writes PNG paths -- so the tests assert on figures/paths directly. These tests pin:

* a dataset producing distribution / correlation / target-vs-feature figures, and
  honoring a single ``kind``;
* figures saved to disk when ``save_dir`` is given (paths exist, descriptor carries
  them);
* a run leaderboard producing metric-over-nodes + leaderboard figures, and an empty
  leaderboard producing none;
* the descriptors round-tripping through JSON.

``pandas`` + ``numpy`` + ``matplotlib`` are required (``importorskip``). Figures are
closed after each assertion to keep the matplotlib state clean. The CLI arg-parse
tests are pure-Python: they exercise the argparse wiring for ``loom viz`` (the
mutually-exclusive --dataset / --run).
"""

from __future__ import annotations

import json

import pytest

from loom.cli import _build_parser


# ---------------------------------------------------------------------------
# Pure plotting logic (no Metaflow).
# ---------------------------------------------------------------------------

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
pytest.importorskip("matplotlib")


def _close(figures) -> None:
    """Close matplotlib figures to keep global state clean between tests."""
    import matplotlib.pyplot as plt

    for fig in figures:
        plt.close(fig)


def _frame(n: int = 120, seed: int = 0) -> "pd.DataFrame":
    """A small frame with numeric features, a categorical, and a binary target."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "cat": rng.choice(["p", "q"], size=n),
            "target": rng.integers(0, 2, n),
        }
    )


def test_plot_dataframe_all_kinds_returns_figures() -> None:
    """kind='all' returns distribution, correlation, and target-vs-feature figures."""
    from flows.viz import plot_dataframe

    figures, desc = plot_dataframe(_frame(), target="target", kind="all")
    names = [p["name"] for p in desc["plots"]]
    assert "distributions" in names
    assert "correlation" in names
    assert "target_vs_feature" in names
    assert len(figures) == len(desc["plots"])
    assert desc["source"] == "dataset"
    _close(figures)


def test_plot_dataframe_single_kind() -> None:
    """A single kind produces only that figure family."""
    from flows.viz import plot_dataframe

    figures, desc = plot_dataframe(_frame(), target="target", kind="correlation")
    assert [p["name"] for p in desc["plots"]] == ["correlation"]
    assert len(figures) == 1
    _close(figures)


def test_plot_dataframe_returns_figure_objects() -> None:
    """The returned figures are real matplotlib Figure objects (embeddable)."""
    from matplotlib.figure import Figure

    from flows.viz import plot_dataframe

    figures, _ = plot_dataframe(_frame(), kind="distributions")
    assert figures and all(isinstance(f, Figure) for f in figures)
    _close(figures)


def test_plot_dataframe_saves_paths_when_save_dir(tmp_path) -> None:
    """With save_dir, each figure is written to a PNG and its path recorded."""
    import os

    from flows.viz import plot_dataframe

    figures, desc = plot_dataframe(
        _frame(), target="target", kind="all", save_dir=str(tmp_path)
    )
    paths = [p["path"] for p in desc["plots"]]
    assert paths and all(p is not None for p in paths)
    for p in paths:
        assert os.path.isfile(p)
        assert os.path.getsize(p) > 0
    _close(figures)


def test_plot_dataframe_no_paths_without_save_dir() -> None:
    """Without save_dir, paths are None but figures are still returned."""
    from flows.viz import plot_dataframe

    figures, desc = plot_dataframe(_frame(), kind="distributions")
    assert all(p["path"] is None for p in desc["plots"])
    assert figures
    _close(figures)


def test_plot_dataframe_descriptor_is_json_able() -> None:
    """The descriptor round-trips through JSON (suitable for a RunResult summary)."""
    from flows.viz import plot_dataframe

    figures, desc = plot_dataframe(_frame(), target="target", kind="all")
    assert json.loads(json.dumps(desc)) == desc
    _close(figures)


def test_plot_run_metrics_returns_figures() -> None:
    """A scored leaderboard yields metric-over-nodes + leaderboard figures."""
    from flows.viz import plot_run_metrics

    lb = [
        {"node_id": "n1", "metric": 0.80},
        {"node_id": "n2", "metric": 0.90},
        {"node_id": "n3", "metric": 0.85},
    ]
    figures, desc = plot_run_metrics(lb)
    names = [p["name"] for p in desc["plots"]]
    assert names == ["metric_over_nodes", "leaderboard"]
    assert len(figures) == 2
    assert desc["source"] == "run"
    assert desc["n_rows"] == 3
    _close(figures)


def test_plot_run_metrics_empty_leaderboard_no_figures() -> None:
    """An empty / unscored leaderboard produces no figures (but a valid descriptor)."""
    from flows.viz import plot_run_metrics

    figures, desc = plot_run_metrics([])
    assert figures == []
    assert desc["plots"] == []
    assert desc["n_rows"] == 0

    # Rows without a numeric metric also produce no metric figures.
    figures2, desc2 = plot_run_metrics([{"node_id": "n1", "metric": None}])
    assert figures2 == []
    assert desc2["n_rows"] == 1


def test_plot_run_metrics_saves_paths(tmp_path) -> None:
    """With save_dir, the run figures are written to PNGs and recorded."""
    import os

    from flows.viz import plot_run_metrics

    lb = [{"node_id": "n1", "metric": 0.5}, {"node_id": "n2", "metric": 0.7}]
    figures, desc = plot_run_metrics(lb, save_dir=str(tmp_path))
    for p in desc["plots"]:
        assert p["path"] is not None and os.path.isfile(p["path"])
    _close(figures)


# ---------------------------------------------------------------------------
# CLI arg-parsing (pure-Python, no matplotlib/Metaflow).
# ---------------------------------------------------------------------------


def test_cli_viz_parses_dataset() -> None:
    """`loom viz --dataset ... --kind ...` parses into the viz handler."""
    from loom.cli import _cmd_viz

    parser = _build_parser()
    args = parser.parse_args(
        ["viz", "--dataset", "IngestDataset/123", "--target", "y", "--kind", "all"]
    )
    assert args.command == "viz"
    assert args.dataset == "IngestDataset/123"
    assert args.run is None
    assert args.target == "y"
    assert args.kind == "all"
    assert args.func is _cmd_viz


def test_cli_viz_parses_run() -> None:
    """`loom viz --run PATHSPEC` parses the run input."""
    parser = _build_parser()
    args = parser.parse_args(["viz", "--run", "EvalCandidate/42"])
    assert args.run == "EvalCandidate/42"
    assert args.dataset is None


def test_cli_viz_requires_one_source() -> None:
    """`loom viz` needs exactly one of --dataset / --run."""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["viz"])  # neither
    with pytest.raises(SystemExit):
        parser.parse_args(["viz", "--dataset", "d", "--run", "r"])  # both
