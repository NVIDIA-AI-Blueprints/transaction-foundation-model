"""Loom's read-only visualization Metaflow flow.

This module defines the single static ``FlowSpec`` -- :class:`VizFlow` -- that the
``loom viz`` command runs (via the Metaflow MLOps interface) to **plot** a data
object or a run's results. Viz is the **read-only tier** of the approval matrix
(design-spec §3): it trains nothing and writes nothing back; it renders matplotlib
figures, embeds them in a Metaflow ``@card``, and emits a Metaflow run. It NEVER
prompts.

The input is either a **data object** referenced by ``dataset_ref`` (a pathspec
like ``"IngestDataset/123"``) -- for distribution / correlation / target-vs-feature
plots -- or a **run** referenced by ``run_pathspec`` -- for metric-over-nodes /
leaderboard plots. Data is read through the Metaflow **Client API** only
(:func:`loom.dataio.materialize_dataset` for a data object; ``metaflow.Run`` for a
run); Loom never touches the underlying datastore (local or S3/minio) directly.

Flow shape::

    start --> plot --> end

* ``start`` -- materialize the data object into a tmp ``./input`` (data-object
               input) or resolve the run and read its leaderboard artifacts (run
               input), via the Client API. READ-ONLY.
* ``plot``  -- generate the figures via the pure, unit-testable
               :func:`plot_dataframe` / :func:`plot_run_metrics` (which return
               matplotlib figures + a small descriptor without needing Metaflow),
               embed each figure as a card image, and store the descriptor on
               ``self.viz``.
* ``end``   -- carry ``self.viz`` forward so ``Run.data.viz`` exposes it to the
               MLOps interface's Client-API read.

The plotting *logic* is factored into the module-level pure functions
:func:`plot_dataframe` and :func:`plot_run_metrics` so they are unit-testable on a
small in-memory DataFrame / leaderboard with no Metaflow involved: each returns the
matplotlib ``Figure`` objects (and, optionally, saves them to a directory) plus a
JSON-able descriptor. The flow step is a thin wrapper that calls them and embeds
the figures.

``matplotlib`` uses the non-interactive ``Agg`` backend (set before pyplot is
imported) so figures render headlessly inside a flow. ``pandas`` / ``numpy`` /
``matplotlib`` and ``loom`` are imported *inside* the functions/steps so the flow
file parses even where they are not yet importable until the Runner subprocess sets
up the environment.
"""

from __future__ import annotations

from typing import Any

from metaflow import FlowSpec, Parameter, card, current, step

#: Maximum number of numeric columns to show in the correlation heatmap / per the
#: distribution grid, so a wide frame does not produce an unreadable figure.
_MAX_PLOT_COLUMNS = 20


def _new_agg_pyplot():
    """Return ``matplotlib.pyplot`` with the headless ``Agg`` backend selected.

    Selecting ``Agg`` before importing pyplot lets figures render with no display
    (inside a flow / on CI), which is exactly the read-only, headless context viz
    runs in.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_dataframe(
    train: Any,
    target: str | None = None,
    kind: str = "all",
    save_dir: str | None = None,
) -> tuple[list, dict]:
    """Plot a DataFrame into matplotlib figures (pure function).

    This is the unit-testable core of :class:`VizFlow` for a data-object input:
    given a pandas DataFrame it builds the standard read-only plots with no Metaflow
    involved, and (optionally) saves them. It is domain-neutral -- it never assumes
    a task type or a column meaning; it plots only what the data and the (optional)
    declared target say.

    The plots, by ``kind``:

    * ``"distributions"`` -- one histogram per numeric column (capped grid);
    * ``"correlation"`` -- a correlation heatmap over the numeric columns;
    * ``"target"`` -- for a declared target, target-vs-feature views (a box/strip of
      each top numeric feature against the target, or a scatter for a continuous
      target);
    * ``"all"`` (default) -- all of the above that apply.

    Args:
        train: The DataFrame to plot.
        target: Optional declared target column for the target-vs-feature plots.
        kind: Which plot family to produce (see above).
        save_dir: When given, each figure is also saved as a PNG there and its path
            recorded in the descriptor; the figures are always returned regardless.

    Returns:
        ``(figures, descriptor)`` -- ``figures`` is the list of matplotlib
        ``Figure`` objects (so a caller can embed them); ``descriptor`` is a
        JSON-able dict ``{"kind", "source": "dataset", "plots": [{"name", "title",
        "path"|None}...]}`` describing each figure.
    """
    import os

    import numpy as np
    import pandas as pd  # noqa: F401  (used via the DataFrame API)

    plt = _new_agg_pyplot()

    numeric_cols = [
        c for c in train.columns if pd.api.types.is_numeric_dtype(train[c])
    ][:_MAX_PLOT_COLUMNS]

    figures: list = []
    plots: list[dict] = []

    def _emit(fig, name: str, title: str) -> None:
        path = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"{name}.png")
            fig.savefig(path, bbox_inches="tight", dpi=100)
        figures.append(fig)
        plots.append({"name": name, "title": title, "path": path})

    want = {"distributions", "correlation", "target"} if kind == "all" else {kind}

    # Distributions: a histogram grid over numeric columns.
    if "distributions" in want and numeric_cols:
        ncols = min(3, len(numeric_cols))
        nrows = int(np.ceil(len(numeric_cols) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
        axes = np.atleast_1d(axes).ravel()
        for ax, col in zip(axes, numeric_cols):
            ax.hist(train[col].dropna().to_numpy(), bins=20, color="#4C78A8")
            ax.set_title(str(col), fontsize=9)
        for ax in axes[len(numeric_cols):]:
            ax.set_visible(False)
        fig.suptitle("Feature distributions")
        fig.tight_layout()
        _emit(fig, "distributions", "Feature distributions")

    # Correlation heatmap over numeric columns.
    if "correlation" in want and len(numeric_cols) >= 2:
        import warnings

        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            corr = train[numeric_cols].corr(numeric_only=True)
        fig, ax = plt.subplots(
            figsize=(0.6 * len(numeric_cols) + 2, 0.6 * len(numeric_cols) + 2)
        )
        im = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(numeric_cols)))
        ax.set_yticks(range(len(numeric_cols)))
        ax.set_xticklabels([str(c) for c in numeric_cols], rotation=90, fontsize=8)
        ax.set_yticklabels([str(c) for c in numeric_cols], fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Correlation heatmap")
        fig.tight_layout()
        _emit(fig, "correlation", "Correlation heatmap")

    # Target-vs-feature: box/strip per top numeric feature, or scatter for a
    # continuous target.
    if "target" in want and target and str(target) in train.columns:
        feat_cols = [c for c in numeric_cols if str(c) != str(target)][:6]
        if feat_cols:
            tcol = train[target]
            continuous_target = (
                pd.api.types.is_numeric_dtype(tcol)
                and int(tcol.nunique(dropna=True)) > 20
            )
            ncols = min(3, len(feat_cols))
            nrows = int(np.ceil(len(feat_cols) / ncols))
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(4 * ncols, 3 * nrows)
            )
            axes = np.atleast_1d(axes).ravel()
            for ax, col in zip(axes, feat_cols):
                if continuous_target:
                    ax.scatter(
                        train[col].to_numpy(),
                        tcol.to_numpy(),
                        s=8,
                        alpha=0.5,
                        color="#E45756",
                    )
                    ax.set_xlabel(str(col), fontsize=8)
                    ax.set_ylabel(str(target), fontsize=8)
                else:
                    groups = train.groupby(tcol, dropna=False)[col]
                    data = [g.dropna().to_numpy() for _, g in groups]
                    labels = [str(k) for k, _ in groups]
                    if data:
                        # matplotlib 3.9 renamed boxplot's ``labels`` ->
                        # ``tick_labels``; pass positionally + set ticks ourselves
                        # to stay compatible across versions without a deprecation.
                        ax.boxplot(data)
                        ax.set_xticks(range(1, len(labels) + 1))
                        ax.set_xticklabels(labels, fontsize=8)
                    ax.set_title(f"{col} by {target}", fontsize=9)
            for ax in axes[len(feat_cols):]:
                ax.set_visible(False)
            fig.suptitle(f"Target ({target}) vs. features")
            fig.tight_layout()
            _emit(fig, "target_vs_feature", f"Target ({target}) vs. features")

    descriptor = {"kind": kind, "source": "dataset", "plots": plots}
    return figures, descriptor


def plot_run_metrics(
    leaderboard: list[dict],
    save_dir: str | None = None,
) -> tuple[list, dict]:
    """Plot a run's leaderboard / metric-over-nodes into figures (pure function).

    This is the unit-testable core of :class:`VizFlow` for a run input: given the
    leaderboard rows a provider's ``runs()`` returns (or a run's recorded node
    metrics), it builds a metric-over-nodes line and a ranked leaderboard bar with
    no Metaflow involved.

    Each row is read best-effort for a numeric ``metric`` and an identifier
    (``node_id`` / ``run_id`` / ``pathspec``); rows without a numeric metric are
    skipped for the metric plots (but counted in the descriptor).

    Args:
        leaderboard: The leaderboard rows to plot.
        save_dir: When given, each figure is also saved as a PNG there and its path
            recorded in the descriptor.

    Returns:
        ``(figures, descriptor)`` -- ``figures`` is the matplotlib ``Figure`` list;
        ``descriptor`` is ``{"kind": "run", "source": "run", "n_rows", "plots":
        [...]}``.
    """
    import os

    plt = _new_agg_pyplot()

    scored = [
        r for r in (leaderboard or []) if isinstance(r.get("metric"), (int, float))
    ]
    figures: list = []
    plots: list[dict] = []

    def _emit(fig, name: str, title: str) -> None:
        path = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"{name}.png")
            fig.savefig(path, bbox_inches="tight", dpi=100)
        figures.append(fig)
        plots.append({"name": name, "title": title, "path": path})

    if scored:
        metrics = [float(r["metric"]) for r in scored]
        labels = [
            str(r.get("node_id") or r.get("run_id") or r.get("pathspec") or i)
            for i, r in enumerate(scored)
        ]

        # Metric over nodes (search trajectory: the best-so-far envelope is the
        # story, so plot the per-node metric in order).
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(range(len(metrics)), metrics, marker="o", color="#4C78A8")
        ax.set_xlabel("node / run (in order)")
        ax.set_ylabel("metric")
        ax.set_title("Metric over nodes")
        fig.tight_layout()
        _emit(fig, "metric_over_nodes", "Metric over nodes")

        # Leaderboard bar (best-first).
        order = sorted(range(len(metrics)), key=lambda i: -metrics[i])
        top = order[:15]
        fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(top))))
        ax.barh(
            [labels[i] for i in top][::-1],
            [metrics[i] for i in top][::-1],
            color="#54A24B",
        )
        ax.set_xlabel("metric")
        ax.set_title("Leaderboard (best first)")
        fig.tight_layout()
        _emit(fig, "leaderboard", "Leaderboard (best first)")

    descriptor = {
        "kind": "run",
        "source": "run",
        "n_rows": len(leaderboard or []),
        "plots": plots,
    }
    return figures, descriptor


class VizFlow(FlowSpec):
    """Read-only plots of a data object or a run, emitted as a Metaflow ``@card``.

    For a data-object input (``dataset_ref``) it materializes the data via the
    Client API and produces distribution / correlation / target-vs-feature plots
    with :func:`plot_dataframe`; for a run input (``run_pathspec``) it reads the
    run's leaderboard and produces metric-over-nodes / leaderboard plots with
    :func:`plot_run_metrics`. The figures are embedded as card images and a small
    descriptor is carried on ``self.viz`` so the MLOps interface reads it back from
    ``Run.data``. READ-ONLY: trains nothing, writes nothing back.
    """

    #: Metaflow **pathspec** of the data object to plot (e.g. ``IngestDataset/123``).
    #: One of ``dataset_ref`` / ``run_pathspec`` is required.
    dataset_ref = Parameter(
        "dataset_ref",
        default="",
        type=str,
        help="Metaflow pathspec of the data object to plot (e.g. IngestDataset/123).",
    )

    #: Metaflow **pathspec** of a run to plot (metric-over-nodes / leaderboard),
    #: e.g. ``EvalCandidate/42``. Alternative to ``dataset_ref``.
    run_pathspec = Parameter(
        "run_pathspec",
        default="",
        type=str,
        help="Metaflow pathspec of a run to plot (e.g. EvalCandidate/42).",
    )

    #: Optional declared target column for the target-vs-feature plots (data-object
    #: input only). Inferred from the data object's schema when omitted.
    target = Parameter(
        "target",
        default="",
        type=str,
        help="Optional target column for target-vs-feature plots.",
    )

    #: Which plot family to produce for a data-object input
    #: (``distributions``/``correlation``/``target``/``all``). Ignored for a run.
    kind = Parameter(
        "kind",
        default="all",
        type=str,
        help="Plot family for a dataset (distributions|correlation|target|all).",
    )

    @step
    def start(self) -> None:
        """Resolve the input (data object or run) into plot data, via the Client API.

        Data-object input: materialize ``train`` into a tmp ``./input`` and load it
        with pandas. Run input: resolve the run and read its leaderboard artifacts.
        READ-ONLY in both cases.
        """
        import os
        import tempfile

        import pandas as pd

        from loom.dataio import dataset_schema, materialize_dataset

        dataset_ref = (self.dataset_ref or "").strip()
        run_pathspec = (self.run_pathspec or "").strip()

        self._mode = "run" if (run_pathspec and not dataset_ref) else "dataset"
        self._train_df = None
        self._leaderboard = []
        self._resolved_target = (self.target or "").strip() or None

        if self._mode == "dataset":
            workspace = tempfile.mkdtemp(prefix="loom-viz-")
            input_dir = os.path.join(workspace, "input")
            os.makedirs(input_dir, exist_ok=True)
            materialize_dataset(dataset_ref, input_dir)
            self._train_df = pd.read_csv(os.path.join(input_dir, "train.csv"))
            self.workspace_dir = workspace
            if not self._resolved_target:
                try:
                    self._resolved_target = (
                        str(dataset_schema(dataset_ref).get("target") or "") or None
                    )
                except Exception:  # pragma: no cover - schema read edge case
                    self._resolved_target = None
        else:
            self._leaderboard = self._read_run_leaderboard(run_pathspec)

        self.next(self.plot)

    @staticmethod
    def _read_run_leaderboard(run_pathspec: str) -> list[dict]:
        """Read a run's recorded leaderboard / node metrics via the Client API.

        Best-effort: resolves the run and reads a ``leaderboard``/``nodes`` artifact
        off ``.data`` when present, falling back to a single-row leaderboard built
        from a ``best_metric``/``metric`` artifact. Any failure yields ``[]`` so a
        missing artifact never fails the plot.
        """
        from metaflow import Run, namespace

        try:
            namespace(None)
        except Exception:  # pragma: no cover - namespace API edge case
            pass

        try:
            run = Run(run_pathspec)
            data = run.data
        except Exception:  # noqa: BLE001 - unresolvable run / metadata down
            return []
        if data is None:  # pragma: no cover - a successful run has data
            return []

        for name in ("leaderboard", "nodes"):
            value = getattr(data, name, None)
            if isinstance(value, list) and value:
                return [dict(r) for r in value if isinstance(r, dict)]

        for name in ("best_metric", "metric"):
            value = getattr(data, name, None)
            if isinstance(value, (int, float)):
                return [{"node_id": run_pathspec, "metric": float(value)}]
        return []

    @card
    @step
    def plot(self) -> None:
        """Generate the figures and embed them as card images.

        Delegates to the pure :func:`plot_dataframe` (data object) or
        :func:`plot_run_metrics` (run), so the plotting logic is unit-testable
        without Metaflow, then embeds each returned figure as an ``@card`` image and
        stores the descriptor on ``self.viz``.
        """
        if self._mode == "dataset" and self._train_df is not None:
            figures, descriptor = plot_dataframe(
                self._train_df,
                target=self._resolved_target,
                kind=(self.kind or "all").strip() or "all",
            )
        else:
            figures, descriptor = plot_run_metrics(self._leaderboard)

        self.viz = descriptor
        self._render_card(figures, descriptor)
        self.next(self.end)

    def _render_card(self, figures: list, descriptor: dict) -> None:
        """Render the ``@card``: a header + each figure embedded as an image.

        Args:
            figures: The matplotlib ``Figure`` objects from the pure plotter.
            descriptor: The JSON-able plot descriptor (titles per figure).
        """
        from metaflow.cards import Image, Markdown

        current.card.append(Markdown("# Loom visualization"))
        source = descriptor.get("source")
        ref = self.dataset_ref if source == "dataset" else self.run_pathspec
        current.card.append(
            Markdown(
                f"**source:** {source} `{ref}`  \n"
                f"**plots:** {len(descriptor.get('plots') or [])}"
            )
        )

        if not figures:
            current.card.append(
                Markdown(
                    "_No plottable data found (no numeric columns / no scored "
                    "runs)._"
                )
            )
            return

        plots = descriptor.get("plots") or []
        for fig, meta in zip(figures, plots):
            current.card.append(Markdown(f"## {meta.get('title', meta.get('name'))}"))
            try:
                current.card.append(Image.from_matplotlib(fig))
            except Exception:  # pragma: no cover - card image edge case
                current.card.append(Markdown("_(figure could not be embedded)_"))

    @step
    def end(self) -> None:
        """Carry ``self.viz`` forward so ``Run.data.viz`` exposes it.

        Metaflow persists step artifacts, so ``self.viz`` (set in ``plot``) is
        already on ``Run.data``; the MLOps interface reads it back for the
        command's summary. Nothing else to do.
        """
        pass


if __name__ == "__main__":
    VizFlow()
