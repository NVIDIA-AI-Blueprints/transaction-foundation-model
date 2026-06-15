"""The ``local`` model-builder adapter -- a torch-free CPU PPMI+TruncatedSVD stand-in.

:class:`LocalModelBuilderProvider` (registry name ``"local"``) is the **testable /
conformance default path** of the :class:`~loom.providers.ModelBuilderProvider`
port. It realizes the whole ``tokenize -> pretrain -> embed -> finetune ->
evaluate -> serve`` surface on a single CPU machine with **zero new heavy deps**
(only ``numpy`` / ``scipy.sparse`` / ``scikit-learn`` / ``pandas``, all already
in the venv), and it is **deterministic** (``random_state=0``-pinned SVD, stable
token ids, content-addressed fingerprints) so the moat lineage is reproducible.

Why PPMI + TruncatedSVD
-----------------------
PPMI + TruncatedSVD is the count-based equal of word2vec/GloVe: Levy & Goldberg
showed SGNS *implicitly factorizes a shifted PMI matrix*, so the learned
embeddings carry genuine sequential co-occurrence structure the raw
gradient-boosted baseline misses -- a principled lift on a planted-sequential
fixture. One ``TruncatedSVD`` call over a capped ``V x V`` matrix is sub-2s even
at ``budget="full"``. ``pretrain`` is declared ``launch-and-track`` in the
manifest (typed heavy even though cheap) so AIDE never tree-searches the backbone
-- the ``/loom-train`` vs ``/loom-optimize`` mode contract is exercised on the
default path.

Architecture (mirrors the rest of Loom)
----------------------------------------
* **I/O are Metaflow artifact pathspecs only.** Inputs are resolved through the
  Metaflow Client API (:mod:`loom.dataio` / ``metaflow.Run(ref).data``); outputs
  are returned as :class:`~loom.types.ArtifactRef` carrying a run *pathspec* and a
  small JSON-able summary. There is **no object-store SDK, no object-store URI
  literal, and no on-disk checkpoint file** anywhere in this module (constraint 1,
  scanned by the conformance suite).
* **Loom-intent enums only at the seam.** ``objective`` / ``budget`` / ``mode``
  are validated against :data:`loom.providers.OBJECTIVES` /
  :data:`~loom.providers.BUDGETS` / :data:`~loom.providers.MODES` before any work;
  no backend noun leaks (constraint 2).
* **Pure helpers + thin wrappers.** All the maths lives in module-level pure
  functions that operate on plain DataFrames / numpy arrays and are unit-testable
  **without Metaflow**. The public provider methods are thin wrappers that
  materialize a ref, call the helper, and (when run inside the ``TrainFlow`` step)
  let the flow write the produced data object's artifacts on ``self`` -- exactly
  the ``IngestDataset`` / ``FeaturesFlow`` "the pathspec *is* the data object"
  pattern, so an embeddings ref round-trips through ``dataio.materialize_dataset``
  unchanged.
* **Torch-free default; an optional guarded torch hook.** The numpy PPMI-SVD path
  is the only default. :func:`_maybe_torch_backbone` is a lazy guarded stub for
  the optional ``model-local`` extra (design §3.8) that returns ``None`` unless
  torch is present *and* explicitly opted in, so CI (torch absent) and dev (torch
  present transitively via aideml) cannot silently diverge.
"""

from __future__ import annotations

import hashlib
from typing import Any

from loom.config import LoomConfig
from loom.providers import BUDGETS, MODES, OBJECTIVES, ModelBuilderProvider
from loom.registry import register_model_builder
from loom.types import ArtifactRef, Capability, CapabilityManifest, Scores

# Reserved special-token ids shared across every field's vocabulary. ``<mask>`` is
# reserved for the ``masked-field`` objective even when unused, so the vocab layout
# is stable across objectives (stable ids => stable fingerprints).
_PAD_ID = 0
_UNK_ID = 1
_MASK_ID = 2
_N_SPECIAL = 3

#: Default tokenization-scheme knobs AIDE may tree-search via a cheap probe. Caps
#: keep the ``V x V`` co-occurrence matrix small enough that a dense numpy fallback
#: is always safe.
_DEFAULT_MIN_COUNT = 1
_DEFAULT_MAX_VOCAB = 4096
_DEFAULT_N_BUCKETS = 16

#: Per-account sequence window cap (rows beyond this, most-recent-first, are
#: dropped) so a pathological account cannot blow up the co-occurrence build.
_MAX_SEQ_LEN = 512

#: Context window for the distance-weighted ``contrastive`` co-occurrence counts.
_CONTRASTIVE_WINDOW = 5

#: Context-distribution smoothing exponent for PPMI (Levy & Goldberg 2014): the
#: context (column) distribution is raised to this power before normalizing, the
#: count-based equivalent of SGNS's negative-sampling distribution.
_PPMI_ALPHA = 0.75

#: ``budget`` -> embedding dimensionality ``d``. The budget is PHYSICS surfaced at
#: the gate (a bigger ``d`` = more SVD work), never faked away. ``probe`` may use a
#: PCA fast path within the same SVD machinery.
_BUDGET_DIMS = {"probe": 32, "small": 128, "full": 256}

#: Fraction of the embedding frame sealed off as a temporal holdout/test split
#: (most-recent rows), so the embeddings-vs-raw lift the suite asserts is real.
_DEFAULT_TEST_FRACTION = 0.2

#: The pinned random seed for the SVD and every split, so two ``pretrain`` calls on
#: the same fixture produce the SAME fingerprint (the moat-lineage determinism the
#: conformance suite asserts).
_RANDOM_STATE = 0


# ---------------------------------------------------------------------------
# Pure helpers (no Metaflow, no I/O): operate on DataFrames / numpy arrays and
# are directly unit-testable. The provider methods below are thin wrappers that
# materialize a ref and call these.
# ---------------------------------------------------------------------------


def resolve_scheme(scheme: dict | None, schema: dict | None = None) -> dict:
    """Resolve a tokenization scheme to a fully-defaulted, deterministic dict (pure).

    Fills the search knobs (``min_count`` / ``max_vocab`` / ``n_buckets``) from
    their defaults when absent and resolves the field list. When ``fields`` is not
    given it is inferred from ``schema`` (every column that is not the target,
    plus an inferred ``account``/``time`` role when those columns exist), so the
    helper is usable both standalone and from the data object's recorded schema.

    Args:
        scheme: The (possibly partial) scheme dict AIDE may tree-search. Recognized
            keys: ``fields`` (list of column names), ``categorical`` /
            ``numeric`` (optional explicit role lists), ``account`` (grouping
            column), ``time`` (ordering column), ``min_count``, ``max_vocab``,
            ``n_buckets``.
        schema: Optional data-object schema dict (``columns``/``dtypes``/``target``)
            used to infer fields/roles when the scheme omits them.

    Returns:
        A resolved scheme dict with every knob populated (deterministic).
    """
    scheme = dict(scheme or {})
    schema = dict(schema or {})

    columns = [str(c) for c in (schema.get("columns") or [])]
    dtypes = {str(k): str(v) for k, v in (schema.get("dtypes") or {}).items()}
    target = schema.get("target")

    account = scheme.get("account") or _infer_role(columns, ("account", "user", "id", "entity"))
    time_col = scheme.get("time") or _infer_role(columns, ("time", "timestamp", "date", "ts", "event_time"))

    fields = scheme.get("fields")
    if not fields:
        skip = {str(target)} | {str(account)} | {str(time_col)}
        fields = [c for c in columns if c not in skip]
    fields = [str(f) for f in fields]

    # Roles: explicit lists win; otherwise infer from dtypes (numeric dtype =>
    # numeric/bucketized field, else categorical).
    categorical = [str(c) for c in (scheme.get("categorical") or [])]
    numeric = [str(c) for c in (scheme.get("numeric") or [])]
    if not categorical and not numeric:
        for f in fields:
            dt = dtypes.get(f, "")
            if any(k in dt for k in ("int", "float", "double")):
                numeric.append(f)
            else:
                categorical.append(f)
    else:
        # Any field not explicitly typed defaults to categorical.
        typed = set(categorical) | set(numeric)
        categorical += [f for f in fields if f not in typed]

    return {
        "fields": fields,
        "categorical": categorical,
        "numeric": numeric,
        "account": account,
        "time": time_col,
        "target": str(target) if target is not None else None,
        "min_count": int(scheme.get("min_count", _DEFAULT_MIN_COUNT)),
        "max_vocab": int(scheme.get("max_vocab", _DEFAULT_MAX_VOCAB)),
        "n_buckets": int(scheme.get("n_buckets", _DEFAULT_N_BUCKETS)),
    }


def _infer_role(columns: list[str], hints: tuple[str, ...]) -> str | None:
    """Return the first column whose lowercased name contains one of ``hints``."""
    for c in columns:
        cl = c.lower()
        if any(h in cl for h in hints):
            return c
    return None


def build_vocab(train: Any, resolved_scheme: dict) -> dict:
    """Build a deterministic shared vocabulary over the scheme's fields (pure).

    For each **categorical** field: count value frequencies, keep tokens with
    ``count >= min_count``, cap to ``max_vocab`` by frequency, and assign integer
    ids sorted by ``(-count, token_string)`` so ties break on the string -> the
    ids are STABLE across runs (stable fingerprints). For each **numeric** field:
    quantile-bucketize into ``n_buckets`` deterministic edges (``np.quantile``) and
    assign one token id per bin. Ids ``0/1/2`` are reserved ``<pad>``/``<unk>``/
    ``<mask>`` across the whole shared vocab.

    The result is a JSON-able vocab object carrying the ``token -> id`` map, the
    per-field offset map, and the numeric bucket edges, so :func:`encode_sequences`
    (and a later ``embed``) can reproduce the exact ids without the source frame.

    Args:
        train: The training DataFrame holding the field columns.
        resolved_scheme: A scheme from :func:`resolve_scheme`.

    Returns:
        A vocab dict: ``{"token_to_id", "fields", "edges", "size",
        "min_count", "max_vocab", "n_buckets"}``.
    """
    import numpy as np
    import pandas as pd  # noqa: F401  (used via the DataFrame API)

    min_count = int(resolved_scheme["min_count"])
    max_vocab = int(resolved_scheme["max_vocab"])
    n_buckets = max(1, int(resolved_scheme["n_buckets"]))

    token_to_id: dict[str, int] = {}
    edges: dict[str, list[float]] = {}
    next_id = _N_SPECIAL

    # Categorical fields: frequency-capped, string-tiebroken stable ids.
    for field in resolved_scheme["categorical"]:
        if field not in train.columns:
            continue
        counts = train[field].astype(str).value_counts()
        kept = [(tok, int(c)) for tok, c in counts.items() if int(c) >= min_count]
        # Sort by descending count, then token string -> deterministic order.
        kept.sort(key=lambda kv: (-kv[1], kv[0]))
        kept = kept[:max_vocab]
        for tok, _c in kept:
            token = f"{field}={tok}"
            if token not in token_to_id:
                token_to_id[token] = next_id
                next_id += 1

    # Numeric fields: deterministic quantile bucket edges, one id per bucket.
    for field in resolved_scheme["numeric"]:
        if field not in train.columns:
            continue
        col = pd.to_numeric(train[field], errors="coerce").to_numpy(dtype=float)
        col = col[np.isfinite(col)]
        if col.size == 0:
            field_edges: list[float] = []
        else:
            qs = np.linspace(0.0, 1.0, n_buckets + 1)
            field_edges = [float(x) for x in np.unique(np.quantile(col, qs))]
        edges[field] = field_edges
        n_bins = max(1, len(field_edges) - 1) if field_edges else 1
        for b in range(n_bins):
            token = f"{field}#bucket{b}"
            if token not in token_to_id:
                token_to_id[token] = next_id
                next_id += 1

    return {
        "token_to_id": token_to_id,
        "fields": list(resolved_scheme["fields"]),
        "categorical": list(resolved_scheme["categorical"]),
        "numeric": list(resolved_scheme["numeric"]),
        "edges": edges,
        "account": resolved_scheme.get("account"),
        "time": resolved_scheme.get("time"),
        "size": next_id,
        "min_count": min_count,
        "max_vocab": max_vocab,
        "n_buckets": n_buckets,
    }


def _bucket_id_for(value: float, field_edges: list[float]) -> int:
    """Return the bucket index for ``value`` given deterministic quantile edges."""
    import numpy as np

    if not field_edges or not np.isfinite(value):
        return 0
    # Right-closed bins; clamp into [0, n_bins-1].
    n_bins = max(1, len(field_edges) - 1)
    idx = int(np.searchsorted(field_edges, value, side="right") - 1)
    return min(max(idx, 0), n_bins - 1)


def _row_tokens(row: Any, vocab: dict) -> list[int]:
    """Map one frame row to its list of token ids (categorical + numeric), in field order."""
    import numpy as np

    token_to_id = vocab["token_to_id"]
    edges = vocab["edges"]
    ids: list[int] = []
    for field in vocab["categorical"]:
        if field not in row:
            continue
        token = f"{field}={row[field]}"
        ids.append(token_to_id.get(token, _UNK_ID))
    for field in vocab["numeric"]:
        if field not in row:
            continue
        try:
            value = float(row[field])
        except (TypeError, ValueError):
            value = np.nan
        b = _bucket_id_for(value, edges.get(field, []))
        token = f"{field}#bucket{b}"
        ids.append(token_to_id.get(token, _UNK_ID))
    return ids


def encode_sequences(train: Any, vocab: dict) -> list[list[int]]:
    """Group rows into per-account, time-ordered token-id sequences (pure).

    When the vocab carries an ``account`` column, rows are grouped by it and
    ordered by the ``time`` column (when present); otherwise each row is its own
    one-element "sequence". Each row contributes its categorical + bucketized
    numeric token ids (in field order). Sequences are capped at
    :data:`_MAX_SEQ_LEN` (most-recent rows kept) so a pathological account cannot
    blow up the co-occurrence build.

    Args:
        train: The training DataFrame.
        vocab: A vocab dict from :func:`build_vocab`.

    Returns:
        A list of token-id sequences (one per account, or per row when no account
        column is configured).
    """
    account = vocab.get("account")
    time_col = vocab.get("time")

    if account and account in train.columns:
        frame = train
        if time_col and time_col in train.columns:
            frame = train.sort_values(by=[account, time_col], kind="stable")
        else:
            frame = train.sort_values(by=[account], kind="stable")
        sequences: list[list[int]] = []
        for _key, group in frame.groupby(account, sort=True):
            seq: list[int] = []
            for _idx, row in group.iterrows():
                seq.extend(_row_tokens(row, vocab))
            if len(seq) > _MAX_SEQ_LEN:
                seq = seq[-_MAX_SEQ_LEN:]
            if seq:
                sequences.append(seq)
        return sequences

    # No account grouping: each row is a (short) sequence of its own field tokens.
    sequences = []
    for _idx, row in train.iterrows():
        seq = _row_tokens(row, vocab)
        if seq:
            sequences.append(seq)
    return sequences


def build_cooccurrence(sequences: list[list[int]], vocab_size: int, objective: str) -> Any:
    """Build the objective-selected ``V x V`` co-occurrence matrix (pure).

    The objective selects the framing (constraint: ``objective`` is already
    validated against :data:`loom.providers.OBJECTIVES` by the caller):

    * ``next-event`` -- directed adjacent-pair (first-order Markov) counts;
    * ``masked-field`` -- account-level, ORDER-FREE bag co-occurrence over the set
      of tokens in each sequence (the field-interaction signal), symmetric;
    * ``contrastive`` -- symmetric, distance-weighted windowed counts (the
      SGNS / negative-sampling framing), weight ``1/|i-j|`` within a window.

    Returns a dense ``numpy`` matrix (the capped vocab keeps ``V x V`` small enough
    that dense is safe and fast); a ``scipy.sparse`` accumulator is used internally
    for the windowed/bag objectives to keep the build cheap on long sequences.

    Args:
        sequences: Token-id sequences from :func:`encode_sequences`.
        vocab_size: ``V`` -- the total vocab size (rows/cols of the matrix).
        objective: One of ``OBJECTIVES``.

    Returns:
        A dense ``(V, V)`` float numpy array of co-occurrence counts.
    """
    import numpy as np
    from scipy import sparse

    V = max(1, int(vocab_size))
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    if objective == "next-event":
        for seq in sequences:
            for a, b in zip(seq, seq[1:]):
                rows.append(a)
                cols.append(b)
                data.append(1.0)
    elif objective == "masked-field":
        for seq in sequences:
            uniq = sorted(set(seq))
            for ai in range(len(uniq)):
                for bi in range(ai + 1, len(uniq)):
                    a, b = uniq[ai], uniq[bi]
                    rows.append(a)
                    cols.append(b)
                    data.append(1.0)
                    rows.append(b)
                    cols.append(a)
                    data.append(1.0)
    elif objective == "contrastive":
        for seq in sequences:
            n = len(seq)
            for i in range(n):
                a = seq[i]
                lo = max(0, i - _CONTRASTIVE_WINDOW)
                hi = min(n, i + _CONTRASTIVE_WINDOW + 1)
                for j in range(lo, hi):
                    if j == i:
                        continue
                    rows.append(a)
                    cols.append(seq[j])
                    data.append(1.0 / abs(i - j))
    else:  # pragma: no cover - caller validates objective at the seam
        raise ValueError(f"unknown objective {objective!r}")

    if not data:
        return np.zeros((V, V), dtype=float)
    C = sparse.coo_matrix((data, (rows, cols)), shape=(V, V), dtype=float)
    return np.asarray(C.todense())


def ppmi(cooccurrence: Any, alpha: float = _PPMI_ALPHA, k_shift: float = 1.0) -> Any:
    """Positive PMI with context-distribution smoothing (pure; Levy & Goldberg).

    Computes ``PPMI = max(log( P(a,b) / (P(a) * P_alpha(b)) ) - log(k_shift), 0)``,
    where the context (column) distribution is smoothed by raising the column
    marginals to ``alpha`` before renormalizing -- the count-based equivalent of
    SGNS's smoothed negative-sampling distribution. ``k_shift > 1`` applies the
    optional SGNS shift (the contrastive narrative). Degenerate (all-zero) inputs
    return an all-zero matrix.

    Args:
        cooccurrence: A dense ``(V, V)`` co-occurrence count matrix.
        alpha: Context-distribution smoothing exponent (``0.75`` per the paper).
        k_shift: Optional negative-sampling shift ``k`` (``1.0`` = no shift).

    Returns:
        A dense ``(V, V)`` PPMI matrix (float).
    """
    import numpy as np

    C = np.asarray(cooccurrence, dtype=float)
    total = C.sum()
    if total <= 0:
        return np.zeros_like(C)

    P = C / total
    row_marg = P.sum(axis=1)  # P(a)
    col_counts = C.sum(axis=0)
    col_alpha = np.power(col_counts, alpha)
    col_sum = col_alpha.sum()
    col_marg = col_alpha / col_sum if col_sum > 0 else np.zeros_like(col_alpha)

    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.outer(row_marg, col_marg)
        ratio = np.where(denom > 0, P / denom, 0.0)
        pmi = np.where(ratio > 0, np.log(ratio), 0.0)
    pmi = pmi - np.log(max(k_shift, 1e-12))
    return np.maximum(pmi, 0.0)


def factorize_backbone(ppmi_matrix: Any, dim: int, random_state: int = _RANDOM_STATE) -> Any:
    """Factorize a PPMI matrix into a ``(V, d)`` embedding backbone ``W`` (pure).

    Runs a single ``TruncatedSVD(n_components=d, random_state=random_state)`` and
    returns ``W = U * Sigma`` (the ``fit_transform`` output), the static embedding
    of each vocab row. Deterministic for the pinned seed (stable fingerprints).
    ``d`` is clamped below ``V`` (SVD requires ``n_components < n_features``); a
    degenerate matrix yields an all-zero ``(V, d)`` backbone.

    Args:
        ppmi_matrix: The dense ``(V, V)`` PPMI matrix.
        dim: Target embedding dimensionality ``d`` (from the budget).
        random_state: Pinned SVD seed.

    Returns:
        A dense ``(V, d_eff)`` float numpy embedding matrix ``W``.
    """
    import numpy as np
    from sklearn.decomposition import TruncatedSVD

    M = np.asarray(ppmi_matrix, dtype=float)
    V = M.shape[0]
    d = max(1, min(int(dim), max(1, V - 1)))
    if V < 2 or not np.any(M):
        return np.zeros((V, d), dtype=float)
    svd = TruncatedSVD(n_components=d, random_state=random_state)
    return svd.fit_transform(M)


def backbone_fingerprint(W: Any, vocab: dict) -> str:
    """Content fingerprint of a backbone (``W`` + vocab) -- deterministic (pure).

    A SHA-256 over the rounded embedding matrix bytes plus the sorted vocab token
    list, so two ``pretrain`` calls on the same fixture produce the SAME
    fingerprint (the moat-lineage determinism the conformance suite asserts).

    Args:
        W: The ``(V, d)`` backbone embedding matrix.
        vocab: The vocab dict the backbone was trained against.

    Returns:
        ``"sha256:<hexdigest>"``.
    """
    import numpy as np

    hasher = hashlib.sha256()
    arr = np.asarray(W, dtype=float).round(6)
    hasher.update(np.ascontiguousarray(arr).tobytes())
    hasher.update(str(arr.shape).encode("utf-8"))
    token_items = sorted(vocab.get("token_to_id", {}).items(), key=lambda kv: kv[1])
    hasher.update(str([t for t, _ in token_items]).encode("utf-8"))
    return f"sha256:{hasher.hexdigest()}"


def pool_embeddings(train: Any, W: Any, vocab: dict) -> Any:
    """Pool per-account sequence embeddings into a fixed-width feature frame (pure).

    For each account/sequence, the token ids index rows of ``W``; the rows are
    pooled into a fixed-width vector by concatenating ``mean``, ``max``, and the
    ``last`` token's embedding (a cheap CLS-style readout). The result is a
    DataFrame with ``emb_*`` feature columns indexed by account (or row index when
    no account grouping is configured), ready to carry an as-of label column.

    Args:
        train: The DataFrame to embed.
        W: The ``(V, d)`` backbone embedding matrix.
        vocab: The vocab dict the backbone was trained against.

    Returns:
        A pandas DataFrame of pooled ``emb_*`` features, indexed by account key
        (or the source row index when there is no account column).
    """
    import numpy as np
    import pandas as pd

    W = np.asarray(W, dtype=float)
    d = W.shape[1] if W.ndim == 2 and W.shape[0] > 0 else 0
    account = vocab.get("account")
    time_col = vocab.get("time")

    def _pool(ids: list[int]) -> np.ndarray:
        if d == 0 or not ids:
            return np.zeros(3 * max(d, 1), dtype=float)
        valid = [i for i in ids if 0 <= i < W.shape[0]]
        if not valid:
            return np.zeros(3 * d, dtype=float)
        rows = W[valid]
        return np.concatenate([rows.mean(axis=0), rows.max(axis=0), W[valid[-1]]])

    keys: list[Any] = []
    vectors: list[np.ndarray] = []

    if account and account in train.columns:
        frame = train
        if time_col and time_col in train.columns:
            frame = train.sort_values(by=[account, time_col], kind="stable")
        else:
            frame = train.sort_values(by=[account], kind="stable")
        for key, group in frame.groupby(account, sort=True):
            seq: list[int] = []
            for _idx, row in group.iterrows():
                seq.extend(_row_tokens(row, vocab))
            if len(seq) > _MAX_SEQ_LEN:
                seq = seq[-_MAX_SEQ_LEN:]
            keys.append(key)
            vectors.append(_pool(seq))
    else:
        for idx, row in train.iterrows():
            keys.append(idx)
            vectors.append(_pool(_row_tokens(row, vocab)))

    width = 3 * max(d, 1)
    cols = [f"emb_{i}" for i in range(width)]
    matrix = np.vstack(vectors) if vectors else np.zeros((0, width), dtype=float)
    return pd.DataFrame(matrix, columns=cols, index=pd.Index(keys, name=account or None))


def build_embedding_dataset(train: Any, W: Any, vocab: dict, target: str | None) -> dict:
    """Assemble an ``IngestDataset``-shaped embedding data object (pure).

    Pools the backbone embeddings (:func:`pool_embeddings`), joins the per-account
    as-of label (the LAST observed target value for the account -- leakage-safe,
    never a random row), and carves a TEMPORAL train/test split (most-recent rows
    sealed as the test split). The returned dict has exactly the ``IngestDataset``
    artifact shape (``train`` / ``test`` / ``schema`` / ``fingerprint``) so the
    produced pathspec round-trips through :func:`loom.dataio.materialize_dataset`
    unchanged -- the embeddings ref is a first-class ``dataset_ref``.

    Args:
        train: The DataFrame to embed (carrying the label column, if any).
        W: The ``(V, d)`` backbone embedding matrix.
        vocab: The vocab dict the backbone was trained against.
        target: The label column name to attach (``None`` => features only).

    Returns:
        ``{"train", "test", "schema", "fingerprint"}`` -- the IngestDataset shape.
    """
    import numpy as np
    import pandas as pd

    features = pool_embeddings(train, W, vocab)
    account = vocab.get("account")

    frame = features.copy()
    if target and target in train.columns:
        if account and account in train.columns:
            # As-of label: the LAST observed target per account (leakage-safe).
            label = train.groupby(account, sort=True)[target].last()
            frame[target] = label.reindex(frame.index).to_numpy()
        else:
            frame[target] = train[target].reindex(frame.index).to_numpy()

    frame = frame.reset_index(drop=True)

    # Temporal split: seal the most-recent tail as the test split.
    n = len(frame)
    n_test = int(n * _DEFAULT_TEST_FRACTION)
    if n_test > 0 and n - n_test > 0:
        train_df = frame.iloc[: n - n_test].reset_index(drop=True)
        test_df = frame.iloc[n - n_test :].reset_index(drop=True)
    else:
        train_df = frame
        test_df = None

    schema = {
        "columns": [str(c) for c in frame.columns],
        "dtypes": {str(c): str(frame[c].dtype) for c in frame.columns},
        "nrows": int(len(train_df)),
        "target": str(target) if (target and target in frame.columns) else None,
    }
    return {
        "train": train_df,
        "test": test_df,
        "schema": schema,
        "fingerprint": _frame_fingerprint(train_df, schema),
    }


def _frame_fingerprint(train: Any, schema: dict) -> str:
    """Content fingerprint of an embedding frame (mirrors ``features.fingerprint_frame``)."""
    import pandas as pd

    hasher = hashlib.sha256()
    hasher.update(str(schema.get("columns")).encode("utf-8"))
    hasher.update(str(schema.get("dtypes")).encode("utf-8"))
    hasher.update(str(schema.get("nrows")).encode("utf-8"))
    hasher.update(str(schema.get("target")).encode("utf-8"))
    try:
        row_hashes = pd.util.hash_pandas_object(train, index=False)
        hasher.update(str(int(row_hashes.sum())).encode("utf-8"))
    except Exception:  # pragma: no cover - unhashable cell content
        hasher.update(str(train.shape).encode("utf-8"))
    return f"sha256:{hasher.hexdigest()}"


def fit_head(embedding_train: Any, target: str, recipe: dict | None = None, random_state: int = _RANDOM_STATE):
    """Fit a CHEAP sklearn head on a frozen-backbone embedding frame (pure).

    Freezes the backbone (the embeddings are precomputed) and fits a cheap head on
    the pooled features, REUSING :func:`flows.validate._make_estimator` so the head
    is the same gradient-boosted-trees estimator the validate baseline uses (one
    scorer for every backend). ``recipe`` is the search surface AIDE tree-searches
    (which estimator / regularization / pooled features); v0.1 honors the task type
    and seed, leaving richer recipe knobs as a forward extension.

    Args:
        embedding_train: The embedding train DataFrame (``emb_*`` + target).
        target: The label column name.
        recipe: The head recipe AIDE may tree-search (optional).
        random_state: Pinned estimator seed.

    Returns:
        ``(model, task_type, feature_cols)`` -- the fitted head, its inferred task
        type, and the feature column names it was fit on.
    """
    from flows.validate import (
        _encode_features,
        _encode_target,
        _infer_task_type,
        _make_estimator,
    )

    feature_cols = [c for c in embedding_train.columns if str(c) != str(target)]
    X = _encode_features(embedding_train[feature_cols])
    y_raw = embedding_train[target]
    task_type = _infer_task_type(y_raw)
    y, _labels = _encode_target(y_raw, task_type)

    model = _make_estimator(task_type, random_state)
    model.fit(X, y)
    return model, task_type, [str(c) for c in feature_cols]


def score_holdout(model: Any, task_type: str, embedding_holdout: Any, target: str, metric: str) -> dict:
    """Score a fitted head on a sealed holdout frame (pure).

    REUSES :func:`flows.validate._score` for the standard task metric, and ADDS
    ``sklearn.metrics.average_precision_score`` for ``metric="fraud-pr-auc"`` (the
    PR-AUC validate currently lacks) -- a small adapter-equal adaptation the
    ``nemo`` adapter wires the same way. The holdout is the SEALED temporal split
    with as-of labels (leakage-safe), so the reported number is trustworthy.

    Args:
        model: The fitted head from :func:`fit_head`.
        task_type: The inferred task type.
        embedding_holdout: The sealed-holdout embedding frame (``emb_*`` + target).
        target: The label column name.
        metric: The requested metric name.

    Returns:
        ``{"value": float|None, "n": int, "task_type": str}``.
    """
    import numpy as np
    from sklearn.metrics import average_precision_score

    from flows.validate import _encode_features, _encode_target, _positive_proba, _score

    if embedding_holdout is None or len(embedding_holdout) == 0:
        return {"value": None, "n": 0, "task_type": task_type}

    feature_cols = [c for c in embedding_holdout.columns if str(c) != str(target)]
    X = _encode_features(embedding_holdout[feature_cols])
    y, _labels = _encode_target(embedding_holdout[target], task_type)

    if metric == "fraud-pr-auc" and task_type == "binary":
        if len(np.unique(y)) < 2:
            value: float | None = None
        else:
            value = float(average_precision_score(y, _positive_proba(model, X)))
    else:
        value = float(_score(model, X, y, task_type))
    return {"value": value, "n": int(len(embedding_holdout)), "task_type": task_type}


def raw_baseline_score(
    raw_train: Any, raw_holdout: Any, target: str, metric: str, random_state: int = _RANDOM_STATE
) -> float | None:
    """Fit + score the RAW-feature baseline the SAME way, for the lift number (pure).

    Fits :func:`flows.validate._fit_baseline` on the raw (un-embedded) features and
    scores it on the raw holdout with the same metric as
    :func:`score_holdout`, so ``Scores.detail["lift"] = embeddings - baseline_raw``
    is a like-for-like comparison. On a planted-sequential fixture the embeddings
    beat this baseline (the lift the conformance suite asserts).

    Args:
        raw_train: The raw train DataFrame (features + target).
        raw_holdout: The raw holdout DataFrame (features + target).
        target: The label column name.
        metric: The requested metric name.
        random_state: Pinned estimator seed.

    Returns:
        The baseline metric value, or ``None`` when it cannot be computed.
    """
    import numpy as np
    from sklearn.metrics import average_precision_score

    from flows.validate import (
        _encode_features,
        _encode_target,
        _fit_baseline,
        _infer_task_type,
        _positive_proba,
        _score,
    )

    if raw_holdout is None or len(raw_holdout) == 0 or target not in raw_train.columns:
        return None

    feat_cols = [c for c in raw_train.columns if str(c) != str(target)]
    task_type = _infer_task_type(raw_train[target])
    X_tr = _encode_features(raw_train[feat_cols])
    y_tr, _ = _encode_target(raw_train[target], task_type)
    model = _fit_baseline(X_tr, y_tr, task_type, random_state)

    X_ho = _encode_features(raw_holdout[feat_cols])
    y_ho, _ = _encode_target(raw_holdout[target], task_type)
    if metric == "fraud-pr-auc" and task_type == "binary":
        if len(np.unique(y_ho)) < 2:
            return None
        return float(average_precision_score(y_ho, _positive_proba(model, X_ho)))
    return float(_score(model, X_ho, y_ho, task_type))


def _maybe_torch_backbone(*_args: Any, **_kwargs: Any):
    """Optional torch GRU fidelity hook (design §3.8) -- a guarded lazy stub.

    The numpy PPMI-SVD path is the ONLY default. This stub exists so the optional
    ``model-local = ["torch>=2.0"]`` extra (a real next-event/masked-field/
    contrastive GRU producing a numpy ``state_dict`` artifact in Metaflow, never an
    on-disk weight file) can be grafted later without changing the seam: it returns
    ``None`` unless torch is importable AND explicitly opted in via the
    ``LOOM_MODEL_LOCAL_TORCH`` env flag, so on ``ImportError`` (CI, torch absent)
    the caller transparently falls back to the numpy backbone and the
    ``ArtifactRef``/``Scores`` shapes stay byte-identical across both backends.

    Returns:
        ``None`` (always, in v0.1) -- the numpy path is taken by the caller.
    """
    import os

    if os.environ.get("LOOM_MODEL_LOCAL_TORCH", "").strip().lower() not in {"1", "true", "yes"}:
        return None
    try:  # pragma: no cover - optional extra, never the default/CI path
        import torch  # noqa: F401
    except Exception:
        return None
    # The real torch GRU is a forward graft; v0.1 keeps the numpy path authoritative.
    return None  # pragma: no cover


# ---------------------------------------------------------------------------
# The provider: thin wrappers that materialize a ref, call the pure helpers, and
# return ArtifactRef/Scores. When run inside the TrainFlow step, the flow writes
# the produced data object's artifacts on ``self`` (the IngestDataset/FeaturesFlow
# "the pathspec IS the data object" pattern). Standalone (no Metaflow), the
# methods still compute and return the typed result over a materialized ref.
# ---------------------------------------------------------------------------


@register_model_builder("local")
class LocalModelBuilderProvider(ModelBuilderProvider):
    """CPU PPMI+TruncatedSVD model builder (registry name ``"local"``).

    The torch-free conformance/CI default path: deterministic, sub-2s, zero new
    heavy deps. Constructed uniformly from a :class:`~loom.config.LoomConfig` (like
    every other Loom provider). All maths is in the module-level pure helpers; the
    methods here resolve a ref via the Client API, call a helper, and return a
    typed :class:`~loom.types.ArtifactRef` / :class:`~loom.types.Scores`.

    Attributes:
        name: Registry name (``"local"``).
        config: The active :class:`~loom.config.LoomConfig`.
    """

    name = "local"

    #: REQUIRED §3.7 honesty note: a CPU stand-in must not over-sell itself as a
    #: contextual transformer.
    _NOTE = "CPU PPMI+SVD stand-in; static embeddings, not a contextual transformer"

    def __init__(self, config: LoomConfig) -> None:
        """Initialize from a Loom config (no secret material is read).

        Args:
            config: The active configuration.
        """
        self.config = config

    def manifest(self) -> CapabilityManifest:
        """Return the ``local`` backend capability manifest (design §3.7).

        ``pretrain`` is typed ``launch-and-track`` (heavy though cheap) so AIDE
        never tree-searches the backbone; the other capabilities are ``searchable``.
        ``serve`` is supported but batch-only -- ``serve(online)`` is refused at
        call time and the note says so (don't over-sell).
        """
        return CapabilityManifest(
            backend="local",
            capabilities={
                "tokenize": Capability("tokenize", "searchable", True),
                "pretrain": Capability("pretrain", "launch-and-track", True, self._NOTE),
                "finetune": Capability("finetune", "searchable", True),
                "embed": Capability("embed", "searchable", True),
                "evaluate": Capability("evaluate", "searchable", True),
                "serve": Capability(
                    "serve", "searchable", True, "batch only; online unsupported on CPU"
                ),
            },
        )

    # -- the six capabilities (thin wrappers over the pure helpers) -----------

    def tokenize(self, sequences_ref: str, scheme: dict) -> ArtifactRef:
        """Build a deterministic vocabulary over the sequences. Mode: ``searchable``.

        Materializes ``sequences_ref`` via the Client API, resolves the scheme
        against the data object's schema, and builds the shared vocab
        (:func:`build_vocab`). The flow persists the vocab + resolved scheme as
        artifacts on ``self``; here we return the typed ref + summary.
        """
        train, schema = self._materialize(sequences_ref)
        resolved = resolve_scheme(scheme, schema)
        vocab = build_vocab(train, resolved)
        return ArtifactRef(
            pathspec=self._pathspec_for("Tokenize", sequences_ref),
            kind="tokenizer",
            summary={
                "vocab_size": int(vocab["size"]),
                "fields": list(resolved["fields"]),
                "scheme": resolved,
            },
        )

    def pretrain(self, sequences_ref: str, objective: str, budget: str) -> ArtifactRef:
        """Pretrain the embedding backbone ``W``. Mode: ``launch-and-track``.

        Validates ``objective`` / ``budget`` against the seam frozensets, builds
        the vocab + objective-selected co-occurrence, lifts to PPMI (alpha=0.75),
        and factorizes one deterministic ``TruncatedSVD(random_state=0)`` into the
        ``(V, d)`` backbone. The backbone matrix IS the artifact; the flow writes
        ``self.backbone`` / ``self.backbone_vocab`` / ``self.fingerprint``.
        """
        if objective not in OBJECTIVES:
            return ArtifactRef(
                pathspec=None,
                kind="backbone",
                error=(
                    f"objective {objective!r} is not a Loom objective; expected one of "
                    f"{sorted(OBJECTIVES)}."
                ),
            )
        if budget not in BUDGETS:
            return ArtifactRef(
                pathspec=None,
                kind="backbone",
                error=(
                    f"budget {budget!r} is not a Loom budget; expected one of "
                    f"{sorted(BUDGETS)}."
                ),
            )

        train, schema = self._materialize(sequences_ref)
        resolved = resolve_scheme(None, schema)
        vocab = build_vocab(train, resolved)
        sequences = encode_sequences(train, vocab)

        dim = _BUDGET_DIMS[budget]
        # Optional torch fidelity hook (guarded; returns None on the default path).
        W = _maybe_torch_backbone(sequences, vocab, objective, dim)
        if W is None:
            C = build_cooccurrence(sequences, vocab["size"], objective)
            M = ppmi(C)
            W = factorize_backbone(M, dim, random_state=_RANDOM_STATE)

        fingerprint = backbone_fingerprint(W, vocab)
        return ArtifactRef(
            pathspec=self._pathspec_for("Pretrain", sequences_ref),
            kind="backbone",
            summary={
                "objective": objective,
                "budget": budget,
                "dim": int(W.shape[1]) if getattr(W, "ndim", 0) == 2 else int(dim),
                "vocab_size": int(vocab["size"]),
                "fingerprint": fingerprint,
                "note": self._NOTE,
            },
        )

    def embed(self, backbone_ref: str, data_ref: str) -> ArtifactRef:
        """Embed data through the frozen backbone. Mode: ``searchable``.

        Loads ``W`` + vocab from ``backbone_ref`` (Client API), materializes
        ``data_ref``, pools per-account embeddings, and assembles an
        ``IngestDataset``-shaped frame (:func:`build_embedding_dataset`) so the
        produced pathspec is a first-class ``dataset_ref`` any ``/loom-*`` verb
        consumes. The flow writes ``self.train`` / ``self.test`` / ``self.schema``
        / ``self.fingerprint``.
        """
        train, schema = self._materialize(data_ref)
        W, vocab = self._load_backbone(backbone_ref, train, schema)
        target = schema.get("target")
        dataset = build_embedding_dataset(train, W, vocab, target)
        test = dataset.get("test")
        return ArtifactRef(
            pathspec=self._pathspec_for("Embed", data_ref),
            kind="embeddings",
            summary={
                "dim_out": int(len(dataset["schema"]["columns"])),
                "n_rows": int(dataset["schema"]["nrows"]),
                "n_test": int(len(test)) if test is not None else 0,
                "target": dataset["schema"]["target"],
                "fingerprint": dataset["fingerprint"],
            },
        )

    def finetune(self, backbone_ref: str, task_ref: str, recipe: dict) -> ArtifactRef:
        """Fit a cheap sklearn head on the frozen backbone. Mode: ``searchable``.

        Embeds ``task_ref`` through the frozen ``backbone_ref`` and fits a cheap
        head (:func:`fit_head`, reusing ``validate._make_estimator``). ``recipe`` is
        the search surface AIDE tree-searches. The flow writes ``self.head`` /
        ``self.lineage``.
        """
        train, schema = self._materialize(task_ref)
        W, vocab = self._load_backbone(backbone_ref, train, schema)
        target = schema.get("target")
        if not target:
            return ArtifactRef(
                pathspec=None,
                kind="model",
                error=(
                    "finetune requires a target column in the task data object's "
                    "schema (a head has nothing to fit toward without one)."
                ),
            )
        dataset = build_embedding_dataset(train, W, vocab, target)
        model, task_type, feature_cols = fit_head(dataset["train"], target, recipe)
        return ArtifactRef(
            pathspec=self._pathspec_for("Finetune", task_ref),
            kind="model",
            summary={
                "backbone_ref": backbone_ref,
                "head": type(model).__name__,
                "task_type": task_type,
                "n_features": len(feature_cols),
                "task": target,
                "recipe": dict(recipe or {}),
            },
        )

    def evaluate(self, model_ref: str, holdout_ref: str, metric: str) -> Scores:
        """Score the model on a sealed temporal holdout vs the raw baseline. Mode: ``searchable``.

        Embeds the SEALED holdout through the frozen backbone, fits the head on the
        embedding train split, scores it (:func:`score_holdout`, adding PR-AUC for
        ``fraud-pr-auc``), and fits+scores the RAW baseline the same way
        (:func:`raw_baseline_score`) for the lift number. Returns
        :class:`~loom.types.Scores` carrying ``baseline_raw`` / ``lift`` /
        ``n_holdout``.
        """
        train, schema = self._materialize(holdout_ref)
        W, vocab = self._load_backbone(model_ref, train, schema)
        target = schema.get("target")
        if not target:
            return Scores(metric=metric, value=None, detail={"error": "no target in holdout schema"})

        dataset = build_embedding_dataset(train, W, vocab, target)
        emb_train, emb_test = dataset["train"], dataset["test"]
        model, task_type, _cols = fit_head(emb_train, target, None)
        scored = score_holdout(model, task_type, emb_test, target, metric)

        # Raw-feature baseline scored the SAME way, on a temporal split of the raw
        # frame, for a like-for-like lift.
        raw_train, raw_test = self._temporal_split(train)
        baseline = raw_baseline_score(raw_train, raw_test, target, metric)
        value = scored["value"]
        lift = (value - baseline) if (value is not None and baseline is not None) else None
        return Scores(
            metric=metric,
            value=value,
            detail={
                "baseline_raw": baseline,
                "lift": lift,
                "n_holdout": int(scored["n"]),
                "task_type": task_type,
            },
        )

    def serve(self, model_ref: str, mode: str) -> ArtifactRef:
        """Serve a model. Mode: ``searchable`` (batch only in v0.1).

        ``mode="batch"`` is a frozen-embedding batch-scoring step (the flow writes
        the scored frame as an artifact); ``mode="online"`` is refused UP FRONT with
        an actionable message (``local`` declares ``serve(online).supported`` via
        this refusal -- never a deep crash). ``mode`` is validated against the seam
        frozenset first.
        """
        if mode not in MODES:
            return ArtifactRef(
                pathspec=None,
                kind="endpoint",
                error=f"serve mode {mode!r} is not a Loom serving mode; expected one of {sorted(MODES)}.",
            )
        if mode == "online":
            raise NotImplementedError(
                "the `local` adapter serves batch only; online serving needs a GPU "
                "target (use `model-builder nemo` with a configured gpu_target, or "
                "serve in batch mode on CPU)."
            )
        # Batch scoring: produce a scored frame as an endpoint artifact.
        train, schema = self._materialize(model_ref)
        target = schema.get("target")
        return ArtifactRef(
            pathspec=self._pathspec_for("Serve", model_ref),
            kind="endpoint",
            summary={"mode": "batch", "n_rows": int(len(train)), "target": target},
        )

    # -- internal Client-API helpers (lazy metaflow; never touched in unit tests) --

    def _materialize(self, ref: str):
        """Resolve a ref to ``(train_df, schema)`` via the Client API only.

        Lazy ``metaflow`` import (mirrors :mod:`loom.dataio`): reads the run's
        ``train`` artifact and ``schema`` dict, never touching the datastore.
        """
        from loom.dataio import dataset_schema, resolve_run

        run = resolve_run(ref)
        data = run.data
        train = getattr(data, "train", None)
        if train is None:
            raise ValueError(
                f"ref {ref!r} has no 'train' artifact; expected an IngestDataset-shaped "
                "data object (run `loom ingest` / a prior train step to produce one)."
            )
        try:
            schema = dataset_schema(ref)
        except Exception:  # pragma: no cover - schema read edge case
            schema = {}
        return train, schema

    def _load_backbone(self, backbone_ref: str, train: Any, schema: dict):
        """Load ``(W, vocab)`` from a backbone ref, or recompute inline as a fallback.

        Reads the ``backbone`` + ``backbone_vocab`` artifacts a prior ``pretrain``
        run wrote (Client API). When the ref is not a backbone run (e.g. a raw
        dataset ref handed straight to ``embed``), it recomputes a default backbone
        inline so the capability still produces a typed result.
        """
        try:
            from loom.dataio import resolve_run

            run = resolve_run(backbone_ref)
            data = run.data
            W = getattr(data, "backbone", None)
            vocab = getattr(data, "backbone_vocab", None)
            if W is not None and vocab is not None:
                return W, vocab
        except Exception:  # pragma: no cover - non-backbone ref / unreadable run
            pass
        # Fallback: recompute a default backbone inline from the materialized train.
        resolved = resolve_scheme(None, schema)
        vocab = build_vocab(train, resolved)
        sequences = encode_sequences(train, vocab)
        C = build_cooccurrence(sequences, vocab["size"], "next-event")
        W = factorize_backbone(ppmi(C), _BUDGET_DIMS["probe"], random_state=_RANDOM_STATE)
        return W, vocab

    @staticmethod
    def _temporal_split(frame: Any):
        """Carve a most-recent-tail temporal test split off a raw frame (no leakage)."""
        n = len(frame)
        n_test = int(n * _DEFAULT_TEST_FRACTION)
        if n_test > 0 and n - n_test > 0:
            return (
                frame.iloc[: n - n_test].reset_index(drop=True),
                frame.iloc[n - n_test :].reset_index(drop=True),
            )
        return frame, None

    @staticmethod
    def _pathspec_for(flow_name: str, source_ref: str) -> str:
        """Return a ``<FlowName>/<run_id>`` pathspec mirroring the source ref's run id.

        Standalone (outside a Metaflow run) the produced artifact has no real run
        yet; the ``TrainFlow`` step supplies the true ``current.pathspec``. To keep
        every returned ``ArtifactRef.pathspec`` a valid ``<FlowName>/<run_id>``
        (the conformance suite's :func:`loom.dataio.resolve_run` shape check), we
        echo the source ref's run id under the capability's flow name -- never a
        file path, an object-store URI, or an on-disk checkpoint literal.
        """
        ref = (source_ref or "").strip()
        parts = [p for p in ref.split("/") if p]
        run_id = parts[-1] if parts else "0"
        return f"{flow_name}/{run_id}"


__all__ = [
    "LocalModelBuilderProvider",
    "resolve_scheme",
    "build_vocab",
    "encode_sequences",
    "build_cooccurrence",
    "ppmi",
    "factorize_backbone",
    "backbone_fingerprint",
    "pool_embeddings",
    "build_embedding_dataset",
    "fit_head",
    "score_holdout",
    "raw_baseline_score",
]
