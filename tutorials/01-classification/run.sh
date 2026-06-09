#!/usr/bin/env bash
# =============================================================================
# tutorials/01-classification/run.sh
#
# "Your first model: tabular classification" -- the tested, KEYLESS companion to
# this tutorial's README.md. It walks the core Loom lifecycle on a clean, labeled
# binary-classification table:
#
#     generate synthetic data  ->  ingest (UNIQUE name)  ->  eda  ->  features
#       ->  validate (CV + sealed holdout, emits a VERDICT)  ->  report (model-card)
#
# Every verb runs with --json and the outcome is ASSERTED inline; ANY regression
# exits NONZERO. It is fully self-contained: it sources the local cluster env (so
# the keyless verbs talk to the live local Metaflow + minio datastore), it
# generates its own deterministic data inline (no downloads), and it ingests under
# a UNIQUE dataset name so concurrent / repeat runs never collide.
#
# It NEVER runs a model-using verb (`run` / `optimize` / the agentic NL flow) --
# those need an LLM key and cost money; the README narrates them as the next step.
#
# Run it:
#     cd /Users/anub/Work/Loom && bash tutorials/01-classification/run.sh
# =============================================================================
set -euo pipefail

# --- Make the keyless verbs self-contained -----------------------------------
# Source the local cluster env (minio endpoint + Metaflow datastore profile) so
# this script works on its own, exactly as the examples bed does.
if [ -f /tmp/loom-cluster-env.sh ]; then
  # shellcheck disable=SC1091
  source /tmp/loom-cluster-env.sh
fi

# --- LOOM: the verb entrypoint (venv module form; no PATH needed) ------------
LOOM="${LOOM:-/Users/anub/Work/Loom/.venv/bin/python -m loom}"
PY="${PY:-/Users/anub/Work/Loom/.venv/bin/python}"

# --- Where this tutorial lives + a private scratch dir (cleaned on exit) ------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/loom-tutorial-01-XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

# --- A UNIQUE dataset name so concurrent / repeat runs never collide ---------
RUN_TAG="tut01-$$-${RANDOM}-$(date +%s)"
DATASET_NAME="tut-${RUN_TAG}"

# =============================================================================
# _assert_json -- the ONE assert helper (no jq dependency; mirrors examples bed).
#
# Usage: _assert_json "<json-string>" "<python-bool-expr-over-`o`>" "<message>"
# Parses the JSON envelope into `o` (a dict) and evaluates the boolean expr. On a
# falsy result (or a parse error) it prints the message + the offending JSON to
# stderr and exits NONZERO. Asserts on the stable envelope fields:
#   o['status'], o['VERDICT'], o['pathspec'], o['summary'][...].
# =============================================================================
_assert_json() {
  local json="$1" expr="$2" msg="$3"
  printf '%s' "$json" | "$PY" -c '
import sys, json
expr = sys.argv[1]
msg = sys.argv[2]
raw = sys.stdin.read()
try:
    o = json.loads(raw)
except Exception as e:  # noqa: BLE001
    sys.stderr.write(f"ASSERT PARSE FAIL: {msg}\n  expr: {expr}\n  err: {e}\n  json: {raw!r}\n")
    sys.exit(1)
_ns = {
    "bool": bool, "len": len, "any": any, "all": all, "str": str, "int": int,
    "float": float, "abs": abs, "min": min, "max": max, "sum": sum,
    "sorted": sorted, "set": set, "list": list, "isinstance": isinstance,
}
try:
    ok = bool(eval(expr, {"__builtins__": {}}, {"o": o, **_ns}))
except Exception as e:  # noqa: BLE001
    sys.stderr.write(f"ASSERT EVAL FAIL: {msg}\n  expr: {expr}\n  err: {e}\n  json: {json.dumps(o)}\n")
    sys.exit(1)
if not ok:
    sys.stderr.write(f"ASSERT FAIL: {msg}\n  expr: {expr}\n  json: {json.dumps(o)}\n")
    sys.exit(1)
sys.stderr.write(f"  ok: {msg}\n")
' "$expr" "$msg"
}

# _field -- pull one top-level JSON field to stdout (for capturing pathspecs).
_field() {
  local json="$1" key="$2"
  printf '%s' "$json" | "$PY" -c "import sys,json; print(json.load(sys.stdin)[\"$key\"])"
}

echo "== tutorial: 01-classification  dataset=${DATASET_NAME}  work=${WORK}"

# =============================================================================
# (1) GENERATE the deterministic synthetic data inline (no downloads).
#     A clean, learnable binary-classification table: an `id`, 12 float
#     `feature_*` columns (some carry real signal, some are noise), and an integer
#     `target` in {0, 1} with a mild ~60/40 imbalance so ROC-AUC stays meaningful.
#     Seeded -> a repeat run is byte-identical. No planted leak (clean data).
# =============================================================================
"$PY" - "$WORK/train.csv" <<'PYEOF'
import sys
from pathlib import Path

import pandas as pd
from sklearn.datasets import make_classification

out = Path(sys.argv[1])
out.parent.mkdir(parents=True, exist_ok=True)

N_ROWS = 1500
N_FEATURES = 12
N_INFORMATIVE = 5

X, y = make_classification(
    n_samples=N_ROWS,
    n_features=N_FEATURES,
    n_informative=N_INFORMATIVE,
    n_redundant=3,
    n_classes=2,
    weights=[0.6, 0.4],   # mild imbalance -> ROC-AUC is meaningful
    flip_y=0.02,          # a little label noise so the baseline is non-trivial
    random_state=0,       # fixed seed -> deterministic, reproducible
)

frame = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(N_FEATURES)])
frame.insert(0, "id", range(len(frame)))
frame["target"] = y.astype(int)
frame.to_csv(out, index=False)
print(f"wrote {len(frame)} rows x {frame.shape[1]} cols -> {out}")
PYEOF

# =============================================================================
# (2) INGEST it under the UNIQUE name -> capture the data-object pathspec.
#     This is the one external->Metaflow boundary; everything downstream consumes
#     the returned `IngestDataset/<id>` pathspec.
# =============================================================================
INGEST_JSON="$($LOOM ingest --source "$WORK" --name "$DATASET_NAME" --json)"
_assert_json "$INGEST_JSON" "o['status'] == 'ok'" "ingest succeeded"
_assert_json "$INGEST_JSON" "bool(o['pathspec'])" "ingest produced a dataset pathspec"
DATASET="$(_field "$INGEST_JSON" pathspec)"
echo "   dataset_ref = ${DATASET}"

# Confirm it really landed in the catalog under our unique name.
DATASETS_JSON="$($LOOM datasets --json)"
_assert_json "$DATASETS_JSON" "o['status'] == 'ok'" "datasets listing succeeded"
_assert_json "$DATASETS_JSON" \
  "any(d['pathspec'] == '${DATASET}' for d in o['summary']['datasets'])" \
  "the ingested dataset appears in the catalog"
_assert_json "$DATASETS_JSON" \
  "any(d['name'] == '${DATASET_NAME}' for d in o['summary']['datasets'])" \
  "the catalog entry carries our unique dataset name"

# =============================================================================
# (3a) EDA --target: read-only profile. shape + target balance + leakage scan.
#      The data is CLEAN, so leakage_flags MUST be empty. EDA carries no VERDICT
#      (it is read-only); we keep its run pathspec so `features` can compose on it.
# =============================================================================
EDA_JSON="$($LOOM eda --dataset "$DATASET" --target target --json)"
_assert_json "$EDA_JSON" "o['status'] == 'ok'" "eda succeeded"
_assert_json "$EDA_JSON" "bool(o['pathspec'])" "eda produced an EdaFlow run pathspec"
_assert_json "$EDA_JSON" "o['summary'].get('nrows', 0) > 0" "eda saw rows (nrows)"
_assert_json "$EDA_JSON" "o['summary'].get('ncols', 0) > 0" "eda saw columns (ncols)"
_assert_json "$EDA_JSON" "o['summary'].get('target') == 'target'" "eda resolved the target column"
_assert_json "$EDA_JSON" "bool(o['summary'].get('target_balance'))" "eda reported target balance"
_assert_json "$EDA_JSON" "len(o['summary'].get('leakage_flags') or []) == 0" \
  "clean data -> eda flags NO leakage"
EDA_RUN="$(_field "$EDA_JSON" pathspec)"
echo "   eda_run = ${EDA_RUN}"

# =============================================================================
# (3b) FEATURES --from <eda-run>: engineer a NEW, leakage-aware feature object.
#      Composing on the EDA run makes feature-building leakage-aware: it reads the
#      upstream profile's leakage_flags and refuses/drops anything flagged before
#      building. On clean data nothing is dropped, it BUILDs, and it writes a brand
#      -new FeaturesFlow/<id> pathspec (distinct from the source) for validate.
# =============================================================================
FEATURES_JSON="$($LOOM features --dataset "$DATASET" --target target --from "$EDA_RUN" --json)"
_assert_json "$FEATURES_JSON" "o['status'] == 'ok'" "features succeeded"
_assert_json "$FEATURES_JSON" "o['VERDICT'] == 'BUILT'" "features VERDICT is BUILT"
_assert_json "$FEATURES_JSON" "bool(o['pathspec']) and o['pathspec'] != '${DATASET}'" \
  "features produced a NEW data-object pathspec (distinct from the source)"
_assert_json "$FEATURES_JSON" "o['summary'].get('n_features_after', 0) > 0" \
  "features summary reports n_features_after"
_assert_json "$FEATURES_JSON" "o['summary'].get('refused_leakage') == False" \
  "features did not refuse on leakage (clean data)"
FEATURES="$(_field "$FEATURES_JSON" pathspec)"
echo "   features_ref = ${FEATURES}"

# =============================================================================
# (3c) VALIDATE: fit a gradient-boosted-trees baseline on the engineered features
#      and evaluate it rigorously -- stratified cross-validation PLUS a sealed
#      holdout the CV never saw. Emits a VERDICT (PASS / REVIEW). We keep its run
#      pathspec to feed the report's model-card.
# =============================================================================
VALIDATE_JSON="$($LOOM validate --dataset "$FEATURES" --target target --json)"
_assert_json "$VALIDATE_JSON" "o['status'] == 'ok'" "validate succeeded"
_assert_json "$VALIDATE_JSON" "o['VERDICT'] in ('PASS', 'REVIEW')" \
  "validate emitted a VERDICT (PASS or REVIEW)"
_assert_json "$VALIDATE_JSON" "(o['summary'].get('cv') or {}).get('mean') is not None" \
  "validate reported cross-validation numbers"
_assert_json "$VALIDATE_JSON" "(o['summary'].get('holdout') or {}).get('score') is not None" \
  "validate reported sealed-holdout numbers"
_assert_json "$VALIDATE_JSON" "o['summary'].get('holdout_fraction') is not None" \
  "validate reported the holdout fraction"
_assert_json "$VALIDATE_JSON" "bool(o['pathspec'])" "validate produced a ValidateFlow run pathspec"
VALIDATE_RUN="$(_field "$VALIDATE_JSON" pathspec)"
echo "   validate_run = ${VALIDATE_RUN}"

# =============================================================================
# (3d) REPORT --runs <validate-run>: assemble a read-only model-card -- the
#      validated baseline's run, its metric, and its lineage gathered into one
#      structured card. Read-only (trains nothing). VERDICT OK means at least one
#      successful run was assembled into the card.
# =============================================================================
REPORT_JSON="$($LOOM report --runs "$VALIDATE_RUN" --json)"
_assert_json "$REPORT_JSON" "o['status'] == 'ok'" "report succeeded"
_assert_json "$REPORT_JSON" "o['VERDICT'] == 'OK'" "report VERDICT is OK (assembled a model-card)"
_assert_json "$REPORT_JSON" "bool(o['pathspec'])" "report produced a ReportFlow run pathspec"
_assert_json "$REPORT_JSON" "o['summary'].get('n_runs', 0) >= 1" "report gathered at least one run"
_assert_json "$REPORT_JSON" "o['summary'].get('n_successful', 0) >= 1" \
  "report saw at least one successful run (our validated baseline)"
_assert_json "$REPORT_JSON" \
  "any(r.get('pathspec') == '${VALIDATE_RUN}' for r in (o['summary'].get('leaderboard') or []))" \
  "the validated baseline run appears in the report leaderboard"

echo "== PASS: tutorial 01-classification"
