#!/usr/bin/env bash
# =============================================================================
# tutorials/02-regression/run.sh -- "Predicting a number: regression".
#
# A self-checking, KEYLESS walkthrough of the core Loom lifecycle on a table
# whose target is a CONTINUOUS number (a regression task). It is the sibling of
# tutorials/01 (classification): the ONLY thing that changes is the target's
# shape, and that single change flips Loom's auto-detection from ROC-AUC
# (classification) to RMSE (regression).
#
# Like the eval-bed examples it: generates deterministic synthetic data,
# ingests it under a UNIQUELY-NAMED dataset (so concurrent / repeat runs never
# collide), runs the keyless verb sequence with --json, and ASSERTS each
# outcome inline. Any regression in the contract exits NONZERO.
#
# Run it directly (the cluster env is sourced below if present, so it is
# self-contained):
#     bash tutorials/02-regression/run.sh
#
# KEYLESS ONLY. The model-using verbs (loom run / loom optimize / the agentic
# NL flow) cost an API key and money -- they are described in the README as a
# "needs an LLM key" next step and are NEVER invoked here.
# =============================================================================
set -euo pipefail

# --- Make the script self-contained: source the live cluster env if present. -
# This points Metaflow at the local minio datastore so the keyless verbs run
# against the already-up engine.
if [ -f /tmp/loom-cluster-env.sh ]; then
  # shellcheck disable=SC1091
  source /tmp/loom-cluster-env.sh
fi

# --- LOOM: the verb entrypoint -----------------------------------------------
# Verbs are invoked as `$LOOM <verb> ... --json`. LOOM defaults to the venv
# module form so no PATH / console-script is needed. The harness may override it.
LOOM="${LOOM:-/Users/anub/Work/Loom/.venv/bin/python -m loom}"
PY="${PY:-/Users/anub/Work/Loom/.venv/bin/python}"

# --- Where this tutorial lives + a private scratch dir -----------------------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/loom-tutorial-XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

# --- A UNIQUE dataset name so concurrent / repeat runs never collide ---------
# Stable-prefixed (the tutorial dir) + pid + a random suffix.
RUN_TAG="$(basename "$HERE")-$$-${RANDOM}"
DATASET_NAME="tut-${RUN_TAG}"

# =============================================================================
# _assert_json -- the ONE assert helper (no jq dependency).
#
# Usage:  _assert_json "<json-string>" "<python-bool-expr-over-`o`>" "<message>"
#
# It parses the JSON envelope into `o` (a dict) and evaluates the boolean
# expression. On a falsy result (or a parse error) it prints the message + the
# offending JSON to stderr and exits NONZERO, failing the run. Assertions read
# the verb envelope's stable fields: o['status'], o['VERDICT'], o['pathspec'],
# o['summary'][...].
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

echo "== tutorial: $(basename "$HERE")  dataset=${DATASET_NAME}  work=${WORK}"

# =============================================================================
# (1) GENERATE the deterministic synthetic REGRESSION data into $WORK.
#     A clean tabular table whose `target` is a CONTINUOUS float -- that is the
#     one change vs tutorial 01 that makes the whole pipeline a regression.
# =============================================================================
"$PY" "$HERE/make_data.py" --out-dir "$WORK"

# =============================================================================
# (2) INGEST it under the UNIQUE name -> capture the data-object pathspec.
# =============================================================================
INGEST_JSON="$($LOOM ingest --source "$WORK" --name "$DATASET_NAME" --json)"
_assert_json "$INGEST_JSON" "o['status'] == 'ok'" "ingest succeeded"
_assert_json "$INGEST_JSON" "bool(o['pathspec'])" "ingest produced a dataset pathspec"
DATASET="$(printf '%s' "$INGEST_JSON" | "$PY" -c 'import sys,json; print(json.load(sys.stdin)["pathspec"])')"
echo "   dataset_ref = ${DATASET}"

# =============================================================================
# (3) The keyless verb sequence + inline assertions on the --json.
# =============================================================================

# --- 3a. datasets: the newly-ingested data object appears in the catalog. ----
DATASETS_JSON="$($LOOM datasets --json)"
_assert_json "$DATASETS_JSON" "o['status'] == 'ok'" "datasets listing succeeded"
_assert_json "$DATASETS_JSON" \
  "any(d['pathspec'] == '${DATASET}' for d in o['summary']['datasets'])" \
  "the ingested dataset appears in the catalog"
_assert_json "$DATASETS_JSON" \
  "any(d['name'] == '${DATASET_NAME}' for d in o['summary']['datasets'])" \
  "the catalog entry carries our unique dataset name"

# --- 3b. eda --target: profile the table. shape + target + leakage:none. -----
# Read-only profile. The data is CLEAN, so leakage_flags must be empty. Because
# `target` is CONTINUOUS, the meaningful target profile lives in
# numeric_describe[target] (mean/std/min/max) rather than a tidy class balance.
EDA_JSON="$($LOOM eda --dataset "$DATASET" --target target --json)"
_assert_json "$EDA_JSON" "o['status'] == 'ok'" "eda succeeded"
_assert_json "$EDA_JSON" "o['summary'].get('nrows', 0) > 0" "eda saw rows (nrows)"
_assert_json "$EDA_JSON" "o['summary'].get('ncols', 0) > 0" "eda saw columns (ncols)"
_assert_json "$EDA_JSON" "o['summary'].get('target') == 'target'" "eda resolved the target column"
_assert_json "$EDA_JSON" \
  "'target' in (o['summary'].get('numeric_describe') or {})" \
  "eda profiled the CONTINUOUS target as a numeric column"
_assert_json "$EDA_JSON" \
  "(o['summary']['numeric_describe']['target'].get('std') or 0) > 0" \
  "the continuous target actually varies (numeric_describe.target.std > 0)"
_assert_json "$EDA_JSON" "len(o['summary'].get('leakage_flags') or []) == 0" \
  "clean data -> eda flags NO leakage"

# --- 3c. features: build engineered features into a NEW data object. ---------
FEATURES_JSON="$($LOOM features --dataset "$DATASET" --target target --json)"
_assert_json "$FEATURES_JSON" "o['status'] == 'ok'" "features succeeded"
_assert_json "$FEATURES_JSON" "o['VERDICT'] == 'BUILT'" "features VERDICT is BUILT"
_assert_json "$FEATURES_JSON" "bool(o['pathspec']) and o['pathspec'] != '${DATASET}'" \
  "features produced a NEW data-object pathspec (distinct from the source)"
_assert_json "$FEATURES_JSON" "o['summary'].get('n_features_after', 0) > 0" \
  "features summary reports n_features_after"
_assert_json "$FEATURES_JSON" "o['summary'].get('refused_leakage') == False" \
  "features did not refuse on leakage (clean data)"
FEATURES="$(printf '%s' "$FEATURES_JSON" | "$PY" -c 'import sys,json; print(json.load(sys.stdin)["pathspec"])')"
echo "   features_ref = ${FEATURES}"

# --- 3d. validate: a rigorous baseline -- but a REGRESSION one. --------------
# Same verb as tutorial 01, but because the target is continuous, validate
# auto-infers task_type == "regression", fits a HistGradientBoostingRegressor,
# and scores with RMSE (LOWER is better) instead of ROC-AUC. We assert the task
# type + metric explicitly so a silent flip back to classification fails here.
VALIDATE_JSON="$($LOOM validate --dataset "$FEATURES" --target target --json)"
_assert_json "$VALIDATE_JSON" "o['status'] == 'ok'" "validate succeeded"
_assert_json "$VALIDATE_JSON" "o['VERDICT'] in ('PASS', 'REVIEW')" \
  "validate emitted a VERDICT (PASS or REVIEW)"
_assert_json "$VALIDATE_JSON" "o['summary'].get('task_type') == 'regression'" \
  "validate auto-detected a REGRESSION task (continuous target)"
_assert_json "$VALIDATE_JSON" "o['summary'].get('metric') == 'rmse'" \
  "validate scored with RMSE (the regression metric, lower-is-better)"
_assert_json "$VALIDATE_JSON" "(o['summary'].get('cv') or {}).get('mean') is not None" \
  "validate reported cross-validation RMSE (cv.mean)"
_assert_json "$VALIDATE_JSON" "((o['summary'].get('cv') or {}).get('mean') or -1) >= 0" \
  "the CV RMSE is a non-negative error (a real error magnitude, not an accuracy)"
_assert_json "$VALIDATE_JSON" "(o['summary'].get('holdout') or {}).get('score') is not None" \
  "validate reported sealed-holdout RMSE"
_assert_json "$VALIDATE_JSON" "o['summary'].get('holdout_fraction') is not None" \
  "validate reported the holdout fraction"
VALIDATE_RUN="$(printf '%s' "$VALIDATE_JSON" | "$PY" -c 'import sys,json; print(json.load(sys.stdin)["pathspec"])')"
echo "   validate_run = ${VALIDATE_RUN}"

# --- 3e. report: assemble the validate run into a read-only model-card. ------
# Report gathers the run(s), their metric, and lineage. best_metric is our
# holdout RMSE; n_runs/n_successful confirm the run was picked up.
REPORT_JSON="$($LOOM report --runs "$VALIDATE_RUN" --json)"
_assert_json "$REPORT_JSON" "o['status'] == 'ok'" "report succeeded"
_assert_json "$REPORT_JSON" "o['summary'].get('n_runs', 0) >= 1" \
  "report gathered at least the validate run"
_assert_json "$REPORT_JSON" "o['summary'].get('n_successful', 0) >= 1" \
  "report saw the validate run as successful"
_assert_json "$REPORT_JSON" "o['summary'].get('best_metric') is not None" \
  "report surfaced the run's metric (the RMSE) as best_metric"
_assert_json "$REPORT_JSON" \
  "any(r['pathspec'] == '${VALIDATE_RUN}' for r in (o['summary'].get('leaderboard') or []))" \
  "the validate run appears on the report leaderboard"

echo "== PASS: $(basename "$HERE")"
