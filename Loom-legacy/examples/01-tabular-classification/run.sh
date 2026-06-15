#!/usr/bin/env bash
# =============================================================================
# examples/_template/run.sh -- the CANONICAL self-checking recipe skeleton.
#
# Every examples/<NN-name>/run.sh is a copy of THIS file with section (3) filled
# in. It is the regression eval bed: it generates deterministic synthetic data,
# ingests it under a UNIQUELY-NAMED dataset (so concurrent / repeat runs never
# collide), runs the keyless verb sequence with --json, and ASSERTS the outcomes
# inline. Any regression exits NONZERO -- that is the whole point.
#
# Run it directly (after sourcing the cluster env, see examples/README.md):
#     source /tmp/loom-cluster-env.sh
#     bash examples/<NN-name>/run.sh
#
# tests/test_examples.py replays this exact script and asserts exit 0.
# =============================================================================
set -euo pipefail

# --- LOOM: the verb entrypoint -----------------------------------------------
# Verbs are invoked as `$LOOM <verb> ... --json`. LOOM defaults to the venv
# module form so no PATH / console-script is needed. The harness may override it.
LOOM="${LOOM:-/Users/anub/Work/Loom/.venv/bin/python -m loom}"
PY="${PY:-/Users/anub/Work/Loom/.venv/bin/python}"

# --- Where this example lives + a private scratch dir ------------------------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/loom-example-XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

# --- A UNIQUE dataset name so concurrent / repeat runs never collide ---------
# Keep it stable-prefixed (the example dir) + a random suffix.
RUN_TAG="$(basename "$HERE")-$$-${RANDOM}"
DATASET_NAME="ex-${RUN_TAG}"

# =============================================================================
# _assert_json -- the ONE assert helper every example uses (no jq dependency).
#
# Usage:
#     _assert_json "<json-string>" "<python-bool-expr-over-`o`>" "<message>"
#
# It parses the JSON envelope into `o` (a dict) with the venv python and
# evaluates the boolean expression. On a falsy result (or a parse error) it
# prints the message + the offending JSON to stderr and exits NONZERO, failing
# the run. Examples assert on the verb envelope's stable fields:
#     o['status'], o['VERDICT'], o['pathspec'], o['summary'][...], o['gate'][...]
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
# The expression is REPO-AUTHORED (not untrusted input), so the eval namespace
# exposes the small set of builtins assertions need (bool/len/any/all/...) plus
# the parsed envelope as `o`.
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

echo "== example: $(basename "$HERE")  dataset=${DATASET_NAME}  work=${WORK}"

# =============================================================================
# (1) GENERATE the deterministic synthetic data into $WORK.
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
# (3) RUN the keyless verb sequence with --json + ASSERT each outcome.
#     -- THE PER-EXAMPLE BODY. Replace this block in each examples/<NN-name>.
#     -- Use ONLY keyless verbs (no model key): ingest / datasets / eda /
#        features / validate / viz / report / ops / train[local] /
#        deploy[plan] / collab[build] / telemetry / doctor.
#     -- LLM verbs (run / optimize / pipeline's optimize stage / the agentic
#        NL flow) belong in the README walkthrough prose, NOT here.
#     -- For train use:  LOOM_MODEL_BUILDER_PROVIDER=local $LOOM train ...
# =============================================================================

# --- 3a. datasets: the newly-ingested data object appears in the catalog. ----
# Confirms the ingest boundary landed a readable, named data object.
DATASETS_JSON="$($LOOM datasets --json)"
_assert_json "$DATASETS_JSON" "o['status'] == 'ok'" "datasets listing succeeded"
_assert_json "$DATASETS_JSON" \
  "any(d['pathspec'] == '${DATASET}' for d in o['summary']['datasets'])" \
  "the ingested dataset appears in the catalog"
_assert_json "$DATASETS_JSON" \
  "any(d['name'] == '${DATASET_NAME}' for d in o['summary']['datasets'])" \
  "the catalog entry carries our unique dataset name"

# --- 3b. eda --target: profile it. shape + target + leakage:none. ------------
# Read-only profile. The data is CLEAN, so leakage_flags must be empty.
EDA_JSON="$($LOOM eda --dataset "$DATASET" --target target --json)"
_assert_json "$EDA_JSON" "o['status'] == 'ok'" "eda succeeded"
_assert_json "$EDA_JSON" "o['summary'].get('nrows', 0) > 0" "eda saw rows (nrows)"
_assert_json "$EDA_JSON" "o['summary'].get('ncols', 0) > 0" "eda saw columns (ncols)"
_assert_json "$EDA_JSON" "o['summary'].get('target') == 'target'" "eda resolved the target column"
_assert_json "$EDA_JSON" "bool(o['summary'].get('target_balance'))" "eda reported target balance"
_assert_json "$EDA_JSON" "len(o['summary'].get('leakage_flags') or []) == 0" \
  "clean data -> eda flags NO leakage"
EDA_RUN="$(printf '%s' "$EDA_JSON" | "$PY" -c 'import sys,json; print(json.load(sys.stdin)["pathspec"])')"
echo "   eda_run = ${EDA_RUN}"

# --- 3c. features: build engineered features into a NEW data object. ---------
# Produces a brand-new FeaturesFlow/<id> pathspec (distinct from the source),
# VERDICT == BUILT, with before/after feature counts.
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

# --- 3d. validate: a rigorous baseline. VERDICT PASS/REVIEW; CV + holdout. ---
# Fits a gradient-boosted-trees baseline on the engineered features and emits a
# VERDICT with cross-validation + sealed-holdout numbers.
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

echo "== PASS: $(basename "$HERE")"
