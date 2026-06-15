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

# The two columns make_data.py plants as leaks (kept in sync with make_data.py).
LEAK_NUMERIC="leak_score"        # near_perfect_predictor (numeric near-dup of target)
LEAK_CATEGORICAL="leak_flag"     # duplicate_of_target (categorical relabel of target)

# --- 3a. eda --target: the profile must FLAG both planted leaks ----------------
# Read-only profile. With the target declared, EDA's leakage check fires:
# leak_score trips near_perfect_predictor (|corr|~0.99) and leak_flag trips
# duplicate_of_target. Expected: status ok, leakage True, both planted columns in
# leakage_flags (and the benign `id` / `feature_*` columns are NOT flagged).
EDA_JSON="$($LOOM eda --dataset "$DATASET" --target target --json)"
_assert_json "$EDA_JSON" "o['status'] == 'ok'" "eda succeeded"
_assert_json "$EDA_JSON" "bool(o['pathspec'])" "eda produced an EdaFlow run pathspec"
_assert_json "$EDA_JSON" "o['summary'].get('target') == 'target'" "eda resolved the declared target"
_assert_json "$EDA_JSON" "len(o['summary'].get('leakage_flags') or []) >= 2" "eda flagged the planted leak(s)"
_assert_json "$EDA_JSON" "o['summary'].get('leakage') is True" "eda set the leakage bool"
_assert_json "$EDA_JSON" \
  "'${LEAK_NUMERIC}' in {f['column'] for f in o['summary']['leakage_flags']}" \
  "eda named the numeric leak column (${LEAK_NUMERIC})"
_assert_json "$EDA_JSON" \
  "'${LEAK_CATEGORICAL}' in {f['column'] for f in o['summary']['leakage_flags']}" \
  "eda named the categorical leak column (${LEAK_CATEGORICAL})"
_assert_json "$EDA_JSON" \
  "'near_perfect_predictor' in {f['kind'] for f in o['summary']['leakage_flags']}" \
  "eda detected the near_perfect_predictor kind"
_assert_json "$EDA_JSON" \
  "'duplicate_of_target' in {f['kind'] for f in o['summary']['leakage_flags']}" \
  "eda detected the duplicate_of_target kind"
_assert_json "$EDA_JSON" \
  "'id' not in {f['column'] for f in o['summary']['leakage_flags']}" \
  "eda did NOT false-positive the benign id column"
# Capture the EdaFlow run pathspec so features can compose on it via --from.
EDA_RUN="$(printf '%s' "$EDA_JSON" | "$PY" -c 'import sys,json; print(json.load(sys.stdin)["pathspec"])')"
echo "   eda_run = ${EDA_RUN}"

# --- 3b. features --from <eda-run>: DROP the flagged columns -------------------
# The eda -> features composition gate: features reads the upstream EDA run's
# leakage_flags and drops exactly those columns before building, then writes a NEW
# data object. Expected: VERDICT BUILT, refused_leakage True, both planted leaks
# in dropped_columns, n_features_after reflects the drop, and a new FeaturesFlow
# pathspec downstream verbs can consume.
FEAT_JSON="$($LOOM features --dataset "$DATASET" --target target --from "$EDA_RUN" --json)"
_assert_json "$FEAT_JSON" "o['status'] == 'ok'" "features succeeded"
_assert_json "$FEAT_JSON" "o['VERDICT'] == 'BUILT'" "features VERDICT is BUILT"
_assert_json "$FEAT_JSON" "bool(o['pathspec'])" "features produced a new FeaturesFlow pathspec"
_assert_json "$FEAT_JSON" "o['summary'].get('refused_leakage') is True" "features refused the leakage (dropped flagged cols)"
_assert_json "$FEAT_JSON" \
  "'${LEAK_NUMERIC}' in (o['summary'].get('dropped_columns') or [])" \
  "features dropped the numeric leak column (${LEAK_NUMERIC})"
_assert_json "$FEAT_JSON" \
  "'${LEAK_CATEGORICAL}' in (o['summary'].get('dropped_columns') or [])" \
  "features dropped the categorical leak column (${LEAK_CATEGORICAL})"
_assert_json "$FEAT_JSON" \
  "len(o['summary'].get('dropped_columns') or []) >= 2" \
  "features dropped both planted leak columns"
FEATURES="$(printf '%s' "$FEAT_JSON" | "$PY" -c 'import sys,json; print(json.load(sys.stdin)["pathspec"])')"
echo "   features_ref = ${FEATURES}"

# --- 3c. validate the de-leaked feature set: a clean baseline ------------------
# Validate the NEW (leak-free) data object. Expected: status ok, a VERDICT
# (PASS / REVIEW), and CV + holdout numbers present -- a real baseline, not the
# ~perfect score the leaks would have produced.
VAL_JSON="$($LOOM validate --dataset "$FEATURES" --target target --json)"
_assert_json "$VAL_JSON" "o['status'] == 'ok'" "validate succeeded"
_assert_json "$VAL_JSON" "o['VERDICT'] in ('PASS', 'REVIEW')" "validate emitted a VERDICT"
_assert_json "$VAL_JSON" "bool(o['summary'].get('cv'))" "validate reported CV numbers"
_assert_json "$VAL_JSON" "bool(o['summary'].get('holdout'))" "validate reported a holdout score"

echo "== PASS: $(basename "$HERE")"
