#!/usr/bin/env bash
# =============================================================================
# tutorials/03-leakage-safe-features/run.sh -- self-checking tutorial recipe.
#
# Teaching goal: LEAKAGE-AWARE FEATURE ENGINEERING. We ingest a synthetic loan
# table that hides two "leak" columns (values you only know AFTER the outcome,
# so they cannot exist at prediction time), then watch the keyless lifecycle:
#
#     ingest  -> the data crosses into Metaflow once, under a UNIQUE name
#     eda     -> profiles the data WITH the target declared, so the leakage
#                check fires and surfaces `leakage_flags`
#     features --from <eda-run>
#             -> reads those flags and DROPS exactly the leaky columns before
#                building, writing a NEW (clean) data object
#     validate-> scores the clean feature set -> an HONEST VERDICT (not the
#                ~perfect score the leaks would have manufactured)
#
# It is self-contained: it sources the live cluster env, generates deterministic
# data inline, uses a UNIQUE dataset name (so repeat / concurrent runs never
# collide), asserts every outcome by parsing the --json envelopes, prints clear
# PASS/FAIL, and exits NONZERO on any regression.
#
# Run it:
#     bash tutorials/03-leakage-safe-features/run.sh
#
# KEYLESS ONLY -- no `run`/`optimize`/model-using verbs (those cost an API key
# and money); the optimize step is described in README.md as a next step only.
# =============================================================================
set -euo pipefail

# --- Make the script self-contained: source the live local cluster env -------
# (datastore endpoint + credentials for the already-up Metaflow + minio stack.)
if [ -f /tmp/loom-cluster-env.sh ]; then
  # shellcheck disable=SC1091
  source /tmp/loom-cluster-env.sh
fi

# --- LOOM: the verb entrypoint -----------------------------------------------
LOOM="${LOOM:-/Users/anub/Work/Loom/.venv/bin/python -m loom}"
PY="${PY:-/Users/anub/Work/Loom/.venv/bin/python}"

# --- Where this tutorial lives + a private scratch dir (cleaned on exit) ------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/loom-tutorial-XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

# --- A UNIQUE dataset name so concurrent / repeat runs never collide ---------
RUN_TAG="$(basename "$HERE")-$$-${RANDOM}"
DATASET_NAME="tut-${RUN_TAG}"

# =============================================================================
# _assert_json -- the ONE assert helper (no jq dependency).
#
# Usage: _assert_json "<json-string>" "<python-bool-expr-over-`o`>" "<message>"
# Parses the JSON envelope into `o` (a dict) and evaluates the boolean expr. On a
# falsy result (or parse error) it prints the message + offending JSON to stderr
# and exits NONZERO -- failing the whole run, which is the point of a tested
# tutorial.
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

# The two columns make_data.py plants as leaks (kept in sync with make_data.py).
LEAK_NUMERIC="recovery_amount"        # near_perfect_predictor (numeric near-dup of target)
LEAK_CATEGORICAL="collections_status" # duplicate_of_target (categorical relabel of target)
ID_COLUMN="application_id"            # benign high-cardinality row index (MUST NOT be flagged)

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
# (3) THE LEAKAGE-SAFE FEATURE FLOW (all keyless) + ASSERT each outcome.
# =============================================================================

# --- 3a. eda --target: the profile must SURFACE the planted leaks --------------
# Read-only profile. Declaring the target turns on EDA's leakage check:
# recovery_amount trips near_perfect_predictor (|corr|~0.99) and collections_status
# trips duplicate_of_target. We assert the flags fire, name both leak columns and
# both leak KINDs, and do NOT false-positive the benign application_id.
EDA_JSON="$($LOOM eda --dataset "$DATASET" --target defaulted --json)"
_assert_json "$EDA_JSON" "o['status'] == 'ok'" "eda succeeded"
_assert_json "$EDA_JSON" "bool(o['pathspec'])" "eda produced an EdaFlow run pathspec"
_assert_json "$EDA_JSON" "o['summary'].get('target') == 'defaulted'" "eda resolved the declared target"
_assert_json "$EDA_JSON" "o['summary'].get('leakage') is True" "eda set the leakage bool (a leak is present)"
_assert_json "$EDA_JSON" "len(o['summary'].get('leakage_flags') or []) >= 2" "eda surfaced >=2 leakage_flags"
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
  "'${ID_COLUMN}' not in {f['column'] for f in o['summary']['leakage_flags']}" \
  "eda did NOT false-positive the benign id column (${ID_COLUMN})"
EDA_RUN="$(printf '%s' "$EDA_JSON" | "$PY" -c 'import sys,json; print(json.load(sys.stdin)["pathspec"])')"
echo "   eda_run = ${EDA_RUN}"

# --- 3b. features --from <eda-run>: DROP the flagged columns -------------------
# The composition gate: --from points at the EDA run, so features reads its
# leakage_flags and drops exactly those columns before building, then writes a NEW
# data object. We assert it BUILT, refused the leakage, and dropped BOTH leaks.
FEAT_JSON="$($LOOM features --dataset "$DATASET" --target defaulted --from "$EDA_RUN" --json)"
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

# --- 3c. validate the de-leaked feature set: an HONEST baseline ----------------
# Validate the NEW (leak-free) data object. Expected: status ok, a real VERDICT
# (PASS / REVIEW), and CV + holdout numbers present -- a believable baseline, not
# the ~perfect score the leaks would have produced.
VAL_JSON="$($LOOM validate --dataset "$FEATURES" --target defaulted --json)"
_assert_json "$VAL_JSON" "o['status'] == 'ok'" "validate succeeded"
_assert_json "$VAL_JSON" "o['VERDICT'] in ('PASS', 'REVIEW')" "validate emitted an honest VERDICT"
_assert_json "$VAL_JSON" "bool(o['summary'].get('cv'))" "validate reported CV numbers"
_assert_json "$VAL_JSON" "bool(o['summary'].get('holdout'))" "validate reported a holdout score"

echo "== PASS: $(basename "$HERE")"
