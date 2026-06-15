#!/usr/bin/env bash
# =============================================================================
# examples/05-ops-and-drift/run.sh -- the CANONICAL self-checking recipe.
#
# Every examples/<NN-name>/run.sh is a copy of the _template skeleton with
# section (3) filled in. It is the regression eval bed: it generates
# deterministic synthetic data, ingests it under a UNIQUELY-NAMED dataset (so
# concurrent / repeat runs never collide), runs the keyless verb sequence with
# --json, and ASSERTS the outcomes inline. Any regression exits NONZERO -- that
# is the whole point.
#
# Run it directly (after sourcing the cluster env, see examples/README.md):
#     source /tmp/loom-cluster-env.sh
#     bash examples/05-ops-and-drift/run.sh
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
#     make_data.py writes TWO ingest sources: $WORK/reference and $WORK/shifted
#     (the same schema; only the shifted frame's numeric distributions moved).
# =============================================================================
"$PY" "$HERE/make_data.py" --out-dir "$WORK"

# =============================================================================
# (2) INGEST the REFERENCE (baseline) under the UNIQUE name -> its pathspec.
# =============================================================================
INGEST_JSON="$($LOOM ingest --source "$WORK/reference" --name "$DATASET_NAME" --json)"
_assert_json "$INGEST_JSON" "o['status'] == 'ok'" "ingest (reference) succeeded"
_assert_json "$INGEST_JSON" "bool(o['pathspec'])" "ingest (reference) produced a dataset pathspec"
DATASET="$(printf '%s' "$INGEST_JSON" | "$PY" -c 'import sys,json; print(json.load(sys.stdin)["pathspec"])')"
echo "   reference_ref = ${DATASET}"

# =============================================================================
# (3) RUN the keyless ops/monitoring sequence with --json + ASSERT each outcome.
#     -- Ingest the SHIFTED variant as its own data object.
#     -- Seed run-health with a REAL ValidateFlow run (so ops --flow has counts).
#     -- ops --flow ValidateFlow      : run-health view (counts + status).
#     -- ops --dataset --reference    : the drift check (shifted vs reference).
#     -- All keyless: ops is the read-only tier; validate runs without a key.
#     -- LLM verbs (loom run / optimize / the NL flow) live in README prose only.
# =============================================================================

# (3a) Ingest the SHIFTED variant as a second data object (same schema, moved).
SHIFT_INGEST_JSON="$($LOOM ingest --source "$WORK/shifted" --name "${DATASET_NAME}-shifted" --json)"
_assert_json "$SHIFT_INGEST_JSON" "o['status'] == 'ok'" "ingest (shifted) succeeded"
_assert_json "$SHIFT_INGEST_JSON" "bool(o['pathspec'])" "ingest (shifted) produced a dataset pathspec"
SHIFTED="$(printf '%s' "$SHIFT_INGEST_JSON" | "$PY" -c 'import sys,json; print(json.load(sys.stdin)["pathspec"])')"
echo "   shifted_ref   = ${SHIFTED}"

# (3b) Seed run health: a REAL ValidateFlow run so `ops --flow` has runs to count.
#      validate is keyless (no model key); we only need it to LAND a finished run.
VALIDATE_JSON="$($LOOM validate --dataset "$DATASET" --target target --json)"
_assert_json "$VALIDATE_JSON" "o['status'] == 'ok'" "validate (seed run) succeeded"
_assert_json "$VALIDATE_JSON" "bool(o['VERDICT'])" "validate produced a VERDICT"
echo "   seeded ValidateFlow run for the run-health view"

# (3c) ops --flow: the run-health view. Reads recent ValidateFlow runs via the
#      Client API (read-only), reports success/failure counts + a status.
#      Expected: status ok; summary.health present with at least our seeded run.
OPS_HEALTH_JSON="$($LOOM ops --flow ValidateFlow --json)"
_assert_json "$OPS_HEALTH_JSON" "o['status'] == 'ok'" "ops --flow succeeded"
_assert_json "$OPS_HEALTH_JSON" "o['summary'].get('health') and 'n_runs' in o['summary']['health']" \
  "ops --flow reported a run-health block"
_assert_json "$OPS_HEALTH_JSON" "o['summary']['health'].get('flow_name') == 'ValidateFlow'" \
  "ops --flow health is scoped to ValidateFlow"
_assert_json "$OPS_HEALTH_JSON" \
  "isinstance(o['summary']['health'].get('n_runs'), int) and o['summary']['health']['n_runs'] >= 1" \
  "ops --flow counted at least the seeded run"
_assert_json "$OPS_HEALTH_JSON" \
  "o['summary']['health']['n_runs'] == o['summary']['health']['n_successful'] + o['summary']['health']['n_failed']" \
  "ops --flow run counts reconcile (n_runs == ok + failed)"
_assert_json "$OPS_HEALTH_JSON" \
  "o['VERDICT'] in ('OK', 'ATTENTION', 'EMPTY')" \
  "ops --flow carried a status VERDICT"

# (3d) ops --dataset --reference: the DRIFT check. Materializes both data objects
#      via the Client API and compares their summary stats. The shifted frame's
#      numeric features moved well past the drift threshold, so:
#      Expected: status ok; summary.drift present, drift==True, status DRIFT,
#                a non-empty drift_flags list (mean_shift on the feature_* cols).
OPS_DRIFT_JSON="$($LOOM ops --dataset "$SHIFTED" --reference "$DATASET" --json)"
_assert_json "$OPS_DRIFT_JSON" "o['status'] == 'ok'" "ops --dataset --reference succeeded"
_assert_json "$OPS_DRIFT_JSON" "o['summary'].get('drift') and 'status' in o['summary']['drift']" \
  "ops drift check reported a drift block"
_assert_json "$OPS_DRIFT_JSON" "o['summary']['drift'].get('drift') is True" \
  "ops detected the planted distribution drift"
_assert_json "$OPS_DRIFT_JSON" "o['summary']['drift'].get('status') == 'DRIFT'" \
  "ops drift status is DRIFT"
_assert_json "$OPS_DRIFT_JSON" \
  "isinstance(o['summary']['drift'].get('drift_flags'), list) and len(o['summary']['drift']['drift_flags']) > 0" \
  "ops drift listed at least one drifted column"
_assert_json "$OPS_DRIFT_JSON" \
  "any(f.get('column','').startswith('feature_') for f in o['summary']['drift']['drift_flags'])" \
  "ops drift flagged a shifted feature_* column"
_assert_json "$OPS_DRIFT_JSON" "o['VERDICT'] == 'ATTENTION'" \
  "ops drift degrades the overall status to ATTENTION"

echo "== PASS: $(basename "$HERE")"
