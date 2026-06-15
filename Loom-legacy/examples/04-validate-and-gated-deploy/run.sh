#!/usr/bin/env bash
# =============================================================================
# examples/04-validate-and-gated-deploy/run.sh -- the self-checking recipe.
#
# A copy of examples/_template/run.sh with section (3) filled in for the
# deploy-gate use case. It is the regression eval bed: it generates deterministic
# synthetic data, ingests it under UNIQUELY-NAMED datasets (so concurrent /
# repeat runs never collide), runs the keyless verb sequence with --json, and
# ASSERTS the outcomes inline. Any regression exits NONZERO -- that is the point.
#
# This example proves the cross-verb EXIT GATE + safe-by-default: deploy gates on
# the upstream `loom validate` VERDICT. A clean dataset validates PASS -> the gate
# ALLOWs (STAGED); a planted-leak dataset validates REVIEW -> the gate BLOCKs.
# Either way --apply is OFF by default (no external mutation).
#
# Run it directly (after sourcing the cluster env, see examples/README.md):
#     source /tmp/loom-cluster-env.sh
#     bash examples/04-validate-and-gated-deploy/run.sh
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
#     This example writes TWO variants under $WORK: clean/train.csv (a learnable
#     but honest signal -> validate PASS) and leaky/train.csv (a planted
#     near-perfect predictor -> validate REVIEW). They drive the two gate branches.
# =============================================================================
"$PY" "$HERE/make_data.py" --out-dir "$WORK"

# =============================================================================
# (2) INGEST both variants under UNIQUE names -> capture each data-object pathspec.
# =============================================================================
CLEAN_INGEST_JSON="$($LOOM ingest --source "$WORK/clean" --name "${DATASET_NAME}-clean" --json)"
_assert_json "$CLEAN_INGEST_JSON" "o['status'] == 'ok'" "ingest (clean) succeeded"
_assert_json "$CLEAN_INGEST_JSON" "bool(o['pathspec'])" "ingest (clean) produced a dataset pathspec"
CLEAN_DATASET="$(printf '%s' "$CLEAN_INGEST_JSON" | "$PY" -c 'import sys,json; print(json.load(sys.stdin)["pathspec"])')"
echo "   clean_dataset_ref = ${CLEAN_DATASET}"

LEAKY_INGEST_JSON="$($LOOM ingest --source "$WORK/leaky" --name "${DATASET_NAME}-leaky" --json)"
_assert_json "$LEAKY_INGEST_JSON" "o['status'] == 'ok'" "ingest (leaky) succeeded"
_assert_json "$LEAKY_INGEST_JSON" "bool(o['pathspec'])" "ingest (leaky) produced a dataset pathspec"
LEAKY_DATASET="$(printf '%s' "$LEAKY_INGEST_JSON" | "$PY" -c 'import sys,json; print(json.load(sys.stdin)["pathspec"])')"
echo "   leaky_dataset_ref = ${LEAKY_DATASET}"

# =============================================================================
# (3) RUN the keyless verb sequence with --json + ASSERT each outcome.
#     The deploy-gate use case: validate -> deploy --validate (PLAN only, no
#     --apply). The gate decision MUST match the upstream validate VERDICT:
#       PASS  -> ALLOW  -> STAGED   (apply still False; no external mutation)
#       REVIEW-> BLOCK  -> BLOCKED  (apply still False; no external mutation)
#     deploy is the cross-verb exit gate; --apply is OFF by default (safe-by-default).
# =============================================================================

# --- The ALLOW branch: clean dataset -> validate PASS -> gate ALLOW -> STAGED ---
echo "-- branch A (clean -> PASS -> ALLOW/STAGED) --"
VALIDATE_CLEAN_JSON="$($LOOM validate --dataset "$CLEAN_DATASET" --target target --json)"
_assert_json "$VALIDATE_CLEAN_JSON" "o['status'] == 'ok'" "validate (clean) succeeded"
_assert_json "$VALIDATE_CLEAN_JSON" "o['VERDICT'] == 'PASS'" "validate (clean) VERDICT is PASS"
_assert_json "$VALIDATE_CLEAN_JSON" "len(o['summary'].get('leakage_flags') or []) == 0" "validate (clean) found no leakage"
_assert_json "$VALIDATE_CLEAN_JSON" "bool(o['summary'].get('cv')) and bool(o['summary'].get('holdout'))" "validate (clean) has CV + holdout"
_assert_json "$VALIDATE_CLEAN_JSON" "bool(o['pathspec'])" "validate (clean) produced a ValidateFlow run pathspec"
VALIDATE_CLEAN_RUN="$(printf '%s' "$VALIDATE_CLEAN_JSON" | "$PY" -c 'import sys,json; print(json.load(sys.stdin)["pathspec"])')"
echo "   validate_clean_run = ${VALIDATE_CLEAN_RUN}"

# deploy --validate <PASS run>  (PLAN only; NO --apply). Gate must ALLOW -> STAGED.
DEPLOY_CLEAN_JSON="$($LOOM deploy --validate "$VALIDATE_CLEAN_RUN" --json)"
_assert_json "$DEPLOY_CLEAN_JSON" "o['status'] == 'ok'" "deploy (clean) run completed"
_assert_json "$DEPLOY_CLEAN_JSON" "o['gate'] is not None and o['gate']['decision'] == 'ALLOW'" "deploy gate ALLOWed (clean PASS)"
_assert_json "$DEPLOY_CLEAN_JSON" "o['VERDICT'] == 'STAGED'" "deploy (clean) VERDICT is STAGED"
_assert_json "$DEPLOY_CLEAN_JSON" "o['summary'].get('status') == 'PLANNED'" "deploy (clean) status is PLANNED (staged manifest, no mutation)"
_assert_json "$DEPLOY_CLEAN_JSON" "o['summary'].get('apply') is False" "deploy (clean) apply is OFF by default"

# --- The BLOCK branch: leaky dataset -> validate REVIEW -> gate BLOCK -> BLOCKED ---
echo "-- branch B (leaky -> REVIEW -> BLOCK/BLOCKED) --"
VALIDATE_LEAKY_JSON="$($LOOM validate --dataset "$LEAKY_DATASET" --target target --json)"
_assert_json "$VALIDATE_LEAKY_JSON" "o['status'] == 'ok'" "validate (leaky) succeeded"
_assert_json "$VALIDATE_LEAKY_JSON" "o['VERDICT'] == 'REVIEW'" "validate (leaky) VERDICT is REVIEW (sub-threshold)"
_assert_json "$VALIDATE_LEAKY_JSON" "len(o['summary'].get('leakage_flags') or []) > 0" "validate (leaky) flagged leakage"
_assert_json "$VALIDATE_LEAKY_JSON" "any(f.get('column') == 'leak_feature' for f in o['summary'].get('leakage_flags') or [])" "validate (leaky) named the planted leak_feature"
_assert_json "$VALIDATE_LEAKY_JSON" "bool(o['pathspec'])" "validate (leaky) produced a ValidateFlow run pathspec"
VALIDATE_LEAKY_RUN="$(printf '%s' "$VALIDATE_LEAKY_JSON" | "$PY" -c 'import sys,json; print(json.load(sys.stdin)["pathspec"])')"
echo "   validate_leaky_run = ${VALIDATE_LEAKY_RUN}"

# deploy --validate <REVIEW run>  (PLAN only; NO --apply). Gate must BLOCK -> BLOCKED.
DEPLOY_LEAKY_JSON="$($LOOM deploy --validate "$VALIDATE_LEAKY_RUN" --json)"
_assert_json "$DEPLOY_LEAKY_JSON" "o['status'] == 'ok'" "deploy (leaky) run completed"
_assert_json "$DEPLOY_LEAKY_JSON" "o['gate'] is not None and o['gate']['decision'] == 'BLOCK'" "deploy gate BLOCKed (leaky REVIEW)"
_assert_json "$DEPLOY_LEAKY_JSON" "len(o['gate'].get('reasons') or []) > 0" "deploy gate gave blocking reasons"
_assert_json "$DEPLOY_LEAKY_JSON" "o['VERDICT'] == 'BLOCKED'" "deploy (leaky) VERDICT is BLOCKED"
_assert_json "$DEPLOY_LEAKY_JSON" "o['summary'].get('status') == 'BLOCKED'" "deploy (leaky) status is BLOCKED"
_assert_json "$DEPLOY_LEAKY_JSON" "o['summary'].get('apply') is False" "deploy (leaky) apply is OFF by default (no external mutation)"

echo "== PASS: $(basename "$HERE")"
