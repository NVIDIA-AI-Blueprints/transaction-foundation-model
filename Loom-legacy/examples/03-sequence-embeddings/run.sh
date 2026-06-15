#!/usr/bin/env bash
# =============================================================================
# examples/03-sequence-embeddings/run.sh -- the CANONICAL self-checking recipe.
#
# A copy of examples/_template/run.sh with section (3) filled in. It is the
# regression eval bed: it generates deterministic synthetic data, ingests it
# under a UNIQUELY-NAMED dataset (so concurrent / repeat runs never collide),
# runs the keyless verb sequence with --json, and ASSERTS the outcomes inline.
# Any regression exits NONZERO -- that is the whole point.
#
# Run it directly (after sourcing the cluster env, see examples/README.md):
#     source /tmp/loom-cluster-env.sh
#     bash examples/03-sequence-embeddings/run.sh
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
#     -- The model-builder (CPU) use case: build an embedding backbone from the
#        per-account event sequences with the torch-free `local` adapter, then
#        show the production GPU path (nemo) gates cleanly with no GPU target.
# =============================================================================

# (3a) BUILD the backbone on CPU with the torch-free `local` adapter.
#      LOOM_MODEL_BUILDER_PROVIDER=local selects the PPMI+TruncatedSVD stand-in
#      (no GPU, sub-2s); --objective next-event learns the planted adjacent-pair
#      (first-order Markov) co-occurrence; --budget probe is the cheapest dim.
#      Expected: VERDICT == "BUILT", a fingerprint, and a backbone run pathspec.
TRAIN_LOCAL_JSON="$(LOOM_MODEL_BUILDER_PROVIDER=local $LOOM train \
  --dataset "$DATASET" --objective next-event --budget probe --json)"
_assert_json "$TRAIN_LOCAL_JSON" "o['status'] == 'ok'" "local train run succeeded"
_assert_json "$TRAIN_LOCAL_JSON" "o['VERDICT'] == 'BUILT'" \
  "local train BUILT the backbone (the CPU adapter actually builds)"
_assert_json "$TRAIN_LOCAL_JSON" "o['summary'].get('status') == 'BUILT'" \
  "local train summary status is BUILT"
_assert_json "$TRAIN_LOCAL_JSON" "bool(o['pathspec'])" \
  "local train produced a backbone run pathspec (a first-class dataset_ref)"
_assert_json "$TRAIN_LOCAL_JSON" "bool(o['summary'].get('fingerprint'))" \
  "local train recorded a deterministic backbone fingerprint"
_assert_json "$TRAIN_LOCAL_JSON" "str(o['summary'].get('fingerprint')).startswith('sha256:')" \
  "the backbone fingerprint is a content hash (sha256:...)"
_assert_json "$TRAIN_LOCAL_JSON" "o['summary'].get('objective') == 'next-event'" \
  "local train used the next-event objective"
_assert_json "$TRAIN_LOCAL_JSON" "o['summary'].get('budget') == 'probe'" \
  "local train used the probe budget"
BACKBONE="$(printf '%s' "$TRAIN_LOCAL_JSON" | "$PY" -c 'import sys,json; print(json.load(sys.stdin)["pathspec"])')"
echo "   backbone_ref = ${BACKBONE}"

# (3b) Show the PRODUCTION path (nemo, the default model_builder_provider) GATES
#      cleanly: with no configured gpu_target it REFUSES up front rather than
#      launching. This is the safe-by-default heavy-launch gate (the deploy
#      --apply posture). The run still completes (the gate status IS the correct
#      outcome) -- it just refuses to consume GPU.
#      Expected: VERDICT == "REFUSED_NO_GPU_TARGET", no backbone pathspec.
TRAIN_NEMO_JSON="$($LOOM train \
  --dataset "$DATASET" --objective next-event --budget probe --json)"
_assert_json "$TRAIN_NEMO_JSON" "o['status'] == 'ok'" \
  "the nemo gate run completed (a clean refusal is a success, not a crash)"
_assert_json "$TRAIN_NEMO_JSON" "o['VERDICT'] == 'REFUSED_NO_GPU_TARGET'" \
  "nemo (the GPU production path) REFUSES with no gpu_target -- the heavy-launch gate"
_assert_json "$TRAIN_NEMO_JSON" "o['summary'].get('status') == 'REFUSED_NO_GPU_TARGET'" \
  "nemo train summary status is REFUSED_NO_GPU_TARGET"
_assert_json "$TRAIN_NEMO_JSON" "not o['summary'].get('artifact_pathspec')" \
  "the refused nemo run produced NO backbone artifact (it did not launch)"
_assert_json "$TRAIN_NEMO_JSON" "o['summary'].get('launch') == False" \
  "the heavy GPU launch is OFF by default (--launch not set)"

echo "== PASS: $(basename "$HERE")"
