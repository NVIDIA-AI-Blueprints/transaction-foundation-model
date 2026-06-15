#!/usr/bin/env bash
# =============================================================================
# tutorials/06-monitoring-drift/run.sh
#
# "Monitoring data drift in production" -- a self-checking, KEYLESS walkthrough
# of Loom's read-only monitoring tier. It:
#
#   1. generates two deterministic synthetic frames inline (no downloads):
#        - a REFERENCE (v1) baseline distribution, and
#        - a CURRENT  (v2) batch whose numeric features have drifted.
#   2. ingests each as its own Metaflow data object under a UNIQUE name
#      (timestamp + pid + random suffix) so repeat / concurrent runs never
#      collide.
#   3. lands one real keyless `validate` run so the run-health view has
#      something to count.
#   4. runs the two monitoring verbs with --json and ASSERTS the outcomes:
#        - `ops --flow ValidateFlow`              -> run-health rollup
#        - `ops --dataset <v2> --reference <v1>`  -> DATA DRIFT comparison
#   5. prints a clear PASS / FAIL and exits NONZERO on any regression.
#
# Everything here is KEYLESS: `ops` and `validate` are the read-only / no-model
# tier. The key-gated `loom run` / `loom optimize` (re-build against the new
# distribution) are described in README.md as the "needs an LLM key" next step
# and are deliberately NOT executed here.
#
# Run it (from the repo root):
#     cd /Users/anub/Work/Loom
#     bash tutorials/06-monitoring-drift/run.sh
# =============================================================================
set -euo pipefail

# --- Make the script self-contained: bring up the cluster env if present. ----
# /tmp/loom-cluster-env.sh points Metaflow at the local minio (s3) datastore.
if [[ -f /tmp/loom-cluster-env.sh ]]; then
  # shellcheck disable=SC1091
  source /tmp/loom-cluster-env.sh
fi

# --- LOOM: the verb entrypoint (venv module form; no PATH needed). -----------
LOOM="${LOOM:-/Users/anub/Work/Loom/.venv/bin/python -m loom}"
PY="${PY:-/Users/anub/Work/Loom/.venv/bin/python}"

# --- Where this tutorial lives + a private scratch dir (cleaned on exit). -----
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/loom-tutorial06-XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

# --- A UNIQUE dataset name so concurrent / repeat runs never collide. --------
# Stable prefix (this tutorial) + timestamp + pid + random suffix.
RUN_TAG="$(basename "$HERE")-$(date +%s)-$$-${RANDOM}"
DATASET_NAME="tut-${RUN_TAG}"

# =============================================================================
# _assert_json -- the ONE assert helper (no jq dependency).
#
# Usage: _assert_json "<json-string>" "<python-bool-expr-over-`o`>" "<message>"
# Parses the JSON envelope into `o` (a dict) and evaluates the boolean expr.
# On a falsy result (or parse error) it prints the message + offending JSON to
# stderr and exits NONZERO -- failing the run. That is the whole point.
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

# _pathspec -- pull the data-object pathspec out of an ingest --json envelope.
_pathspec() { printf '%s' "$1" | "$PY" -c 'import sys,json; print(json.load(sys.stdin)["pathspec"])'; }

echo "== tutorial: $(basename "$HERE")  dataset=${DATASET_NAME}  work=${WORK}"

# =============================================================================
# (1) GENERATE the two deterministic synthetic frames INLINE (no downloads).
#
#     Both frames share the IDENTICAL schema -- id, feature_0..feature_7, target
#     -- so the drift check compares like-for-like. Only the CURRENT (v2) frame's
#     numeric features are shifted: a +3.0 additive mean offset and a 1.8x spread
#     inflation, large enough to clear Loom's relative-mean-shift threshold (0.25)
#     on every feature, so the drift check fires unambiguously. Seeded -> a repeat
#     run is byte-identical; the two frames use distinct derived seeds (independent
#     draws, not the same rows).
# =============================================================================
mkdir -p "$WORK/reference" "$WORK/current"
"$PY" - "$WORK" <<'PYGEN'
import sys
from pathlib import Path
import numpy as np
import pandas as pd

out = Path(sys.argv[1])
N_ROWS, N_FEATURES = 2000, 8
SHIFT_MEAN_OFFSET, SHIFT_STD_SCALE = 3.0, 1.8  # the planted drift on v2

def build(seed: int, mean_offset: float = 0.0, std_scale: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(size=(N_ROWS, N_FEATURES))
    feats = base * std_scale + mean_offset
    cols = [f"feature_{i}" for i in range(N_FEATURES)]
    df = pd.DataFrame(feats, columns=cols)
    df.insert(0, "id", range(N_ROWS))
    # A learnable label off the pre-shift signal -- keeps the schema realistic.
    logits = base[:, 0] - 0.5 * base[:, 1]
    df["target"] = (logits > np.median(logits)).astype(int)
    return df

# v1 reference: standard-normal baseline.
build(seed=0).to_csv(out / "reference" / "train.csv", index=False)
# v2 current: same schema, numeric features moved (independent draw, seed+1).
build(seed=1, mean_offset=SHIFT_MEAN_OFFSET, std_scale=SHIFT_STD_SCALE).to_csv(
    out / "current" / "train.csv", index=False
)
print("  generated reference/ (v1 baseline) + current/ (v2 shifted)")
PYGEN

# =============================================================================
# (2) INGEST the REFERENCE (v1 baseline) -> capture its data-object pathspec.
#     `ingest` is the one external -> Metaflow boundary. Expected: status ok +
#     a non-empty pathspec (the reference dataset_ref).
# =============================================================================
REF_INGEST_JSON="$($LOOM ingest --source "$WORK/reference" --name "${DATASET_NAME}-v1" --json)"
_assert_json "$REF_INGEST_JSON" "o['status'] == 'ok'" "ingest (v1 reference) succeeded"
_assert_json "$REF_INGEST_JSON" "bool(o['pathspec'])" "ingest (v1 reference) produced a pathspec"
REFERENCE="$(_pathspec "$REF_INGEST_JSON")"
echo "   reference_ref (v1) = ${REFERENCE}"

# =============================================================================
# (3) INGEST the CURRENT (v2 shifted) batch as its own data object.
#     Expected: status ok + a non-empty pathspec (the current dataset_ref).
# =============================================================================
CUR_INGEST_JSON="$($LOOM ingest --source "$WORK/current" --name "${DATASET_NAME}-v2" --json)"
_assert_json "$CUR_INGEST_JSON" "o['status'] == 'ok'" "ingest (v2 current) succeeded"
_assert_json "$CUR_INGEST_JSON" "bool(o['pathspec'])" "ingest (v2 current) produced a pathspec"
CURRENT="$(_pathspec "$CUR_INGEST_JSON")"
echo "   current_ref   (v2) = ${CURRENT}"

# =============================================================================
# (4) SEED run health: land one REAL keyless ValidateFlow run on the baseline so
#     `ops --flow` has at least one finished run to count. validate needs no key.
#     Expected: status ok + a VERDICT (PASS / REVIEW).
# =============================================================================
VALIDATE_JSON="$($LOOM validate --dataset "$REFERENCE" --target target --json)"
_assert_json "$VALIDATE_JSON" "o['status'] == 'ok'" "validate (seed run) succeeded"
_assert_json "$VALIDATE_JSON" "bool(o['VERDICT'])" "validate produced a VERDICT"
echo "   seeded a ValidateFlow run for the run-health view"

# =============================================================================
# (5a) ops --flow ValidateFlow -- the RUN-HEALTH view. Reads recent ValidateFlow
#      runs through the Metaflow Client API (read-only) and rolls up success /
#      failure counts. Expected: status ok; summary.health scoped to ValidateFlow
#      with n_runs >= 1 and n_runs == n_successful + n_failed; VERDICT is one of
#      OK / ATTENTION / EMPTY.
# =============================================================================
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
_assert_json "$OPS_HEALTH_JSON" "o['VERDICT'] in ('OK', 'ATTENTION', 'EMPTY')" \
  "ops --flow carried a status VERDICT"

# =============================================================================
# (5b) ops --dataset <v2> --reference <v1> -- the DATA DRIFT check. Materializes
#      both data objects via the Client API and compares their summary stats. The
#      v2 frame's numeric features moved well past the threshold, so:
#      Expected: status ok; summary.drift present with drift == True,
#      status == "DRIFT", a non-empty drift_flags list naming the moved feature_*
#      columns, and a comparison over shared columns; the overall VERDICT degrades
#      to ATTENTION.
# =============================================================================
OPS_DRIFT_JSON="$($LOOM ops --dataset "$CURRENT" --reference "$REFERENCE" --json)"
_assert_json "$OPS_DRIFT_JSON" "o['status'] == 'ok'" "ops --dataset --reference succeeded"
_assert_json "$OPS_DRIFT_JSON" "o['summary'].get('drift') and 'status' in o['summary']['drift']" \
  "ops drift check reported a drift block"
_assert_json "$OPS_DRIFT_JSON" \
  "isinstance(o['summary']['drift'].get('n_shared_columns'), int) and o['summary']['drift']['n_shared_columns'] > 0" \
  "ops drift compared the two data objects over shared columns"
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
