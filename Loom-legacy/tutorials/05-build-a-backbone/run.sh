#!/usr/bin/env bash
# =============================================================================
# tutorials/05-build-a-backbone/run.sh
#
# "Building a model backbone with no GPU" -- a self-checking, KEYLESS tutorial
# for Loom's model-builder (`train`) seam.
#
# It generates a deterministic, domain-neutral per-account EVENT-SEQUENCE
# fixture INLINE (no downloads, no separate data script), ingests it under a
# UNIQUELY-NAMED dataset (so concurrent / repeat runs never collide), then runs
# the keyless verb sequence with --json and ASSERTS the outcomes inline. Any
# regression exits NONZERO -- that is the whole point.
#
# Run it directly (after the local Metaflow + minio datastore is up):
#     source /tmp/loom-cluster-env.sh   # done automatically below if present
#     bash tutorials/05-build-a-backbone/run.sh
#
# HARD RULE: KEYLESS ONLY. The CPU backbone build uses the torch-free `local`
# model-builder adapter (LOOM_MODEL_BUILDER_PROVIDER=local) -- no GPU, no model
# key, no spend. The real GPU pretrain (`train --launch` with LOOM_GPU_TARGET)
# is NEVER run here; this script only proves it GATES cleanly with no target.
# =============================================================================
set -euo pipefail

# --- Self-contained cluster env ----------------------------------------------
# Source the local Metaflow + minio datastore env so the script runs on its own.
if [ -f /tmp/loom-cluster-env.sh ]; then
  # shellcheck disable=SC1091
  source /tmp/loom-cluster-env.sh
fi

# --- LOOM: the verb entrypoint -----------------------------------------------
# Verbs are invoked as `$LOOM <verb> ... --json`. Defaults to the venv module
# form so no PATH / console-script is needed. Overridable via $LOOM / $PY.
LOOM="${LOOM:-/Users/anub/Work/Loom/.venv/bin/python -m loom}"
PY="${PY:-/Users/anub/Work/Loom/.venv/bin/python}"

# --- Where this tutorial lives + a private scratch dir -----------------------
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/loom-tutorial-XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

# --- A UNIQUE dataset name so concurrent / repeat runs never collide ---------
# Stable-prefixed (the tutorial dir) + pid + a random + a timestamp suffix.
RUN_TAG="$(basename "$HERE")-$$-${RANDOM}-$(date +%s)"
DATASET_NAME="tut-${RUN_TAG}"

# =============================================================================
# _assert_json -- the ONE assert helper (no jq dependency).
#
# Usage: _assert_json "<json-string>" "<python-bool-expr-over-`o`>" "<message>"
# Parses the JSON envelope into `o` (a dict) and evaluates the boolean
# expression. On a falsy result (or a parse error) it prints the message + the
# offending JSON to stderr and exits NONZERO, failing the run.
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

# A tiny PASS/FAIL banner so a human running it sees the verdict at a glance.
_pass() { echo "== PASS: $(basename "$HERE") -- backbone built keyless, GPU gate honored"; }
_fail() { echo "== FAIL: $(basename "$HERE") -- $1" >&2; exit 1; }

echo "== tutorial: $(basename "$HERE")  dataset=${DATASET_NAME}  work=${WORK}"

# =============================================================================
# (1) GENERATE the deterministic synthetic data INLINE into $WORK.
#
# A domain-neutral, per-account EVENT-SEQUENCE fixture with a PLANTED next-event
# signal: positive accounts (label=1) follow a first-order Markov chain
# A->B->C->D->E->A with 85% probability at each step; negative accounts (label=0)
# emit i.i.d. random events. That adjacent-event co-occurrence is exactly what
# the pooled PPMI+SVD CPU backbone learns. Single seeded numpy RNG => a repeat
# run is byte-identical. No downloads.
# =============================================================================
"$PY" - "$WORK" <<'PYEOF'
import sys
from pathlib import Path
import numpy as np
import pandas as pd

out_dir = Path(sys.argv[1])
out_dir.mkdir(parents=True, exist_ok=True)

SEED = 0
N_ACCOUNTS = 160
ALPHABET = ["A", "B", "C", "D", "E"]
CHAIN = {"A": "B", "B": "C", "C": "D", "D": "E", "E": "A"}
CHAIN_PROB = 0.85
MIN_SEQ_LEN, MAX_SEQ_LEN = 8, 16

rng = np.random.default_rng(SEED)
rows = []
for acct in range(N_ACCOUNTS):
    positive = acct % 2 == 0
    seq_len = int(rng.integers(MIN_SEQ_LEN, MAX_SEQ_LEN))
    if positive:
        cur = ALPHABET[int(rng.integers(0, len(ALPHABET)))]
        events = [cur]
        for _ in range(seq_len - 1):
            if rng.random() < CHAIN_PROB:
                cur = CHAIN[cur]
            else:
                cur = ALPHABET[int(rng.integers(0, len(ALPHABET)))]
            events.append(cur)
    else:
        events = [ALPHABET[int(rng.integers(0, len(ALPHABET)))] for _ in range(seq_len)]
    for t, ev in enumerate(events):
        rows.append(
            {
                "account": f"acct-{acct:03d}",
                "t": t,
                "event": ev,
                "amount": float(rng.normal(10.0, 2.0)),
                "label": int(positive),
            }
        )

frame = pd.DataFrame(rows)
path = out_dir / "train.csv"
frame.to_csv(path, index=False)
print(f"Wrote {len(frame)} rows across {N_ACCOUNTS} accounts -> {path}")
PYEOF

# =============================================================================
# (2) INGEST it under the UNIQUE name -> capture the data-object pathspec.
#     The one external->Metaflow boundary; the CSV becomes a versioned data
#     object that everything downstream consumes by pathspec.
# =============================================================================
INGEST_JSON="$($LOOM ingest --source "$WORK" --name "$DATASET_NAME" --json)"
_assert_json "$INGEST_JSON" "o['status'] == 'ok'" "ingest succeeded"
_assert_json "$INGEST_JSON" "bool(o['pathspec'])" "ingest produced a dataset pathspec"
DATASET="$(printf '%s' "$INGEST_JSON" | "$PY" -c 'import sys,json; print(json.load(sys.stdin)["pathspec"])')"
echo "   dataset_ref = ${DATASET}"

# =============================================================================
# (3) BUILD the backbone on CPU with the torch-free `local` adapter.
#
#     LOOM_MODEL_BUILDER_PROVIDER=local selects the PPMI+TruncatedSVD stand-in
#     (no GPU, no torch, sub-2s, deterministic). --objective next-event learns
#     the planted adjacent-pair (first-order Markov) co-occurrence; --budget
#     probe is the cheapest embedding dimensionality. The real heavy GPU launch
#     stays OFF (no --launch), so this consumes ZERO GPU-hours.
#
#     Expected: status ok, STATUS/VERDICT == "BUILT", a deterministic sha256:
#     fingerprint, and a backbone run pathspec (a first-class dataset_ref).
# =============================================================================
TRAIN_LOCAL_JSON="$(LOOM_MODEL_BUILDER_PROVIDER=local $LOOM train \
  --dataset "$DATASET" --objective next-event --budget probe --json)"
_assert_json "$TRAIN_LOCAL_JSON" "o['status'] == 'ok'" "local train run succeeded"
_assert_json "$TRAIN_LOCAL_JSON" "o['VERDICT'] == 'BUILT'" \
  "local train BUILT the backbone (the CPU adapter actually builds, no GPU)"
_assert_json "$TRAIN_LOCAL_JSON" "o['summary'].get('status') == 'BUILT'" \
  "local train summary STATUS is BUILT"
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
_assert_json "$TRAIN_LOCAL_JSON" "o['summary'].get('launch') == False" \
  "the real GPU launch stayed OFF for the CPU build (zero GPU-hours)"
BACKBONE="$(printf '%s' "$TRAIN_LOCAL_JSON" | "$PY" -c 'import sys,json; print(json.load(sys.stdin)["pathspec"])')"
echo "   backbone_ref = ${BACKBONE}"

# =============================================================================
# (4) Show the PRODUCTION GPU path GATES cleanly with no target.
#
#     The default model-builder provider is `nemo` (the GPU lowering compiler).
#     With no configured LOOM_GPU_TARGET it REFUSES up front rather than
#     launching -- the safe-by-default heavy-launch posture. The run still
#     COMPLETES (a clean refusal IS the correct outcome, not a crash); it just
#     refuses to consume GPU. The real GPU pretrain is `train --launch` with a
#     LOOM_GPU_TARGET set -- deferred, never run here.
#
#     Expected: status ok, VERDICT == "REFUSED_NO_GPU_TARGET", no artifact.
# =============================================================================
TRAIN_GPU_JSON="$($LOOM train \
  --dataset "$DATASET" --objective next-event --budget probe --json)"
_assert_json "$TRAIN_GPU_JSON" "o['status'] == 'ok'" \
  "the GPU gate run completed (a clean refusal is a success, not a crash)"
_assert_json "$TRAIN_GPU_JSON" "o['VERDICT'] == 'REFUSED_NO_GPU_TARGET'" \
  "the GPU production path REFUSES with no LOOM_GPU_TARGET -- the heavy-launch gate"
_assert_json "$TRAIN_GPU_JSON" "o['summary'].get('status') == 'REFUSED_NO_GPU_TARGET'" \
  "GPU train summary STATUS is REFUSED_NO_GPU_TARGET"
_assert_json "$TRAIN_GPU_JSON" "not o['summary'].get('artifact_pathspec')" \
  "the refused GPU run produced NO backbone artifact (it did not launch)"
_assert_json "$TRAIN_GPU_JSON" "o['summary'].get('launch') == False" \
  "the heavy GPU launch is OFF by default (--launch not set)"

_pass
