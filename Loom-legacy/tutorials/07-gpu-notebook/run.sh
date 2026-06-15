#!/usr/bin/env bash
# =============================================================================
# tutorials/07-gpu-notebook/run.sh
#
# "Interactive GPU notebooks: `loom notebook`" -- a self-checking smoke for the
# notebook launcher's PLANNING + ROUTING path.
#
# HARD RULE: FREE + NO SPEND. This NEVER launches a GPU. It only exercises the
# `--dry-run`/`--json` planning path (which builds the submission and emits it
# WITHOUT touching Modal) and the non-Modal-target refusal. The real launch
# (`loom notebook`, which bills an H100) is interactive and is NOT run here --
# see README.md "Validate everything works" for the live checklist.
#
# Run it directly (no datastore or Modal needed -- planning is pure):
#     bash tutorials/07-gpu-notebook/run.sh
# =============================================================================
set -euo pipefail

LOOM="${LOOM:-/Users/anub/Work/Loom/.venv/bin/python -m loom}"
PY="${PY:-/Users/anub/Work/Loom/.venv/bin/python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- _assert_json: the ONE assert helper (no jq) -----------------------------
_assert_json() {
  local json="$1" expr="$2" msg="$3"
  printf '%s' "$json" | "$PY" -c '
import sys, json
expr, msg = sys.argv[1], sys.argv[2]
raw = sys.stdin.read()
try:
    o = json.loads(raw)
except Exception as e:  # noqa: BLE001
    sys.stderr.write(f"ASSERT PARSE FAIL: {msg}\n  expr: {expr}\n  err: {e}\n  json: {raw!r}\n")
    sys.exit(1)
_ns = {"bool": bool, "len": len, "str": str, "int": int}
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

echo "== tutorial: $(basename "$HERE")  (FREE planning smoke -- never launches a GPU)"

# =============================================================================
# (1) The default plan: --json emits the submission spec WITHOUT launching.
#     Expected: status ok, VERDICT PLANNED, GPU=H100, the NeMo image, port 8888,
#     datastore forwarded, a Modal app name.
# =============================================================================
PLAN_JSON="$($LOOM notebook --json)"
_assert_json "$PLAN_JSON" "o['status'] == 'ok'" "notebook plan emitted (no launch)"
_assert_json "$PLAN_JSON" "o['VERDICT'] == 'PLANNED'" "the plan is PLANNED, not launched"
_assert_json "$PLAN_JSON" "o['summary'].get('gpu') == 'H100'" "plan requests an H100"
_assert_json "$PLAN_JSON" "bool(o['summary'].get('image'))" "plan names a container image"
_assert_json "$PLAN_JSON" "o['summary'].get('port') == 8888" "plan forwards the Jupyter port 8888"
_assert_json "$PLAN_JSON" "o['summary'].get('mount_datastore') == True" "plan forwards the datastore by default"
_assert_json "$PLAN_JSON" "bool(o['summary'].get('app_name'))" "plan resolves a Modal app name"

# =============================================================================
# (2) --no-datastore is carried into the plan.
# =============================================================================
NODS_JSON="$($LOOM notebook --no-datastore --json)"
_assert_json "$NODS_JSON" "o['summary'].get('mount_datastore') == False" \
  "--no-datastore turns off datastore forwarding in the plan"

# =============================================================================
# (3) modal://<app> routes to that named app.
# =============================================================================
APP_JSON="$($LOOM notebook --gpu modal://team-burst --json)"
_assert_json "$APP_JSON" "o['summary'].get('app_name') == 'team-burst'" \
  "modal://team-burst plans under the 'team-burst' app"

# =============================================================================
# (4) A non-Modal target REFUSES up front (no launch, actionable). The verb exits
#     NONZERO on an unsupported target (a usage error), so tolerate that here with
#     `|| true` -- we assert on the emitted JSON, not the exit code.
# =============================================================================
REFUSE_JSON="$($LOOM notebook --gpu slurm-cluster-x --json || true)"
_assert_json "$REFUSE_JSON" "o['status'] == 'error'" "a non-Modal target is an error"
_assert_json "$REFUSE_JSON" "o['VERDICT'] == 'REFUSED'" "non-Modal target REFUSED (no launch)"

echo "== PASS: $(basename "$HERE") -- planning + routing verified, zero GPU spend"
