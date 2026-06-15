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

# --- 3a. ISOLATE the telemetry corpus into $WORK + turn capture ON. ----------
# The telemetry layer assembles trajectories from the COMMAND-LEVEL ROLLOUTS
# every verb records (joined on the rollout's task.experiment_id) plus the
# telemetry events. Both the rollout flywheel and the event log default to
# files anchored at the cwd, so we point all four signals at this run's private
# scratch dir: the example then sees ONLY the trajectories ITS OWN verbs
# generate -- no collision with a developer's real corpus or with a concurrent
# example. LOOM_TELEMETRY=1 turns event capture on (rollouts are recorded
# regardless; the JOIN works either way). Exported here, before any verb that
# records a rollout. (ingest records none, so section 2 stays out of the corpus.)
export LOOM_TELEMETRY=1
export LOOM_TELEMETRY_PATH="${WORK}/telemetry/events.jsonl"
export LOOM_TRAJECTORIES_PATH="${WORK}/telemetry/trajectories.jsonl"
export LOOM_LEARNINGS_PATH="${WORK}/learnings/rollouts.jsonl"
export LOOM_PROXY_LOG_PATH="${WORK}/learnings/proxy_calls.jsonl"

# --- 3b. eda --target: a read-only profile that records a ROLLOUT. -----------
# The keyless EDA verb. Beyond profiling the data, it appends a command="eda"
# rollout whose task.experiment_id is THIS dataset_ref -- a trajectory the
# telemetry layer can assemble + distill. Clean data -> no leakage flags.
EDA_JSON="$($LOOM eda --dataset "$DATASET" --target target --json)"
_assert_json "$EDA_JSON" "o['status'] == 'ok'" "eda succeeded (records a rollout)"
_assert_json "$EDA_JSON" "o['summary'].get('nrows', 0) > 0" "eda saw rows (nrows)"
_assert_json "$EDA_JSON" "len(o['summary'].get('leakage_flags') or []) == 0" \
  "clean data -> eda flags NO leakage"

# --- 3c. a SECOND dataset + validate: a DISTINCT trajectory. -----------------
# Each verb keys its rollout on task.experiment_id (the dataset_ref for eda /
# validate), so eda + validate on the SAME dataset would collide into one
# trajectory. We ingest a second data object (a distinct pathspec, hence a
# distinct experiment_id) and validate it -> a second, separate trajectory.
# This proves a couple of keyless verbs each produce their own distillable
# rollout. validate also carries a real reward (the holdout score / PASS).
DATASET_NAME_2="${DATASET_NAME}-b"
INGEST_JSON_2="$($LOOM ingest --source "$WORK" --name "$DATASET_NAME_2" --json)"
_assert_json "$INGEST_JSON_2" "o['status'] == 'ok'" "second ingest succeeded"
DATASET_2="$(printf '%s' "$INGEST_JSON_2" | "$PY" -c 'import sys,json; print(json.load(sys.stdin)["pathspec"])')"
echo "   dataset_ref_2 = ${DATASET_2}"

VALIDATE_JSON="$($LOOM validate --dataset "$DATASET_2" --target target --json)"
_assert_json "$VALIDATE_JSON" "o['status'] == 'ok'" "validate succeeded (records a rollout)"
_assert_json "$VALIDATE_JSON" "o['VERDICT'] in ('PASS', 'REVIEW')" \
  "validate emitted a VERDICT (the rollout's reward signal)"

# --- 3d. telemetry status: the read-only corpus snapshot. --------------------
# Summarizes the assembled corpus: events, trajectories, and the IP-boundary
# split (general vs tenant-owned). Our two general rollouts -> >=2 general
# trajectories, 0 tenant-owned. No VERDICT (a pure read).
TELEMETRY_JSON="$($LOOM telemetry status --json)"
_assert_json "$TELEMETRY_JSON" "o['verb'] == 'telemetry-status'" "telemetry status verb tag"
_assert_json "$TELEMETRY_JSON" "o['status'] == 'ok'" "telemetry status succeeded"
_assert_json "$TELEMETRY_JSON" "isinstance(o['summary'].get('n_events'), int)" \
  "telemetry status reports an event count"
_assert_json "$TELEMETRY_JSON" "o['summary'].get('n_trajectories', 0) >= 2" \
  "telemetry assembled >=2 trajectories (eda + validate)"
_assert_json "$TELEMETRY_JSON" \
  "bool(o['summary'].get('ip_boundary')) and o['summary']['ip_boundary'].get('general', 0) >= 2" \
  "the IP boundary counts our >=2 general trajectories"
_assert_json "$TELEMETRY_JSON" \
  "o['summary']['ip_boundary'].get('tenant_owned', 0) == 0" \
  "no tenant-owned trajectories leaked into the corpus"

# --- 3e. telemetry export: the general-only, REDACTED SFT corpus. ------------
# `telemetry export` has NO --json flag, so we parse its prose AND read the
# JSONL it writes. The export is general-only by default (the IP boundary) and
# content is REDACTED BY DEFAULT (the prompt-hygiene invariant): the SFT rows
# carry the <REDACTED:...> sentinel, never raw rows. Each row is one LOOM-DS-1
# SFT example: context (chat messages) -> teacher_output + tools_trajectory +
# reward/weight, tagged owned_by=general.
EXPORT_OUT="${WORK}/loom-ds-1.jsonl"
EXPORT_PROSE="$($LOOM telemetry export --out "$EXPORT_OUT")"
printf '%s\n' "$EXPORT_PROSE"

# Assert the prose: >=1 example written + content REDACTED by default.
printf '%s' "$EXPORT_PROSE" | "$PY" -c '
import sys, re
prose = sys.stdin.read()
m = re.search(r"examples written:\s*(\d+)", prose)
n = int(m.group(1)) if m else -1
if n < 1:
    sys.stderr.write(f"ASSERT FAIL: telemetry export wrote >=1 example\n  prose:\n{prose}\n")
    sys.exit(1)
if "REDACTED (default)" not in prose:
    sys.stderr.write(f"ASSERT FAIL: telemetry export content is REDACTED by default\n  prose:\n{prose}\n")
    sys.exit(1)
sys.stderr.write(f"  ok: telemetry export wrote {n} example(s), content REDACTED (default)\n")
'

# Assert the JSONL: every row is a general-only, redacted SFT example with the
# context / teacher_output / tools_trajectory / reward / weight shape, and NO
# raw row content leaked (a <REDACTED:...> sentinel is present, never bulk data).
"$PY" - "$EXPORT_OUT" <<'PYEOF'
import json, sys

path = sys.argv[1]
rows = []
with open(path, encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

def fail(msg):
    sys.stderr.write(f"ASSERT FAIL: {msg}\n  path: {path}\n  rows: {json.dumps(rows)[:800]}\n")
    sys.exit(1)

if len(rows) < 1:
    fail("export JSONL has >=1 SFT example")

required = {"trajectory_id", "verb", "context", "teacher_output",
            "tools_trajectory", "reward", "weight", "owned_by"}
seen_redaction = False
for r in rows:
    missing = required - set(r)
    if missing:
        fail(f"SFT example missing keys {sorted(missing)}")
    if r["owned_by"] != "general":
        fail(f"IP boundary: every example must be owned_by=general, got {r['owned_by']!r}")
    if not isinstance(r["context"], list) or not r["context"]:
        fail("SFT example context must be a non-empty list of chat messages")
    if not all(isinstance(m, dict) and "role" in m and "content" in m for m in r["context"]):
        fail("SFT example context messages must have role + content")
    if not isinstance(r["tools_trajectory"], list):
        fail("SFT example tools_trajectory must be a list")
    # Prompt hygiene: content is redacted by default -> the typed sentinel
    # appears somewhere in the serialized example, never raw rows.
    if "<REDACTED:" in json.dumps(r):
        seen_redaction = True

if not seen_redaction:
    fail("redaction: no <REDACTED:...> sentinel found -- content was NOT redacted by default")

sys.stderr.write(
    f"  ok: {len(rows)} SFT example(s), all owned_by=general, redacted (sentinel present), "
    "shape context/teacher_output/tools_trajectory/reward/weight\n"
)
PYEOF

echo "== PASS: $(basename "$HERE")"
