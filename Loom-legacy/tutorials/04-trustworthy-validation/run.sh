#!/usr/bin/env bash
# =============================================================================
# tutorials/04-trustworthy-validation/run.sh
#
# "Validation you can trust + a model card."
#
# A self-contained, KEYLESS, self-checking tutorial run. It:
#   1. generates a small deterministic synthetic tabular dataset that carries a
#      *sensitive attribute* column (`group`) alongside an honest learnable
#      signal -- inline, no downloads;
#   2. ingests it into a live Metaflow data object under a UNIQUE name (so
#      concurrent / repeat runs never collide);
#   3. runs `loom validate` with the rigorous-eval switches -- stratified K-fold
#      CV + a sealed holdout + probability calibration + per-slice FAIRNESS
#      (`--sensitive group`) + leakage checks -- and ASSERTS the --json envelope
#      actually carries every one of those signals plus a VERDICT;
#   4. runs `loom report` on that validate run to assemble the MODEL CARD and
#      ASSERTS the card was built.
#
# It uses ONLY keyless verbs (ingest / validate / report). The LLM-driven search
# verbs (`optimize` / `run`) cost money and are intentionally NOT run here -- see
# the README for how they slot in as a "needs an LLM key" next step.
#
# Run it directly:
#     cd /Users/anub/Work/Loom
#     bash tutorials/04-trustworthy-validation/run.sh
#
# Exits 0 on success; NONZERO on any regression.
# =============================================================================
set -euo pipefail

# --- Make the run self-contained: pick up the live cluster env if present ----
if [ -f /tmp/loom-cluster-env.sh ]; then
  # shellcheck disable=SC1091
  source /tmp/loom-cluster-env.sh
fi

# --- LOOM: the verb entrypoint (venv module form; no PATH/console-script) -----
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
# _assert_json -- parse a --json envelope into `o` and evaluate a bool expr.
# On a falsy result (or parse error) it prints the message + offending JSON to
# stderr and exits NONZERO, failing the run.
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

# =============================================================================
# (1) GENERATE deterministic synthetic data WITH a sensitive attribute column.
#
# A handful of informative `feature_*` columns drive a logit; the binary label
# is a seeded Bernoulli draw on that logit, so a baseline learns a REAL (not
# perfect) signal -> a healthy PASS validate. A `group` column (values "A"/"B")
# is the *sensitive attribute* we ask validate to slice fairness on. It is
# generated INDEPENDENTLY of the label, so it is not predictive and is not a
# leak -- exactly what you want when auditing for disparate performance.
# =============================================================================
cat > "$WORK/make_data.py" <<'PY'
import argparse, os
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-rows", type=int, default=1600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n, k = args.n_rows, 5
    X = rng.normal(size=(n, k))
    weights = np.array([1.3, -1.0, 0.8, -0.6, 0.4])
    logit = X @ weights
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(size=n) < prob).astype(int)

    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(k)])
    # Sensitive attribute, drawn INDEPENDENTLY of the label (not a predictor,
    # not a leak) -- the column we slice fairness on.
    df["group"] = rng.choice(["A", "B"], size=n)
    df["target"] = y

    os.makedirs(args.out_dir, exist_ok=True)
    df.to_csv(os.path.join(args.out_dir, "train.csv"), index=False)
    print(f"wrote {n} rows -> {args.out_dir}/train.csv")

if __name__ == "__main__":
    main()
PY

"$PY" "$WORK/make_data.py" --out-dir "$WORK/data"

# =============================================================================
# (2) INGEST it under the UNIQUE name -> capture the data-object pathspec.
# =============================================================================
INGEST_JSON="$($LOOM ingest --source "$WORK/data" --name "$DATASET_NAME" --json)"
_assert_json "$INGEST_JSON" "o['status'] == 'ok'" "ingest succeeded"
_assert_json "$INGEST_JSON" "bool(o['pathspec'])" "ingest produced a dataset pathspec"
DATASET="$(printf '%s' "$INGEST_JSON" | "$PY" -c 'import sys,json; print(json.load(sys.stdin)["pathspec"])')"
echo "   dataset_ref = ${DATASET}"

# =============================================================================
# (3) VALIDATE rigorously -- CV + sealed holdout + calibration + FAIRNESS
#     (--sensitive group) + leakage checks. Assert the --json carries every
#     rigorous-eval signal AND a VERDICT.
# =============================================================================
echo "-- validate (CV + sealed holdout + calibration + fairness + leakage) --"
VALIDATE_JSON="$($LOOM validate --dataset "$DATASET" --target target --sensitive group --json)"
_assert_json "$VALIDATE_JSON" "o['status'] == 'ok'" "validate succeeded"
_assert_json "$VALIDATE_JSON" "o['VERDICT'] in ('PASS', 'REVIEW', 'FAIL')" "validate emitted a VERDICT"
_assert_json "$VALIDATE_JSON" "o['VERDICT'] == 'PASS'" "validate VERDICT is PASS (honest learnable signal, no leak)"

# Cross-validation: a stratified K-fold mean +/- std over multiple folds.
_assert_json "$VALIDATE_JSON" "bool(o['summary'].get('cv'))" "validate carries a CV block"
_assert_json "$VALIDATE_JSON" "len(o['summary']['cv'].get('scores') or []) >= 2" "CV ran multiple folds"
_assert_json "$VALIDATE_JSON" "isinstance(o['summary']['cv'].get('mean'), (int, float))" "CV reports a mean score"
_assert_json "$VALIDATE_JSON" "isinstance(o['summary']['cv'].get('std'), (int, float))" "CV reports a std (fold spread)"

# Sealed holdout: a separate slice never seen by CV, with its own score + size.
_assert_json "$VALIDATE_JSON" "bool(o['summary'].get('holdout'))" "validate carries a sealed-holdout block"
_assert_json "$VALIDATE_JSON" "isinstance(o['summary']['holdout'].get('score'), (int, float))" "holdout reports a score"
_assert_json "$VALIDATE_JSON" "int(o['summary']['holdout'].get('n', 0)) > 0" "holdout has a nonzero size"

# Probability calibration: reliability bins + a Brier score.
_assert_json "$VALIDATE_JSON" "bool(o['summary'].get('calibration'))" "validate carries a calibration block"
_assert_json "$VALIDATE_JSON" "len(o['summary']['calibration'].get('bins') or []) > 0" "calibration has reliability bins"
_assert_json "$VALIDATE_JSON" "isinstance(o['summary']['calibration'].get('brier'), (int, float))" "calibration reports a Brier score"

# Fairness / per-slice metrics for the sensitive attribute we passed.
_assert_json "$VALIDATE_JSON" "bool(o['summary'].get('slice_metrics'))" "validate carries per-slice fairness metrics"
_assert_json "$VALIDATE_JSON" "set(o['summary']['slice_metrics'].keys()) == {'A', 'B'}" "fairness sliced on the sensitive 'group' values A and B"
_assert_json "$VALIDATE_JSON" "all(s.get('score') is not None for s in o['summary']['slice_metrics'].values())" "each fairness slice has its own score"
_assert_json "$VALIDATE_JSON" "isinstance(o['summary']['slice_metrics']['A'].get('score'), (int, float))" "fairness slice A score is numeric"
_assert_json "$VALIDATE_JSON" "isinstance(o['summary']['slice_metrics']['B'].get('score'), (int, float))" "fairness slice B score is numeric"

# Leakage checks: a clean dataset has NO leakage flags.
_assert_json "$VALIDATE_JSON" "'leakage_flags' in o['summary']" "validate ran leakage checks"
_assert_json "$VALIDATE_JSON" "len(o['summary'].get('leakage_flags') or []) == 0" "no leakage flagged on the clean dataset"
_assert_json "$VALIDATE_JSON" "o['summary'].get('leakage') is False" "leakage verdict is False (clean)"

# The validate run pathspec feeds the model-card report.
_assert_json "$VALIDATE_JSON" "bool(o['pathspec'])" "validate produced a ValidateFlow run pathspec"
VALIDATE_RUN="$(printf '%s' "$VALIDATE_JSON" | "$PY" -c 'import sys,json; print(json.load(sys.stdin)["pathspec"])')"
echo "   validate_run = ${VALIDATE_RUN}"

# =============================================================================
# (4) REPORT -- assemble the MODEL CARD from the validate run. Assert the card
#     was built and reflects the validated run.
# =============================================================================
echo "-- report (build the model card) --"
REPORT_JSON="$($LOOM report --runs "$VALIDATE_RUN" --json)"
_assert_json "$REPORT_JSON" "o['status'] == 'ok'" "report succeeded"
_assert_json "$REPORT_JSON" "o['VERDICT'] == 'OK'" "report VERDICT is OK"
_assert_json "$REPORT_JSON" "bool(o['card_path'])" "report built a model card (card_path present)"
_assert_json "$REPORT_JSON" "bool(o['pathspec'])" "report produced a ReportFlow run pathspec"
_assert_json "$REPORT_JSON" "int(o['summary'].get('n_runs', 0)) == 1 and int(o['summary'].get('n_successful', 0)) == 1" "model card covers the 1 successful validate run"
_assert_json "$REPORT_JSON" "any(r.get('pathspec') == '$VALIDATE_RUN' for r in o['summary'].get('leaderboard') or [])" "model card's leaderboard names our validate run"
_assert_json "$REPORT_JSON" "isinstance(o['summary'].get('best_metric'), (int, float))" "model card records the best metric"

echo "== PASS: $(basename "$HERE")"
