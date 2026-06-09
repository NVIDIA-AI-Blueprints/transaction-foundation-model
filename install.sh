#!/usr/bin/env bash
#
# Loom installer. Prefers an AI assistant (claude / codex) to DRIVE the install —
# it follows INSTALL.md, adapts to your machine, watches for errors, and verifies
# with `loom doctor`. With neither assistant present, it runs the same steps as a
# plain script.
#
# Usage:
#   ./install.sh            auto — use an assistant if available, else scripted
#   ./install.sh --shell    force the scripted path (no assistant)
#   ./install.sh --agent    force the assistant path (errors if none found)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

MODE="auto"
case "${1:-}" in
  --shell) MODE="shell" ;;
  --agent) MODE="agent" ;;
  "")      MODE="auto" ;;
  -h|--help) echo "usage: ./install.sh [--shell|--agent]   (default: drive via claude/codex if present, else scripted)"; exit 0 ;;
  *) echo "usage: ./install.sh [--shell|--agent]" >&2; exit 2 ;;
esac

detect_agent() {
  if command -v claude >/dev/null 2>&1; then echo "claude"
  elif command -v codex >/dev/null 2>&1; then echo "codex"
  else echo ""; fi
}

AGENT_PROMPT='Install Loom on this macOS machine by following ./INSTALL.md exactly, in order, from the repo root. Run each step. After the Python step, CONFIRM that `pip install -e .` fully completed — `python -c "import metaflow"` must succeed — before continuing; a partial install (often a build failure in the heavy AIDE dependency) is the most common problem. Finish by starting the datastore port-forward and running `loom doctor`, iterating on any FAIL until it prints "VERDICT: PASS". Ask me before anything destructive or any sudo.'

run_agent() {
  local agent="$1"
  echo "==> Found '$agent' — letting it install Loom for you."
  echo "    It follows INSTALL.md, adapts to your machine, and verifies with 'loom doctor'."
  echo "    Approve its steps as it goes (it will ask before anything destructive)."
  echo
  exec "$agent" "$AGENT_PROMPT"
}

run_shell() {
  echo "==> Scripted install (no AI assistant used)."

  echo "--> [1/3] engine (Python)"
  command -v python3.12 >/dev/null 2>&1 || { echo "    python3.12 not found. Run: brew install python@3.12" >&2; exit 1; }
  [ -d .venv ] || python3.12 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install -e .
  python -c "import metaflow" >/dev/null 2>&1 || {
    echo "    Metaflow is not importable after 'pip install -e .' — the install did not complete." >&2
    echo "    Re-run 'pip install -e .' from the repo root and read the first error." >&2
    exit 1
  }
  export LOOM_PYTHON="$REPO_ROOT/.venv/bin/python"

  echo "--> [2/3] the 'loom' command (Node)"
  command -v node >/dev/null 2>&1 || { echo "    node not found. Run: brew install node" >&2; exit 1; }
  ( cd cli && npm install && npm run build && npm link )

  echo "--> [3/3] local datastore"
  bash scripts/setup_metaflow_minikube.sh

  cat <<NEXT

==> Engine + CLI + datastore are installed.

   1) Add this to your shell profile so every new shell finds the engine:
        export LOOM_PYTHON="$REPO_ROOT/.venv/bin/python"

   2) Start the datastore port-forward (keep it running) and verify:
        kubectl port-forward -n loom svc/minio 9000:9000 9001:9001 &
        source .env.metaflow && loom doctor        # expect VERDICT: PASS

   3) Run it:
        loom
NEXT
}

case "$MODE" in
  agent)
    a="$(detect_agent)"
    [ -n "$a" ] || { echo "no claude/codex CLI found on PATH — install one, or run: ./install.sh --shell" >&2; exit 1; }
    run_agent "$a"
    ;;
  shell)
    run_shell
    ;;
  auto)
    a="$(detect_agent)"
    if [ -n "$a" ]; then
      run_agent "$a"
    else
      echo "==> No claude/codex CLI found — falling back to the scripted install."
      echo "    (Install Claude Code or Codex and re-run for a guided install.)"
      echo
      run_shell
    fi
    ;;
esac
