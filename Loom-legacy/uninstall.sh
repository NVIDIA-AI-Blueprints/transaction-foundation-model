#!/usr/bin/env bash
#
# Loom uninstaller — reverses ./install.sh. Gates every destructive step (y/N).
# It does NOT delete the repo clone itself; remove the directory yourself if you
# want it gone. It also can't edit your shell profile — remove any LOOM_PYTHON
# export you added there by hand.
#
# Usage:
#   ./uninstall.sh          confirm each step
#   ./uninstall.sh --yes    assume yes (no prompts)
set -uo pipefail   # NOT -e: keep going past a piece that's already gone

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

ASSUME_YES=0
case "${1:-}" in
  --yes|-y) ASSUME_YES=1 ;;
  "") ;;
  -h|--help) echo "usage: ./uninstall.sh [--yes]   (reverses install.sh; gates each destructive step)"; exit 0 ;;
  *) echo "usage: ./uninstall.sh [--yes]" >&2; exit 2 ;;
esac

confirm() {  # confirm "<question>" -> 0 (yes) / 1 (no)
  [ "$ASSUME_YES" = 1 ] && return 0
  read -r -p "$1 [y/N] " ans
  [ "$ans" = "y" ] || [ "$ans" = "Y" ]
}

echo "Uninstalling Loom from: $REPO_ROOT"
echo

# 1 — the global `loom` command (npm link / install)
if confirm "Remove the global 'loom' command (npm)?"; then
  if npm rm -g @loom/cli >/dev/null 2>&1; then :; else ( cd cli && npm unlink >/dev/null 2>&1 || true ); fi
  echo "   removed the 'loom' command."
fi

# 2 — the local datastore (minio namespace; then optionally the whole cluster)
if command -v kubectl >/dev/null 2>&1 && kubectl get namespace loom >/dev/null 2>&1; then
  if confirm "Delete the 'loom' namespace (the minio datastore) from minikube?"; then
    kubectl delete namespace loom >/dev/null 2>&1 || true
    echo "   deleted the 'loom' namespace."
  fi
fi
if command -v minikube >/dev/null 2>&1 && minikube status >/dev/null 2>&1; then
  if confirm "Delete the minikube cluster entirely? (only if Loom is its only user)"; then
    minikube delete >/dev/null 2>&1 || true
    echo "   minikube deleted."
    if command -v colima >/dev/null 2>&1 && confirm "Stop colima too?"; then
      colima stop >/dev/null 2>&1 || true
      echo "   colima stopped."
    fi
  fi
fi

# 3 — the Python venv
if [ -d .venv ] && confirm "Remove the Python venv (.venv)?"; then
  rm -rf .venv && echo "   removed .venv."
fi

# 4 — generated datastore config
if [ -f .env.metaflow ] && confirm "Remove .env.metaflow?"; then
  rm -f .env.metaflow && echo "   removed .env.metaflow."
fi

# 5 — the agent home runtime (installed Pi packages, sessions, login, materialized agents)
if confirm "Reset the agent home runtime (cli/home: installed packages, sessions, login, agents)?"; then
  rm -rf cli/home/npm cli/home/agents cli/home/sessions cli/home/.cache \
         cli/home/auth.json cli/home/.loom-bootstrap.json cli/home/mcp-npx-cache.json \
         cli/home/trust.json cli/home/models.json 2>/dev/null || true
  echo "   reset the agent home runtime."
fi

echo
echo "Done — Loom's installed pieces are removed; the repo clone is left in place."
echo "If you added 'export LOOM_PYTHON=...' to your shell profile, remove it by hand."
