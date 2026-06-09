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

# ---------------------------------------------------------------------------
# Preflight repair: Homebrew python@3.12 vs. the system libexpat.
#
# On recent macOS, the Homebrew python@3.12 bottle's `pyexpat` extension imports
# newer expat symbols but its load command points at the system
# /usr/lib/libexpat.1.dylib, which lacks them. The result: `import pyexpat` ->
# `import pip`/`ensurepip` -> `python3.12 -m venv` all fail, so the very first
# install step cannot even start (the failure surfaces as a cryptic ensurepip
# CalledProcessError). Repair it by relinking the affected extension modules
# against Homebrew's expat (which has the symbols) and re-signing ad-hoc.
#
# No-op when pyexpat already imports, when not on macOS, or when python3.12 is
# not yet installed. Non-sudo and reversible — each patched .so is backed up
# alongside itself as <name>.orig.bak (restore by hand, or `brew reinstall
# python@3.12`).
# ---------------------------------------------------------------------------
ensure_macos_python_expat() {
  [ "$(uname -s)" = "Darwin" ] || return 0
  command -v python3.12 >/dev/null 2>&1 || return 0

  local err
  if err="$(python3.12 -c 'import pyexpat' 2>&1)"; then
    return 0  # pyexpat already imports — nothing to do.
  fi
  case "$err" in
    *expat*) : ;;  # the known breakage; anything else is surfaced verbatim below.
    *) echo "    python3.12 cannot import pyexpat (unexpected error):" >&2; printf '%s\n' "$err" >&2; return 1 ;;
  esac

  echo "==> Repairing python3.12 pyexpat (Homebrew bottle links the system libexpat, which lacks newer symbols)."
  if ! command -v brew >/dev/null 2>&1; then
    echo "    Homebrew is required to repair this but 'brew' was not found — install it (https://brew.sh) and re-run." >&2
    return 1
  fi
  local tool
  for tool in install_name_tool codesign otool; do
    command -v "$tool" >/dev/null 2>&1 || {
      echo "    '$tool' is required — install the Xcode Command Line Tools (xcode-select --install) and re-run." >&2
      return 1
    }
  done

  brew list expat >/dev/null 2>&1 || brew install expat
  local brew_expat; brew_expat="$(brew --prefix expat)/lib/libexpat.1.dylib"
  [ -f "$brew_expat" ] || { echo "    expat is installed but $brew_expat is missing — cannot repair." >&2; return 1; }

  # The extension modules live in python's DESTSHARED (lib-dynload). Importing
  # sysconfig does NOT need pyexpat, so this resolves even while pyexpat is broken.
  local dynload; dynload="$(python3.12 -c 'import sysconfig; print(sysconfig.get_config_var("DESTSHARED"))')"
  local fixed=0 so cur
  for so in "$dynload"/*.so; do
    [ -e "$so" ] || continue
    cur="$(otool -L "$so" 2>/dev/null | awk '/libexpat/{print $1; exit}')" || true
    [ -n "$cur" ] || continue
    case "$cur" in /usr/lib/*|/System/*) : ;; *) continue ;; esac  # only repoint system links
    cp -n "$so" "$so.orig.bak" 2>/dev/null || true
    chmod u+w "$so" 2>/dev/null || true
    # A half-applied change (relinked but not re-signed) is unloadable, so on any
    # failure restore the backup and fail loudly rather than aborting mid-relink.
    if ! install_name_tool -change "$cur" "$brew_expat" "$so" \
        || ! codesign --force --sign - "$so" >/dev/null 2>&1; then
      [ -f "$so.orig.bak" ] && cp -f "$so.orig.bak" "$so"
      echo "    failed to relink/re-sign $(basename "$so") — restored the original." >&2
      return 1
    fi
    echo "    relinked $(basename "$so") -> $brew_expat"
    fixed=1
  done

  [ "$fixed" -eq 1 ] || { echo "    no extension module linked the system libexpat; cannot auto-repair. Original error:" >&2; printf '%s\n' "$err" >&2; return 1; }
  python3.12 -c 'import pyexpat' >/dev/null 2>&1 || { echo "    pyexpat still failing after relink. Original error:" >&2; printf '%s\n' "$err" >&2; return 1; }
  echo "    pyexpat repaired (backups left as *.orig.bak in $dynload)."
}

# Persist LOOM_PYTHON to the user's shell profile, idempotently. INSTALL.md marks
# this export as required so every new shell's `loom` finds the engine venv.
persist_loom_python() {
  local line profile
  line="export LOOM_PYTHON=\"$REPO_ROOT/.venv/bin/python\""
  profile="${ZDOTDIR:-$HOME}/.zshrc"
  case "${SHELL:-}" in *bash*) profile="$HOME/.bash_profile" ;; esac
  if [ -f "$profile" ] && grep -qF "$line" "$profile"; then
    echo "    LOOM_PYTHON already set correctly in $profile"
  else
    # Drop any stale export + marker (e.g. a prior install from another directory)
    # before appending the current one, so a reinstall never leaves a wrong path.
    if [ -f "$profile" ]; then
      grep -vE '^[[:space:]]*export LOOM_PYTHON=|^# Loom: point the loom CLI at the engine venv' "$profile" > "$profile.tmp" 2>/dev/null || true
      cat "$profile.tmp" > "$profile" && rm -f "$profile.tmp"
    fi
    printf '\n# Loom: point the loom CLI at the engine venv interpreter (added by install.sh)\n%s\n' "$line" >> "$profile"
    echo "    set LOOM_PYTHON in $profile (open a new shell or 'source $profile' to pick it up)"
  fi
}

AGENT_PROMPT='Install Loom on this macOS machine by following ./INSTALL.md exactly, in order, from the repo root. Run each step. After the Python step, CONFIRM that `pip install -e .` fully completed — `python -c "import metaflow"` must succeed — before continuing; a partial install (often a build failure in the heavy AIDE dependency) is the most common problem. If `python3.12 -m venv` or pip fails with a pyexpat/libexpat symbol error, repair it first: `brew install expat`, then relink the python3.12 pyexpat .so against "$(brew --prefix expat)/lib/libexpat.1.dylib" with install_name_tool and re-sign it with `codesign --force --sign -`. Finish by starting the datastore port-forward and running `loom doctor`, iterating on any FAIL until it prints "VERDICT: PASS". Ask me before anything destructive or any sudo.'

run_agent() {
  local agent="$1"
  # Clear the macOS pyexpat breakage up front so the assistant starts from a
  # python3.12 whose venv/pip actually work.
  ensure_macos_python_expat || true

  # Run the routine install steps WITHOUT a prompt-per-command. Installing Loom is
  # a bounded, reversible (./uninstall.sh), user-initiated local task, so we
  # pre-approve the tools it actually needs (shell + file edits) rather than make
  # you confirm every pip/npm/brew/kubectl call. Unexpected tools still prompt, and
  # the prompt tells the assistant to ask before anything destructive or sudo.
  # Set LOOM_INSTALL_SUPERVISED=1 to keep the per-step approvals.
  local -a cmd=("$agent")
  if [ -z "${LOOM_INSTALL_SUPERVISED:-}" ]; then
    case "$agent" in
      claude) cmd+=(--allowedTools Bash Edit Write Read --permission-mode acceptEdits) ;;
      codex)  cmd+=(--full-auto) ;;
    esac
  fi
  cmd+=("$AGENT_PROMPT")

  echo "==> Found '$agent' — installing Loom with minimal prompting."
  echo "    It follows INSTALL.md and verifies with 'loom doctor'; routine steps run"
  echo "    without asking, and it still flags anything risky before doing it."
  echo "    (Set LOOM_INSTALL_SUPERVISED=1 to approve every step instead.)"
  echo
  exec "${cmd[@]}"
}

run_shell() {
  echo "==> Scripted install (no AI assistant used)."

  echo "--> [1/3] engine (Python)"
  command -v python3.12 >/dev/null 2>&1 || { echo "    python3.12 not found. Run: brew install python@3.12" >&2; exit 1; }
  ensure_macos_python_expat

  # (Re)create the venv. A previous failed run can leave a venv with no working
  # pip (e.g. an ensurepip failure), which `[ -d .venv ]` would silently reuse —
  # rebuild from scratch in that case.
  if [ -d .venv ] && ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
    echo "    existing .venv has no working pip — recreating it."
    rm -rf .venv
  fi
  [ -d .venv ] || python3.12 -m venv .venv
  # Drive the venv interpreter directly — more robust than 'activate' under set -u.
  .venv/bin/python -m pip install -e .
  .venv/bin/python -c "import metaflow" >/dev/null 2>&1 || {
    echo "    Metaflow is not importable after 'pip install -e .' — the install did not complete." >&2
    echo "    Re-run '.venv/bin/python -m pip install -e .' from the repo root and read the first error." >&2
    exit 1
  }
  export LOOM_PYTHON="$REPO_ROOT/.venv/bin/python"
  persist_loom_python

  echo "--> [2/3] the 'loom' command (Node)"
  command -v node >/dev/null 2>&1 || { echo "    node not found. Run: brew install node" >&2; exit 1; }
  ( cd cli && npm install && npm run build && npm link )

  echo "--> [3/3] local datastore"
  bash scripts/setup_metaflow_minikube.sh

  cat <<NEXT

==> Engine + CLI + datastore are installed (LOOM_PYTHON persisted to your shell profile).

   1) Start the datastore port-forward in a separate terminal (keep it running):
        kubectl port-forward -n loom svc/minio 9000:9000 9001:9001 &

   2) Source the datastore env and verify (expect VERDICT: PASS):
        source .env.metaflow && "$REPO_ROOT/.venv/bin/python" -m loom doctor

   3) Open a new shell (or 'source' your profile) so 'loom' is on PATH, then run:
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
