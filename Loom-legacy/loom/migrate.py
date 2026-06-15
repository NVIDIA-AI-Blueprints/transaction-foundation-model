"""``loom update``'s migration advisor — the LLM-guided half of upgrades.

The mechanical update (``git pull`` + npm rebuild + ``pip install -e .``) stays in
``cli/src/update.ts`` and is deterministic. THIS module owns the *variable* part: a
release may need a machine brought from its current state to a new desired state
(re-run the cluster setup, re-source the env, add port-forwards…), and that reasoning
differs by release and by the machine's actual state. So a release ships a
``migrations/<version>.yaml`` (the ``loom-migration/1`` format — see
``migrations/FORMAT.md``) declaring the DESIRED end-state, and this advisor hands the
applicable manifests + the machine's live ``loom doctor --json`` to an LLM (claude,
else codex) to reconcile — reusing the exact handoff pattern ``loom doctor --fix``
uses (``os.execvp``, no auto-approve flags, the prompt's own ask-before-destructive
gate). When no assistant is on PATH it falls back to printing the applicable
manifests so the user can follow them by hand.

``loom update`` invokes this after the mechanical steps; it is a silent no-op when no
manifest is newer than the installed version (the common case), so the LLM only
engages when there is a real migration to reason about.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

# loom/ -> repo root; migrations/ lives at the repo root next to scripts/.
REPO_ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS_DIR = REPO_ROOT / "migrations"
INDEX_FILE = MIGRATIONS_DIR / "INDEX.yaml"


def _installed_version() -> Optional[str]:
    """The pip-installed loom version (the migration version-of-record), or None."""
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _dist_version

    try:
        return _dist_version("loom")
    except PackageNotFoundError:  # not pip-installed (partial install)
        return None


def has_applicable() -> bool:
    """Whether any migration manifest applies to the installed version.

    A cheap, side-effect-free check (no doctor run, no handoff) so callers like
    ``loom doctor`` can decide whether ``loom migrate`` is the relevant remedy.
    Returns False when loom isn't pip-installed or no manifest matches. Swallows
    any error (missing deps / malformed index) — a nudge must never crash doctor.
    """
    try:
        installed = _installed_version()
        if installed is None:
            return False
        return bool(_applicable(_load_index(), installed))
    except Exception:  # noqa: BLE001 - the nudge is best-effort
        return False


def _load_index() -> list[dict]:
    """Parse ``migrations/INDEX.yaml`` into its ascending ``releases`` list."""
    if not INDEX_FILE.is_file():
        return []
    import yaml

    data = yaml.safe_load(INDEX_FILE.read_text()) or {}
    releases = data.get("releases") or []
    # Require BOTH `to` and `manifest`: downstream (_advisor_prompt) subscripts
    # rel["manifest"], so a malformed hand-edited row with `to` but no `manifest`
    # must be dropped here rather than crash the migrate with a KeyError.
    return [r for r in releases if isinstance(r, dict) and r.get("to") and r.get("manifest")]


def _applicable(releases: list[dict], installed: str) -> list[dict]:
    """Select the manifests whose desired-state this machine should satisfy.

    Two kinds:

    * the BASELINE — the newest manifest with ``to <= installed`` (the release the
      machine is ON). Selected so that an install whose VERSION advanced (e.g.
      ``loom update``'s ``pip install -e .`` bumps it to the target) but whose
      desired-state was never reached (no cluster setup) still gets reconciled.
      Keying selection on ``to > installed`` alone was the bug behind "loom update
      did nothing": pip moves the version to the target BEFORE the advisor runs, so
      a pure version-delta finds nothing to do even on a machine that never set the
      stack up. Desired-state, checked via ``loom doctor`` in :func:`run_migrate`,
      decides whether the baseline actually needs work (a healthy baseline no-ops).
    * PENDING upgrades — manifests with ``to > installed`` whose ``from`` PEP 440
      predicate the installed version satisfies (a release whose CODE you haven't
      installed yet; uncommon in the editable-install flow but possible).

    Returns the baseline (if any) then pending, ascending by ``to``.

    PEP 440 — NOT SemVer: ``0.1.0.dev0`` is valid PEP 440 (a pre-release of
    ``0.2.0``) but invalid SemVer, so ``packaging`` must do the comparison.
    """
    from packaging.specifiers import SpecifierSet
    from packaging.version import InvalidVersion, Version

    try:
        cur = Version(installed)
    except InvalidVersion:
        return []

    baseline: Optional[tuple] = None
    pending: list[tuple] = []
    for rel in releases:
        try:
            to = Version(str(rel["to"]))
        except InvalidVersion:
            continue
        if to > cur:
            # A pending upgrade: honor the manifest's `from` predicate.
            from_pred = _manifest_from_predicate(rel)
            if from_pred is not None:
                try:
                    applies = SpecifierSet(from_pred).contains(cur, prereleases=True)
                except Exception:  # noqa: BLE001 - unparseable predicate -> fail closed
                    applies = False
                if not applies:
                    continue  # NOT-APPLICABLE to this installed version -> skip
            pending.append((to, rel))
        elif baseline is None or to > baseline[0]:
            baseline = (to, rel)  # newest manifest at/below the installed version
    pending.sort(key=lambda t: t[0])
    out: list[dict] = [baseline[1]] if baseline is not None else []
    out.extend(rel for _, rel in pending)
    return out


def _manifest_from_predicate(rel: dict) -> Optional[str]:
    """Read the ``from`` PEP 440 predicate out of a release's manifest file."""
    mpath = REPO_ROOT / str(rel.get("manifest", ""))
    if not mpath.is_file():
        return None
    import yaml

    try:
        m = yaml.safe_load(mpath.read_text()) or {}
    except Exception:  # noqa: BLE001
        return None
    frm = m.get("from")
    return str(frm) if frm else None


def _doctor_json() -> Optional[dict]:
    """Capture the machine's live ``loom doctor --json`` (best-effort).

    doctor exits 1 when a check FAILs but still prints its JSON envelope, so we
    capture regardless of exit code. Returns None if it could not be run/parsed (the
    advisor prompt then instructs the LLM to run it itself).
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "loom", "doctor", "--json"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except Exception:  # noqa: BLE001 - doctor missing/slow; the LLM can re-run it
        return None
    out = (proc.stdout or "").strip()
    if not out:
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


def _read(paths: list[Path]) -> str:
    """Concatenate readable files with a header per file (for the prompt)."""
    blocks = []
    for p in paths:
        if p.is_file():
            # os.path.relpath never raises (unlike Path.relative_to, which throws
            # when an absolute/escaping manifest path resolves outside the repo).
            label = os.path.relpath(p, REPO_ROOT)
            blocks.append(f"===== {label} =====\n{p.read_text()}")
    return "\n\n".join(blocks)


def _advisor_prompt(installed: str, applicable: list[dict], doctor: Optional[dict], apply: bool) -> str:
    """Build the LLM prompt: the format + the applicable manifests + the live state."""
    manifest_paths: list[Path] = [MIGRATIONS_DIR / "FORMAT.md"]
    for rel in applicable:
        manifest_paths.append(REPO_ROOT / str(rel["manifest"]))
        if rel.get("notes"):
            manifest_paths.append(REPO_ROOT / str(rel["notes"]))
    versions = ", ".join(str(r["to"]) for r in applicable)
    doctor_blob = (
        json.dumps(doctor, indent=2)
        if doctor is not None
        else "(could not capture; run `loom doctor --json` yourself first)"
    )
    mode = (
        "MODE: APPLY. Reconcile the machine to the desired state, but PAUSE for my "
        "explicit confirmation before EVERY transition with `mutation: cluster` or "
        "`confirm: true`, and run the others only after showing me the command."
        if apply
        else "MODE: ADVISORY (default). Do NOT execute anything that mutates the "
        "cluster or the environment. Produce the exact ordered plan — which "
        "transitions are unmet, which are already-satisfied (skipped), the commands "
        "to run, and the expectations — and let me run/approve them."
    )
    return (
        "You are `loom update`'s migration advisor. The mechanical update (git pull, "
        "npm rebuild, pip install) has already run; your job is to bring this machine "
        f"to the desired state of the newer release(s): {versions}.\n\n"
        f"Installed loom version: {installed}\n\n"
        "Read migrations/FORMAT.md for the loom-migration/1 format and the advisor "
        "loop, then reconcile using the applicable manifest(s) below against the live "
        "machine state. RULES (from the format):\n"
        "- Evaluate each `desired` assertion against `loom doctor --json`; run ONLY "
        "the transitions whose assertions are unmet.\n"
        "- A transition whose `guard` is already true is ALREADY APPLIED -> SKIP it.\n"
        "- A `mutation: cluster` or `confirm: true` transition is NEVER run "
        "unattended — present it and wait for my approval.\n"
        "- `expectations` with `do_not_fix: true` are EXPECTED side-effects (e.g. old "
        "~/.metaflow runs won't appear) — state them, never try to 'repair' them.\n"
        "- The `rollback` block is human-invoked ONLY; never auto-select it.\n"
        "- A transition's `verify` must hold after its `run`, or stop and report "
        "`on_fail`.\n\n"
        f"{mode}\n\n"
        "Ask me before anything destructive or sudo.\n\n"
        f"--- LIVE STATE (loom doctor --json) ---\n{doctor_blob}\n\n"
        f"--- FORMAT + APPLICABLE MANIFESTS ---\n{_read(manifest_paths)}\n"
    )


def _needs_reconcile(applicable: list[dict], installed: str, doctor: Optional[dict]) -> tuple[bool, str]:
    """Decide whether there is migration WORK, given the applicable manifests + the
    live ``loom doctor --json``.

    The desired-state source of truth is ``loom doctor``. Engage when either:
    - a PENDING upgrade exists (``to > installed`` — new desired-state to reach), or
    - the baseline release's stack is NOT healthy (doctor VERDICT != PASS), meaning
      the version is at the target but the machine never reached the desired state
      (exactly the "loom update did nothing / metaflow not set up" case).

    A healthy baseline with no pending upgrade is a genuine no-op. Returns
    ``(needs, reason)``.
    """
    from packaging.version import InvalidVersion, Version

    try:
        cur = Version(installed)
        pending = [r for r in applicable if Version(str(r["to"])) > cur]
    except (InvalidVersion, KeyError):
        pending = []
    if pending:
        return True, f"pending upgrade(s): {', '.join(str(r['to']) for r in pending)}"
    verdict = (doctor or {}).get("VERDICT")
    if verdict != "PASS":
        # No pending upgrade, but the stack isn't healthy -> the current release's
        # desired-state isn't met (or a port-forward/env is down). Reconcile.
        return True, f"`loom doctor` is not healthy (VERDICT: {verdict or 'unknown'})"
    return False, "stack healthy (loom doctor VERDICT: PASS)"


def run_migrate(apply: bool = False) -> int:
    """Reconcile the machine toward the current/pending release's desired state.

    Selects the applicable manifests (the baseline release the machine is on, plus
    any pending upgrade — see :func:`_applicable`), then uses live ``loom doctor``
    health to decide if there is actually work (so a healthy stack is a no-op even
    though a baseline manifest matches). With an assistant on PATH it hands off
    (``os.execvp``, replacing this process); otherwise it prints the applicable
    manifests for manual follow-through.

    Args:
        apply: Let the assistant EXECUTE the reconcile (still confirming cluster
            mutations), versus the default advisory (propose-only) mode.

    Returns:
        Process exit code (0 unless the handoff launch fails).
    """
    installed = _installed_version()
    if installed is None:
        print(
            "loom is not pip-installed in this interpreter, so the migration version "
            "cannot be read. Run `pip install -e .` (or `loom update`) first."
        )
        return 0
    releases = _load_index()
    applicable = _applicable(releases, installed)
    if not applicable:
        print(f"Migrations: no manifest applies to installed loom {installed}.")
        return 0

    # Desired-state, not version-delta, decides whether there's work: a stack that
    # is already healthy needs nothing even though a baseline manifest matches.
    doctor = _doctor_json()
    needs, reason = _needs_reconcile(applicable, installed, doctor)
    if not needs:
        print(f"Migrations: up to date (installed loom {installed}; {reason}).")
        return 0

    versions = ", ".join(str(r["to"]) for r in applicable)
    summaries = "; ".join(f"{r['to']}: {r.get('summary', '')}" for r in applicable)
    print(f"\nReconciling to release(s) {versions} -- {reason}.\n  {summaries}")

    # Reuse doctor --fix's assistant detection + handoff pattern (no auto-approve
    # flags; the prompt carries the ask-before-destructive gate).
    from loom.cli import _detect_repair_assistant

    assistant = _detect_repair_assistant()
    prompt = _advisor_prompt(installed, applicable, doctor, apply)
    if assistant is None:
        print(
            "\nNo `claude` or `codex` CLI on PATH for guided migration. Follow the "
            "applicable manifest(s) by hand -- read their human notes (the .md "
            "sidecars) and reconcile against `loom doctor`:\n"
        )
        for rel in applicable:
            print(f"  - {rel.get('manifest')}  (notes: {rel.get('notes')})")
        print("\n  See migrations/FORMAT.md for how to read them.")
        return 0

    mode_word = "apply" if apply else "advisory"
    print(
        f"\nHanding the migration to `{assistant}` ({mode_word} mode) -- it reconciles "
        "this machine to the release's desired state, asking before any cluster "
        "change. (Re-run with no assistant to print the manual steps instead.)"
    )
    sys.stdout.flush()
    try:
        os.execvp(assistant, [assistant, prompt])
    except OSError as exc:  # noqa: BLE001 - actionable, no traceback
        print(f"\nCould not launch `{assistant}`: {exc}. Follow the manifests by hand.")
        return 1
    return 0  # unreachable on success (process replaced)
