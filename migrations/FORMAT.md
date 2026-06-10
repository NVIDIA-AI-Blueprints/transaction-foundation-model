# Loom migration manifests — `loom-migration/1`

An **LLM-targeted release-note format**. A maintainer writes one
`migrations/<to_version>.yaml` per release that declares the **desired end-state**
of a user's machine after that release — *not* an imperative upgrade script.
`loom update`'s advisor (Claude, else Codex) reads it, diffs desired-vs-actual
against the machine's live `loom doctor --json`, and runs **only** the transitions
whose desired assertions are currently unmet. The maintainer describes the target;
the reconciler computes the path. This is why a fixed `loom update` script can't do
it: the mechanical update (pull + rebuild + reinstall) is the same every release,
but *what a given machine needs to reach the new desired state* differs by release
and by the machine's current state.

It is a sibling of `loom doctor --fix`: same idea (hand the hard reasoning to an
LLM), same handoff machinery, applied to upgrades instead of diagnosis.

## Core model — current-state → desired-state

| block | meaning |
|---|---|
| `probes` | how to **read** actual state — `loom doctor --json` checks by exact name, or read-only `shell`/`http` |
| `desired` | **assertions** that must hold after the release (the end-state contract) |
| `transitions` | guarded state-changers that make a desired assertion true; idempotent via `guard`, re-checked via `verify` |
| `expectations` | side-effects the advisor must **expect and never try to remediate** |
| `rollback` | human-invoked-only escape hatch; **never auto-selected** during a forward reconcile |

## The advisor loop (what the LLM does)

1. **Read installed version.** `loom doctor --json` → `summary.loom_version`
   (preferred), falling back to parsing `summary.checks[0].detail` (`loom (\S+) @`).
2. **Select manifests.** After `git pull --ff-only`, the newly-pulled set is
   `git -C <repo> diff --name-only <old>..<new> -- migrations/`. Keep those whose
   `from` PEP 440 predicate the installed version satisfies and whose `to` is newer;
   sort ascending by `to`; apply in order. Non-matching manifests are
   **NOT-APPLICABLE → skipped silently** (this is how a newer machine no-ops).
3. **Read actual state.** Run `loom doctor --json` once + any referenced
   `shell`/`http` probes → ACTUAL. Evaluate every `desired.expect`.
4. **Reconcile.** For each **unmet** assertion, take its transitions (those whose
   `satisfies` lists the assertion). For each: evaluate `guard` — true ⇒ already
   applied ⇒ **SKIP**. Else gate on `mutation`/`confirm` (below), run, then evaluate
   `verify` — not true ⇒ stop the chain and emit `on_fail`.
5. **Confirm convergence.** Re-run `loom doctor --json`; require every
   `desired.expect` true. Surface `expectations` to the human as *"expected, not
   failures."*
6. **Degrade safely.** On any failure, fall back to `update.ts`'s manual-fallback
   text and the `on_fail` hints — never leave the machine silently half-migrated.

## Evaluation grammar (deliberately tiny, so the LLM can't drift)

- `probe:<id>.status` → `PASS|WARN|FAIL` (from the named doctor check)
- `probe:<id>.match` → bool (the probe's `regex` matched)
- `probe:<id>.capture.<group>` → captured string (e.g. the version token)
- `probe:<id>.exit` → int; `probe:<id>.stdout_contains("…")` → bool
- `probe:<id>.expect_status` → the `http` probe's status
- operators: `==` `!=` `&&` `||` `>=` ( `>=` is **PEP 440**-aware when both sides
  look like versions — `0.1.0.dev0` is valid PEP 440 but *invalid* SemVer, so the
  comparator MUST be PEP 440, never SemVer )

An `expect`/`guard`/`verify` is one boolean expression over the above. `all:`/`any:`
list forms are sugar for `&&`/`||`.

### Probe kinds
- `doctor_check`: `{ check: "<exact name>", read: status|detail, regex?: "…" }`
  (named regex groups → `.capture.*`; presence → `.match`)
- `doctor_version`: sugar for check `python/venv + import loom`, exposing `.capture.v`
- `shell`: `{ cmd: [argv…], read_only: true }` — MUST be side-effect-free
- `http`: `{ url, expect_status }`

### Transition fields
- `id`, `satisfies: [assertion-id, …]`
- `guard` — boolean; **true ⇒ already applied ⇒ skip** (idempotency)
- `mutation` — `none | local | cluster`
- `confirm` — bool. **Forced `true` whenever `mutation: cluster`** or the step is
  destructive; the advisor will not run a `confirm: true` step unattended.
- `run` — one of `{ shell: [argv…], background?: bool }` ·
  `{ edit: { file, ensure_lines: [...] } }` (append-if-absent, idempotent) ·
  `{ doc: "instruction the LLM carries out / relays" }`
- `verify` — boolean re-checked after `run`. **Validator rule: `verify` must be at
  least as strict as the desired assertion it satisfies** (so a coarse guard can't
  let a half-applied state pass).
- `on_fail` — human hint; becomes part of the advisor prompt / fallback.
- `background: true` runs a long-lived process (a port-forward) detached; `verify`
  then polls the matching `http` probe with retry rather than blocking on the run.

### Expectation fields
- `id`, `statement`, `do_not_fix: true` (the runner is **mechanically forbidden**
  from "repairing" it), optional `probe` + `expected_value` documenting the benign
  reading. Phrase statements with the literal strings `loom doctor` emits.

### Rollback
- A typed block with `confirm: true` **and** `never_auto_select: true`. Destructive
  recovery (e.g. `minikube delete`) is offered only on explicit human request — the
  forward reconcile can never choose it.

## Validation the advisor enforces before running anything
- every `transitions[*].satisfies` id exists in `desired`
- every `mutation: cluster` step has `confirm: true`
- every `verify` is ≥ as strict as the assertion(s) it satisfies
- no `expectation` has a transition (you cannot "fix" an expectation)
- `rollback` is `never_auto_select` and not referenced by any `satisfies`
- every `doctor_check` probe names a check that exists in `loom doctor --json`

> **Recommended hardening (not yet implemented):** the six doctor check names are
> currently inline string literals in `loom/cli.py`, and a manifest references them
> by string. Promoting them to a shared constant that both the engine and the
> advisor reference would turn a rename into a build/test failure instead of a
> silently mis-evaluated guard. Until then, a renamed check is caught by the
> manifest's own tests, not the type system.

## Safety invariants (learned from stress-testing this format)
- The mechanical update (`git pull` / npm rebuild / `pip install -e .`) stays
  **deterministic and outside the manifest**. The manifest only carries the variable
  *migration* reasoning.
- Handoff to claude/codex reuses doctor's existing pattern (`os.execvp`, prompt's
  built-in *"ask before anything destructive or sudo"*). It does **not** bolt on
  `install.sh`'s `--full-auto`/`--allowedTools acceptEdits` pre-approve flags — an
  unattended full-auto agent on a cluster-connected box is the single most dangerous
  path and is explicitly out.
- The advisor loop is **bounded** (a max-iteration cap); it never retries forever.
- `shell` probes are read-only; state-mutating `shell` lives only in `run`.

## Conventions
- One `migrations/<to_version>.yaml` per release, keyed to the exact version literal
  in `loom/__init__.py` / `pyproject.toml` (e.g. `migrations/0.2.0.yaml`).
- A **required** human sidecar `migrations/<to_version>.md` carries the WHY/intent a
  maintainer writes once and reviews in the PR; the advisor fetches contract + prose
  together at every gate.
- `migrations/INDEX.yaml` lists every release ascending so the chain is enumerable
  without globbing.
- Append-only: a manifest is **frozen the moment its version ships** and never
  edited afterward — only superseded by the next version's file. Pre-release edits
  are fine. No date prefixes (these are version-keyed, not chronological).
