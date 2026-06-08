# CLAUDE.md

Repository invariants for **Loom**. These are hard constraints — keep every
change conforming to them. Loom is private product code; commercial, customer,
and pricing strategy live elsewhere and must never be added here.

## What Loom is
- A **general-purpose, domain-neutral** automated ML engine. The metric is the
  spec. **Never fit Loom to a specific customer, dataset, or vertical**, and
  never base a file on a pre-existing/sandbox/example flow. Build to the spec.

## Architecture — ports and adapters ("providers")
- `loom-core` defines provider **interfaces** (ports); concrete **adapters**
  implement them and are selected purely by configuration. Two seams in v0.1:
  - `SearchProvider` — the "brain" (default adapter `aide`).
  - `ExecutionProvider` — the MLOps "muscle" (default `metaflow`; `local` for a
    Metaflow-free dev path).
- Providers are resolved **by name** through `loom/registry.py`. The controller
  resolves `config.search_provider` / `config.mlops_provider` to classes via the
  registry and **never imports concrete adapters directly**.
- An `ExecutionProvider` is **callable** with the exec-callback signature
  `(code, reset_session) -> ExecutionResult` (`__call__` aliases `execute`), so
  it can be passed straight to a `SearchProvider`. Do not break this seam.
- Adding a provider = one adapter class + a `@register_search` /
  `@register_execution` decorator. No core edits.

## Hard invariants — do not violate
- **`ExecutionResult` is field-identical to AIDE's**
  `aide.interpreter.ExecutionResult`
  (`term_out`, `exec_time`, `exc_type`, `exc_info`, `exc_stack`). Field parity is
  what lets a Loom result convert to/from AIDE's type by a straight
  `aide.interpreter.ExecutionResult(**dataclasses.asdict(result))`. If you touch
  it, keep parity exactly.
- **AIDE is pinned by SHA `40dcf28`** (`aideml @ git+...@40dcf28` in
  `pyproject.toml`). Do **not** float it to a tag/branch — the agent/journal API
  and ExecutionResult shape are anchored to that revision. **Do not edit AIDE**;
  the `aide` adapter *drives* AIDE's public loop, it does not patch internals.
- **One static Metaflow flow.** `EvalCandidate(FlowSpec)` is generated once;
  **the candidate enters as data** (`IncludeFile` + `Parameter`), never as a
  newly generated flow per candidate. Use standard Metaflow APIs only.
- **Lazy-import heavy/optional deps.** `import loom` and `import loom.types`
  must succeed with nothing but the standard library installed. AIDE, Metaflow,
  pandas, etc. are imported inside functions/methods; each built-in adapter
  registers under its own guarded import in `loom/providers/__init__.py`.
- **Secrets via environment only.** Never hardcode keys/tokens/endpoints. Read
  `NVIDIA_API_KEY`, `OPENAI_BASE_URL`, `ANTHROPIC_API_KEY`, `METAFLOW_*` from the
  environment at the point of use. `LoomConfig` records non-secret routing
  values only; never persist secrets to the corpus or any artifact.
- **Multi-tenant IP boundary.** A `NodeRecord` carries `owned_by`; records owned
  by a specific tenant are tagged, and only `owned_by == "general"` records may
  feed a cross-tenant model (`Corpus.general()`). Preserve this boundary.

## Conventions
- Python ≥ 3.10, type hints, docstrings, idiomatic style.
- The vendored interpreter (`loom/providers/_interpreter.py`) is the single
  dependency-light execution primitive shared by `local_exec` and the Metaflow
  `evaluate` step, so neither hard-depends on AIDE internals. Keep its
  `exception_summary` behavior faithful to AIDE's interpreter.
- Keep this repo **domain-neutral** and free of commercial/customer/$ content.
