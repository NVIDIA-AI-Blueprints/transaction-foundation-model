> **Status:** FINAL Loom product & UX design spec — the decided interaction surface. **Canonical location: the build repo at `Loom/DESIGN.md` in `ZKAI-Network/transaction-foundation-model`** (this is a build artifact, kept with the code). Supersedes the earlier `Loom/DESIGN.md` draft (2026-06-15) and the verb-set / v0.1-interface placement in the strategy dossier's `command-surface.md` and `v0.1-plan.md` §3/§7 (see §11 Reconciliation). Primary spine = composable typed verbs over versioned data-objects (P4), with coherent grafts from Experiment-as-Code (P2), the Conversational Director's capabilities (P1), and Notebook-native cards (P3). Produced by a 4-design / 3-judge / synthesize→critique→revise design panel. **Last updated:** 2026-06-15

# Loom — Product & UX Design Spec

**Loom is an agent harness for training state-of-the-art foundation models: a small set of sharp, typed verbs you *compile before you spend*, driven identically by a human at a terminal and a Claude/Codex agent, where contract violations surface as named diffs caught for free and money is the only thing a human must press a button for — and that button enforces a binding budget envelope, not just a one-time "go."**

Its first driving use case is a **transaction foundation model (TFM)**: a decoder-only transformer pretrained with causal language modeling on sequences of financial/blockchain transactions, whose frozen embeddings power downstream tasks (fraud detection, next-trade prediction, credit scoring). The design is general (FM-training) but TFM-first.

This document is the decided interaction surface — the verb grammar, the cost/launch safety model, the contract UX, the failure/resume/concurrency lifecycle, the lineage/report model, telemetry, and the v0.1 build slice. It is opinionated on purpose, and it closes the safety holes a prior draft left open (each fix is flagged inline as **`[FIX]`**).

---

## 0. The decision in one paragraph

**The spine is six sharp verbs over versioned, lineage-carrying data objects addressed by pathspec** — `ingest → tokenize → pretrain → embed → evaluate → report` — each doing exactly one thing, emitting a machine-checkable VERDICT, and being **1:1 a typed agent tool**. This *is* the reference repo's usage contract made ergonomic (every verb / pathspec / VERDICT / capability-mode / approval-matrix concept transfers 1:1); dual-driver symmetry is mechanical (one typed contract generates both faces, so the **result envelope** of `loom tokenize --json` is byte-identical to `loom.tokenize()` — the *approval handshake* differs by driver and is specified explicitly in §5.3); tier-as-a-verb-property plus an un-delegable, budget-bounded launch button is the cleanest GPU-cost-safety story; and six verbs over the existing NeMo + Metaflow seam is the smallest genuinely-useful TFM-first v0.1. Onto that spine we graft: **(P2)** compile-time contract checking (the tokenizer is a spec the engine compiles in <1s, so C1/C2/C3 violations are caught before a GPU-second), the contract-aware `diff`, signature-asserting `replay`, and signature-stamped checkpoints; **(P1)** plan-time goal↔metric↔split coherence, the free CPU rehearsal of the whole DAG before any launch, and steering-trajectory telemetry for the moat; **(P3)** contract-violation-as-a-named-card and staleness propagation, expressed as terminal cards + a lineage invariant. We explicitly **reject** P1's NL-planner in the hot path, P3's JupyterLab-extension/monkey-patch surface, and P2's mandatory "DSL + PR for everything" ceremony (see §12).

---

## 1. Core metaphor

> **Six sharp verbs piping versioned data-objects by pathspec, with a live read-only dashboard watching the lineage — `git` plumbing meets `kubectl`, for foundation-model training. Each verb is a contract you compile before you spend; the contract is the type system that makes silent-garbage failures impossible; and every spend is bounded by an envelope the human approved, not a label the agent chose.**

You are never "inside" Loom and never "talking to" Loom. You **run verbs**; verbs produce immutable objects; objects thread by reference; a dashboard (`loom top`) projects the live state. A human runs the verbs at a terminal; a Claude/Codex agent calls the identical verbs as typed tools. Money and shipping are the only things a human must press a button for — and the agent has no tool that can press it.

Four properties make this trustworthy rather than reckless:

1. **The verb is the spec.** A tokenizer change is a declarative spec the engine *compiles* (derives `vocab_size`, `vocab_hash`, `chunk_size`, `tokens_per_txn`; checks injectivity; estimates cost from the compiled plan) with no side effects and no GPU. The inner loop is sub-second; you iterate on the spec dozens of times before the first dollar.
2. **Tier is a property of the verb, not a flag.** `read-only / workspace-write / expensive / irreversible` and `searchable / launch-and-track` are baked into each verb's type and tool schema — so "the search brain spawns twenty 8-GPU jobs" is *unrepresentable*, not merely discouraged.
3. **Cost is derived, and approval is an envelope.** Every cost number is computed from the compiled plan (token count × model FLOPs/token × target $/GPU-h), never a hardcoded label. Approving a run approves a **binding `{max_steps, max_$, max_wall_clock}` envelope** the orchestrator hard-kills at — "approve ≈$86" means "stop at ~$86," not "start something estimated at $86."
4. **Every state is an object with a lifecycle.** Objects aren't only clean `PASS` artifacts; a `Checkpoint` can be `RUNNING / SUCCEEDED / FAILED / PREEMPTED / STOPPED_AT_BUDGET`, carrying a resume token, so failure and resume are first-class, not a re-spend-from-scratch dead end.

---

## 2. The command surface

### 2.1 The narrow waist: one typed contract, two faces

Each verb is declared **once** as a typed contract — a JSON-Schema'd argument spec + a structured result envelope + a tier. From that single declaration we generate the **human CLI** (`loom tokenize …`, pretty TTY renderer) and the **agent tool** (`loom.tokenize(…)`, JSON in / JSON out). The TTY-pretty stream and the `--json` machine envelope are two renderers over the same result object.

**What is identical, and what is not** *(`[FIX]` — the prior draft overclaimed "byte-identical … one set of gate behavior," which is false at the interactive gates)*:

- **Identical:** the argument set, the **result envelope**, the tier, the VERDICTs, the exit codes, the contract diagnostics. `loom tokenize --json`'s result is byte-identical to `loom.tokenize()`'s result. There is one engine code path; neither face has a private capability.
- **Different by design — the approval handshake.** An interactive prompt has no byte-identical JSON analog, so we *specify* the difference rather than pretend it away (see §5.3): a human PLAN ends with `Proceed? [y/N]`; the agent receives `status:"PLAN"` + a `confirm_token` and must make a **second** call (`loom.tokenize({…, confirm_token})`) to proceed. Same data, mirrored round-trip.
- **`stage_baseline` is a first-class argument, not an interactive-only affordance** *(`[FIX]` — previously the human was nudged with `[B] stage baseline` but the agent had no equivalent, so an agent could silently ship a report with no control, violating house rule #3).* Both faces set `stage_baseline: true|false`; when a tokenize step changes the vocab and `stage_baseline` is unset, **both** faces are blocked from `report` until a baseline exists in the experiment (the human is prompted, the agent gets `REFUSED_NO_BASELINE` with the one-line fix). House rule #3 binds both drivers.

```
verb declaration  ──┬──►  argparse CLI  (TTY renderer)   →  loom tokenize …      [y/N] gate
                    └──►  agent tool     (JSON renderer)  →  loom.tokenize(…)     confirm_token gate
                         (same args · same RESULT envelope · same tier · same VERDICTs · APPROVAL handshake differs, §5.3)
```

### 2.2 The six core verbs + inspectors — with corrected tiers

Every verb: `loom <verb> [pathspec-in] [--flags] [--json] [--launch]`. Each is also `loom.<verb>(...)`.

```
loom ingest    <source>          → IngestDataset/<n>   raw events → versioned object (schema sniff + EDA leakage gate; no transform)
loom tokenize  <IngestDataset/n> → Corpus/<n>          THE contract-bearing verb (compiles the tokenizer spec; C1/C2/C3)
loom pretrain  <Corpus/n>        → Checkpoint/<n>      launch-and-track · GPU · IRREVERSIBLE · budget-bounded (CLM)
loom embed     <Checkpoint/n>    → Embeddings/<n>      forward pass of the frozen backbone → vectors
loom evaluate  <Embeddings/n>    → Scores/<n>          multi-task battery + VERDICT vs baseline (C6)
loom report    --experiment <id> → Report/<n>          assemble model card: runs + metrics + lineage + contract ledger
```

**Tier table (single source of truth — resolves the contradictory labels in the prior draft):**

| Verb | Tier | Capability mode | Cost-PLAN'd? | Human button required? | Agent may auto-proceed? |
|---|---|---|---|---|---|
| `ingest` | workspace-write | n/a | only if a remote pull is metered | no (idempotent, §6) | yes |
| `tokenize` | workspace-write (cheap, CPU) | n/a | PLAN shown, free | no | yes (non-destructive) |
| `pretrain` | **irreversible / external** | **launch-and-track** | **yes, derived** | **yes** | **never** |
| **`embed`** | **expensive (GPU) / external** *(`[FIX]`)* | **launch-and-track above threshold** | **yes when N≥threshold** | only above the budget cap | only below the row/GPU-sec budget |
| `evaluate` | workspace-write (CPU) | searchable | free | no | yes |
| `report` | read-only; `--send` is irreversible | n/a | free; `--send` gated | only for `--send` | never for `--send` |

> **`[FIX] embed is real GPU work, not a free workspace-write.** Embedding is a forward pass of every sequence through the 29M-param transformer, and the reference tokenizer pipeline is GPU-only (level-400 sharp edge #8). For D5 that is ~41.2M events → ~tens of millions of sequences. So `embed` is a **budgeted, launch-and-track model-builder call** exactly like the reference's `loom train --capability embed --budget probe` — NOT a free write:
> - It requires a `LOOM_GPU_TARGET` (no silent CPU path), same as `pretrain`.
> - It prints a **derived cost PLAN** when `N_sequences` or estimated GPU-seconds exceed a threshold (default: a few GPU-minutes), with the inputs shown.
> - The agent **may auto-proceed only below a row/GPU-second budget** (small eval frames). Above it, `embed` returns `PLAN` and requires the same envelope approval as `pretrain`. An agent can never `loom.embed()` an arbitrarily large corpus with no PLAN and no ceiling.

Read-only inspectors (never prompt, never need a key):

```
loom ls [Type] [--stale] [--status …]   list objects     loom show <pathspec> [--lineage] [--sample] [--eda]
loom diff <a> <b>                        contract-aware diff  loom explain <verb|contract|pathspec>
loom plan <spec|cmd>                     compile + cost PLAN  loom replay <run-pathspec>   reproducibility receipt
loom top                                 live dashboard       loom doctor                  environment check
loom search <Corpus/n> …                 the AIDE brain — scoped to `searchable` work ONLY (pooling, heads, PCA)
```

Lifecycle verbs (gated; introduced in §6/§7):

```
loom resume  <Checkpoint/n>   resume a FAILED/PREEMPTED run, quoting cost-to-completion (gated like pretrain)
loom kill    <pathspec>       stop a RUNNING job — IRREVERSIBLE, confirmation + loss summary, disable-model-invocation
loom approve <PendingLaunch/n> mint a launch from an authenticated interactive session (§5.4)
loom gc      <query>          retention / garbage-collect immutable objects (gated, §7.6)
```

### 2.3 The campaign spec — "the metric is the spec"

A campaign is a tiny declarative file (or inline flags). **No resolvable metric → no run.** The same file is authored by a human in `$EDITOR` and emitted by an agent — the one shared, diffable artifact.

```yaml
# experiments/tfm-t1-timedelta.loom        — the metric IS the spec
experiment: tfm-t1-timedelta
dataset:   IngestDataset/201
goal:      "Add inter-transaction time-delta tokens so burst-vs-lull dynamics become visible to the TFM."
metric:    "Improve fraud AUPRC on the TEMPORAL test split, vs the no-time-delta baseline checkpoint."
baseline:  Checkpoint/187                  # the shipped control, re-scored in-experiment on identical splits

tokenizer:                                 # the declarative tokenizer spec — compiled, not hand-numbered
  pipeline: financial
  merchant_hash_size: 2000
  amount_strategy: fixed                   # C2: fixed (deterministic). quantile/kmeans => fitted-artifact burden, warned + persisted.
  include_time_delta: true                 # ← the T1 change. adds TDIF_* (32 log-bins)
  # vocab_size, vocab_hash, tokens_per_txn, chunk_size are DERIVED outputs — never written here.

pretrain:
  objective: next-event                    # CLM
  budget:    small                         # probe | small | full — a STARTING POINT; the envelope below is what binds
  envelope:                                # ← [FIX] the binding, hard-enforced ceiling the human approves (§4.6)
    max_steps:      3000                   # reference repo's real-run value (08-data Step 7)
    max_usd:        100                    # hard kill ceiling; derived estimate must be ≤ this to launch
    max_wall_clock: 8h
```

`loom <verb> --experiment tfm-t1-timedelta` threads every run together. A verb run without a resolvable metric refuses: `REFUSED_NO_METRIC: declare a one-sentence measurable metric (--metric or an .loom campaign).` We deliberately do **not** let an agent silently author the metric sentence (P1's weakness); an agent may *draft* it into the file for a human to confirm.

### 2.4 Composition by pathspec — and non-interactive safety

Pathspecs are Loom's "everything is a file." `-q` makes a verb print only its output pathspec on stdout (everything human goes to stderr), so the DAG is pipeable and agent-composable.

```bash
loom ingest ./data/decoder_corpus_t1 --name tfm-corpus-t1 -q \
  | xargs loom tokenize --include-time-delta --experiment tfm-t1-timedelta -q \
  | xargs loom pretrain --objective next-event --budget small --metric fraud-auprc --experiment tfm-t1-timedelta
#                                              ^ prints a machine-readable PENDING_LAUNCH and exits 0; does NOT launch
```

> **`[FIX] Pretrain in a non-interactive/piped context never hangs and never buries a launch.** When stdin is not a TTY, `loom pretrain` (a) emits a single-line machine-readable `PENDING_LAUNCH <pathspec> cost=$… envelope=…` on stdout and **exits 0** (it does not block waiting for confirmation, and does not break the pipe with a non-zero code), and (b) **refuses `--launch --yes`** with `REFUSED_NONINTERACTIVE_LAUNCH: a launch must come from an interactive TTY, `loom approve`, or `[L]` in loom top`. A launch can never be buried mid-pipeline in a script or cron job. The same rule applies to `embed` above its budget and to `report --send`.

---

## 3. Worked examples

### 3.1 T1 — turn on the time-delta token (changes vocab → retrain) · C1 + C3

```
# HUMAN
$ loom tokenize IngestDataset/201 --include-time-delta --experiment tfm-t1-timedelta
PLAN  tokenize IngestDataset/201   (workspace-write · cheap · compiles in 0.4s)
  step change   + time_delta_s  (TimeDeltaTokenizer, 32 log-bins → TDIF_*)
  C3            tokens/txn 12 → 13   ⇒  chunk_size 315 → 292   (auto-derived: 4096 // (13+1); APPLIED)
  C1            vocab_size 6251 → 6283 (+32 TDIF)   ⇒  NEW signature; shipped Checkpoint/187 is INCOMPATIBLE
                (retrain required — `embed` will refuse Checkpoint/187 against this corpus)
  C2            determinism OK — TDIF bins are config-only, no fitted artifact
Proceed? [y/N] y
✓ Corpus/204   verdict=PASS  vocab=6283  tokens/txn=13  chunk_size=292  sig=sha256:9af3…
  next: loom pretrain Corpus/204 --objective next-event --budget small --launch
```

```jsonc
// AGENT — same engine, same result envelope; the approval handshake is the confirm_token round-trip (§5.3).
loom.tokenize({ in:"IngestDataset/201", include_time_delta:true, experiment:"tfm-t1-timedelta" })
// → { status:"PLAN", tier:"workspace-write", confirm_token:"ct_8b1f…",
//     contract_effects:[
//       { contract:"C3", change:"tokens_per_txn 12→13", derived:{ chunk_size:"315→292" } },
//       { contract:"C1", change:"vocab_size 6251→6283", consequence:"retrain_required",
//         invalidates:["Checkpoint/187"] },
//       { contract:"C2", status:"ok" }],
//     result_preview:{ pathspec:"Corpus/204", verdict:"PASS", vocab_size:6283, vocab_hash:"sha256:9af3…" } }
// agent then: loom.tokenize({ …same args…, confirm_token:"ct_8b1f…" }) → committed result envelope (byte-identical to CLI --json)
```

The eyeball check is one verb — `loom show Corpus/204 --sample` prints a real corpus line (`… MONTH_02 TDIF_18 <sep> AMT_3 …`). For T1 the baseline is simply the shipped `Checkpoint/187`, threaded into the experiment via `baseline:`; for a from-scratch change `stage_baseline:true` re-tokenizes without the change under the same `--experiment` (house rule #3, enforced for both drivers per §2.1).

### 3.2 T2 — drop the CUST token (deployability ablation) · C1 + C6

```
# HUMAN
$ loom tokenize IngestDataset/201 --drop-step cust --eval-split entity-disjoint \
        --experiment tfm-t2-drop-cust
PLAN  tokenize IngestDataset/201   (workspace-write · cheap)
  step change   − cust  (CUST_0..CUST_2999)
  C1            vocab_size 6251 → 3251 (−3000)   ⇒  retrain required; invalidates Checkpoint/187
  C3            tokens/txn 12 → 11   ⇒  chunk_size 315 → 341   (auto-derived)
  C6            eval split  temporal → ENTITY-DISJOINT (hold out 600 whole users)
                COHERENCE: your stated goal is "deployable to unseen users". A temporal split would
                let train/test users overlap → offline AUPRC would be optimistic. Entity-disjoint
                makes the generalization claim honest. (This is exactly backlog item E2.)
Proceed? [y/N] y
✓ Corpus/205   vocab=3251  tokens/txn=11  chunk_size=341  eval_split=entity-disjoint  verdict=PASS
```

The **goal↔metric↔split coherence check** (grafted from P1, run at compile/EDA time, not by an NL planner) notices "unseen users" contradicts a temporal split, folds in the entity-disjoint split, and *refuses to publish a number whose split contradicts the stated goal* at `evaluate` time. Agent call is identical args + the `confirm_token` round-trip.

### 3.3 D5 — next-trade prediction on internal multi-chain trade streams (first production-facing run; T2-from-day-one) · new data, C1 + C6

```
# Pattern C new-data pipeline as plain verb composition. Note: re-ingesting the same source is IDEMPOTENT (§6).
$ loom ingest ./data/zkai_internal/dex_trades --name zkai-trades-multichain \
        --entity wallet --event trade --experiment tfm-d5-next-trade
✓ IngestDataset/310   rows=41.2M  wallets=1.9M  chains=[eth,base,sol,hyperliquid,polymarket,kalshi]
  eda VERDICT: REVIEW
    · freshness OK (latest event 2026-06-13)
    · ⚠ whale skew: 3 wallets carry 18% of volume — see `loom show IngestDataset/310 --eda`
    · chain exports arrive broken: 0.4% of rows dropped (null ts) — logged in object.ingest_report

$ loom tokenize IngestDataset/310 --schema chain --no-identity-token --eval-split entity-disjoint \
        --experiment tfm-d5-next-trade           # identity lives in the GROUPING, not the vocab (T2 baked in)
✓ Corpus/311  vocab=5355  tokens/event=10  chunk_size=372  eval_split=entity-disjoint(by=wallet)  verdict=PASS
  (5 specials + 8 AMT + 2 DIR + 5000 CTPY + 256 MTH + 2 ST + 7 GAS + 24 HOUR + 7 DOW + 12 MONTH + 32 TDIF)

# Free CPU rehearsal of the WHOLE DAG before any GPU button (grafted from P1; the torch-free PPMI+SVD model-builder).
$ loom pretrain Corpus/311 --model-builder local --budget probe --experiment tfm-d5-next-trade
✓ rehearsal complete in 22s on CPU. End-to-end DAG green.
  next-trade Prec@5 (PPMI+SVD baseline) 0.31 vs repeat-last-item 0.22 — embeddings carry real signal.
  → approve the real GPU run against THIS green dry run, not a hope.

# The irreversible-tier launch — explicit target + DERIVED cost + a binding envelope.
$ export LOOM_GPU_TARGET=modal LOOM_NEMO_IMAGE=nvcr.io/nvidia/nemo:25.09.01
$ loom pretrain Corpus/311 --objective next-event --budget small \
        --metric next-trade-prec@5 --baseline popularity,repeat-last-item \
        --experiment tfm-d5-next-trade --launch
┌─ LAUNCH GATE  ·  tfm-d5-next-trade  ·  stage: pretrain ─────────────────────────────────────┐
│ tier         IRREVERSIBLE / EXTERNAL  (real GPUs)                                            │
│ what         next-event CLM pretrain on Corpus/311                                           │
│ COST (derived, not a label):                                                                │
│   tokens          1.93e10  (15.3M seqs × 372 events × 3.4 tok/event-incl-sep, from Corpus/311)│
│   params          29.0M     seq_len 4096    FLOPs/token ≈ 6·N·… → 3.36e20 train FLOPs        │
│   target          modal · 4×H100 @ $2.20/GPU-h · ~38% MFU                                    │
│   ESTIMATE        ≈ 142 GPU-h ≈ $312        confidence: MEDIUM (MFU assumed)                  │
│   ⚠ THIS IS 3.6× YOUR LARGEST PRIOR RUN in this workspace ($86, Checkpoint/420)  ← anomaly    │
│ ENVELOPE (binding — orchestrator hard-kills at the ceiling, §4.6):                          │
│   max_steps 3000   max_$ 360 (estimate +15%)   max_wall_clock 8h                            │
│ produces     Checkpoint → HF safetensors (C5)                                               │
│ flag         PRODUCTION-FACING → a SECOND gate at `report --send` (customer artifact)        │
│ approve →    estimate $312 > $200 high-gate threshold: re-run with --launch --confirm-usd 312 │
│              (or press [L] in loom top and TYPE the amount)                                  │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
$ loom pretrain Corpus/311 … --launch --confirm-usd 312
✓ Checkpoint/420  status=RUNNING  job=modal://…   envelope={3000 steps, $360, 8h}   watch: loom top

# embed is GPU work → it ALSO prints a derived PLAN (this corpus is large). [FIX]
$ loom embed Checkpoint/420 --corpus Corpus/311 --pool last-token --experiment tfm-d5-next-trade
PLAN  embed  (expensive · GPU · launch-and-track above threshold)
  N_sequences 15.3M   est 1×H100 × 1.4h ≈ $3.1   target: modal
  exceeds auto-proceed budget ($1) → requires --launch (or agent envelope approval)
$ loom embed Checkpoint/420 --corpus Corpus/311 … --launch --yes
✓ Embeddings/421  (last-token pool, dim=512)  row-ID aligned (C6 asserted)  cost=$3.0

$ loom evaluate Embeddings/421 --task next-item-preck --k 5 \
        --baseline repeat-last-item --experiment tfm-d5-next-trade
✓ Scores/422  next-trade Prec@5 = 0.41 vs repeat-last-item 0.22 on WALLET-DISJOINT split
  VERDICT: PASS  (metric beats baseline as specified)
```

Because D5 is **production-facing**, `report --send` (customer artifact) is a second irreversible gate requiring a PASS verdict, a non-stale lineage (§7.3 makes staleness a **hard FAIL** here), and a human button — never agent-fired.

> Note how the D5 cost (`≈$312`) differs from the T1 cost (`≈$86`): D5's corpus is ~40× larger, so the **derived** estimate is ~3.6× higher (steps capped at 3000 bound it). The number is computed from `Corpus/311`'s token count, not copied from a template. *(`[FIX]` — the prior draft quoted an identical hardcoded `$86` for both T1 and the 41.2M-row D5; that fictional number is gone.)*

---

## 4. Cost / launch gating — the safety perimeter, made structural

Tier is a **property of the verb**, so it can't be bypassed by a flag. There are now **six** structural locks (the prior three, plus three that close the holes the critique found).

### 4.1 Tier locks (unchanged, restated)

- **read-only** (`ls`, `show`, `diff`, `plan`, `explain`, `replay`, `top`, `doctor`, `report` without `--send`, `watch`): never prompt, no key.
- **workspace-write** (`ingest`, `tokenize`, `evaluate`): run in the workspace; cheap PLAN; human gets `[y/N]` only on destructive overwrites; agent auto-proceeds when non-destructive.
- **expensive / external** (`embed` above threshold, `pretrain` planning, AIDE `search`/`optimize`): **always** prints a derived cost PLAN and stops; never proceeds on its own above its budget.
- **irreversible / external** (`pretrain --launch`, `embed` above the budget cap, `report --send`, `kill`, `gc`, future `deploy --apply`): **human button only**, `disable-model-invocation: true`.

### 4.2 `[FIX]` Lock 4 — the cost PLAN is DERIVED from the compiled plan, never a label

The estimate is a function of the actual run, with every input shown so it is **auditable, not magic**:

```
estimate_$  =  train_FLOPs / (target_FLOPs_per_s · MFU)  ·  $/GPU-h_per_device · n_devices  / 3600
train_FLOPs ≈ 6 · params · tokens          # tokens = Corpus.token_count (carried on the object)
                                           # params from the model config; both scale A1 (seq_len↑) and A2 (params↑)
```

So A1's context scaling (315→630→1260 transactions, longer `seq_len`, ~quadratic attention term added to the FLOPs model) and A2's parameter scaling change the number; a 40×-larger corpus (D5/D1) changes the number. The gate card prints `tokens, params, seq_len, steps, $/GPU-h, target, MFU, confidence`. Low-confidence estimates (unknown MFU, spot pricing) are flagged `confidence: LOW` and **widen the envelope's `max_$` automatically** so the human approves a ceiling that covers the uncertainty.

### 4.3 `[FIX]` Lock 5 — approval is a BINDING ENVELOPE, hard-enforced, not a one-time "go"

The human/`loom approve` does not approve "a tier" or "a launch" — it approves `{max_steps, max_$, max_wall_clock}`. The orchestrator **hard-kills** (not warns) at the first ceiling reached and writes a `STOPPED_AT_BUDGET` verdict with the partial `Checkpoint` (resumable, §6). `loom top` shows the ceiling and live burn, and the COST pane alert at 90% is a heads-up — **the kill at 100% is automatic and non-optional**. "Approve ≈$312" now means "stop at $360 (estimate +15% / confidence-widened)," full stop.

### 4.4 `[FIX]` Lock 6 — magnitude-aware gate + per-workspace spend cap (bounds the SIZE of what reaches the button)

The remaining threat is an agent proposing **one enormous** job a human rubber-stamps. We make a huge job *feel* huge:

- **Tiered approval by estimated $:** below `$200` (default) a launch is a single `--launch --yes` / `[L]`. **At or above the threshold**, the launch requires `--launch --confirm-usd <N>` where `<N>` must equal the derived estimate (the human/`approve` *types the dollar amount* — a second factor that can't be fat-fingered), and in `loom top` the `[L]` path prompts for the typed amount.
- **Anomaly banner:** if the estimate is `≥ Nx` (default 3×) the largest prior run in the experiment/workspace, the gate card shows `⚠ THIS IS N× YOUR LARGEST PRIOR RUN` (see §3.3). The agent's `PendingLaunch` payload carries the same `anomaly:{ratio, baseline_run}` so the agent must surface it when it hands off.
- **Per-workspace/day spend cap:** a hard `LOOM_SPEND_CAP_USD_PER_DAY` ceiling. A launch that would breach it is `REFUSED_SPEND_CAP` and raising the cap is itself a human-escalation action (not an agent tool). So even a sequence of individually-approved runs can't quietly run away.

### 4.5 Locks 1–3 (the original three), restated

(1) `pretrain`/`embed`-above-threshold refuse with `REFUSED_NO_GPU_TARGET` (the launch card renders as a $0 clean park) until `LOOM_GPU_TARGET` is set — **a new user structurally cannot spend money on day one.** (2) A derived cost PLAN is printed and the verb stops; a bare `loom pretrain` never launches. (3) Firing requires a second explicit gesture from an authenticated interactive source (§5.4): `--launch --yes`/`--confirm-usd` from a human TTY, `[L]` in `loom top`, or `loom approve PendingLaunch/n`. An agent-originated launch is refused and auto-queued as a `PendingLaunch`. `searchable` vs `launch-and-track` is enforced by the tool schema, so `loom.search` literally cannot accept a launch-and-track capability.

**This whole GPU-cost-safety story lives in the type system and the orchestrator, not in prompts.**

### 4.6 The envelope in `loom top`

The COST pane shows the binding ceiling and the hard-stop, not just an alert (see §7.5). A `STOPPED_AT_BUDGET` job appears with its partial checkpoint and a `loom resume` affordance.

---

## 5. The dual-driver contract: how a human and an agent share one surface

A Claude/Codex agent drives the **exact same verbs** as typed tools. Its leverage is judgment and composition; its hard limits are the same gates. It never free-types shell — it calls typed tools.

### 5.1 Human walkthrough — zero to a trained, validated TFM (T1)

1. **Install & doctor.** `npx @zkailabs.com/loom@latest` bootstraps; `loom doctor` checks adapters → `VERDICT: PASS`. No key for read-only verbs.
2. **Declare the campaign.** `loom new experiment tfm-t1-timedelta` scaffolds `experiments/tfm-t1-timedelta.loom` and opens `$EDITOR`. You write the two sentences + dataset + the `envelope`. Without a metric, every mutating verb refuses.
3. **Ingest + EDA gate.** `loom ingest ./data/decoder_corpus_t1 --name tfm-corpus-t1 --experiment tfm-t1-timedelta` → `IngestDataset/201` (idempotent — re-running returns the same object, §6). `loom show IngestDataset/201 --eda` shows the leakage table (the `CUST_*` family flagged).
4. **Tokenize** (the contract verb). PLAN compiles in <1s; C3 derives `chunk 315→292`; C1 says vocab `6251→6283` and `Checkpoint/187` is incompatible. Confirm → `Corpus/204` with a signature hash. `loom show Corpus/204 --sample` for the eyeball check.
5. **Stage the baseline** (house rule #3). For T1 the baseline is the shipped `Checkpoint/187` (already threaded). For a from-scratch change, set `stage_baseline:true`.
6. **Plan the pretrain — see the money.** With no GPU target → `REFUSED_NO_GPU_TARGET`. Set `LOOM_GPU_TARGET=modal`, re-run → the **derived** cost PLAN (tokens, params, $/GPU-h → `≈$86` for this 263M-token corpus), the binding envelope, the step-0 loss target `ln(6283)=8.75`, and **stop**.
7. **Press the button.** `loom pretrain … --launch --yes` (under the $200 high-gate threshold, so no typed amount needed) or `[L]` in `loom top`. `Checkpoint/420` goes `RUNNING` inside its envelope.
8. **Watch.** `loom top` (§7.5): GPU job, live loss, **$ burned vs the binding ceiling**, the step-0 loss canary (`8.74 ≈ ln 6283 ✓`). Ctrl-C the dashboard freely; it's read-only, the job keeps running.
9. **Embed + evaluate.** `loom embed Checkpoint/420 …` → for T1's eval frame this is a small N, so it auto-proceeds below the budget; → `Embeddings/421`. `loom evaluate …` → `Scores/422` with a VERDICT, AUPRC + CIs on the temporal split vs baseline. `evaluate` enforces C6 (row-ID join).
10. **Report.** `loom report --experiment tfm-t1-timedelta` assembles the model card (runs, deltas, lineage DAG, cost actuals, the C1–C6 ledger). `--md > t1-card.md` for the KB, including negative results (house rule #6).

Total: ~10 verbs, two confirmations (one cheap tokenize, one launch), one dashboard you dip in and out of.

### 5.2 Agent walkthrough — an agent-driven campaign and the launch handoff (D5)

1. **Author the campaign.** `loom.new_experiment({ id, goal, metric })`. Omit the metric → every later tool returns `REFUSED_NO_METRIC`. The agent *drafts* the metric; the human confirms it for a production-facing run.
2. **Ingest + read the EDA verdict.** `loom.ingest({…})` → `{ pathspec:"IngestDataset/310", eda:{ verdict:"REVIEW", flags:[{type:"whale_skew",…}] } }`. The agent sees REVIEW as data and states its reasoning.
3. **Tokenize with contracts as data.** `loom.tokenize({…})` → PLAN with `contract_effects` + `confirm_token`; non-destructive, so the agent makes the second confirm call → `{ pathspec:"Corpus/311", verdict:"PASS" }`.
4. **Rehearse free on CPU.** `loom.pretrain({ model_builder:"local", budget:"probe" })` → DAG green in ~22s — *evidence* before asking for money.
5. **Plan the real pretrain — and hit the wall.** `loom.pretrain({ objective, budget:"small", metric })` → `{ status:"PLAN", tier:"irreversible", cost:{ derived:true, usd:312, confidence:"medium", inputs:{tokens,params,seq_len,…} }, envelope:{max_steps,max_$,max_wall_clock}, anomaly:{ratio:3.6,baseline:"Checkpoint/420"}, requires_human_button:true }`. **There is no argument the agent can pass to fire a real launch.** `launch:true` → `REFUSED_AGENT_CANNOT_LAUNCH: queued as PendingLaunch/77`. `loom.approve` **does not exist** as an agent tool (`disable-model-invocation:true`). The AIDE brain *structurally cannot* tree-search `pretrain` (manifest-typed `launch-and-track`).
6. **Hand off the button.** `loom.request_approval({ pending:"PendingLaunch/77" })` pushes a human-facing gate card to an **authenticated** channel (§5.4), carrying the derived cost, the envelope, and the anomaly banner. The agent prepares every gate but mints none.
7. **Poll, don't block.** `loom.show({ pathspec:"Checkpoint/420" })` polls `RUNNING → SUCCEEDED | FAILED | PREEMPTED | STOPPED_AT_BUDGET`; or `loom.watch(... )` registers a webhook. The agent never holds a blocking loop on a 6-hour job. On `FAILED/PREEMPTED` it may *propose* a `loom resume` (which is itself gated; §6).
8. **Finish the searchable tail.** After SUCCEEDED: `loom.embed(...)` (below-budget eval frames auto-proceed; large corpora hit the same PLAN as a human), `loom.evaluate(...)` → VERDICT, then `loom.report({format:"md"})`. For `report --send` it again surfaces a gate card. It writes the run's typed, redacted learnings rows.

**Trust model in one line:** full agent autonomy over everything reversible and within-budget (proposing, ingesting, EDA, tokenizing, evaluating, rehearsing on CPU, small embeds, drafting reports); **zero authority over money above the auto-proceed budget, over shipping, or over destruction.**

### 5.3 `[FIX]` The approval handshake, specified per driver (not claimed identical)

| Step | Human (TTY) | Agent (tool) |
|---|---|---|
| PLAN | rendered card, ends `Proceed? [y/N]` | `{ status:"PLAN", confirm_token, …}` |
| Confirm a cheap/non-destructive verb | type `y` | second call `loom.<verb>({…, confirm_token})` |
| Confirm a launch (< $200) | `--launch --yes` / `[L]` | **cannot** — `REFUSED_AGENT_CANNOT_LAUNCH`, queues `PendingLaunch` |
| Confirm a launch (≥ $200) | `--launch --confirm-usd <N>` / type amount in `[L]` | **cannot** |
| Result | committed result envelope | **byte-identical** committed result envelope |

The **result** is identical across drivers; the **approval** is a mirrored data round-trip for cheap verbs and a hard human-only wall for launches. `confirm_token` is single-use, scoped to the exact compiled plan (its hash), and expires (default 15 min) so an agent can't replay a stale approval against a changed spec.

### 5.4 `[FIX]` Approval-channel trust model (the Slack/PR hole)

The prior draft let a launch be approved by "a human's Slack identity" — a soft boundary in an org that runs unattended Slack bots and a headless Slack→Linear loop. Closed:

- A `confirm_token` for a launch is **minted only by an authenticated interactive action**: a human TTY session, or a click in an app the human is logged into. It is **never** minted by "a message that appears to come from a Slack handle."
- If Slack approval is offered, it is **notify-only by default**: the gate card posts to Slack but the human must paste a **signed, single-use, expiring token** back into a TTY or click an authenticated deep link — a free-text "approve" message a bot could emit does **not** mint a launch.
- **Automated/bot Slack identities are explicitly excluded** from minting tokens (allow-list of human principals; the triage bot's identity is denied by construction).
- PR-comment "approvals" are advisory only and never mint a token.

---

## 6. `[FIX]` Failure, resume, and idempotency — the lifecycle the prior draft was missing

The single most common real-world event is a 6-hour pretrain dying at step 1840/3000. The immutable-object model accommodates this rather than fighting it.

**Object status is first-class.** A `Checkpoint` is `RUNNING | SUCCEEDED | FAILED | PREEMPTED | STOPPED_AT_BUDGET`, each carrying a **resume token** (the orchestrator's last consolidated step + optimizer state ref). A partial checkpoint is a real, addressable object with `verdict=INCOMPLETE` — it is *not* a clean `PASS`, and nothing downstream (`embed`/`evaluate`) will consume an `INCOMPLETE` object (refused with a named card).

**`loom resume <Checkpoint/n>`** (gated exactly like `pretrain`, `disable-model-invocation` on the actual launch) re-attaches to the resume token and **quotes cost-to-completion (remaining steps), not the full cost**:

```
$ loom resume Checkpoint/420            # died PREEMPTED at step 1840/3000
PLAN  resume Checkpoint/420  (irreversible · GPU)
  from step 1840 → 3000   remaining 1160 steps
  COST (derived)  ≈ 9.2 GPU-h ≈ $20    (not the full $86 — you already spent $66)
  envelope (remaining)  max_$ 30   max_wall_clock 3h
  approve → loom resume Checkpoint/420 --launch --yes
```

A user who loses a $66 run to a node preemption pays ~$20 to finish, not $86 to restart. On `FAILED` (a real error, not preemption), `loom show --lineage` shows the failure cause and the experiment records the FAIL; resume is offered only when the resume token is valid.

**Idempotency of `ingest`/`tokenize`.** Re-running `ingest` or `tokenize` with the **same source + same spec** returns the **existing object** (content-addressed by `source_fingerprint + spec_hash`), it does **not** fork a garbage twin. D5's "0.4% of broken rows dropped on each pull" is handled deterministically: the drop rule is part of the spec, so the same export yields the same `IngestDataset`. A *changed* source or spec produces a new object (and marks downstream stale, §7.3). `loom ingest --force` is the explicit escape hatch to re-pull a moving source as a new object.

**What FAIL does to lineage/experiment.** A FAILED/STOPPED stage is recorded in the experiment with its verdict and cost actuals (so the report and telemetry reflect reality, including spent-then-failed money); it does not silently vanish, and it does not block sibling runs.

---

## 7. Lineage, contracts-as-UX, diff/replay, reports, dashboard

### 7.1 Data objects & pathspecs

Every verb output is an immutable, versioned object addressed by `Type/<n>`. Each carries: parent pathspecs, the exact verb+args that made it, the contract signatures it satisfies, its VERDICT and **status**, the experiment id, and (for GPU objects) **cost actuals + the envelope it ran under**. **C1 is enforced structurally:** corpus + tokenizer-config + `vocab_size` travel as *one* ingested object, so the lineage physically cannot claim a checkpoint was trained on a tokenizer it wasn't.

```
$ loom show Embeddings/421 --lineage
Embeddings/421  (last-token pool, dim=512)  experiment=tfm-t1-timedelta
└─ Checkpoint/420   pretrain --budget small   loss=3.91  cost=$84/24GPU-h  envelope=$86  sig=sha256:9af3…  PASS
   └─ Corpus/204    tokenize --include-time-delta   vocab=6283 tok/txn=13 chunk=292   PASS
      └─ IngestDataset/201   ingest ./data/decoder_corpus_t1   eda=REVIEW
```

### 7.2 Contract violations as great UX — caught at compile time, surfaced as a named diff

The unifying rule: **a contract is either auto-renegotiated and announced, or it blocks with a named cause and an offered one-line fix — never a stack trace, never a silent pass.** The check runs during `tokenize`/`plan` in <1s, before any data is tokenized or dollar spent. The same diagnostic renders as a terminal card (human) and a `diagnostics[]` array (agent).

#### (a) The MONTH_12 / CARD_0 collision — the real reference bug (C1, vocab integrity)

Loom lays out each step's id-range from the spec, asserts the ranges are **disjoint and total to `vocab_size`** (injectivity), and on collision **refuses to write the Corpus**:

```
$ loom tokenize IngestDataset/201 --reorder-step card:first --experiment tfm-t5-fieldorder
✗ CONTRACT VIOLATION  C1 (vocabulary is not injective) — refusing to write Corpus, $0 spent

  Two distinct tokens resolve to the SAME id — the model could never tell them apart:

      token      step    offset        id
      MONTH_12   month   base 6178 +11 → 6189
      CARD_0     card    base 6189 +0  → 6189     ⚠ COLLISION

  Cause: --reorder-step moved `card` ahead of `month`, but the `month` step's base
         offset was not recomputed, so its top id overlaps `card`'s range. Classic silent
         failure: training would run and produce a garbage-but-plausible loss.

  Fix:   re-derive offsets so every (step,value) gets a unique id:
             loom tokenize IngestDataset/201 --reorder-step card:first --recompute-offsets
         ⚠ NOTE (C1): reordering shifts EVERY token's id ⇒ vocab_hash changes (even though
           vocab_size is unchanged) ⇒ existing checkpoints are INVALIDATED, retrain required.
           This fixes the collision; it does NOT make the reorder free. (backlog T5)
  Explain: loom explain C1
  VERDICT: FAIL   (no Corpus written; nothing downstream can proceed)
```

> `[FIX]` The fix line now states the C1 consequence honestly: `--recompute-offsets` produces a *valid* corpus but a *new* `vocab_hash`, so the reorder still requires a retrain — it is not a vocab-preserving, $0 fix. The agent payload carries `retrain_required:true` alongside the fix.

#### (b) Tokenizer ↔ checkpoint signature mismatch (C1, the E2 hardening)

Every Checkpoint stores its tokenizer **signature** `{config hash, vocab_size, tokens_per_txn, vocab_hash, encode_path}` as a first-class field. `embed`/`pretrain` assert the Corpus↔Checkpoint signatures match *before any GPU forward pass*:

```
$ loom embed Checkpoint/187 --corpus Corpus/204
✗ CONTRACT VIOLATION  C1 (signature mismatch) — refusing to embed, $0 spent
  Checkpoint/187 : vocab=6251 tokens/txn=12 sig=sha256:4b7c…  encode_path=corpus-line  (time-delta OFF)
  Corpus/204     : vocab=6283 tokens/txn=13 sig=sha256:9af3…  encode_path=corpus-line  (time-delta ON)
  The embedding table has 6251 rows; Corpus/204 emits ids up to 6282 — IDs would index the
  wrong embedding rows. The model would run and return mush, with NO error.
  Fix:   (1) embed against Checkpoint/420 (matching sig 9af3…), or
         (2) retrain on the new tokenizer:  loom pretrain Corpus/204 --budget small --launch (≈$86 derived)
  VERDICT: FAIL
```

#### The rest of the contracts get the same treatment

- **C2 (determinism):** `amount_strategy=quantile|kmeans` warns you've taken on a fitted-artifact burden and **persists the binner state into the Corpus object** (`get_state()`), so determinism is recoverable — announced, not silent.
- **C3 (corpus grammar):** `chunk_size` is *derived* from `tokens_per_txn` and announced; forcing a bad value shows the truncation math and refuses.
- **C4 (`{input_ids, labels}` with −100 masking):** validated on the free CPU rehearsal before any GPU.
- **C5 (HF safetensors):** `pretrain` writes consolidated safetensors; `embed` asserts `AutoModelForCausalLM.from_pretrained` loads.
- **C6 (eval hygiene):** `evaluate` refuses to score if embeddings/labels don't join on row-IDs, and the goal↔split coherence check refuses to publish a number whose split contradicts the stated goal.

The step-0 **loss canary** (`first-step loss 8.74 ≈ ln(6283) ✓`) is surfaced as live narration in `loom top`; far-off → the run narration pauses in plain language and offers a halt.

#### `[FIX]` Spec-shaped vs code-shaped experiments (the C2/replay boundary)

`loom replay` proves reproducibility from spec text alone for **spec-shaped** experiments (the declarative tokenizer; T1/T2/A1/A2/E2). Several backlog items are **code-shaped** — O1 (custom loss/collator), A4 (field-fusion modeling code), T3(c) (numeric side-channel) — where the compiler can't re-derive behavior from spec text. For these:

- The experiment **pins the modeling code's git-sha into the signature** (`code_sha` alongside `vocab_hash`), and `loom replay` falls back to **git-provenance replay** (assert the recorded `code_sha` is checked out, recompile what is spec-shaped, re-run the pinned code).
- The Corpus signature records **which `encode_path`** produced it (corpus-line vs per-transaction — level-400 sharp edge #3), so embeddings from different encode paths **cannot be compared as if identical** (`evaluate`/`report --compare` refuse a cross-encode-path comparison). The "reproducibility is provable" claim is scoped explicitly to spec-shaped work; code-shaped work is "provable up to a pinned git-sha."

### 7.3 `[FIX]` Staleness BLOCKS by default — the silent-wrong-number path is closed

Re-running an upstream `tokenize` marks every downstream `Checkpoint`/`Embeddings`/`Scores` **stale**. The prior draft only *warned*; a warning is dismissible and is the wrong default for a silent-correctness hazard. New behavior:

- `evaluate`/`report` **refuse** a stale object by default, with a named card and a single explicit `--allow-stale` override **recorded in lineage** (so the report shows the number was computed against a superseded upstream).
- `report --send` (the customer artifact) treats staleness as a **hard FAIL with no override** — a customer-facing number can never be computed from a superseded tokenizer.
- Staleness distinguishes **direct-parent** staleness (THIS object's parent re-tokenized → block) from **sibling** staleness (an unrelated branch in the same experiment re-tokenized → informational only), so the block isn't over-broad.

`loom ls --stale` lists stale objects; `loom top` flags them (⚠). It's a property of the lineage graph, not a fragile notebook gutter mark.

### 7.4 Contract-aware diff + reproducibility receipt

```
$ loom diff Corpus/203 Corpus/204
  + step time_delta_s  (TimeDeltaTokenizer, 32 log-bins)
  IMPLIES:
    vocab_size  6251 → 6283  (+32)          # C1
    tokens/txn  12 → 13   ⇒  chunk 315 → 292 # C3
    vocab_hash  4a2b… → 9af3…  →  RETRAIN REQUIRED
```

`loom replay Checkpoint/420` re-derives the spec, recompiles it, and asserts the resulting `vocab_hash` equals the one the checkpoint recorded → `REPLAY: vocab_hash matches (sha256:9af3…). Spec reproduces.` CI runs exactly this on every change (with the code-sha fallback of §7.2 for code-shaped runs).

### 7.5 The live dashboard: `loom top`

A four-pane live dashboard — pure projection over the data objects. **The only mutating keys are the launch button and the gated kill** *(`[FIX]` — kill is no longer an unconfirmed read-only keystroke)*:

```
┌ LEADERBOARD (experiment: tfm-t1-timedelta) ─┐┌ LINEAGE ───────────────────────────┐
│ run         metric    AUPRC  Δbase verdict   ││ Ingest/201 ─┬─ Corpus/203 ─ Ckpt/187 │
│ ▸time-delta fraud-AP  0.659 +.047  PASS  ★   ││             └─ Corpus/204 ─ Ckpt/420▸│
│  baseline   fraud-AP  0.612  —     PASS      ││  ▸ = running   ⚠ = STALE             │
└──────────────────────────────────────────────┘└──────────────────────────────────────┘
┌ GPU JOBS ───────────────────────────────────┐┌ COST / ENVELOPE ───────────────────┐
│ Ckpt/420 modal 4×H100 step 1840/3000 RUNNING ││ Ckpt/420  $52 / cap $86  [60%]       │
│   loss 3.91  (step-0 was 8.74 ✓ ≈ ln 6283)   ││   HARD-KILL at $86 / 3000 steps / 8h │
│ PendingLaunch/77  D5 next-trade  ~$312 ⚠3.6× ││ workspace today  $52 / cap $500      │
│   [L]aunch (≥$200 → type amount)             ││ month  $612                          │
└──────────────────────────────────────────────┘└──────────────────────────────────────┘
nav (read-only): [enter]=show  [/]=filter  q=quit
ACTIONS (gated): [L]aunch a PendingLaunch (auth + typed-$ above threshold)   [K]ill a job (confirm + loss summary)
```

> `[FIX]` **Kill is gated like any irreversible verb.** `[K]` (capitalized, separated from read-only nav) opens a confirmation showing exactly what is lost — `$ spent, steps completed, whether resumable` — and requires explicit confirmation (typed run-id above a spend threshold). The agent **cannot kill**: `loom kill` is `disable-model-invocation:true`, so a human-approved run can't be terminated by the agent. A fat-finger on the wrong row can't destroy a job. The agent observes via `loom top --json` / `loom ls --status running --json`; it polls, it does not own a TTY and has no kill tool.

### 7.6 `[FIX]` Object retention / GC (the ever-growing store)

The immutable store would otherwise grow without bound (multi-GB checkpoints + embeddings). `loom gc` (gated, `disable-model-invocation`) applies a retention policy: **never** GC an object referenced by a shipped `Report` or a non-stale leaf; **offer** to GC superseded intermediates (stale embeddings, `STOPPED_AT_BUDGET` partials older than N days) with a freed-storage summary; checkpoints behind a customer-facing report are pinned. `loom show --storage` surfaces per-object size and total store cost so storage is visible, not invisible.

### 7.7 `--experiment` threading & reports

`--experiment <id>` is the join key; the baseline lives under the same id (house rule #3) so a report always contains its own control. Loom refuses to fold an unrelated hypothesis into an existing experiment (house rule #1).

```
$ loom report --experiment tfm-t1-timedelta
EXPERIMENT  tfm-t1-timedelta   "Add inter-transaction time-delta tokens"
metric: fraud AUPRC, temporal test split (higher = better)
  run          checkpoint      vocab  AUPRC        Δ vs baseline   verdict
  baseline     Checkpoint/187  6251   0.612 ±.011  —               PASS
  time-delta   Checkpoint/420  6283   0.659 ±.010  +0.047  ✓       PASS
lineage: IngestDataset/201 → {Corpus/203, Corpus/204} → {Ckpt/187, Ckpt/420} → Emb → Scores
cost: $84 (this experiment, 24 GPU-h, envelope $86)   contracts: C1–C6 all green   staleness: none
→ loom report --experiment tfm-t1-timedelta --md > t1-card.md
```

> `[FIX]` **Cross-run / cross-tokenizer comparison is normalized or refused.** `loom report --compare A B` **refuses to declare a naive winner across mismatched splits, tokenizers, or encode paths** and surfaces the mismatch as a caveat. For A1's 315 vs 630 vs 1260-context runs (different `chunk_size`, different cost), the report **normalizes the axis it can** (metric per the *same* eval split) and **labels what it cannot** (cost is reported per-run, not summed; sequence-coverage differs), with the plain-language caveat: *"these three runs use different chunk_size and per-run cost; ranked by fraud-AUPRC on the shared temporal split only."* Comparisons across different `encode_path` (§7.2) are hard-refused.

---

## 8. Telemetry capture — the moat (LOOM-DS-1)

Every verb invocation appends a **typed, content-redacted** record to the learnings corpus (`learnings/rollouts.jsonl`): `{ verb, spec_git_sha, code_sha, vocab_hash, contract_effects, verdict, status, metric, Δ_vs_baseline, cost_estimate, cost_actuals, envelope, parent/stage lineage, owned_by }`. Metaflow content-addresses it for free.

The high-value signal is the **interaction trajectory**: `intent → drafted spec → contract diagnostic fired → human's fix-button choice / steer → gate decision → verdict`. A labeled rollout reads *"agent proposed `include_time_delta`; C1 surfaced a MISMATCH vs Checkpoint/187; human chose Plan retrain; pretrain PASS, +0.047 AUPRC."* That distills into **LOOM-DS-1** (proposal/critic). Capture is `owned_by`-tagged and redacted — **only `owned_by=general` trajectories train the model**; customer goals and internal data shapes never do. `loom telemetry status` shows what a workspace contributed; `loom telemetry export` distills the SFT corpus.

> `[FIX]` **Opt-out, redaction-miss, and revocation are first-class flows** (the moat corpus is append-only, so these need an explicit path). (1) **Opt-out:** `LOOM_TELEMETRY=off` (or per-experiment `telemetry: off`) suppresses capture entirely; a workspace defaults to `owned_by=tenant` (never trains the model) unless explicitly set to `general`. (2) **Mis-tag / redaction miss:** every record carries a `redaction_schema_version`; `loom telemetry export` re-runs redaction at export time and **quarantines** any record that fails the current redactor or whose `owned_by` is ambiguous (it is excluded from the SFT corpus, not silently shipped). (3) **Revocation after capture:** `loom telemetry revoke --owner <id>` writes a **tombstone** that the append-only log honors at export — revoked owners' trajectories are filtered out of every future `export`, and the next distillation excludes them. Customer data stays in the tenant perimeter; only metrics/code/diffs from `owned_by=general` leave the box.

---

## 9. `[FIX]` Concurrency — two drivers in one workspace

Per the "Loom concurrent agents" reality (a human + an agent, or two agents, acting at once), the workspace is not single-writer:

- **Object numbering is atomic.** `Type/<n>` ids are minted by the metadata service via an atomic counter (Metaflow's run-id allocation); two concurrent `tokenize` calls get distinct `Corpus/<n>` ids, never a collision.
- **`PendingLaunch` is single-owner.** A `PendingLaunch` is owned by the principal who can approve it; a second driver cannot approve another's pending launch. While a human is mid-approval (`confirm_token` minted, unspent), the pending is **locked**; the agent that queued it cannot re-queue or mutate it.
- **No lost work.** Because objects are immutable and content-addressed (§6 idempotency), two drivers running the same `ingest`/`tokenize` spec converge on the **same** object rather than forking; two drivers running *different* specs produce two objects, both tracked. The launch gate's `confirm_token` is bound to a specific compiled plan hash, so a human can't accidentally approve plan A with a token minted for plan B.
- **`loom top` shows who owns what** (which principal queued a `PendingLaunch`, which is running) so concurrent drivers don't step on each other.

---

## 10. The v0.1 slice — TFM-first, smallest useful

**Goal of v0.1:** prove the verb-and-pathspec spine end-to-end on the TFM domain with the **least translation from the reference contract**, on the existing NeMo + Metaflow seam, where the contract checks (C1/C3) and the cost/launch safety model are real and the dual-driver symmetry is mechanical. Build six things in order; ship when the first four work.

| # | Build | Done-when |
|---|---|---|
| **1** | **`tokenize` + the contract compiler.** Declarative tokenizer spec → derives `vocab_size`/`vocab_hash`/`tokens_per_txn`/`chunk_size`; runs **injectivity (C1) + signature (C1) + chunk-derivation (C3) + determinism (C2)**; emits the named-diff card on violation. One typed contract → `loom tokenize` + `loom.tokenize()` with the **confirm_token handshake** (§5.3). | Reproduces vocab 6283 / chunk 292 for T1 and vocab 3251 / chunk 341 for T2; the **MONTH_12/CARD_0 collision is caught in <1s as a named diff and refuses to write the Corpus**, with the honest retrain-required note (§7.2a); `loom tokenize --json` result is byte-identical to the tool result. |
| **2** | **`ingest` + EDA leakage gate + idempotency** over the TabFormer/chain pipelines → `IngestDataset/<n>` carrying corpus + tokenizer-config + `vocab_size` as one object (C1 structural); content-addressed re-ingest (§6). | `loom ingest` of `decoder_corpus_t1` produces a versioned object with an `eda` VERDICT and the `CUST_*` family flagged; re-ingesting the same source returns the same object; agent gets the same JSON. |
| **3** | **`pretrain` PLAN + the full safety perimeter** (no real GPU yet): **derived** cost PLAN (§4.2), `REFUSED_NO_GPU_TARGET`, the **binding envelope** (§4.3), the **magnitude/anomaly/spend-cap gate** (§4.4), the un-delegable `--launch`/`PendingLaunch` queue with `disable-model-invocation`, **non-interactive `PENDING_LAUNCH` behavior** (§2.4), and the **authenticated approval channel** (§5.4). Wire the **free CPU PPMI+SVD rehearsal** (`--model-builder local`) so the whole DAG runs green before any GPU. | A bare `loom pretrain` never launches; an agent-originated launch is refused and queued; the cost number is computed from `Corpus.token_count` (T1 ≈ $86, a 40×-larger corpus is visibly higher); a piped `pretrain` emits `PENDING_LAUNCH` and exits 0; the local rehearsal runs ingest→tokenize→(local model)→embed→evaluate in seconds and beats a classical baseline. |
| **4** | **`embed` (as budgeted GPU work) + `evaluate` + the C1 signature handshake + C6 row-ID/coherence + staleness-blocks.** `embed` prints a derived PLAN above threshold and requires a target (§2.2 `[FIX]`); `evaluate`/`report` **refuse** stale objects by default (§7.3). | `embed` against a mismatched checkpoint refuses with the C1 signature card; a large-corpus `embed` prints a cost PLAN and won't agent-auto-fire above budget; `evaluate` refuses a non-row-ID join and an incoherent goal/split; a stale upstream blocks `evaluate` (override recorded in lineage). |
| **5** | **`pretrain --launch` on real GPU** via the NeMo adapter + `LOOM_GPU_TARGET=modal`; the **envelope hard-kill** (§4.3), the step-0 loss canary + live narration, and the `RUNNING/FAILED/PREEMPTED/STOPPED_AT_BUDGET` lifecycle + **`loom resume`** with cost-to-completion (§6). | T1 retrain produces `Checkpoint/420` (HF safetensors, C5) with the launch card, loss-canary, full lineage, a real hard-kill at the ceiling, and a recoverable resume after a simulated preemption. |
| **6** | **`report` + `loom diff` + `loom replay` (with code-sha fallback) + `loom top` (kill gated, envelope pane) + telemetry append (opt-out/revoke).** | `loom report --experiment tfm-t1-timedelta` shows baseline-vs-T1 with lineage + the contract ledger + staleness state; `report --send` hard-fails on stale; `loom replay` asserts `vocab_hash` in CI; `[K]ill` is confirmed and agent-blocked; every node lands in the redacted, owned_by-gated learnings corpus. |

**The exact first verbs to build, in order:** `loom tokenize` (the contract compiler — highest value, catches the silent-failure class, sets the typed-contract + confirm_token pattern every other verb inherits), then `loom ingest`, then `loom pretrain` (derived PLAN + full safety perimeter + CPU rehearsal, *before* any real GPU). Those three give a human or agent a fully gated, contract-checked, CPU-rehearsable T1/T2 dry run with **zero GPU spend** and the entire cost-safety model exercised against a real (derived) number — the smallest thing that proves the whole thesis. `embed`/`evaluate`/`report` and the real GPU launch follow.

**Explicitly out of v0.1:** the AIDE `search` verb (searchable tier), `deploy`/`report --send` to external customers, multi-tenant BYO-Metaflow endpoints, the full multi-task E1 battery (start with fraud-AUPRC + next-item Prec@K), and richer `loom top` panes (a minimal status + envelope view suffices). `loom gc` retention ships as a stub policy in v0.1 and is hardened in v0.2.

---

## 11. `[FIX]` Reconciliation with prior locked decisions (per the house-rule status convention)

The brief said the spine was "consistent with locked decisions." Two on-disk docs partially contradict it; this section resolves the contradiction explicitly rather than asserting a consistency that isn't there. **This DESIGN-UX is the canonical interaction surface and supersedes the conflicting parts below; `command-surface.md` and `v0.1-plan.md` should carry a status header pointing here.**

**(1) Verb set — supersedes `command-surface.md`'s 12-verb catalog.** `command-surface.md` commits to `connect/eda/features/pipeline/train/optimize/validate/viz/report/deploy/ops/collab`. This spec collapses the *TFM-training core* to six verbs and maps the rest:

| command-surface.md verb | Maps to here |
|---|---|
| `connect` | `ingest` (the `IngestDataset` producer) |
| `eda` | folded **into `ingest`** as the leakage-gate VERDICT (read-only, inline) |
| `train` (pretrain/embed) | **split** into `pretrain` (launch-and-track) **and** `embed` (budgeted GPU) — making `embed`'s cost explicit was a `[FIX]` |
| `optimize` | `search` (AIDE, searchable-only) — out of v0.1 |
| `validate` | folded **into `evaluate`**, which **carries the PASS-before-promote gate (house rule #5)** explicitly — `report --send`/`deploy` assert an `evaluate` PASS verdict, so the gate is *not* lost, just renamed |
| `report` / `viz` | `report` (viz is a render mode of the model card) |
| `deploy` / `ops` / `collab` | post-v0.1; `deploy --apply` is the same irreversible tier; `collab`/`ops` are read/share over the same objects |
| `features` / `pipeline` | not part of the TFM-training core (the corpus *is* the feature pipeline, via `tokenize`); revisit for the general-DS surface |

`tokenize` and `embed` are new top-level verbs because the TFM domain makes the tokenizer the load-bearing contract and embedding a distinct GPU cost — both were implicit (and `embed`'s cost was dangerously implicit) in the 12-verb catalog.

**(2) v0.1 form factor and `pretrain` placement — supersedes `v0.1-plan.md` §3/§7.** `v0.1-plan.md` decided a **Claude Code skill-pack** v0.1 with the bespoke binary as end-state, and placed `/loom-train` (pretrain) in **v0.2**. Two reconciliations:

- **Form factor:** the **typed-contract narrow waist** (§2.1) is the real commitment — *one declaration, two faces*. The skill-pack and the standalone `loom` binary are **two renderers of the same contract**, not a v0.1-vs-end-state fork. v0.1 can ship the agent face as Claude Code skills (inheriting plan-mode/permissions/memory, as decided) **and** a thin `loom` CLI from the same declaration; "skill-pack first" and "verbs over pathspecs" are not in tension once the contract is the single source.
- **`pretrain` placement:** v0.1-plan.md keeps *real GPU* `pretrain` in v0.2, and this spec **agrees** — build item **5** (real GPU launch) is explicitly the rung where GPU/NeMo lands. What v0.1 builds (item **3**) is `pretrain` **PLAN + the safety perimeter + CPU rehearsal with zero GPU spend**. So "pretrain is the third verb to build" means *its planning, gating, and rehearsal surface*, which is exactly the cheapest way to prove the cost-safety thesis before GPU complexity — consistent with v0.1-plan.md's "derisk the seam before GPU" principle, not a contradiction of it.

---

## 12. Why this synthesis, and what we deliberately rejected

**Chosen spine — P4 (Unix verbs + pathspecs).** It is the reference contract made ergonomic (every concept the TFM docs teach transfers 1:1); dual-driver symmetry is mechanically guaranteed (one typed contract → two faces, with the approval handshake specified per driver, §5.3); tier-as-verb-property + the budget-bounded, un-delegable button is the cleanest GPU-safety story; six verbs over Metaflow artifacts is the smallest useful TFM-first v0.1.

**Grafted, coherently:**
- **From P2 (Experiment-as-Code):** tokenizer-as-compiled-spec (C1/C2/C3 as <1s checks with derived `vocab_size`/`chunk_size`/`vocab_hash`), the contract-aware `loom diff`, the signature-asserting `loom replay` + CI (with the code-sha fallback for code-shaped work, §7.2), and signature-stamped checkpoints. We take the *compiler*, not the mandatory PR-per-experiment ceremony.
- **From P1 (Conversational Director):** the goal↔metric↔split coherence check (at compile/EDA time, not by an NL planner), the free CPU rehearsal of the whole DAG before any GPU button, and the steering-trajectory telemetry. We take the *capabilities*, not the NL front door.
- **From P3 (Notebook-native):** contract-violation-as-a-named-card and staleness propagation, as a terminal card + structured agent payload and a lineage invariant — not a JupyterLab extension.

**Rejected:**
- **P1's NL planner in the hot path** — undercuts "the metric is the spec," risks a rubber-stamp gate on real spend, least-buildable v0.1.
- **P3's JupyterLab-extension + constructor monkey-patching** — fragile across the repo's two encode paths, makes headless/CI/agent second-class, clashes with Loom-as-a-CLI-on-the-Pi-harness.
- **P2's "everything is a DSL + a PR"** — too heavy for the inner loop, and its escape hatch reopens the collision class; code-shaped backlog items are handled by git-sha pinning (§7.2), not a DSL.

The result is one coherent product: **a small set of sharp, typed verbs you compile before you spend, that a human and an agent drive identically (same result, mirrored approval), where contract violations are named diffs caught for free, staleness blocks the wrong number, failures resume instead of re-spending, and money is the only thing a human must press a button for — and that button enforces a binding ceiling, not just a one-time "go."**
