> **Status:** DRAFT design — Pi power-user reference + widget-first Loom-on-Pi hosting plan. **Last updated:** 2026-06-16.
> Written against the **live, installed Pi 0.79.0 source** (not just the reader digests). Every load-bearing API claim below was re-verified in the declaration files at the cited paths. Deliverable is **understanding + a buildable design**, not code.
>
> **This revision fixes four issues found in review** (noted inline as `[FIXED #n]`): (1) the launch-gate safety claim was wrongly attributed to a non-existent `ToolDefinition.disableModelInvocation` — corrected to the `tool_call {block}` hook as the only real lock; (2) the live loss-curve widget conflated `@walterra/pi-charts` (a one-shot LLM tool) with the `Image` component driven inside `setWidget` — the two seams are now separated and a no-image fallback is specced; (3) the gate guard `!ctx.hasUI` was wrong (hasUI is true in RPC) — corrected to `ctx.mode === "tui"`; (4) `renderDiff` returns a string, not a Component — the wrapping is now spelled out. Plus pi-tui path, citation-range, and widget-lifecycle fixes.

# Loom on Pi

Two parts. **Part A** is an accurate, actionable mental model of Pi — so we are *power users* who reach for the right primitive instead of hand-rolling. **Part B** is the design for hosting the new Loom Python engine on Pi as its agent runtime, and it is **widget-first**, because that is the entire reason the product owner chose Pi: bespoke, purpose-built UI for the foundation-model-training experience that Claude Code and Codex — with their fixed terminal UIs — simply cannot render.

The verifiable claims below are checked against the installed source tree at
`/Users/anub/Work/Loom/cli/node_modules/@earendil-works/pi-coding-agent@0.79.0` (and the **nested** `…/pi-coding-agent/node_modules/@earendil-works/pi-tui@0.79.0` — see the path note in §A.1), the new engine at `/Users/anub/Work/transaction-foundation-model/Loom/`, and the legacy build at `/Users/anub/Work/transaction-foundation-model/Loom-legacy/cli/`.

---

# PART A — Pi, for power users (as it pertains to Loom)

## A.1 The one-paragraph mental model

Pi (`@earendil-works/pi-coding-agent` + `@earendil-works/pi-ai`, Node ≥ 22.19) is a **minimal, self-extensible agent harness** — "the linux of the agentic world." The core is deliberately tiny: `read/bash/edit/write` tools, an agent loop, a JSONL session tree, and a multi-provider LLM abstraction. Everything we associate with a "real" coding agent — **subagents, plan mode, MCP, todos, permission popups, memory, web search, background bash** — is *explicitly not built in*. It is added through **extensions, skills, prompt-templates, themes, and packages**. The philosophy is *"primitives, not features."* This is the strategic basis for hosting Loom: our capabilities ship as **Pi packages and one extension**, never as a harness fork.

**Package topology (4 layers, layered bottom-up):**

| Package | Role | What Loom touches |
|---|---|---|
| `@earendil-works/pi-ai` | provider/LLM layer: `streamSimple(model, ctx, opts)`, model registry, `registerApiProvider`, env-key map. Also re-exports TypeBox `Type`. | the reasoning-model wiring; **`Type` for tool schemas** |
| `@earendil-works/pi-agent-core` | the loop itself: `Agent`/`AgentHarness`, `AgentTool`, `beforeToolCall`/`afterToolCall`, `AgentEvent` | indirectly (via the loop's seams) |
| `@earendil-works/pi-coding-agent` | the `pi` CLI + **`ExtensionAPI`** + package manager + the **SDK** (`createAgentSession`) + built-in tools + system prompt | **this is what Loom embeds and extends** |
| `@earendil-works/pi-tui` | terminal rendering: `Component`, `TUI`, `Box/Text/Markdown/SelectList/Image` | **the widget layer — the whole point** |

> **Path note (load-bearing, [FIXED minor]).** `@earendil-works/pi-tui` is **NOT a top-level dependency** in the Loom/legacy install. It is bundled **nested inside** `pi-coding-agent`: the only path that exists is
> `…/pi-coding-agent/node_modules/@earendil-works/pi-tui/`.
> The top-level `…/node_modules/@earendil-works/pi-tui` does **not** exist (verified by `ls`). Every pi-tui citation in this document points to the nested location; a builder copying these paths must use the nested form or they will fail. (Inside our extension code, `import { Image } from "@earendil-works/pi-tui"` still resolves — Pi exposes it as a virtual module to extensions — but the *files on disk* are nested.)

> **Naming hazard (load-bearing).** Two npm scopes coexist with identical bundled code: **canonical `@earendil-works/pi-*` (0.79.x)** and **deprecated `@mariozechner/pi-*` (0.73.x)**. The loader hard-aliases both. **Loom pins `@earendil-works/*` at exact versions** — pre-1.0, expect skew. (Legacy already does: `package.json` → `@earendil-works/pi-coding-agent` + `pi-ai` `0.79.0`.)

## A.2 The agent loop / turn lifecycle

Event-stream based (`agent/src/agent-loop.ts`). Works on `AgentMessage[]`, converting to provider `Message[]` only at the LLM boundary. Per turn, `runLoop`:

1. Injects queued **steering** messages before the next assistant call.
2. Streams the assistant response. On `stopReason === "error" | "aborted"` it stops.
3. Collects `toolCall` blocks and executes them via `executeToolCalls` — **parallel by default**, sequential only if `config.toolExecution === "sequential"` or a tool declares `executionMode: "sequential"` (verified `ToolDefinition.executionMode`, types.d.ts:357). Each call runs the **`tool_call`** hook (can `{block}`) → TypeBox validate → `execute` → the **`tool_result`** hook (can rewrite). ⚠ A batch terminates only when **every** call in it sets `terminate: true`.
4. `prepareNextTurn` snapshots model/thinking (both can change mid-run).
5. `shouldStopAfterTurn?` decides termination; on stop, drains the **follow-up** queue.

**Tool contract (durable, load-bearing for Loom):** a tool's `execute()` **throws on failure** — errors are *never* encoded in `content`. The loop catches the throw and produces an error tool-result. This is exactly why the legacy bridge distinguishes a transport failure (throw) from a domain `VERDICT=FAIL` (return); see §B.4.

## A.3 Run modes — and where custom UI exists

Pi's run mode is a first-class enum: `ExtensionMode = "tui" | "rpc" | "json" | "print"` (verified types.d.ts:207), exposed on every handler context as `ctx.mode` (types.d.ts:212).

| `ctx.mode` | Invocation | `ctx.hasUI` | Custom UI (`setWidget`/`custom`)? |
|---|---|---|---|
| **`"tui"`** | `pi` (interactive) | `true` | ✅ **Only mode where custom widgets render.** |
| **`"rpc"`** | `pi --mode rpc` | **`true`** | ❌ **`setWidget`/`custom` are no-ops** despite `hasUI===true` |
| **`"print"`** | `pi -p "…"` | `false` | ❌ no-ops |
| **`"json"`** | `pi --mode json` | `false` | ❌ no-ops |

> **Decisive fact for Loom ([FIXED major] — corrects the old `!ctx.hasUI` guard).** Pi documents `hasUI` as **"true in TUI *and* RPC modes"** (verified types.d.ts:213) — it means *dialog-capable*, not *widget-capable*. Custom widgets/overlays only **render** when `ctx.mode === "tui"` (the type's own comment, types.d.ts:211: *"Use 'tui' to guard terminal-only UI such as custom components"*). Therefore **every widget/overlay guard in Part B is `ctx.mode === "tui"`, NOT `ctx.hasUI`.** Guarding on `hasUI` would let an `--mode rpc` run pass the guard, call `ctx.ui.custom`, get a silent no-op, and proceed *without the human gate* — the exact failure mode §B.5 must prevent. Loom's hosting model spawns the `pi` binary in interactive mode with `stdio:"inherit"` (the legacy "Feynman pattern"), which *is* `mode === "tui"` — so the widget plan lives here; non-tui modes are handled by a hard block (§B.5).

## A.4 The extension API — how Pi is actually extended

An extension is a **default-exported factory** `(pi: ExtensionAPI) => void | Promise<void>`, loaded via **jiti** (raw `.ts` runs with no build step). Discovered from `~/.pi/agent/extensions/`, project `.pi/extensions/`, settings, or CLI `-e`/`--extension`. The four `pi-*` packages + `typebox` are provided as virtual modules.

**Registration methods** (`dist/core/extensions/types.d.ts`):
- **`registerTool(tool: ToolDefinition)`** — an LLM-callable tool. **First registration per name wins.** This is where the verbs land (§B.4).
- `registerCommand(name, {description?, getArgumentCompletions?, handler})` — a `/slash` command.
- `registerShortcut`, `registerFlag` + `getFlag`.
- **`registerMessageRenderer(customType, renderer)`** (verified types.d.ts:855) — custom TUI rendering for `appendEntry`/custom messages (a key widget seam).
- `registerProvider` / `unregisterProvider` — add/override models at runtime.
- `events` — a shared `EventBus` between extensions.

**Action methods** (state/session control): `sendUserMessage`, `sendMessage`, **`appendEntry(customType, data)`** (persist non-LLM state into the session JSONL), `setActiveTools`/`getActiveTools`/`getAllTools`, `setModel`, `setThinkingLevel`, `exec`. Command handlers additionally get `newSession`, `fork`, `navigateTree`, `switchSession`, `reload`, `waitForIdle`.
⚠ Action methods **cannot be called during the load/factory phase** — only `registerX` is safe there.

## A.5 The hook surface — `pi.on(event, handler)`

Handler `(event, ctx) => R | void`; many return a value to modify behavior, chained in registration order. **The four load-bearing hooks for Loom:**

- **`tool_call` → `{block?, reason?}`** — gate/mutate a tool call before it runs, keyed on `event.toolName`. `event.input` is mutable in place (⚠ **not re-validated**). **This is the ONLY real tier/launch lock** (see §B.5 — there is no tool-schema "disable invocation" property; this hook is it).
- **`tool_result` → `{content?, details?, isError?}`** — rewrite a result before the LLM sees it. *Contract-violation rewriting.*
- **`before_agent_start` → `{message?, systemPrompt?}`** — inject a message and/or replace the system prompt. *The Loom persona + per-turn context.*
- **`context{messages}` → `{messages?}`** — rewrite the whole message list before every LLM call (deep-cloned first). *Context engineering / RAG seam.*

Plus lifecycle: `session_start{reason}`, `session_before_{switch,fork,compact,tree}` (each `{cancel:true}`-able), `agent_start/end`, `turn_start/end`, `tool_execution_start/update/end` (observe progress → feed live widgets), `before_provider_request`/`after_provider_response` (inspect/rewrite raw API body), `model_select`, `input`, `user_bash`.

## A.6 Tools, plan mode, permissions, subagents, MCP, memory — what's a primitive vs a package

**THE key architectural fact:** Pi has **no native subagent, no native plan mode, no native MCP, no native permission popup, no native memory.** All are built on `ExtensionAPI` + the SDK. This matters because it tells us exactly which Pi features to *use* vs which to *build*:

- **Tools** — TypeBox-schema'd `ToolDefinition`s (verified shape, types.d.ts §ToolDefinition ~333-364). Built-ins: `read, bash, edit, write, grep, find, ls`; default offered set = `read, bash, edit, write`. The `bash` tool has a pluggable `BashOperations` interface (could redirect to remote/container).
  - **⚠ The full list of `ToolDefinition` properties (verified ~333-364):** `name`, `label`, `description`, **`promptSnippet?`** (one-line entry in the system prompt's "Available tools" section — *omitting it merely drops the tool from that listing; the tool stays callable by name*, see line 340), **`promptGuidelines?`**, `parameters` (TypeBox), **`renderShell?: "default"|"self"`**, `prepareArguments?`, **`executionMode?`**, `execute`, **`renderCall?`** (line 361), **`renderResult?`** (line 363). **There is NO `disableModelInvocation` and no equivalent "hide from model" / "human-only" property on a tool.** (`disableModelInvocation` exists *only* in Pi's **skills** system, skills.d.ts:15 — a different subsystem.) This corrects a load-bearing error in the prior draft (§B.5, [FIXED critical #1]).
- **Plan mode** = the `pi-plan-mode` package (~300 LOC, no subprocess): `registerCommand("plan")` flips the tool set with `setActiveTools(["read","bash"])`, an `on("tool_call")` returns `{block}` for mutating tools, `before_agent_start` injects a header, `appendEntry` persists across reload. **This is the exact recipe for Loom's tier gate** — but we build it into our own extension because `pi-plan-mode` hardcodes `["read","bash"]` and would hide every Loom verb.
- **Permissions / trust** — ⚠ Pi is explicit that trust is an *input-loading guard*, **not** a sandbox or isolation boundary. Real isolation = container/VM. Loom's gate is therefore a `tool_call` hook keyed on tool identity + validated args, **not** a reliance on trust.
- **Subagents** = the `pi-subagents` package (0.28.0). One `registerTool({name:"subagent"})` whose `execute()` **spawns child `pi` subprocesses**; personas are **Markdown + YAML frontmatter** in `.pi/agents/**` (`name, description, tools, thinking, model, output, defaultContext`). Bundled agents: planner/worker/oracle/scout/researcher/reviewer/delegate. Live status via a TUI widget + message renderers.
- **MCP** — none built-in. Add a **bridge extension** registering each MCP tool as a namespaced `mcp__<server>__<tool>` Pi tool (so it falls under the same `tool_call` gate), or adopt `pi-mcp-adapter`.
- **Memory** — not built-in; `@samfp/pi-memory` / `@luxusai/pi-hindsight` / session-search are optional packages. (Loom **declines** these — see §B.8.)

## A.7 The package system

A **Pi package** = an npm/git/local dir bundling four resource types: **extensions** (`.ts`/`.js` — the only type that registers tools/commands/providers/hooks), **skills** (`SKILL.md`), **prompts** (`.md` slash-command templates), **themes** (`.json`). Manifest = the `"pi"` key in `package.json` (the loader reads **only** `extensions/skills/prompts/themes` string arrays; the docs' `video`/`image` are website-gallery only). Auto-discovery works with no manifest.

Install specs: `npm:pkg@ver` (pinned → never auto-updated), `git:host/user/repo@ref`, local paths. Object form allows **per-package resource filtering** (`include` / `!exclude` / `+force-include` / `-force-exclude`) — adopt a package but strip pieces. **Resolution precedence:** project-settings > project-auto > user-settings > user-auto > package. The Feynman distribution model (which Loom copies) uses `withRuntimePeerSpecs` (co-install matching peers so extensions resolve the same runtime types) and `seedBundledWorkspacePackages` (offline symlinks).

## A.8 Model / provider wiring

`streamSimple(model, context, options)` is the single LLM entry point; dispatch is by `model.api` (wire protocol), not provider name, and `Api` is open-ended (`KnownApi | (string & {})`) so **custom APIs are first-class**. Auth via subscription OAuth (`/login`) or API keys (`--api-key` → `auth.json` → env → `models.json`). Custom models via `~/.pi/agent/models.json` or `pi.registerProvider(...)` at runtime.

> **Loom has two model channels that must stay separate:** (1) the **agent's reasoning LLM** (user-chosen via `/model`, lives in pi-ai) and (2) the **engine's optimization providers** (the moat — the actual training/embedding calls inside Python). The agent never sees engine traffic; the engine never sees the agent's key. This separation is automatic because the engine runs as a child Python process behind the verb seam (§B.3).

---

# PART B — Hosting the new Loom on Pi (the design) — WIDGET-FIRST

## B.0 Why Pi, restated as the design constraint

Claude Code and Codex render every tool result into a **fixed, generic terminal transcript**: a colored call header and a text/diff body, with no way for a domain to own the rendering. For foundation-model training that is actively bad — a training run is a *live, multi-dimensional, gated, expensive* process whose state (loss curve, step-0 canary, GPU/$ burn, leaderboard, lineage, contract violations, launch approval) wants **purpose-built visual surfaces**, not JSON blobs scrolling past.

Pi is the only harness in this class that exposes its **rendering substrate** to extensions. The `Component` contract is total and trivial — `render(width: number) => string[]` (verified nested `pi-tui/dist/tui.d.ts:9-29`) — so *anything that emits ANSI lines* is a widget: tables, colored bars, sparklines, ANSI/braille charts, and **inline raster images** via the `Image` component (verified nested `pi-tui/dist/components/image.d.ts`, base64 PNG through the Kitty/iTerm2 graphics protocols — *when the terminal supports them*, see §B.2.3). Pi ships a Snake game and a Doom overlay as extensions — proof of arbitrary interactive rendering. **That capability is the product reason to be on Pi, and the widget plan in §B.2 is the centerpiece of this whole document.**

## B.1 The five rendering seams (the API we become power users of)

All five are verified in the installed source. These are the only tools we need to build every surface in §B.2.

| # | Seam | Signature (verified) | Where it draws |
|---|---|---|---|
| 1 | **Per-tool renderers** | `ToolDefinition.renderCall?(args, theme, ctx) => Component` (types.d.ts:361) and **`renderResult?(result, options, theme, ctx) => Component`** (types.d.ts:363) | inline in the transcript, owning the verb's call/result card. `renderShell:"self"` (types.d.ts:347) opts out of Pi's default colored shell so Loom draws its own framing. |
| 2 | **Persistent panel** | **`ctx.ui.setWidget(key, content, {placement})`** — `content` is `string[]` *or* `(tui, theme) => Component & {dispose?()}` (verified types.d.ts:96-99) | a pinned strip above/below the editor that **stays while you chat**, re-callable with the same `key` to live-update. **The factory return type carries an optional `dispose?()`** — use it to tear down polling intervals (§B.2.3, [FIXED minor]). |
| 3 | **Focus-grabbing overlay** | **`ctx.ui.custom<T>((tui, theme, keybindings, done) => Component & {dispose?()}, {overlay?, overlayOptions?, onHandle?}) => Promise<T>`** (verified types.d.ts:116-126) | a full, keyboard-focused component/overlay returning a typed result; `onHandle` gives an `OverlayHandle` to control visibility. **Renders only when `ctx.mode === "tui"`.** |
| 4 | **Chrome** | `setHeader(factory)` (types.d.ts:110), `setFooter(factory)` (types.d.ts:106), `setStatus(key, text)` (types.d.ts:79), `setWorkingIndicator({frames, intervalMs})` (types.d.ts:92), `setTitle` (types.d.ts:114) — all factory returns carry `dispose?()` | branded banner, live status pill, custom footer ($-burn / GPU step), custom spinner |
| 5 | **Custom session messages** | **`pi.registerMessageRenderer(customType, renderer)`** (types.d.ts:855) paired with `pi.appendEntry(customType, data)` / `pi.sendMessage` | persistent custom entries that survive reload and **re-render on `/resume`** (the launch-approved receipt, the frozen contract-violation card) |

**The `Component` contract** (verified nested `pi-tui/dist/tui.d.ts:9-29`): `render(width): string[]` (line 15), optional `handleInput(data)` (line 19), optional `wantsKeyRelease` (line 24), `invalidate()` (line 29). **Live updates** happen via `TUI.requestRender(force?)` (verified `tui.d.ts:204`) — the differential renderer repaints only what changed, so a widget that recomputes its `render()` output and calls `tui.requestRender()` is a smooth live dashboard.

**Reusable building blocks** (no need to hand-roll): from `pi-tui` — `Text`, `Box` (padding+bg), `Container`, `Spacer`, `Markdown` (themed, syntax-highlighted), `TruncatedText`, `Input`, `Loader`/`CancellableLoader`, `SelectList`, `SettingsList`, and **`Image`**. Also exported from `pi-tui`'s `terminal-image`: **`detectCapabilities()` / `getCapabilities()`** (→ `TerminalCapabilities {images: "kitty"|"iterm2"|null, …}`), **`allocateImageId()`**, `renderImage`, `imageFallback` (verified nested `pi-tui/dist/terminal-image.d.ts:1-30` and `dist/index.d.ts:20`). From `pi-coding-agent` (verified `dist/index.d.ts:25-26`) — **`renderDiff(diffText, opts?) => string`** (red/green intra-line diff highlighting; ⚠ returns a **string**, not a Component — see §B.2.2), `DynamicBorder`, `BorderedLoader`, `CustomEditor`, `getMarkdownTheme()`, `getSettingsListTheme()`, `truncateToVisualLines`. Theme access via `theme.fg(color, text)` / `theme.bg(...)` with named tokens including `success/error/warning`, `toolDiffAdded/Removed/Context`, borders, syntax.

> **What the legacy build did with this surface:** almost nothing. It used only `setHeader` (the LOOM banner — but a *static* one, `invalidate(){}` a no-op), `ui.select` (the gate), `ui.notify`, and `setStatus` (plan mode). **Every verb result was flattened to plain text by `humanize()` (`Loom-legacy/cli/src/manifest.ts:176`)** and `renderResult`/`renderCall` were **never set** (the `ToolDefinition`s at `manifest.ts:246-287` define neither). The structured `details` object was carried for machine gate-asserts but never rendered. That gap *is the rebuild's entire value-add.*

## B.2 The widget UX — DESIGN.md surfaces → concrete Pi widgets (THE HEADLINE)

Each row maps a surface specced in `/Users/anub/Work/transaction-foundation-model/Loom/DESIGN.md` to a Pi rendering primitive from §B.1, with the build recipe and the explicit "what Claude Code/Codex can't do" justification. These are concrete enough to build from.

### B.2.1 The launch / cost gate → an interactive approval-card overlay

**Surface (DESIGN §3.3, §4.4):** before any `expensive`/`irreversible` launch, present a card with the **binding cost envelope**, the **derived cost** (GPU-hours × rate), an **anomaly banner** if inputs look off, and — above a $-threshold — a **typed-confirm** second factor (the user types the dollar amount, not just "yes").

**Pi primitive:** `ctx.ui.custom<{approved:boolean, token?:string}>(factory, {overlay:true})` (seam 3). The factory builds a `Box` containing: a `Text` header, a colored envelope/derived-cost block (`theme.fg("warning", …)`), the anomaly banner row (`theme.fg("error", …)` when present), and an `Input` child for the typed amount when `cost_plan.requires_typed_confirm`. `handleInput` validates the typed value against `cost_plan.cap_usd`; `done({approved:true, token})` resolves the promise. On approval, the extension makes the **second `dispatch` call carrying `confirm_token`** (the round-trip in §B.5).

> **⚠ The lock is the `tool_call` hook, not the overlay ([FIXED critical #1]).** The overlay is the *human experience* of the gate, but it is **not** what stops a model from auto-firing the verb. There is no tool-schema property that hides a tool from the model (see §A.6). The actual enforcement is: a `tool_call` `{block}` hook keyed on the gated verb's `toolName` that, in `mode === "tui"`, opens this overlay and blocks unless it returns `approved`, and in any non-tui mode **hard-blocks** with a `REFUSED_*` reason. The engine's `_loom.disable_model_invocation` flag is a *Loom-internal hint* our extension reads to decide which verbs get this hook (and, as defense-in-depth, to omit `promptSnippet`); it is invisible to Pi. See §B.5 for the full mechanism.

**Run-mode guard ([FIXED major]):** the overlay path is guarded on **`ctx.mode === "tui"`** (not `ctx.hasUI`). In `rpc`/`print`/`json` the gated verb's `tool_call` hook returns `{block:true, reason:"<verb> is irreversible and needs interactive confirmation."}` — there is no surface to mint a confirm, so the call is refused, not silently proceeded.

**The data is already there:** the new envelope carries `cost_plan` and `confirm_token` fields (verified `loom/types.py:179,182`; `VerbResult.to_dict` emits both). A `Status.PLAN` result is the trigger to render this overlay.

**Claude Code / Codex cannot:** render a focus-grabbing modal with a typed-amount field gating an irreversible spend. They surface a yes/no permission prompt at best — no envelope, no derived cost, no anomaly banner, no second factor. This is the single clearest justification for Pi.

### B.2.2 Contract violations → a named-diff card (not a stack trace)

**Surface (DESIGN §7.2):** the MONTH_12 / CARD_0 id-range collision, a signature mismatch — rendered as a **named diff** with the contract name, the offending vs expected id ranges, and the one-line fix.

**Pi primitive:** `ToolDefinition.renderResult` (seam 1) with `renderShell:"self"`, returning a `Box`. ⚠ **`renderDiff(diffText, opts?)` returns a multi-line ANSI `string`, NOT a Component** (verified `pi-coding-agent/dist/.../diff.d.ts:11` — `renderDiff(diffText: string, _options?: RenderDiffOptions): string`). So you **cannot** add it as a Box child directly ([FIXED minor]). The recipe: build the id-range delta text, call `const diff = renderDiff(deltaText)`, then feed `diff` into a `Text` component (or `diff.split("\n")` as the Box's lines), and add `Text` rows for `contract` / `severity` / `fix` alongside. Drive it from the envelope's **`diagnostics[]`** array — each `Diagnostic` carries `{contract, severity, message, fix, data}` (verified `loom/types.py:176` field + `Diagnostic.to_dict`). For a violation that should *persist* across reload (a frozen "this run failed contract X" record), also `appendEntry("loom-violation", diagnostic)` + a `registerMessageRenderer` (seam 5).

⚠ **Renderer robustness:** blocked/aborted/missing-tool results carry `details: {}` — `renderResult` must tolerate empty/missing `diagnostics` and fall back to the text `content`.

**Claude Code / Codex cannot:** turn a domain contract violation into a structured, color-coded named-diff card. They would print the Python traceback or the raw JSON. The whole point of `diagnostics[]` flowing into `renderDiff` is to make a leakage/collision *legible at a glance*.

### B.2.3 `pretrain` → a LIVE training dashboard (the flagship surface)

**Surface (DESIGN §7.5):** a live loss curve, the **step-0 loss canary** (`8.74 ≈ ln 6283 ✓` — the sanity check that initial loss matches ln(vocab)), GPU utilization, **$-burn against the envelope with a hard-kill ceiling at 100%**, step/throughput.

**Pi primitives (combination), with the two image seams cleanly separated ([FIXED critical #2]):**

- **`ctx.ui.setWidget("loom-train", factory, {placement:"belowEditor"})`** (seam 2) — a pinned dashboard strip that stays while the user keeps chatting. The factory's `render(width)` draws, **as its PRIMARY render (always works, any terminal):** an **ANSI block / braille sparkline** loss curve, the step-0 canary line colored `success`/`error` by the `≈ ln(vocab)` check, a `$X / cap $Y` progress bar colored `warning` at 90% and `error` at 100%, and step/throughput counters.

- **The LIVE inline-PNG loss curve — the seam the prior draft left unspecified.** A live, updating PNG is **NOT** done with `@walterra/pi-charts`. That package registers a *one-shot, LLM-callable tool* (`vega_chart`) that shells Python (altair/vl-convert via `uv`), writes a tmp PNG, and returns a single `{type:"image", data:base64}` content block painted **once** into the transcript — it cannot live-update and is not a widget (verified against its `extensions/vega-chart/index.ts`; README tested on pi 0.62.0, not 0.79.0). The correct live seam is to **drive the `Image` component directly inside the `setWidget` factory:**
  1. At job start, `allocateImageId()` once → a stable `imageId`.
  2. Each poll tick, render the new curve to a PNG (cheaply — e.g. the engine emits a base64 PNG in `loom top --json`, or a small local plot), construct `new Image(base64, "image/png", {fallbackColor}, {imageId}, dims)` (verified constructor `image.d.ts`: `(base64Data, mimeType, theme, options?{imageId}, dimensions?)`), and call `tui.requestRender()`. **Reusing the same `imageId`** makes the Kitty/iTerm2 protocol *replace* the prior image in place rather than stacking frames (verified `ImageOptions.imageId` / `ImageRenderOptions.imageId` doc: *"reuses/replaces existing image with this ID"*).
  3. **Mandatory fallback:** the PNG path is gated on `detectCapabilities().images !== null` (verified API; `ImageProtocol = "kitty"|"iterm2"|null`, and `images` is `null` in tmux-without-passthrough, the Linux console, most CI, and many SSH terminals). When `null`, the dashboard renders **only** the ANSI/braille sparkline — never a blank strip. The PNG is an *upgrade*, the sparkline is the floor.

- **`setWorkingIndicator({frames})`** to narrate the canary ("checking step-0 loss…") and **`setFooter`** for the always-visible `$-burn / GPU-step` line.

**Feeding it (the streaming problem — see §B.6):** a one-shot `--json` blob cannot drive a live curve. The dashboard widget is fed by **polling `python -m loom top --json` on a `setInterval`** while a job is `running`, recomputing the widget's `render()` and calling `tui.requestRender()` each tick. **Lifecycle ([FIXED minor]):** the `setWidget` factory **returns a `dispose()`** that `clearInterval()`s the poll loop and `deleteKittyImage(imageId)`s the live PNG; the dashboard is torn down (`setWidget("loom-train", undefined)`) when no job is running and on `session_before_switch` / `agent_end`. (The factory return type's optional `dispose?()` — types.d.ts:98 — exists precisely for this; leaking the interval would poll `loom top --json` forever.) For a *single* long verb call, `execute(toolCallId, params, signal, onUpdate, ctx)`'s `onUpdate` callback streams `tool_execution_update` events for that call's duration.

**Claude Code / Codex cannot:** render a pinned, continuously-updating dashboard *or* an inline live loss-curve image while the conversation continues above it. Their UI is the transcript; a long-running job is just a spinner then a wall of text.

### B.2.4 The experiment leaderboard + lineage → live panes (`loom top`)

**Surface (DESIGN §7.5):** `loom top` — a leaderboard of experiments (metric vs baseline), the lineage DAG (which data object/tokenizer/checkpoint produced which result), running GPU jobs, and the cost/envelope panes.

**Pi primitives:**
- The **pinned strip** version: `setWidget("loom-top", …, {placement:"belowEditor"})` showing the top-N leaderboard rows + a one-line cost-envelope summary, always visible (same `dispose()` discipline as §B.2.3).
- The **full view**: `ctx.ui.custom(factory, {overlay:true})` — a navigable, focus-grabbing leaderboard/lineage pane (arrow-key selection via `handleInput`, `SelectList` for rows), dismissable. Guarded on `ctx.mode === "tui"`.
- **Charts (the one-shot seam, where pi-charts IS appropriate):** the leaderboard bar chart and AUPRC-vs-baseline are **static snapshots**, so the `@walterra/pi-charts` `vega_chart` tool-result image is exactly the right tool here — paint it once when the user asks. **Lineage DAG via `pi-mermaid`** rendered to the terminal.

**Driven by** `python -m loom report --json` (one-shot, for the static panes) + interval polling for the running-jobs section.

**Claude Code / Codex cannot:** provide a `top`-like persistent, navigable dashboard pane *inside the agent*, nor render a lineage DAG or leaderboard chart inline. You'd leave the agent for a browser/CLI.

### B.2.5 `tokenize` PLAN → a compile-result card

**Surface (DESIGN — the "compile before you spend" philosophy):** `tokenize` returns a *plan* — the proposed vocab size, chunk layout, and the contracts it will enforce — before any expensive work.

**Pi primitive:** `ToolDefinition.renderResult` (seam 1) returning a `Box` card: vocab size, chunk count/shape, and a checklist of contracts (each `theme.fg("success", "✓ …")`). Built straight from the verb's `data{}` and `outputs[]` (verified `VerbResult` carries both, `loom/types.py:175,177`). Because `tokenize` is `tier=workspace-write` today, no gate — but the card makes the PLAN legible so the human can eyeball vocab/chunking before `baseline`/`pretrain`.

**Claude Code / Codex cannot:** render a domain "compile result" as a structured card; they'd dump the schema JSON.

### B.2.6 Branding chrome

`setHeader` LOOM banner (carry the legacy `src/branding/header.ts` component — it *already proves* `setHeader` renders an arbitrary width-aware ANSI component), `--theme home/themes/loom.json` (the ZKAI magenta palette), `process.title = "loom"`, redirected config dir `.loom`.

---

## B.3 Architecture: Pi as runtime/harness driving the Python engine

**Shape (carried from the legacy build, rebuilt clean):** a Node `cli/` workspace is a **thin launcher that spawns the unmodified `pi` binary as a child in interactive TUI mode** (`spawn(process.execPath, [wrapper, piMain, ...args], {stdio:"inherit", env})` — legacy `src/pi/launch.ts:43-47`). `stdio:"inherit"` hands the terminal to Pi's TUI (i.e. `mode === "tui"`, the widget-rendering mode). The launcher owns branding, asset-sync, package-ensure, and the one extension that turns verbs into tools. **No harness fork at any layer.**

```
loom (bin/loom.js, Node-version gate)
  └─ src/index.ts main(): parseArgs(strict:false) → resolveInitialPrompt → bootstrap chain → launchPiChat
       └─ spawn pi child (interactive TUI, stdio:inherit  →  ctx.mode === "tui")
            ├─ --extension extensions/loom-tools.ts   ← registers verbs as Pi tools + widgets + gate
            ├─ --system-prompt home/SYSTEM.md          ← DS persona, hides internals
            ├─ --theme home/themes/loom.json           ← brand
            ├─ --prompt-template prompts/              ← /loom-<verb> templates
            └─ env: LOOM_PYTHON, PI_CODING_AGENT_DIR(→.loom home)
                 └─ loom-tools.ts execute(): spawn `$LOOM_PYTHON -m loom <verb> ... --json`
                      └─ Python engine (REGISTRY → verb.fn → VerbResult.to_json())
```

## B.4 The seam decision: shell out to `python -m loom <verb> --json` (recommended)

**Decision: keep the legacy shell-out.** Each verb tool's `execute()` spawns `${LOOM_PYTHON} -m loom <verb> [...flags] --json`, parses the **trailing JSON line** (bottom-up scan, tolerating prose above it), and returns `{content:[{type:"text", text: summary}], details: parsedVerbResult}`. (Legacy reference: `runLoom` `manifest.ts:62-83`, `parseLoomJson` `:90`, `toCliArgs` `:159`, the result contract `:280-283`.)

**Justification, against the engine's real invariants:**
- The engine already guarantees a **dual-driver, byte-identical envelope**: `dispatch(name, input_json).to_json()` is byte-identical to `python -m loom <verb> --json` (verified: `tool_schema` and `dispatch` read the same `REGISTRY` and call the same `verb.fn`; `loom/tools.py:24-86`, `loom/cli.py`). Shelling the CLI exercises the *identical* code path the human uses — the dual-driver invariant stays honest.
- **Process isolation:** a verb crash/segfault cannot take down the Node TUI.
- **Zero IPC to own/version.** The `--json` envelope *is* the contract.
- **Cost:** per-call spawn (~tens of ms) is negligible against CPU verbs (<1s) and GPU verbs (minutes).
- The legacy build resolved the interpreter via `LOOM_PYTHON` (default `…/Loom/.venv/bin/python`) and used `python -m loom` (not a `loom` console script on PATH) — **keep this**; simpler, no PATH collision.

**Alternatives, rejected for v0.1:** a long-lived `python -m loom serve` RPC (a stateful protocol to own, version, and reconnect — defer until polling proves insufficient); in-process via `createAgentSession({customTools})` (loses isolation, couples the engine to Node's lifecycle).

## B.5 Tier / capability → Pi gating mechanism (the future cost/launch gate)

The new envelope already carries the *shape* of gating even though v0.1 leaves it inert (`cost_plan` is an all-null placeholder, `confirm_token` always null, `make/validate_confirm_token` are `NotImplementedError` stubs). Design against the shape now, wire it when `pretrain`/`embed` land.

> **The core correction ([FIXED critical #1]).** The engine's own docstring (`tools.py:27-30`) says `disable-model-invocation` makes it so "an agent structurally cannot fire money/destruction." **That guarantee does NOT come from the tool schema.** Pi has no such property (§A.6). `_loom.disable_model_invocation` is **Loom-internal metadata invisible to Pi** — it only does something if *our extension reads it*. The single real, un-delegable enforcement is the **`tool_call` `{block}` hook keyed on tool identity** (this is exactly what the legacy `installApprovalGate` does: it computes `tierOf(event.toolName)` and returns `{block:true}` — `manifest.ts:299-316`). Everything else (omitting `promptSnippet`, the overlay) is UX / defense-in-depth, not the lock.

| Loom tier / capability (`loom/types.py`) | Pi mechanism |
|---|---|
| `read-only` | tool offered; in the plan-mode allowlist |
| `workspace-write` (the 3 current verbs) | offered; `tool_call` gate passes; agent auto-proceeds |
| `expensive` | `tool_call` gate → `ctx.ui.notify(…, "warning")` + cost widget; above threshold the engine returns `Status.PLAN` |
| `irreversible` + `capability_mode=launch-and-track` | extension reads `_loom.disable_model_invocation` (derived `tier IRREVERSIBLE or capability_mode LAUNCH_AND_TRACK`, `tools.py:31-44`) and on it: **(1) the LOCK** — registers a `tool_call` `{block}` hook for that `toolName` (blocks unless `mode==="tui"` AND the approval overlay returns `approved`); **(2) defense-in-depth** — omits `promptSnippet` so the verb is dropped from the system-prompt "Available tools" listing (cosmetic only — the verb *stays callable by name*, types.d.ts:340; the block hook is what actually stops it). |
| `confirm_token` round-trip | `Status.PLAN` result → approval overlay (`mode==="tui"` only) → on approve, second `dispatch` call carrying the token. In any non-tui mode the block hook hard-refuses; **the token is never minted by a non-interactive/bot path.** |
| **Plan mode** | build Loom's own `/plan` (legacy `installPlanMode` `manifest.ts:356`): `setActiveTools` allowlist **derived from verb tiers** (read-only verbs stay on), `tool_call` veto for writes/expensive/irreversible, state persisted via `appendEntry`, restored on `session_start`. **Do NOT adopt `pi-plan-mode`** — it hardcodes `["read","bash"]` and hides all Loom verbs. |

> **Run-mode guard correction ([FIXED major]).** The legacy `installApprovalGate` guards the interactive branch on `!ctx.hasUI` (`manifest.ts:305`). **That is wrong for `--mode rpc`,** where `hasUI===true` but `ctx.ui.select`/`custom` are no-ops — the gate would silently evaporate. The rebuilt gate must branch on **`ctx.mode === "tui"`**: only then open the overlay; in every other mode return `{block:true, reason}`. (Legacy used `ui.select` for a bare yes/no; the rebuild uses `ui.custom` for the typed-confirm card — same hook, richer surface.)

## B.6 Feeding live widgets (the streaming requirement)

A one-shot `--json` blob **cannot** drive a live loss dashboard — this is the explicit gap §B.2.3 must close. Two complementary feeds:

1. **Single long call:** `execute(toolCallId, params, signal, onUpdate, ctx)` — stream intermediate progress out of a running verb via **`onUpdate(partialResult)`**, emitting `tool_execution_update` events the widget consumes.
2. **Background running job (the real `pretrain`/`top` case):** the agent **polls `python -m loom top --json` / `loom ls --status running --json` on a `setInterval`** (DESIGN already specifies "polls, doesn't block"), recomputes the widget `render()`, and calls `tui.requestRender()` each tick. Launches are queued as `PendingLaunch` and never block the loop. **Each polling widget owns a `dispose()`** that clears its interval and deletes any reused Kitty image (§B.2.3 lifecycle).

Add a streaming RPC only if interval polling proves insufficient. The structured fields the widgets consume — `diagnostics[]`, `cost_plan`, `outputs[]`, `data{}`, `experiment` — are **all already in the envelope** (`loom/types.py:175-182`); only the *rendering* layer is new work.

## B.7 The package/extension layout to BUILD

```
cli/
  package.json                 # bin "loom"→bin/loom.js; deps @earendil-works/pi-coding-agent + pi-ai @0.79.0,
                               #   @sinclair/typebox 0.34.49; files: bin/ dist/ extensions/ prompts/ home/
  bin/loom.js                  # Node-version gate (≥22.19.0) → import dist/index.js   [KEEP from legacy]
  src/
    index.ts                   # main(): parseArgs(strict:false), resolveInitialPrompt, bootstrap, launchPiChat
    pi/{runtime,launch,pi-cli-wrapper}.ts   # resolve pi dist/main.js, spawn child, buildPiEnv/Args   [KEEP]
    pi/ensure-packages.ts      # idempotent `pi install` of the package list   [KEEP, trim]
    manifest.ts                # ★ REBUILD: load schemas, registerLoomTools, runLoom, toCliArgs, parse, gate, plan
    widgets/                   # ★ NEW: launch-gate.ts, contract-diff.ts, train-dashboard.ts, loom-top.ts, plan-card.ts
    branding/{header,logo}.ts  # setHeader banner   [KEEP — already proves arbitrary component rendering]
    bootstrap/sync.ts          # hash-tracked 3-way asset merge   [KEEP]
  extensions/loom-tools.ts     # thin default-export factory: registerLoomTools + gate + plan + header + widgets
  home/
    SYSTEM.md                  # DS persona; NEVER expose internals (Pi, Python, `python -m loom`)   [KEEP+update]
    settings.json              # theme:"loom", quietStartup, packages[]   [KEEP]
    themes/loom.json           # ZKAI magenta palette   [KEEP]
    agents/{data-scout,pipeline-builder,result-reviewer,oracle}.md   # subagent personas   [KEEP+extend]
  prompts/loom-*.md            # /loom-<verb> templates (intake→plan/tier→run→verify→deliver)   [KEEP+update]
```

**`home/settings.json` `packages[]` + `ensureLoomPackages` (the right Pi features for each Loom need):**

| Package | Loom need | Verdict |
|---|---|---|
| `pi-subagents` | DS personas (data-scout / pipeline-builder / result-reviewer / oracle) | **ADOPT — core** |
| `pi-web-access` | research; powers the subagents' `researcher` | **ADOPT — core** |
| `pi-docparser` | ingest/read source docs into reports | **ADOPT — core (docs)** |
| **`@walterra/pi-charts`** | **one-shot snapshot charts** (leaderboard bar, AUPRC-vs-baseline) | **ADOPT — for static snapshots ONLY.** It is a one-shot LLM tool, **not** a live widget; the live loss curve uses the `Image` component directly (§B.2.3). ⚠ pulls a `uv`+altair+vl-convert toolchain; README tested on pi 0.62.0, re-verify on 0.79.0. |
| **`pi-mermaid`** | lineage DAGs | **ADOPT — first-class** |
| `pi-markdown-preview` | rendered report output | ADOPT (reports) |
| `pi-mcp-adapter` | the embed datasets-catalog REST+MCP | CONDITIONAL — prefer a **Loom-owned MCP bridge** that inherits the `tool_call` tier gate + the `loom ingest` data boundary, rather than punching a hole in the abstraction |
| `@kaiserlich-dev/pi-session-search` | session FTS | OPTIONAL preset (native `better-sqlite3` → Node ≤22 guard) |
| `pi-generative-ui` | native-window HTML/SVG charts | OPTIONAL, **darwin-gated** (native dep, fails standalone-bundle compile) — never core; the terminal `Image`/sparkline path is the portable default |

## B.8 Packages we DECLINE — and why

- **`@samfp/pi-memory` / `@luxusai/pi-hindsight`** — **declined.** Two reasons, decided deliberately (not an oversight): (1) **prompt-injection risk** — a memory that auto-injects learned text into future system prompts is an attack surface in a tool that mints irreversible spend approvals; (2) **moat overlap** — durable learning across runs is part of Loom's own product (the distillation/telemetry pipeline), not something to delegate to a generic package. Revisit only if a concrete, bounded durable-memory need is proven.
- **`pi-plan-mode`** — **declined as a dependency** (we build the recipe ourselves) because it hardcodes the read-only allowlist to `["read","bash"]`, which would hide every Loom verb. We reuse its *pattern*, not its code.

## B.9 KEEP vs REBUILD from the legacy `cli/`

The legacy build is the **working pattern** for the spawn-the-binary architecture, branding, packages, the gate, and plan mode — but it drove a *different engine* (an 8-key `LoomResult` from a `loom verbs --json` manifest that the new engine doesn't emit) and **never built the widgets**.

**KEEP (proven, engine-agnostic):**
- The launch/spawn/wrapper chain: `bin/loom.js`, `src/index.ts`, `src/pi/{launch,pi-cli-wrapper,runtime}.ts`.
- `ensureLoomPackages` + `home/settings.json packages[]` + the 3-way asset sync (`bootstrap/sync.ts`).
- Branding: `setHeader` banner (`branding/header.ts`), `loom.json` theme, `process.title`, `.loom` config dir.
- Plan-mode + approval-gate *mechanism* (`installPlanMode`, `installApprovalGate` — `manifest.ts:299-442`) — **but FIX the run-mode guard from `!ctx.hasUI` to `ctx.mode === "tui"`** (§B.5).
- `SYSTEM.md` "never expose internals" invariant + the `/loom-<verb>` prompt templates.
- The `LOOM_PYTHON` + `python -m loom` shell-out and the bottom-up trailing-JSON parse.
- The `Type`-from-`@earendil-works/pi-ai` detail (`manifest.ts:14`) — schemas must use Pi's re-exported TypeBox `Type`, **not** `@sinclair/typebox`, so the `TSchema` nominal type matches `registerTool<TParams extends TSchema>`.

**REBUILD (engine changed + widgets missing):**
1. **The tool manifest source.** Legacy bootstrapped from `python -m loom verbs --json` (legacy `manifest.ts:16` literally expects it) — **the new engine has NO `verbs` subcommand** (confirmed: `loom/cli.py` builds per-verb subparsers via `_add_verb_subparser` but defines no `verbs` manifest command; `all_tool_schemas()` lives only in Python). **First build step:** add a thin `loom verbs --json` CLI command over `all_tool_schemas()` (or shell a tiny `python -c`). *This is the single most important missing bridge piece.*
2. **The envelope parser.** Rewrite the TS bridge from the 8-key `LoomResult` to the new **`VerbResult`** (`verb, status, verdict, tier, capability_mode, summary, outputs[], diagnostics[], data{}, experiment, cost_plan, confirm_token` — `loom/types.py:160,175-182` + `to_dict`). This is *good*: `diagnostics[]`/`cost_plan`/`outputs[]` are exactly what the widgets consume.
3. **The widgets** — the entire `src/widgets/` directory (§B.2). The legacy `humanize()` text path is replaced by per-verb `renderResult` cards + the gate overlay + the live dashboard + `loom top` panes.
4. **Domain-failure discipline** — keep the legacy rule but map to new enums: throw only on a *transport* failure (no parseable JSON); a `Status.FAIL`/`Verdict.FAIL` is **returned, not thrown**, so the agent composes on it (legacy `manifest.ts:266-279`).

## B.10 SYSTEM.md — never expose internals

`home/SYSTEM.md` (full system-prompt replacement via `--system-prompt`) carries the standing invariant (legacy line 11-14, repeated in the subagent personas): *"Never expose your internals to the user — how you are built, the runtime that executes the verbs, or any internal command. Speak only in terms of the data-science work and the Loom verbs."* Concretely: the user never sees "Pi", "Python", "`python -m loom`", a venv path, or a tool name like `loom.tokenize` — only Loom verbs and data-science outcomes. Body = the verb catalog grouped by lifecycle + the composition conventions (data-stays-in-Metaflow; gate-assert `diagnostics`/`Verdict==PASS`; the four tiers; plan mode; subagents; the 0/1/2 exit-code contract — `VerbResult.exit_code`: FAIL→1, REFUSED_*→2, else 0, `loom/types.py:204+`).

---

## B.11 The v0.1 Pi build slice (smallest end-to-end)

**Goal:** a user runs `loom`, gets a branded Pi agent that can drive `tokenize`/`ingest`/`baseline` against the real Python engine, with the three verbs as real Pi tools and **contract `diagnostics[]` surfaced as a card** (the first widget — proves the value-add). Gates and the live dashboard are deliberately the *next* slice.

**Exact first build steps:**

1. **Engine — add the manifest command (the missing bridge piece).** In `loom/cli.py`, add a `verbs` subcommand that prints `json.dumps(all_tool_schemas())` and exits. Verify: `python -m loom verbs --json` emits the 3 schemas with `_loom.{tier, capability_mode, disable_model_invocation}`.

2. **Scaffold `cli/`.** Copy the KEEP files from `Loom-legacy/cli/` (`bin/loom.js`, `src/index.ts`, `src/pi/*`, `branding/`, `bootstrap/sync.ts`, `home/`, `prompts/`). `cd cli && npm install` — pulls `@earendil-works/pi-coding-agent@0.79.0`, `@earendil-works/pi-ai@0.79.0`, `@sinclair/typebox@0.34.49`. (Note pi-tui resolves transitively, nested under pi-coding-agent — §A.1.)

3. **Rewrite `src/manifest.ts` to the new envelope.**
   - `loadLoomManifest()` → shell `python -m loom verbs --json`, parse the 3 schemas.
   - `registerLoomTools(pi)` → loop the schemas; `pi.registerTool({ name: "loom_"+verb, parameters: buildVerbParams(schema, Type) /* Type from @earendil-works/pi-ai */, promptSnippet: gated ? undefined : summary, execute })`. (Remember: omitting `promptSnippet` is cosmetic — the real gate is the `tool_call` hook in step 5b, not here.)
   - `execute()` → `runLoom(["<verb>", ...toCliArgs(args), "--json"])`, parse trailing JSON to a `VerbResult`, return `{content:[{type:"text", text: summary}], details: verbResult}`. Throw only if no JSON parses.

4. **Build the first widget: the contract-diff `renderResult`.** In `src/widgets/contract-diff.ts`, export `renderContractResult(result, options, theme, ctx): Component` returning a `Box` that, when `details.diagnostics` is non-empty, **builds the diff string with `renderDiff(deltaText)` then wraps it in a `Text`** (renderDiff returns a string, not a Component — §B.2.2) plus `Text` rows (`contract`/`severity`/`fix`); else falls back to the summary text. Wire it as `renderResult` on each verb's `ToolDefinition` (with `renderShell:"self"`). **Tolerate empty `details`.**

5. **Branding + persona (+ the gate stub, even though no verb is gated yet).**
   - (a) Confirm `setHeader` LOOM banner, `--theme home/themes/loom.json`, and `home/SYSTEM.md` (the 3-verb catalog + "never expose internals") load. `process.title="loom"`, config dir `.loom`.
   - (b) Port `installApprovalGate` **with the corrected guard** — a `tool_call` hook keyed on `event.toolName` that, for any verb whose `_loom.disable_model_invocation` is true, blocks unless `ctx.mode === "tui"` and an approval is granted. No Phase-0 verb is gated, so this is inert now, but it lands the *correct* lock mechanism so `pretrain`/`embed` inherit it for free.

6. **Run it.** `node bin/loom.js` (or `loom` once linked) → interactive Pi TUI (`mode === "tui"`) with the LOOM banner. Prompt: *"tokenize this dataset and run a baseline."* Verify: the agent calls `loom_tokenize` then `loom_baseline`; each result renders as a card (not raw JSON); a forced contract violation renders the **named-diff card** from `diagnostics[]`. `loom ingest --dataset X` as a `/loom-ingest` template works end-to-end.

**Definition of done for the slice:** three verbs are real Pi tools driven by the live Python engine through the byte-identical `--json` envelope; the contract-diff widget proves the rendering seam works; branding/persona hide all internals; the `tool_call` gate is wired (inert) with the *correct* `mode === "tui"` lock. The launch-gate overlay, the live `pretrain` dashboard (with the `Image`-component PNG + sparkline fallback), and `loom top` are the *next* slices (they need `cost_plan`/`confirm_token` wired live in the engine and the polling feed of §B.6) — but the architecture, the seam, the lock mechanism, and the first widget are all in place and buildable from this document.

---

## Reference paths (cited above)

- **New engine (seam of record):** `/Users/anub/Work/transaction-foundation-model/Loom/{loom/registry.py, loom/types.py, loom/tools.py, loom/cli.py, DESIGN.md, MANUAL.md}`. Key facts verified live: `VerbResult` fields + `to_dict` keys (`types.py:160,175-182`), `disable_model_invocation` is **Loom-internal `_loom` metadata** derived from tier (`tools.py:31-44`, NOT a Pi tool property), **no `verbs` CLI subcommand** (gap #1, `cli.py`).
- **Legacy build (working pattern):** `/Users/anub/Work/transaction-foundation-model/Loom-legacy/cli/{bin/loom.js, src/index.ts, src/pi/{launch,pi-cli-wrapper,runtime,ensure-packages}.ts, src/manifest.ts (runLoom:62, parseLoomJson:90, toCliArgs:159, humanize:176, registerLoomTools:246, installApprovalGate:299, installPlanMode:356), src/branding/header.ts, home/SYSTEM.md, home/themes/loom.json, package.json}`. ⚠ `installApprovalGate` guards on `!ctx.hasUI` — must become `ctx.mode === "tui"` on rebuild.
- **Pi API of record (installed, verified 0.79.0):** `/Users/anub/Work/Loom/cli/node_modules/@earendil-works/pi-coding-agent/dist/core/extensions/types.d.ts` — `ExtensionMode` :207, `mode` :212, `hasUI` :213 ("true in TUI and RPC modes"); UI surface ~:67-191 (`setWidget` :96-99 w/ `dispose?` :98, `setFooter` :106, `setHeader` :110, `custom` :116-126); `ToolDefinition` ~:333-364 (`promptSnippet` :341, `renderShell` :347, `executionMode` :357, `renderCall` :361, `renderResult` :363 — **no `disableModelInvocation`**); `registerMessageRenderer` :855. `disableModelInvocation` lives only in `dist/core/skills.d.ts:15`. `dist/index.d.ts` :25-26 exports `renderDiff` (returns **string** — `dist/.../diff.d.ts:11`), `DynamicBorder`, `getMarkdownTheme`. **Nested** `…/pi-coding-agent/node_modules/@earendil-works/pi-tui/dist/`: `tui.d.ts` (`Component` :9-29 — `render` :15, `handleInput` :19, `invalidate` :29; `requestRender` :204), `components/image.d.ts` (`Image` ctor `(base64, mimeType, theme, options?{imageId}, dimensions?)`, `getImageId()`), `terminal-image.d.ts` :1-30 (`detectCapabilities`/`getCapabilities` → `{images:"kitty"|"iterm2"|null}`, `allocateImageId`, `renderImage`, `imageFallback`).
