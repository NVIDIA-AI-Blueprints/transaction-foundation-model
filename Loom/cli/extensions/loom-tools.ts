/**
 * Loom tools extension — the Pi extension entry point (default-export factory).
 *
 * Pi loads this file directly from source via jiti at runtime (wired through
 * `--extension <abs path>` in src/pi/runtime.ts). It is intentionally thin: all
 * the real logic (manifest fetch, TypeBox schemas, the spawn bridge, the gate,
 * the widget) lives in the tsc-compiled src modules, imported here from the built
 * ../dist/*.js so this entry needs no compilation of its own.
 *
 * Wiring order (load-bearing):
 *   1. loadLoomManifest()            — shell `python -m loom verbs --json`, parse 3 schemas
 *   2. registerLoomTools(pi, …)      — one Pi tool per verb; populates the tier registry
 *   3. installLoomGate(pi)           — the corrected `tool_call` {block} lock (INERT in Phase-0;
 *                                      MUST run after registerLoomTools — reads the tier registry)
 *   4. branding (session_start)      — the LOOM banner header
 *
 * Action methods cannot be called in the factory phase — only registerX + on().
 * The manifest load is async; tool registration happens inside the awaited factory.
 */
import type { ExtensionAPI, ExtensionContext, SessionStartEvent } from "@earendil-works/pi-coding-agent";

import { installLoomHeader } from "../dist/branding/header.js";
import { loadLoomManifest, registerLoomTools } from "../dist/manifest.js";
import { installLoomGate } from "../dist/gate.js";
import { renderContractResult } from "../dist/widgets/contract-diff.js";

export default async function loomTools(pi: ExtensionAPI): Promise<void> {
	// 1+2. One Pi tool per Loom verb (loom_tokenize, loom_ingest, loom_baseline),
	// each shelling out to `python -m loom <verb> --json` and rendering its result
	// through the contract-diff card.
	const schemas = await loadLoomManifest();
	registerLoomTools(pi, schemas, { renderContractResult });

	// 3. The corrected tier/launch gate — a `tool_call` {block} hook guarded on
	// ctx.mode === "tui". Inert for the Phase-0 verbs (none gated), but the lock
	// mechanism is correct so pretrain/embed inherit it. AFTER registerLoomTools.
	installLoomGate(pi);

	// 4. Branding: install the LOOM banner header once the (interactive) session
	// starts and tools/commands are registered.
	pi.on("session_start", (_event: SessionStartEvent, ctx: ExtensionContext) => {
		installLoomHeader(pi, ctx);
	});
}
