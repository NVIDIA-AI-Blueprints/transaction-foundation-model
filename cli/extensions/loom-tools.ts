/**
 * Loom tools extension — the Pi extension entry point.
 *
 * Pi loads this file directly from source via jiti at runtime (wired through
 * `--extension <abs path>` in src/pi/runtime.ts). It is intentionally thin: all
 * the real logic (manifest fetch, TypeBox schemas, the spawn wrapper, the
 * approval gate) lives in the tsc-compiled src/manifest.ts, imported here from
 * the built ../dist/manifest.js so this entry needs no compilation of its own.
 *
 * Mechanism mirrors Feynman's extensions/research-tools.ts (a default-export
 * factory taking the ExtensionAPI).
 */
import type { ExtensionAPI, ExtensionContext, SessionStartEvent } from "@earendil-works/pi-coding-agent";

import { installLoomHeader } from "../dist/branding/header.js";
import { installApprovalGate, registerLoomTools } from "../dist/manifest.js";

export default async function loomTools(pi: ExtensionAPI): Promise<void> {
	// One Pi tool per Loom verb (loom_eda, loom_run, loom_deploy, …), each
	// shelling out to `python -m loom <verb> --json`.
	await registerLoomTools(pi);
	// Layer-B per-call approval gate keyed off the verb tier.
	installApprovalGate(pi);
	// Branding: install the LOOM banner header once the (interactive) session
	// starts and tools/commands are registered.
	pi.on("session_start", (_event: SessionStartEvent, ctx: ExtensionContext) => {
		installLoomHeader(pi, ctx);
	});
}
