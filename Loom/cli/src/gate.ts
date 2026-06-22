/**
 * The Loom launch/tier gate — the ONLY real lock (PI.md §A.5, §B.5).
 *
 * CORRECTED MECHANISM (do not reintroduce the legacy bugs):
 *  - The lock is a `tool_call` hook returning `{ block:true }` keyed on
 *    `event.toolName` (exactly the legacy `installApprovalGate` shape). There is
 *    NO `ToolDefinition.disableModelInvocation` in Pi — `_loom.disable_model_invocation`
 *    is Loom-internal metadata the extension reads (via `isToolGated`) to decide
 *    WHICH verbs this hook locks; the hook itself is the enforcement.
 *  - Guard the interactive (approval) branch on `ctx.mode === "tui"`, NOT
 *    `!ctx.hasUI` (hasUI is true in rpc, where a custom approval UI is a silent
 *    no-op — which would let a gated verb pass without the human gate). In any
 *    non-tui mode a gated verb's hook hard-`{ block:true, reason }`.
 *
 * Phase-0 is INERT: none of tokenize/ingest/baseline is gated
 * (`disable_model_invocation:false`), so `isToolGated` is always false and the hook
 * returns early — but it lands the CORRECT lock so pretrain/embed inherit it for
 * free when they arrive gated.
 *
 * Reads `isToolGated`/`tierOf` from manifest.ts (populated by registerLoomTools),
 * so installLoomGate MUST be called AFTER registerLoomTools.
 */
import type { ExtensionAPI, ExtensionContext, ToolCallEvent, ToolCallEventResult } from "@earendil-works/pi-coding-agent";

import { isToolGated, tierOf } from "./manifest.js";

/**
 * Install the Loom tier/launch gate as a `tool_call` {block} hook. Inert for the
 * Phase-0 verbs (none gated); the mechanism is correct so gated launch verbs
 * inherit it. Must be called after registerLoomTools (reads the tier/gated registries).
 */
export function installLoomGate(pi: ExtensionAPI): void {
	pi.on("tool_call", async (event: ToolCallEvent, ctx: ExtensionContext): Promise<ToolCallEventResult | void> => {
		const toolName = event.toolName;

		// Only Loom verb tools are gated here; let Pi's own gates handle everything
		// else. A tool with no recorded tier is not a Loom verb.
		const tier = tierOf(toolName);
		if (tier === undefined) return;

		// The lock fires ONLY for verbs whose `_loom.disable_model_invocation` is true.
		// Phase-0: no verb is gated, so this returns here and the hook is inert.
		if (!isToolGated(toolName)) return;

		// A gated (launch/irreversible) verb requires an interactive human gate. The
		// approval UI is meaningful ONLY in the interactive terminal (tui). In any
		// other mode (rpc/json/print) a custom approval overlay is a silent no-op, so
		// we must HARD-block rather than let the call slip through ungated.
		if (ctx.mode !== "tui") {
			return {
				block: true,
				reason: `${toolName} is a gated ${tier} Loom verb and can only be launched from the interactive terminal after explicit confirmation.`,
			};
		}

		// Interactive terminal: require an explicit human confirmation before the
		// gated verb runs. A decline (or a dismissed dialog) blocks the call.
		const approved = await ctx.ui.confirm(
			`Loom: confirm launch of ${toolName}`,
			`${toolName} is a gated ${tier} verb. It may incur cost or make irreversible changes. Launch it now?`,
		);
		if (!approved) {
			return { block: true, reason: `User declined to launch the gated Loom verb ${toolName}.` };
		}

		// Approved in tui — allow the call through (no result needed).
		return;
	});
}
