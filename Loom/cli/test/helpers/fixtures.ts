/**
 * Shared test fixtures + tiny utilities.
 *
 * Built against the LOCKED `src/types.ts` envelope so the hand-built results
 * match exactly what the Python engine emits (`loom/types.py:VerbResult.to_dict`)
 * and what `parseLoomJson` yields.
 */
import type { AgentToolResult, Theme, ToolDefinition } from "@earendil-works/pi-coding-agent";
import { Theme as ThemeClass } from "@earendil-works/pi-coding-agent";
import type { TSchema } from "@earendil-works/pi-ai";

import type { Diagnostic, VerbResult } from "../../dist/types.js";

/**
 * Pi's `renderResult` positional params — `[result, options, theme, ctx]` — derived
 * by indexed access (same technique as the widget stub) so we never name the
 * non-exported `ToolRenderContext`/`ToolRenderResultOptions` and still track Pi 0.79.0.
 */
type RenderResult = NonNullable<ToolDefinition<TSchema, VerbResult>["renderResult"]>;
type RenderResultOptions = Parameters<RenderResult>[1];
type RenderContext = Parameters<RenderResult>[3];

/** Strip ANSI SGR escape sequences so substring assertions match the plain text. */
export function stripAnsi(s: string): string {
	// eslint-disable-next-line no-control-regex
	return s.replace(/\x1b\[[0-9;]*m/g, "");
}

/** The full set of foreground theme color keys (Pi 0.79.0 `ThemeColor`). */
const FG_COLORS = [
	"accent", "border", "borderAccent", "borderMuted", "success", "error", "warning",
	"muted", "dim", "text", "thinkingText", "userMessageText", "customMessageText",
	"customMessageLabel", "toolTitle", "toolOutput", "mdHeading", "mdLink", "mdLinkUrl",
	"mdCode", "mdCodeBlock", "mdCodeBlockBorder", "mdQuote", "mdQuoteBorder", "mdHr",
	"mdListBullet", "toolDiffAdded", "toolDiffRemoved", "toolDiffContext", "syntaxComment",
	"syntaxKeyword", "syntaxFunction", "syntaxVariable", "syntaxString", "syntaxNumber",
	"syntaxType", "syntaxOperator", "syntaxPunctuation", "thinkingOff", "thinkingMinimal",
	"thinkingLow", "thinkingMedium", "thinkingHigh", "thinkingXhigh", "bashMode",
];
/** The full set of background theme color keys (Pi 0.79.0 `ThemeBg`). */
const BG_COLORS = [
	"selectedBg", "userMessageBg", "customMessageBg", "toolPendingBg", "toolSuccessBg", "toolErrorBg",
];

/**
 * A real `Theme` instance the widget can call `.fg()/.bg()/.bold()` on. Built from
 * the public `Theme` constructor with a complete color record (every key the widget
 * could reach), so it works regardless of which colors the implementation uses.
 */
export function makeTheme(): Theme {
	const fg = Object.fromEntries(FG_COLORS.map((k) => [k, "#d4d4d4"])) as Record<string, string>;
	const bg = Object.fromEntries(BG_COLORS.map((k) => [k, "#000000"])) as Record<string, string>;
	return new ThemeClass(fg as never, bg as never, "truecolor", { name: "loom-test" }) as Theme;
}

/** Minimal non-partial, non-expanded render options. */
export function makeRenderOptions(): RenderResultOptions {
	return { expanded: false, isPartial: false };
}

/** A minimal render context — only the fields a result render plausibly reads. */
export function makeRenderCtx(): RenderContext {
	return {
		args: {},
		toolCallId: "test-render-1",
		invalidate: () => {},
		lastComponent: undefined,
		state: {},
		cwd: process.cwd(),
		executionStarted: true,
		argsComplete: true,
		isPartial: false,
		expanded: false,
		showImages: false,
		isError: true,
	} as unknown as RenderContext;
}

/** Wrap a `VerbResult` as the `AgentToolResult<VerbResult>` a verb tool returns. */
export function asToolResult(verb: VerbResult): AgentToolResult<VerbResult> {
	return {
		content: [{ type: "text", text: verb.summary }],
		details: verb,
	} as AgentToolResult<VerbResult>;
}

/**
 * A hand-built FAIL envelope carrying one C1 diagnostic with contract/severity/
 * message/fix — the widget test's input. Shaped exactly per `src/types.ts`.
 */
export function failEnvelopeWithC1(): VerbResult {
	const c1: Diagnostic = {
		contract: "C1",
		severity: "error",
		message: "C1 leakage: target column 'is_fraud' is derivable from feature 'fraud_score'.",
		fix: "drop 'fraud_score' from the feature set, or pass --target with a non-derived label.",
		data: { feature: "fraud_score", target: "is_fraud", correlation: 0.998 },
	};
	return {
		verb: "ingest",
		status: "REFUSED_CONTRACT",
		verdict: "FAIL",
		tier: "workspace-write",
		capability_mode: "none",
		summary: "ingest REFUSED — C1 leakage gate failed (1 error).",
		outputs: [],
		diagnostics: [c1],
		data: {},
		experiment: null,
		cost_plan: null,
		confirm_token: null,
	};
}
