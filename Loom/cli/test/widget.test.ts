/**
 * Contract-diff widget test.
 *
 * Calls `renderContractResult` with a hand-built FAIL envelope (one C1 diagnostic
 * carrying contract/severity/message/fix) and asserts the returned Pi `Component`
 * renders (`.render(80) → string[]`) lines that surface the contract id, the
 * message, and the offered fix. Also asserts the empty/missing-`details` fallback
 * (blocked/aborted results carry `details:{}`) renders the text content instead of
 * throwing.
 *
 * No model, no TUI: we build a real `Theme` from the public constructor and a
 * synthetic render context, and read the component's plain (ANSI-stripped) lines.
 *
 * Run: node --test --experimental-strip-types test/widget.test.ts
 */
import test from "node:test";
import assert from "node:assert/strict";

import { renderContractResult } from "../dist/widgets/contract-diff.js";
import type { VerbResult } from "../dist/types.js";
import {
	asToolResult,
	failEnvelopeWithC1,
	makeRenderCtx,
	makeRenderOptions,
	makeTheme,
	stripAnsi,
} from "./helpers/fixtures.ts";

/** Render a tool result through the widget and return plain (ANSI-stripped) lines. */
function renderLines(toolResult: ReturnType<typeof asToolResult>): string[] {
	const component = renderContractResult(toolResult, makeRenderOptions(), makeTheme(), makeRenderCtx());
	const lines = component.render(80);
	assert.ok(Array.isArray(lines), "Component.render(80) must return string[]");
	return lines.map(stripAnsi);
}

test("renders the C1 contract id, message, and fix from a FAIL envelope", () => {
	const env = failEnvelopeWithC1();
	const lines = renderLines(asToolResult(env));
	// The card hard-wraps at the render width, so a phrase can straddle a line
	// boundary. `flat` de-wraps (collapse whitespace) for phrase-level matches;
	// keep short-token matches against the per-line text.
	const text = lines.join("\n");
	const flat = lines.join(" ").replace(/\s+/g, " ");

	assert.match(text, /C1/, "must surface the contract id C1");
	assert.match(text, /leakage/, "must surface the diagnostic message");
	assert.match(flat, /derivable from feature 'fraud_score'/, "must include the message detail");
	assert.match(flat, /fix:/, "must label the offered fix");
	assert.match(flat, /drop 'fraud_score'/, "must surface the offered fix");
});

test("tolerates an empty/missing details and falls back to the text content", () => {
	// A blocked/aborted result: details is {} (not a full VerbResult) and content
	// carries the human-readable text.
	const blocked = {
		content: [{ type: "text", text: "Tool call was blocked by the launch gate." }],
		details: {} as VerbResult,
	} as ReturnType<typeof asToolResult>;

	const lines = renderLines(blocked);
	const text = lines.join("\n");
	assert.match(text, /blocked by the launch gate/, "must fall back to the text content");
});
