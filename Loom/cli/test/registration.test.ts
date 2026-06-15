/**
 * Extension registration + gate-wiring test (mock-pi harness, no model, no TUI).
 *
 * Builds a mock `ExtensionAPI` that captures `registerTool`/`on` calls, feeds the
 * three stub verb schemas (the exact shape `loom verbs --json` emits) through
 * `registerLoomTools` + `installLoomGate`, and asserts:
 *   1. exactly 3 tools are registered, named loom_tokenize / loom_ingest / loom_baseline;
 *   2. each tool carries the locked surface (name/description/parameters/execute/renderResult);
 *   3. the tier registry is populated (tierOf each tool) and Phase-0 gating is INERT
 *      (no verb is gated: disable_model_invocation:false → isToolGated false);
 *   4. a `tool_call` hook is installed by installLoomGate.
 *
 * A second test drives that captured `tool_call` hook with a synthetic event in
 * both `tui` and non-`tui` (`rpc`) modes to prove the corrected mechanism: with no
 * gated verb the hook is inert (returns no block) in BOTH modes — and it never
 * spawns python or invokes a model.
 *
 * Run: node --test --experimental-strip-types test/registration.test.ts
 */
import test from "node:test";
import assert from "node:assert/strict";

import { installLoomGate } from "../dist/gate.js";
import { isToolGated, registerLoomTools, tierOf } from "../dist/manifest.js";
import { renderContractResult } from "../dist/widgets/contract-diff.js";
import type { VerbSchema } from "../dist/types.js";
import { makeCtx, makeMockPi, makeToolCallEvent, runToolCallHandler } from "./helpers/mock-pi.ts";

/** The 3 Phase-0 verb schemas, matching `python -m loom verbs --json` (none gated). */
function stubSchemas(): VerbSchema[] {
	const base = (name: string, capability_mode: VerbSchema["_loom"]["capability_mode"]): VerbSchema => ({
		name,
		description: `stub ${name}`,
		input_schema: {
			type: "object",
			properties: {
				in: { type: "string", description: "input pathspec" },
				preset: { type: "string", enum: ["financial", "chain"], description: "preset" },
				context_len: { type: "integer", description: "context length" },
				include_time_delta: { type: "boolean", description: "T1 flag" },
			},
		},
		_loom: { tier: "workspace-write", capability_mode, disable_model_invocation: false },
	});
	return [
		base("loom.tokenize", "none"),
		base("loom.ingest", "none"),
		base("loom.baseline", "searchable"),
	];
}

test("registerLoomTools registers exactly 3 named verb tools with the locked surface", () => {
	const pi = makeMockPi();
	registerLoomTools(pi.api, stubSchemas(), { renderContractResult });

	assert.deepEqual(pi.toolNames(), ["loom_tokenize", "loom_ingest", "loom_baseline"]);
	assert.equal(pi.tools.length, 3);

	for (const tool of pi.tools) {
		assert.equal(typeof tool.name, "string");
		assert.equal(typeof tool.description, "string");
		assert.ok(tool.parameters, `${tool.name} must carry a TypeBox parameters schema`);
		assert.equal(typeof tool.execute, "function", `${tool.name}.execute must be a function`);
		assert.equal(typeof tool.renderResult, "function", `${tool.name}.renderResult must be wired`);
	}

	// Tier registry populated; Phase-0 gating is inert (nothing gated).
	for (const name of ["loom_tokenize", "loom_ingest", "loom_baseline"]) {
		assert.equal(tierOf(name), "workspace-write", `${name} tier must be recorded`);
		assert.equal(isToolGated(name), false, `${name} must NOT be gated in Phase-0`);
	}
});

test("installLoomGate installs a tool_call hook (after registerLoomTools)", () => {
	const pi = makeMockPi();
	registerLoomTools(pi.api, stubSchemas(), { renderContractResult });
	installLoomGate(pi.api);

	const hook = pi.toolCallHandler();
	assert.ok(hook, "a tool_call hook must be installed");
	assert.equal(hook.event, "tool_call");
	assert.equal(typeof hook.handler, "function");
});

test("the gate hook is INERT for an ungated Phase-0 verb in both tui and rpc modes", async () => {
	const pi = makeMockPi();
	registerLoomTools(pi.api, stubSchemas(), { renderContractResult });
	installLoomGate(pi.api);

	const hook = pi.toolCallHandler();
	assert.ok(hook);

	const event = makeToolCallEvent("loom_tokenize", { preset: "financial" });

	// tui (interactive) — ungated verb: no block.
	const tuiResult = await runToolCallHandler(hook, event, makeCtx("tui"));
	assert.notEqual(tuiResult?.block, true, "ungated verb must not be blocked in tui mode");

	// rpc (non-tui) — ungated verb: still no block. (A GATED verb here must hard-block,
	// but Phase-0 has none — this proves the inert path doesn't accidentally block.)
	const rpcResult = await runToolCallHandler(hook, event, makeCtx("rpc"));
	assert.notEqual(rpcResult?.block, true, "ungated verb must not be blocked in rpc mode");
});

test("the gate hook ignores non-loom tool names", async () => {
	const pi = makeMockPi();
	registerLoomTools(pi.api, stubSchemas(), { renderContractResult });
	installLoomGate(pi.api);

	const hook = pi.toolCallHandler();
	assert.ok(hook);

	const result = await runToolCallHandler(hook, makeToolCallEvent("bash", { command: "ls" }), makeCtx("tui"));
	assert.notEqual(result?.block, true, "a non-loom tool must pass the loom gate untouched");
});
