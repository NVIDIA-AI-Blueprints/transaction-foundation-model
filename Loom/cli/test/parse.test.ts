/**
 * Bridge parse test — the engine seam contract.
 *
 * Runs the real engine ONCE (a CPU-only verb; no model, no GPU, no network) to
 * capture a live `--json` envelope, then asserts `parseLoomJson` (the bridge's
 * trailing-JSON parser) yields a `VerbResult` with the locked field values:
 * status OK / verdict PASS / outputs ["Corpus/1"] / data.vocab_size 6251.
 *
 * Also asserts the trailing-JSON discipline directly: extra non-JSON lines
 * (banner/warnings) before the envelope must not break the bottom-up parse.
 *
 * Run: node --test --experimental-strip-types test/parse.test.ts
 */
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import test from "node:test";
import assert from "node:assert/strict";

import { parseLoomJson } from "../dist/manifest.js";
import type { VerbResult } from "../dist/types.js";

const LOOM_PYTHON =
	process.env.LOOM_PYTHON?.trim() || "/Users/anub/Work/transaction-foundation-model/Loom/.venv/bin/python";

/** Capture the engine's tokenize envelope once (CPU verb; deterministic). */
function captureTokenizeStdout(): string {
	const res = spawnSync(LOOM_PYTHON, ["-m", "loom", "tokenize", "--preset", "financial", "--json"], {
		encoding: "utf8",
	});
	if (res.error) throw res.error;
	assert.equal(res.status, 0, `engine exited ${res.status}; stderr:\n${res.stderr}`);
	return res.stdout;
}

test("parseLoomJson yields a VerbResult from a live tokenize envelope", { concurrency: false }, (t) => {
	if (!existsSync(LOOM_PYTHON)) {
		t.skip(`LOOM_PYTHON not found at ${LOOM_PYTHON} — set LOOM_PYTHON to the Loom venv python`);
		return;
	}
	const stdout = captureTokenizeStdout();
	const result: VerbResult = parseLoomJson(stdout);

	assert.equal(result.verb, "tokenize");
	assert.equal(result.status, "OK");
	assert.equal(result.verdict, "PASS");
	assert.equal(result.tier, "workspace-write");
	assert.equal(result.capability_mode, "none");
	assert.deepEqual(result.outputs, ["Corpus/1"]);
	assert.equal(result.confirm_token, null);
	assert.equal((result.data as { vocab_size?: number }).vocab_size, 6251);
	// The 12 locked top-level keys are all present.
	for (const key of [
		"verb", "status", "verdict", "tier", "capability_mode", "summary",
		"outputs", "diagnostics", "data", "experiment", "cost_plan", "confirm_token",
	]) {
		assert.ok(key in result, `missing envelope key: ${key}`);
	}
});

test("parseLoomJson ignores non-JSON noise before the trailing envelope", () => {
	const envelope: VerbResult = {
		verb: "tokenize",
		status: "OK",
		verdict: "PASS",
		tier: "workspace-write",
		capability_mode: "none",
		summary: "Corpus/1 verdict=PASS",
		outputs: ["Corpus/1"],
		diagnostics: [],
		data: { vocab_size: 6251 },
		experiment: null,
		cost_plan: null,
		confirm_token: null,
	};
	const stdout = ["loom: warning — some banner line", "another non-json log line", JSON.stringify(envelope), ""].join(
		"\n",
	);
	const result = parseLoomJson(stdout);
	assert.equal(result.status, "OK");
	assert.deepEqual(result.outputs, ["Corpus/1"]);
	assert.equal((result.data as { vocab_size?: number }).vocab_size, 6251);
});
