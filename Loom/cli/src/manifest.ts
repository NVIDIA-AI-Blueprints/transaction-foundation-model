/**
 * The Loom bridge — the seam between Pi tools and the Python engine.
 *
 * Contract (from PI.md §B.4, §B.9):
 *  - `loadLoomManifest()` shells `python -m loom verbs --json`, parses the 3
 *    `VerbSchema`s (full JSON-Schema `input_schema` + `_loom` metadata).
 *  - `registerLoomTools(pi, schemas, deps)` loops the schemas and registers one
 *    Pi tool per verb (name `loom_<verb>`), each whose `execute()` shells out via
 *    `runLoom`. Gated verbs (`_loom.disable_model_invocation`) omit `promptSnippet`
 *    (cosmetic); the real lock is the `tool_call` hook in gate.ts.
 *  - `runLoom(args)` spawns `${LOOM_PYTHON} -m loom <args> --json` and returns the
 *    parsed `VerbResult`. It THROWS only on a transport failure (no parseable JSON);
 *    a domain `Status.FAIL`/`Verdict.FAIL` is a well-formed envelope that is RETURNED.
 *  - `toCliArgs(input)` translates validated tool params → CLI argv flags.
 *  - `parseLoomJson(stdout)` bottom-up-scans stdout for the trailing JSON line.
 *
 * Param schemas MUST be built with the `Type` re-exported from
 * `@earendil-works/pi-ai` (so `TSchema` matches `registerTool<TParams extends
 * TSchema>`), NOT `@sinclair/typebox` directly.
 */
import { spawn } from "node:child_process";

import type { AgentToolResult, ExtensionAPI, ToolDefinition } from "@earendil-works/pi-coding-agent";
// Pi validates tool params against the typebox re-exported by pi-ai. We MUST build
// schemas with this `Type`, not @sinclair/typebox, so the `TSchema` that Pi's
// `registerTool<TParams extends TSchema>` expects is the same nominal type.
import { Type, type TSchema } from "@earendil-works/pi-ai";

import type { JsonSchemaObject, JsonSchemaProperty, LoomTier, VerbResult, VerbSchema } from "./types.js";

/** Default venv interpreter on the dev box (mirrors index.ts); override with LOOM_PYTHON. */
const DEFAULT_LOOM_PYTHON = "/Users/anub/Work/transaction-foundation-model/Loom/.venv/bin/python";

function resolveLoomPython(): string {
	const fromEnv = process.env.LOOM_PYTHON?.trim();
	return fromEnv && fromEnv.length > 0 ? fromEnv : DEFAULT_LOOM_PYTHON;
}

/**
 * Dependencies the extension factory threads into `registerLoomTools` — the
 * rendering seam and any host context the tools need. Kept as an explicit bag so
 * the Implement phase can extend it without changing the call site shape.
 */
export interface RegisterToolsDeps {
	/** Per-verb result card (widgets/contract-diff.ts:renderContractResult). */
	renderContractResult: typeof import("./widgets/contract-diff.js").renderContractResult;
}

/** Raw capture of a `python -m loom` child process. */
interface RunResult {
	code: number | null;
	stdout: string;
	stderr: string;
}

/**
 * Spawn `${LOOM_PYTHON} -m loom <...args>` and capture stdout/stderr. Never
 * rejects on a nonzero exit — the caller inspects `code`/`stdout`. Rejects only on
 * a spawn-level failure (interpreter missing, etc.).
 */
function spawnLoom(args: string[], signal?: AbortSignal): Promise<RunResult> {
	const python = resolveLoomPython();
	return new Promise<RunResult>((resolvePromise, reject) => {
		const child = spawn(python, ["-m", "loom", ...args], {
			stdio: ["ignore", "pipe", "pipe"],
			signal,
			env: process.env,
		});
		let stdout = "";
		let stderr = "";
		child.stdout?.on("data", (chunk: Buffer) => {
			stdout += chunk.toString("utf8");
		});
		child.stderr?.on("data", (chunk: Buffer) => {
			stderr += chunk.toString("utf8");
		});
		child.on("error", reject);
		child.on("close", (code) => {
			resolvePromise({ code, stdout, stderr });
		});
	});
}

/**
 * Parse the engine's `--json` output: bottom-up scan for the last line that is a
 * JSON object, returning the parsed `VerbResult`. `--json` prints only the single
 * envelope line, so the trailing-JSON scan is robust against any prose preceding
 * it. Returns the parsed envelope; the caller (`runLoom`) decides throw-vs-return.
 *
 * THROWS when no line parses as a JSON object — that is a transport failure, not a
 * domain outcome (a domain FAIL is a well-formed envelope and parses fine).
 */
export function parseLoomJson(stdout: string): VerbResult {
	const lines = stdout
		.split("\n")
		.map((l) => l.trim())
		.filter((l) => l.length > 0);
	for (let i = lines.length - 1; i >= 0; i--) {
		const line = lines[i];
		if (!line.startsWith("{")) continue;
		try {
			return JSON.parse(line) as VerbResult;
		} catch {
			// keep scanning upward — a prose line may coincidentally start with "{".
		}
	}
	throw new Error("Loom: no parseable JSON envelope in engine output.");
}

/**
 * Spawn `${LOOM_PYTHON} -m loom <...args> --json`, capture stdout, and parse the
 * trailing JSON line into a `VerbResult`. THROWS only on a transport failure (the
 * spawn fails, or no parseable JSON is emitted). A well-formed envelope carrying
 * `status:"FAIL"`/`verdict:"FAIL"` is RETURNED so the agent composes on it.
 *
 * `args` is the full positional argv (verb name first), e.g.
 * `["tokenize", "--preset", "financial"]`; `--json` is appended here.
 */
export async function runLoom(args: string[], signal?: AbortSignal): Promise<VerbResult> {
	const { code, stdout, stderr } = await spawnLoom([...args, "--json"], signal);
	try {
		return parseLoomJson(stdout);
	} catch {
		const reason = stderr.trim() || stdout.trim();
		const verb = args[0] ?? "(unknown verb)";
		throw new Error(
			reason.length > 0
				? `Loom engine failed running '${verb}' (exit ${code ?? "null"}): ${reason.slice(0, 2000)}`
				: `Loom engine '${verb}' exited ${code ?? "null"} with no parseable JSON output.`,
		);
	}
}

/**
 * Run `python -m loom verbs --json` and parse the verb manifest (the 3 schemas).
 * Throws on a missing/garbled manifest (nonzero exit or non-array JSON).
 */
export async function loadLoomManifest(signal?: AbortSignal): Promise<VerbSchema[]> {
	const { code, stdout, stderr } = await spawnLoom(["verbs", "--json"], signal);
	if (code !== 0) {
		throw new Error(`loom verbs --json exited ${code ?? "null"}: ${stderr.trim() || "(no stderr)"}`);
	}
	// The manifest is a JSON array printed as the trailing line; scan bottom-up.
	const lines = stdout
		.split("\n")
		.map((l) => l.trim())
		.filter((l) => l.startsWith("["));
	const payload = lines.length > 0 ? lines[lines.length - 1] : stdout.trim();
	let parsed: unknown;
	try {
		parsed = JSON.parse(payload);
	} catch (error) {
		throw new Error(`loom verbs --json returned non-JSON: ${(error as Error).message}`);
	}
	if (!Array.isArray(parsed)) {
		throw new Error("loom verbs --json did not return an array.");
	}
	return parsed as VerbSchema[];
}

/**
 * Translate validated tool params into CLI argv for `loom <verb> ... --json`.
 * Takes ONLY the params object — the caller prepends the verb name in
 * `runLoom(["<verb>", ...toCliArgs(input)])`. Booleans emit a bare `--flag` only
 * when true; scalars emit `--flag value`; undefined/null/empty are skipped. The
 * engine uses `in` as a param key — a reserved word as a flag is fine (`--in`).
 */
export function toCliArgs(input: Record<string, unknown>): string[] {
	const args: string[] = [];
	for (const [key, value] of Object.entries(input ?? {})) {
		if (value === undefined || value === null) continue;
		// Tool-schema keys are snake_case (include_time_delta); the engine's argparse
		// flags are kebab-case (--include-time-delta). Convert, or multi-word flags fail.
		const flag = `--${key.replace(/_/g, "-")}`;
		if (typeof value === "boolean") {
			if (value) args.push(flag);
			continue;
		}
		const str = String(value);
		if (str.length === 0) continue;
		args.push(flag, str);
	}
	return args;
}

/**
 * Build a TypeBox params object for a verb from its full JSON-Schema `input_schema`.
 * Maps JSON-Schema `type` → `Type.String/Integer/Number/Boolean` (string fallback),
 * honors `enum` on string/number props, and wraps every property NOT in
 * `required[]` in `Type.Optional`. `additionalProperties` defaults to false.
 */
export function buildVerbParams(schema: JsonSchemaObject): TSchema {
	const required = new Set(schema.required ?? []);
	const props: Record<string, TSchema> = {};
	for (const [name, prop] of Object.entries(schema.properties ?? {})) {
		const base = jsonSchemaPropToTypeBox(prop);
		props[name] = required.has(name) ? base : Type.Optional(base);
	}
	return Type.Object(props, { additionalProperties: schema.additionalProperties ?? false });
}

/** Map a single JSON-Schema property to a TypeBox schema (the `Type` from pi-ai). */
function jsonSchemaPropToTypeBox(prop: JsonSchemaProperty): TSchema {
	const description = typeof prop.description === "string" ? prop.description : undefined;
	const options = description ? { description } : {};
	// An `enum` over strings/numbers maps to a union of literals (string fallback).
	if (Array.isArray(prop.enum) && prop.enum.length > 0) {
		return Type.Union(
			prop.enum.map((v) => Type.Literal(v)),
			options,
		);
	}
	switch (prop.type) {
		case "integer":
			return Type.Integer(options);
		case "number":
			return Type.Number(options);
		case "boolean":
			return Type.Boolean(options);
		case "array": {
			const item = prop.items ? jsonSchemaPropToTypeBox(prop.items) : Type.String();
			return Type.Array(item, options);
		}
		case "object":
			return Type.Object({}, { ...options, additionalProperties: true });
		case "string":
		default:
			return Type.String(options);
	}
}

/**
 * Register one Pi tool per Loom verb from the parsed schemas. Each tool's
 * `execute()` shells out via `runLoom(["<verb>", ...toCliArgs(args)])` and returns
 * `{ content:[{type:"text",text:summary}], details: verbResult }`, rendering the
 * result through `deps.renderContractResult`. Gated verbs
 * (`_loom.disable_model_invocation`) omit `promptSnippet` (cosmetic). Populates the
 * tier + gated registries the gate (gate.ts) reads — so installLoomGate MUST run
 * after this.
 */
export function registerLoomTools(pi: ExtensionAPI, schemas: VerbSchema[], deps: RegisterToolsDeps): void {
	const { renderContractResult } = deps;

	for (const schema of schemas) {
		const toolName = toolNameFor(schema.name);
		const meta = schema._loom;
		setToolTier(toolName, meta.tier);
		setToolGated(toolName, meta.disable_model_invocation);

		// The verb name passed to the engine: dotted "loom.tokenize" → bare "tokenize".
		const verb = schema.name.startsWith("loom.") ? schema.name.slice("loom.".length) : schema.name;
		const params = buildVerbParams(schema.input_schema);

		const definition: ToolDefinition<typeof params, VerbResult> = {
			name: toolName,
			label: `loom ${verb}`,
			description: schema.description,
			// Gated verbs are omitted from the "Available tools" prompt section so the
			// model does not auto-offer them (cosmetic — the tool stays callable by
			// name; the real lock is the gate.ts tool_call {block} hook).
			promptSnippet: meta.disable_model_invocation ? undefined : schema.description,
			parameters: params,
			// Loom draws its own contract-diff framing.
			renderShell: "self",
			renderResult: (result, options, theme, ctx) => renderContractResult(result, options, theme, ctx),
			async execute(_toolCallId, args, signal): Promise<AgentToolResult<VerbResult>> {
				// runLoom throws ONLY on a transport failure (no parseable JSON); a domain
				// Status.FAIL/Verdict.FAIL is a well-formed envelope that is RETURNED so the
				// agent reads the diagnostics and composes on them.
				const input: Record<string, unknown> = { ...(args ?? {}) };
				const result = await runLoom([verb, ...toCliArgs(input)], signal);
				return {
					content: [{ type: "text", text: result.summary }],
					details: result,
				};
			},
		};

		pi.registerTool(definition);
	}
}

/** `loom_<verb>` tool name for a dotted engine verb name ("loom.tokenize" → "loom_tokenize"). */
export function toolNameFor(verbName: string): string {
	const bare = verbName.startsWith("loom.") ? verbName.slice("loom.".length) : verbName;
	return `loom_${bare}`;
}

/** Module-level verb tier registry, populated by registerLoomTools, queried by the gate. */
const tierByToolName = new Map<string, LoomTier>();

/** Record a tool's tier (called from registerLoomTools). */
export function setToolTier(toolName: string, tier: LoomTier): void {
	tierByToolName.set(toolName, tier);
}

/** The tier of a registered loom tool, or undefined if not a loom verb tool. */
export function tierOf(toolName: string): LoomTier | undefined {
	return tierByToolName.get(toolName);
}

/** Whether a tool is registered as model-invocation-disabled (gated). Set by registerLoomTools. */
const gatedToolNames = new Set<string>();

export function setToolGated(toolName: string, gated: boolean): void {
	if (gated) gatedToolNames.add(toolName);
	else gatedToolNames.delete(toolName);
}

export function isToolGated(toolName: string): boolean {
	return gatedToolNames.has(toolName);
}
