import { spawn } from "node:child_process";

import type {
	AgentToolResult,
	ExtensionAPI,
	ExtensionContext,
	ToolCallEvent,
	ToolCallEventResult,
	ToolDefinition,
} from "@earendil-works/pi-coding-agent";
// Pi validates tool params against typebox@1.x (re-exported by pi-ai). We MUST
// build schemas with this `Type`, not @sinclair/typebox, so the `TSchema` Pi's
// registerTool<TParams extends TSchema> expects is the same nominal type.
import { Type, type TSchema } from "@earendil-works/pi-ai";

/** One verb as emitted by `python -m loom verbs --json`. */
export interface LoomVerb {
	name: string;
	summary: string;
	required: string[];
	optional: string[];
	tier: LoomTier;
	disable_model_invocation: boolean;
}

export type LoomTier = "read-only" | "workspace-write" | "expensive" | "irreversible";

/** The 8-key object every `loom <verb> --json` prints as its last stdout line. */
export interface LoomResult {
	verb?: string;
	status?: string;
	VERDICT?: string | null;
	pathspec?: string | null;
	card_path?: string | null;
	summary?: unknown;
	gate?: unknown;
	error?: string | null;
}

/**
 * Boolean (store_true) flags in the Python CLI. Everything else in a verb's
 * required/optional list is a string-valued flag. Kept in sync with cli.py's
 * `action="store_true"` arguments that appear in the verb manifest.
 */
const BOOLEAN_FLAGS = new Set(["apply", "send", "launch", "propose"]);

function resolveLoomPython(): string {
	const fromEnv = process.env.LOOM_PYTHON?.trim();
	return fromEnv && fromEnv.length > 0 ? fromEnv : "/Users/anub/Work/Loom/.venv/bin/python";
}

export interface RunResult {
	code: number;
	stdout: string;
	stderr: string;
}

/**
 * Spawn `${LOOM_PYTHON} -m <module> <...args>` and capture stdout/stderr.
 * Never rejects on a nonzero exit — the caller inspects `code`.
 */
export function runLoom(args: string[], signal?: AbortSignal): Promise<RunResult> {
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
			resolvePromise({ code: code ?? 0, stdout, stderr });
		});
	});
}

/**
 * The engine prints prose to stdout in non-JSON modes; in --json mode the
 * machine object is the LAST non-empty stdout line (prose may precede it).
 * Parse defensively from the bottom up.
 */
export function parseLoomJson(stdout: string): LoomResult | null {
	const lines = stdout
		.split("\n")
		.map((l) => l.trim())
		.filter((l) => l.length > 0);
	for (let i = lines.length - 1; i >= 0; i--) {
		const line = lines[i];
		if (!line.startsWith("{") && !line.startsWith("[")) continue;
		try {
			return JSON.parse(line) as LoomResult;
		} catch {
			// keep scanning upward
		}
	}
	return null;
}

/** Run `loom verbs --json` and parse the manifest. Throws on a missing/garbled manifest. */
export async function loadLoomManifest(signal?: AbortSignal): Promise<LoomVerb[]> {
	const { code, stdout, stderr } = await runLoom(["verbs", "--json"], signal);
	if (code !== 0) {
		throw new Error(`loom verbs --json exited ${code}: ${stderr.trim() || "(no stderr)"}`);
	}
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
		throw new Error("loom verbs --json did not return an array");
	}
	return parsed as LoomVerb[];
}

/**
 * Build a TypeBox params object for a verb from its required/optional flag
 * lists. Required flags are non-optional; optional flags are wrapped in
 * Type.Optional; the four store_true flags become booleans, the rest strings.
 */
export function buildVerbParams(verb: LoomVerb): TSchema {
	const props: Record<string, TSchema> = {};
	for (const name of verb.required) {
		props[name] = flagSchema(name, false);
	}
	for (const name of verb.optional) {
		props[name] = flagSchema(name, true);
	}
	return Type.Object(props, { additionalProperties: false });
}

function flagSchema(name: string, optional: boolean): TSchema {
	const isBool = BOOLEAN_FLAGS.has(name);
	const base: TSchema = isBool
		? Type.Boolean({ description: `--${name} flag` })
		: Type.String({ description: `value for --${name}` });
	return optional ? Type.Optional(base) : base;
}

/**
 * Translate validated tool params into CLI argv for `loom <verb> ... --json`.
 * Booleans emit a bare `--flag` only when true; strings emit `--flag value`.
 * Unknown/empty values are skipped.
 */
export function toCliArgs(verb: string, params: Record<string, unknown>): string[] {
	const args: string[] = [verb];
	for (const [key, value] of Object.entries(params ?? {})) {
		if (value === undefined || value === null) continue;
		if (BOOLEAN_FLAGS.has(key)) {
			if (value === true) args.push(`--${key}`);
			continue;
		}
		const str = String(value);
		if (str.length === 0) continue;
		args.push(`--${key}`, str);
	}
	args.push("--json");
	return args;
}

/** A compact, human-facing summary of a verb result for the model's content block. */
export function humanize(verb: string, result: LoomResult | null, raw: string): string {
	if (!result) {
		const trimmed = raw.trim();
		return trimmed.length > 0 ? trimmed.slice(0, 4000) : `loom ${verb} produced no parseable JSON.`;
	}
	const lines: string[] = [];
	const verdict = result.VERDICT ?? undefined;
	const status = result.status ?? undefined;
	const head = [
		`loom ${result.verb ?? verb}`,
		status ? `status=${status}` : undefined,
		verdict ? `VERDICT=${verdict}` : undefined,
	]
		.filter(Boolean)
		.join("  ");
	lines.push(head);
	if (result.pathspec) lines.push(`pathspec: ${result.pathspec}`);
	if (result.card_path) lines.push(`card: ${result.card_path}`);
	if (result.error) lines.push(`error: ${result.error}`);
	if (result.gate && typeof result.gate === "object") {
		const gate = result.gate as { decision?: string; reasons?: unknown };
		if (gate.decision) {
			const reasons = Array.isArray(gate.reasons) && gate.reasons.length > 0 ? ` (${gate.reasons.join("; ")})` : "";
			lines.push(`gate: ${gate.decision}${reasons}`);
		}
	}
	if (result.summary !== undefined && result.summary !== null) {
		const summaryText = summarize(result.summary);
		if (summaryText) lines.push(`summary: ${summaryText}`);
	}
	return lines.join("\n");
}

function summarize(summary: unknown): string {
	if (typeof summary === "string") return summary.slice(0, 1500);
	try {
		const json = JSON.stringify(summary);
		return json.length > 1500 ? `${json.slice(0, 1500)}…` : json;
	} catch {
		return "";
	}
}

/** Module-level verb registry, populated at registration, queried by the gate. */
const tierByToolName = new Map<string, LoomTier>();

export function tierOf(toolName: string): LoomTier | undefined {
	return tierByToolName.get(toolName);
}

/** `loom_<verb>` tool name for a verb (snake-case, schema-safe). */
export function toolNameFor(verb: string): string {
	return `loom_${verb}`;
}

/**
 * Register one Pi tool per Loom verb. Each tool's execute shells out to
 * `${LOOM_PYTHON} -m loom <verb> [...flags] --json`, parses the trailing JSON,
 * and returns { content: [humanized text], details: parsed }. A nonzero exit
 * throws with parsed.error / stderr. Gated (disable_model_invocation) verbs are
 * registered but kept out of the model-offered set (no promptSnippet) so the
 * model can only reach them via the explicit /loom-<verb> slash-command.
 *
 * Returns the verb manifest so callers (the extension entry, the gate) can wire
 * tier-based behavior from the same source of truth.
 */
export async function registerLoomTools(pi: ExtensionAPI): Promise<LoomVerb[]> {
	const verbs = await loadLoomManifest();

	for (const verb of verbs) {
		const toolName = toolNameFor(verb.name);
		tierByToolName.set(toolName, verb.tier);
		const params = buildVerbParams(verb);

		const definition: ToolDefinition<typeof params, LoomResult> = {
			name: toolName,
			label: `loom ${verb.name}`,
			description: `${verb.summary} [tier: ${verb.tier}] Runs the Loom Python engine; pass the verb's flags as fields.`,
			// Gated verbs are omitted from the "Available tools" prompt section so
			// the model does not auto-offer them; they remain reachable via /loom-<verb>.
			promptSnippet: verb.disable_model_invocation ? undefined : verb.summary,
			parameters: params,
			async execute(
				_toolCallId: string,
				args: Record<string, unknown>,
				signal: AbortSignal | undefined,
			): Promise<AgentToolResult<LoomResult>> {
				const argv = toCliArgs(verb.name, args ?? {});
				const { code, stdout, stderr } = await runLoom(argv, signal);
				const parsed = parseLoomJson(stdout);
				// Throw ONLY on a transport/setup failure with no structured result
				// (a crash, garbled output, or a missing engine — typically exit 2).
				// A well-formed result object is a DOMAIN outcome the agent must read
				// and compose on, even when it carries VERDICT=FAIL and exits 1 — e.g.
				// `loom doctor` reporting an unconfigured datastore. Returning it (vs.
				// throwing) preserves the `details` the gate-assert discipline needs.
				if (parsed === null) {
					const reason = stderr.trim() || stdout.trim();
					throw new Error(
						reason.length > 0
							? `loom ${verb.name} failed (exit ${code}): ${reason.slice(0, 2000)}`
							: `loom ${verb.name} exited ${code} with no parseable JSON output.`,
					);
				}
				return {
					content: [{ type: "text", text: humanize(verb.name, parsed, stdout) }],
					details: parsed,
				};
			},
		};

		pi.registerTool(definition);
	}

	return verbs;
}

/**
 * Layer-B approval gate (wired fully in the Persona/Approval phase). Registered
 * here so the tool_call seam is exercised from the Tools phase: irreversible
 * verbs require explicit confirmation, expensive verbs notify. Returns a
 * { block, reason } to veto a call.
 */
export function installApprovalGate(pi: ExtensionAPI): void {
	pi.on("tool_call", async (event: ToolCallEvent, ctx: ExtensionContext): Promise<ToolCallEventResult | void> => {
		const tier = tierOf(event.toolName);
		if (!tier) return; // not a loom verb tool — let Pi's own gates handle it
		if (tier === "irreversible") {
			if (!ctx.hasUI) {
				return { block: true, reason: `${event.toolName} is irreversible and needs interactive confirmation.` };
			}
			const choice = await ctx.ui.select(`Loom: ${event.toolName} is irreversible. Apply?`, ["yes", "no"]);
			if (choice !== "yes") {
				return { block: true, reason: "User declined an irreversible Loom verb." };
			}
		} else if (tier === "expensive") {
			ctx.ui.notify?.(`Loom: ${event.toolName} may be a long/expensive run.`, "warning");
		}
		return;
	});
}
