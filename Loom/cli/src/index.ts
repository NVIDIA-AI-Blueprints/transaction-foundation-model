import { existsSync, mkdirSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { parseArgs } from "node:util";
import { fileURLToPath } from "node:url";

import { LOOM_ASCII_LOGO, LOOM_TAGLINE } from "./branding/logo.js";
import { materializeAgentPersonas, syncBundledAssets } from "./bootstrap/sync.js";
import { runLoomUpdate } from "./update.js";
import { ensureLoomPackages } from "./pi/ensure-packages.js";
import { launchPiChat } from "./pi/launch.js";
import { normalizeLoomSettings } from "./pi/settings.js";
import { validatePiInstallation } from "./pi/runtime.js";

/** Default venv interpreter on the dev box; override with LOOM_PYTHON. */
const DEFAULT_LOOM_PYTHON = "/Users/anub/Work/transaction-foundation-model/Loom/.venv/bin/python";

/**
 * The Loom verbs. The agentic front door accepts `loom <verb> ...` and maps
 * it to the `/loom-<verb>` slash-command prompt. The authoritative source is the
 * verb manifest; this static set is only used for top-level routing before the child
 * boots, so a stale entry just falls through to a freeform prompt.
 *
 * Phase-0 slice: the new clean engine exposes three verbs (tokenize/ingest/baseline).
 */
const LOOM_VERBS = new Set([
	"tokenize",
	"ingest",
	"baseline",
]);

function resolveLoomPython(): string {
	const fromEnv = process.env.LOOM_PYTHON?.trim();
	return fromEnv && fromEnv.length > 0 ? fromEnv : DEFAULT_LOOM_PYTHON;
}

function loadPackageVersion(appRoot: string): string | undefined {
	try {
		const pkg = JSON.parse(readFileSync(resolve(appRoot, "package.json"), "utf8")) as { version?: string };
		return pkg.version;
	} catch {
		return undefined;
	}
}

// ── `loom --help` styling ─────────────────────────────────────────────────────
// ZKAI brand: magenta accent #e01396 (bold headings/logo) + light #ff3eb0 (commands)
// + dim descriptions. Color is suppressed when stdout isn't a TTY or NO_COLOR is set,
// so piped/redirected `--help` stays plain. No pi-tui dependency — just ANSI.
const HELP_COLOR = Boolean(process.stdout.isTTY) && !process.env.NO_COLOR;
const sgr = (code: string, text: string): string => (HELP_COLOR ? `\x1b[${code}m${text}\x1b[0m` : text);
const accent = (text: string): string => sgr("1;38;2;224;19;150", text); // bold magenta
const cmdc = (text: string): string => sgr("38;2;255;62;176", text); //     light pink
const dimc = (text: string): string => sgr("2", text); //                    dim

function printHelp(version: string | undefined): void {
	const out: string[] = [""];
	for (const line of LOOM_ASCII_LOGO) out.push(`  ${accent(line)}`);
	out.push(`  ${dimc(LOOM_TAGLINE)}${version ? dimc(`   v${version}`) : ""}`, "");

	const section = (title: string, rows: Array<[string, string]>): void => {
		out.push(accent(`◆ ${title}`));
		for (const [left, right] of rows) out.push(`  ${cmdc(left.padEnd(24))}${dimc(right)}`);
		out.push("");
	};

	section("Getting started", [
		["loom", "open the agent (interactive)"],
		['loom "profile my data"', "start with a goal, in plain English"],
		["loom doctor", "health-check the local stack"],
		["loom doctor --fix", "let Claude or Codex resolve any issues"],
		["loom update", "pull + rebuild the CLI + refresh the engine"],
	]);
	section("Verbs", [
		["understand", "ingest  eda  validate  viz  datasets  doctor"],
		["build", "features  pipeline  run"],
		["operate", "report  ops"],
		["gated", "deploy  train  collab  skillopt  notebook   (explicit confirm)"],
	]);
	section("Commands", [
		["loom <verb> [--flags]", "jump to a verb workflow (/loom-<verb>)"],
		["loom <goal in words>", "start with a one-shot goal"],
		["loom --help", "show this help"],
		["loom --version", "show version"],
	]);
	section("Tips", [
		["--model <provider/id>", "pick the LLM (or set ANTHROPIC_API_KEY etc.)"],
		["LOOM_PYTHON", "advanced — override the engine runtime path"],
	]);

	console.log(out.join("\n"));
}

/**
 * Map a top-level invocation to the prompt Pi should start with.
 *  - `loom <verb> ...`        -> "/loom-<verb> ..."
 *  - `loom chat <words>`      -> "<words>"
 *  - `loom <freeform words>`  -> "<freeform words>"
 *  - `loom`                   -> undefined (plain interactive)
 */
export function resolveInitialPrompt(command: string | undefined, rest: string[]): string | undefined {
	if (!command) return undefined;
	if (command === "chat") return rest.length > 0 ? rest.join(" ") : undefined;
	if (LOOM_VERBS.has(command)) return [`/loom-${command}`, ...rest].join(" ").trim();
	return [command, ...rest].join(" ").trim();
}

export async function main(): Promise<void> {
	const here = dirname(fileURLToPath(import.meta.url));
	const appRoot = resolve(here, ".."); // dist/ -> package root
	const version = loadPackageVersion(appRoot);

	const { values, positionals } = parseArgs({
		args: process.argv.slice(2),
		allowPositionals: true,
		// strict:false so engine/agent flags (e.g. --dataset) pass through as
		// positionals into the forwarded prompt instead of erroring here.
		strict: false,
		options: {
			help: { type: "boolean", short: "h" },
			version: { type: "boolean", short: "v" },
			cwd: { type: "string" },
			model: { type: "string" },
			thinking: { type: "string" },
			mode: { type: "string" },
			prompt: { type: "string", short: "p" },
			"session-dir": { type: "string" },
		},
	});

	if (values.help) {
		printHelp(version);
		return;
	}
	if (values.version) {
		console.log(version ?? "unknown");
		return;
	}

	const [command, ...rest] = positionals as string[];
	if (command === "update") {
		// Self-update: pull + rebuild the compiled CLI + refresh the engine. Handled
		// here (not as a /loom-<verb> prompt) since it manages the install itself.
		process.exitCode = runLoomUpdate(appRoot, resolveLoomPython());
		return;
	}
	if (command === "help") {
		printHelp(version);
		return;
	}

	const mode = values.mode as "text" | "json" | "rpc" | undefined;
	if (mode !== undefined && mode !== "text" && mode !== "json" && mode !== "rpc") {
		throw new Error("Unknown --mode. Use text, json, or rpc.");
	}

	const loomPython = resolveLoomPython();
	const loomHomeDir = resolve(appRoot, "home");
	const workingDir = resolve((values.cwd as string | undefined) ?? process.cwd());
	const sessionDir = resolve((values["session-dir"] as string | undefined) ?? resolve(loomHomeDir, "sessions"));

	mkdirSync(loomHomeDir, { recursive: true });
	mkdirSync(sessionDir, { recursive: true });
	// Hash-tracked sync of bundled branded assets (the LOOM theme) into the home,
	// preserving any user edits. Then force the
	// branding-related settings keys.
	syncBundledAssets(appRoot, loomHomeDir);
	// Materialize the Loom subagent personas, injecting the host-resolved path to
	// the loom-tools extension so delegated child agents can call the verb tools.
	materializeAgentPersonas(appRoot, loomHomeDir);
	normalizeLoomSettings(resolve(loomHomeDir, "settings.json"));
	// Ensure Loom's bundled Pi capability packages (MCP via pi-mcp-adapter,
	// subagents via pi-subagents) are installed in the home — idempotent, first
	// launch only.
	ensureLoomPackages(appRoot, loomHomeDir);

	const missing = validatePiInstallation(appRoot);
	if (missing.length > 0) {
		throw new Error(
			`Loom cannot start — missing required files:\n  ${missing.join("\n  ")}\n` +
				`Run \`npm install && npm run build\` in ${appRoot}.`,
		);
	}

	// Warn (don't fail) if the engine interpreter is absent — the agent can still
	// boot and report it cleanly; verb tools land in the Tools phase.
	if (!existsSync(loomPython)) {
		process.stderr.write(
			`loom: warning — LOOM_PYTHON not found at ${loomPython}. ` +
				`Set LOOM_PYTHON to your Loom venv's python. Verb tools will fail until then.\n`,
		);
	}

	const oneShot = values.prompt as string | undefined;
	const resolved = oneShot ?? resolveInitialPrompt(command, rest);
	const promptOptions = oneShot
		? { oneShotPrompt: resolved }
		: resolved
			? { initialPrompt: resolved }
			: {};

	await launchPiChat({
		appRoot,
		workingDir,
		sessionDir,
		loomHomeDir,
		loomPython,
		loomVersion: version,
		mode,
		thinkingLevel: values.thinking as string | undefined,
		explicitModelSpec: values.model as string | undefined,
		...promptOptions,
	});
}

main().catch((error) => {
	console.error(error instanceof Error ? error.message : String(error));
	process.exitCode = 1;
});
