import { existsSync, readFileSync } from "node:fs";
import { delimiter, isAbsolute, resolve } from "node:path";
import { pathToFileURL } from "node:url";

/**
 * Runtime path/arg/env resolution for spawning the Pi coding-agent as a child
 * process. Mirrors the Feynman mechanism (feynman-src/src/pi/runtime.ts) but
 * targets Loom's branded home dir and resolves the Python engine via LOOM_PYTHON.
 */

export type PiRuntimeOptions = {
	/** Root of the @loom/cli package (the dir containing package.json). */
	appRoot: string;
	/** cwd the Pi child runs in (the user's project dir). */
	workingDir: string;
	/** Where Pi persists sessions. */
	sessionDir: string;
	/** The branded Pi agent home (PI_CODING_AGENT_DIR target). */
	loomHomeDir: string;
	/** Resolved Python interpreter for the engine (python -m loom ...). */
	loomPython: string;
	loomVersion?: string;
	mode?: "text" | "json" | "rpc";
	thinkingLevel?: string;
	explicitModelSpec?: string;
	oneShotPrompt?: string;
	initialPrompt?: string;
	preLaunchNotice?: string;
};

/**
 * Pi is published under two scopes that are the same project:
 *   @earendil-works/pi-coding-agent (preferred, pinned)
 *   @mariozechner/pi-coding-agent   (legacy fallback)
 * Resolve the first whose dist/main.js exists.
 */
const PI_PACKAGE_SCOPES = ["@earendil-works/pi-coding-agent", "@mariozechner/pi-coding-agent"] as const;

export function resolvePiPaths(appRoot: string) {
	const nodeModules = resolve(appRoot, "node_modules");

	let piPackageRoot: string | undefined;
	for (const scoped of PI_PACKAGE_SCOPES) {
		const candidate = resolve(nodeModules, ...scoped.split("/"));
		if (existsSync(resolve(candidate, "dist", "main.js"))) {
			piPackageRoot = candidate;
			break;
		}
	}
	// Fall back to the preferred scope path even if missing, so error messages
	// from validatePiInstallation point at the expected location.
	if (!piPackageRoot) {
		piPackageRoot = resolve(nodeModules, ...PI_PACKAGE_SCOPES[0].split("/"));
	}

	return {
		piPackageRoot,
		piCliPath: resolve(piPackageRoot, "dist", "cli.js"),
		piMainPath: resolve(piPackageRoot, "dist", "main.js"),
		piCliWrapperPath: resolve(appRoot, "dist", "pi", "pi-cli-wrapper.js"),
		extensionPath: resolve(appRoot, "extensions", "loom-tools.ts"),
		promptTemplatePath: resolve(appRoot, "prompts"),
		systemPromptPath: resolve(appRoot, "home", "SYSTEM.md"),
		themePath: resolve(appRoot, "home", "themes", "loom.json"),
		nodeModulesBinPath: resolve(nodeModules, ".bin"),
	};
}

export type PiPaths = ReturnType<typeof resolvePiPaths>;

export function toNodeImportSpecifier(modulePath: string): string {
	return isAbsolute(modulePath) ? pathToFileURL(modulePath).href : modulePath;
}

export function validatePiInstallation(appRoot: string): string[] {
	const paths = resolvePiPaths(appRoot);
	const missing: string[] = [];
	if (!existsSync(paths.piMainPath)) missing.push(paths.piMainPath);
	if (!existsSync(paths.piCliWrapperPath)) missing.push(paths.piCliWrapperPath);
	return missing;
}

export function buildPiArgs(options: PiRuntimeOptions, paths: PiPaths = resolvePiPaths(options.appRoot)): string[] {
	const args: string[] = ["--session-dir", options.sessionDir];

	// The loom-tools extension is loaded directly from source (.ts); Pi loads
	// extension files via its own loader. Only wire it if it exists (it lands in
	// the Tools phase). Until then, Pi boots with default tools.
	if (existsSync(paths.extensionPath)) {
		args.push("--extension", paths.extensionPath);
	}

	if (existsSync(paths.promptTemplatePath)) {
		args.push("--prompt-template", paths.promptTemplatePath);
	}

	// Full system-prompt replacement (the Loom persona). Passed as text, like Feynman.
	if (existsSync(paths.systemPromptPath)) {
		args.push("--system-prompt", readFileSync(paths.systemPromptPath, "utf8"));
	}

	// Ship the LOOM theme file so `theme: "loom"` in settings.json resolves.
	if (existsSync(paths.themePath)) {
		args.push("--theme", paths.themePath);
	}

	if (options.mode) {
		args.push("--mode", options.mode);
	}
	if (options.explicitModelSpec) {
		args.push("--model", options.explicitModelSpec);
	}
	if (options.thinkingLevel) {
		args.push("--thinking", options.thinkingLevel);
	}
	if (options.oneShotPrompt) {
		args.push("-p", options.oneShotPrompt);
	} else if (options.initialPrompt) {
		args.push(options.initialPrompt);
	}

	return args;
}

export function buildPiEnv(
	options: PiRuntimeOptions,
	paths: PiPaths = resolvePiPaths(options.appRoot),
): NodeJS.ProcessEnv {
	const currentPath = process.env.PATH ?? "";
	const binPath = paths.nodeModulesBinPath;

	return {
		...process.env,
		PATH: `${binPath}${delimiter}${currentPath}`,
		// Redirect Pi's config/auth/models/settings into the branded Loom home.
		// Patched Pi twins read FEYNMAN_*/LOOM_*; upstream Pi reads PI_CODING_AGENT_DIR.
		PI_CODING_AGENT_DIR: options.loomHomeDir,
		LOOM_CODING_AGENT_DIR: options.loomHomeDir,
		// Engine interpreter for `python -m loom <verb> --json`, consumed by the
		// loom-tools extension (Tools phase). Pinned here so the child inherits it.
		LOOM_PYTHON: options.loomPython,
		LOOM_VERSION: options.loomVersion ?? "",
		LOOM_SESSION_DIR: options.sessionDir,
		PI_SKIP_VERSION_CHECK: process.env.PI_SKIP_VERSION_CHECK ?? "1",
		PI_HARDWARE_CURSOR: process.env.PI_HARDWARE_CURSOR ?? "1",
	};
}
