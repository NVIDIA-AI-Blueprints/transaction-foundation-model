import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { constants } from "node:os";

import { buildPiArgs, buildPiEnv, type PiRuntimeOptions, resolvePiPaths } from "./runtime.js";

export function exitCodeFromSignal(signal: NodeJS.Signals): number {
	const signalNumber = constants.signals[signal];
	return typeof signalNumber === "number" ? 128 + signalNumber : 1;
}

/**
 * Spawn the Pi coding-agent child via our wrapper, forwarding the resolved
 * dist/main.js path plus our flags. stdio is inherited so the TUI takes over
 * the terminal. Resolves once the child exits, propagating its exit code.
 */
export async function launchPiChat(options: PiRuntimeOptions): Promise<void> {
	const paths = resolvePiPaths(options.appRoot);
	const { piMainPath, piCliWrapperPath } = paths;

	if (!existsSync(piMainPath)) {
		throw new Error(
			`Pi main module not found: ${piMainPath}\n` +
				`Run \`npm install\` in ${options.appRoot} (expects @earendil-works/pi-coding-agent).`,
		);
	}
	if (!existsSync(piCliWrapperPath)) {
		throw new Error(`Loom Pi wrapper not found: ${piCliWrapperPath}\nRun \`npm run build\` in ${options.appRoot}.`);
	}

	// Clear the screen before the TUI takes over (skip in rpc transport).
	if (process.stdout.isTTY && options.mode !== "rpc") {
		process.stdout.write("\x1b[2J\x1b[3J\x1b[H");
	}
	if (options.preLaunchNotice) {
		process.stdout.write(`${options.preLaunchNotice}\n`);
	}

	const child = spawn(process.execPath, [piCliWrapperPath, piMainPath, ...buildPiArgs(options, paths)], {
		cwd: options.workingDir,
		stdio: "inherit",
		env: buildPiEnv(options, paths),
	});

	await new Promise<void>((resolvePromise, reject) => {
		child.on("error", reject);
		child.on("exit", (code, signal) => {
			if (signal) {
				console.error(`loom: the Pi child exited with signal ${signal}.`);
				process.exitCode = exitCodeFromSignal(signal);
				resolvePromise();
				return;
			}
			process.exitCode = code ?? 0;
			resolvePromise();
		});
	});
}
