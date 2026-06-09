/**
 * Child entry point. Spawned as: node <wrapper> <piMainPath> <...piArgs>
 *
 * Mirrors the installed Pi bin (dist/cli.js): set process.title, mark the
 * PI_CODING_AGENT env flag, silence warnings, then import Pi's main() from the
 * resolved dist/main.js and call it with the forwarded args. We brand the
 * process title as "loom" (the user's command) instead of the runtime default.
 */
import { pathToFileURL } from "node:url";

function handleStdinError(error: unknown): void {
	if (!error || typeof error !== "object") return;
	const code = "code" in error ? (error as { code?: string }).code : undefined;
	const syscall = "syscall" in error ? (error as { syscall?: string }).syscall : undefined;
	if ((code === "EIO" || code === "EBADF") && syscall === "read") {
		return;
	}
	throw error;
}

async function run(): Promise<void> {
	const [piMainPath, ...args] = process.argv.slice(2);
	if (!piMainPath) {
		throw new Error("Missing Pi main module path.");
	}

	process.title = "loom";
	process.env.PI_CODING_AGENT = "true";
	process.emitWarning = (() => undefined) as typeof process.emitWarning;
	process.stdin?.on?.("error", handleStdinError);

	const mod = (await import(pathToFileURL(piMainPath).href)) as {
		main?: (args: string[]) => Promise<void>;
	};
	if (typeof mod.main !== "function") {
		throw new Error(`Pi main module does not export main(): ${piMainPath}`);
	}

	await mod.main(args);
}

try {
	await run();
	process.exit(process.exitCode ?? 0);
} catch (error) {
	console.error(error instanceof Error ? error.message : String(error));
	process.exit(1);
}
