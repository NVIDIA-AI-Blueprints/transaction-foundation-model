import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { resolve } from "node:path";

/**
 * Loom's bundled Pi capability packages, installed into the branded Pi home on
 * first launch so `loom` ships these capabilities without the user running
 * `pi install` by hand. MCP (vendor/cloud data access + cloud ops) is the
 * off-the-shelf **pi-mcp-adapter** — one lazy proxy tool that reads `.mcp.json`,
 * not a custom bridge. Add a capability = add a line here, not a fork.
 */
const LOOM_PI_PACKAGES: { spec: string; marker: string }[] = [
	{ spec: "npm:pi-mcp-adapter", marker: "npm/node_modules/pi-mcp-adapter/package.json" },
	{ spec: "npm:pi-subagents", marker: "npm/node_modules/pi-subagents/package.json" },
	{ spec: "npm:pi-web-access", marker: "npm/node_modules/pi-web-access/package.json" },
];

/**
 * Idempotently ensure Loom's capability packages are installed in the Pi home.
 * Fast path: a present marker file means "already installed" (no per-launch
 * cost). Non-fatal on failure (e.g. offline) — the capability is simply
 * unavailable until `pi install <spec>` succeeds.
 */
export function ensureLoomPackages(appRoot: string, loomHomeDir: string): void {
	const piBin = resolve(appRoot, "node_modules", ".bin", "pi");
	if (!existsSync(piBin)) return; // Pi not installed yet; validatePiInstallation reports it.
	const env = { ...process.env, PI_CODING_AGENT_DIR: loomHomeDir, LOOM_CODING_AGENT_DIR: loomHomeDir };
	for (const { spec, marker } of LOOM_PI_PACKAGES) {
		if (existsSync(resolve(loomHomeDir, marker))) continue; // already installed
		try {
			execFileSync(piBin, ["install", spec], { env, stdio: "ignore" });
		} catch {
			// Non-fatal — the capability is unavailable until the install succeeds.
		}
	}
}
