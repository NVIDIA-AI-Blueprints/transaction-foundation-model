import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, relative, resolve } from "node:path";

/**
 * Hash-tracked asset sync into the Pi home.
 *
 * Loom ships canonical branded assets under `cli/assets/home/` (the theme, the
 * forced settings). The live Pi agent dir is `cli/home/` (PI_CODING_AGENT_DIR).
 * On launch we 3-way-merge the bundled assets into the live home so package
 * upgrades propagate new/updated assets, WITHOUT clobbering a file the user has
 * edited themselves:
 *   - new file in the bundle, absent in home  -> copied
 *   - bundle changed, home untouched since last sync -> updated
 *   - home edited by the user (hash differs from last applied) -> skipped
 *   - bundle no longer ships a file we previously synced, home untouched -> removed
 *
 * The per-file source/target hashes are tracked in a state file inside the home
 * so "untouched since last sync" is decidable. SYSTEM.md and prompts/ are passed
 * to Pi by absolute path (--system-prompt / --prompt-template) and are NOT synced
 * here — only the home-resident theme + settings are.
 */

const STATE_VERSION = 1 as const;

type BootstrapRecord = {
	lastAppliedSourceHash: string;
	lastAppliedTargetHash: string;
};

type BootstrapState = {
	version: typeof STATE_VERSION;
	files: Record<string, BootstrapRecord>;
};

export type BootstrapSyncResult = {
	copied: string[];
	updated: string[];
	skipped: string[];
	removed: string[];
};

function sha256(text: string): string {
	return createHash("sha256").update(text).digest("hex");
}

function readState(path: string): BootstrapState {
	if (!existsSync(path)) return { version: STATE_VERSION, files: {} };
	try {
		const parsed = JSON.parse(readFileSync(path, "utf8")) as Partial<BootstrapState>;
		return { version: STATE_VERSION, files: parsed.files && typeof parsed.files === "object" ? parsed.files : {} };
	} catch {
		return { version: STATE_VERSION, files: {} };
	}
}

function writeState(path: string, state: BootstrapState): void {
	mkdirSync(dirname(path), { recursive: true });
	writeFileSync(path, JSON.stringify(state, null, 2) + "\n", "utf8");
}

function listFiles(root: string): string[] {
	if (!existsSync(root)) return [];
	const files: string[] = [];
	for (const entry of readdirSync(root, { withFileTypes: true })) {
		const path = resolve(root, entry.name);
		if (entry.isDirectory()) {
			files.push(...listFiles(path));
		} else if (entry.isFile()) {
			files.push(path);
		}
	}
	return files.sort();
}

function removeEmptyParents(path: string, stopAt: string): void {
	let current = dirname(path);
	while (current.startsWith(stopAt) && current !== stopAt) {
		if (!existsSync(current)) {
			current = dirname(current);
			continue;
		}
		if (readdirSync(current).length > 0) return;
		rmSync(current, { recursive: true, force: true });
		current = dirname(current);
	}
}

function syncManagedFiles(
	sourceRoot: string,
	targetRoot: string,
	scope: string,
	state: BootstrapState,
	result: BootstrapSyncResult,
	transform?: (text: string) => string,
): void {
	// `transform` lets a bundled file be materialized (e.g. an absolute path
	// injected) before landing in the home. State tracks the SOURCE (template)
	// hash and the TARGET (materialized) hash separately, so we can tell apart a
	// template change, a host-specific re-materialization, and a user edit.
	const apply = transform ?? ((text: string) => text);
	const sourceKeys = new Set(listFiles(sourceRoot).map((p) => relative(sourceRoot, p)));

	// Remove files we previously synced that the bundle no longer ships, but only
	// if the user has not edited them since.
	for (const targetPath of listFiles(targetRoot)) {
		const key = relative(targetRoot, targetPath);
		if (sourceKeys.has(key)) continue;
		const scopedKey = `${scope}:${key}`;
		const previous = state.files[scopedKey];
		if (!previous) continue; // user-authored file we never managed — leave it
		const currentHash = sha256(readFileSync(targetPath, "utf8"));
		if (currentHash !== previous.lastAppliedTargetHash) {
			result.skipped.push(key);
			continue;
		}
		rmSync(targetPath, { force: true });
		removeEmptyParents(targetPath, targetRoot);
		delete state.files[scopedKey];
		result.removed.push(key);
	}

	for (const sourcePath of listFiles(sourceRoot)) {
		const key = relative(sourceRoot, sourcePath);
		const targetPath = resolve(targetRoot, key);
		const sourceText = readFileSync(sourcePath, "utf8");
		const sourceHash = sha256(sourceText);
		const targetText = apply(sourceText);
		const targetHash = sha256(targetText);
		const scopedKey = `${scope}:${key}`;
		const previous = state.files[scopedKey];

		mkdirSync(dirname(targetPath), { recursive: true });

		if (!existsSync(targetPath)) {
			writeFileSync(targetPath, targetText, "utf8");
			state.files[scopedKey] = { lastAppliedSourceHash: sourceHash, lastAppliedTargetHash: targetHash };
			result.copied.push(key);
			continue;
		}

		const currentText = readFileSync(targetPath, "utf8");
		const currentHash = sha256(currentText);

		if (currentHash === targetHash) {
			// Already the materialized content — record the baseline so future edits
			// are detectable (and a changed template/host path re-materializes below).
			state.files[scopedKey] = { lastAppliedSourceHash: sourceHash, lastAppliedTargetHash: targetHash };
			continue;
		}
		if (!previous || currentHash !== previous.lastAppliedTargetHash) {
			// First sight of a pre-existing file, or the user edited it — do not clobber.
			result.skipped.push(key);
			continue;
		}
		writeFileSync(targetPath, targetText, "utf8");
		state.files[scopedKey] = { lastAppliedSourceHash: sourceHash, lastAppliedTargetHash: targetHash };
		result.updated.push(key);
	}
}

/**
 * Sync the bundled branded assets (cli/assets/home) into the live Pi home
 * (cli/home). No-op when the bundle dir is absent.
 */
export function syncBundledAssets(appRoot: string, homeDir: string): BootstrapSyncResult {
	const result: BootstrapSyncResult = { copied: [], updated: [], skipped: [], removed: [] };
	const assetsRoot = resolve(appRoot, "assets", "home");
	if (!existsSync(assetsRoot)) return result;

	const statePath = resolve(homeDir, ".loom-bootstrap.json");
	const state = readState(statePath);

	syncManagedFiles(resolve(assetsRoot, "themes"), resolve(homeDir, "themes"), "themes", state, result);
	// settings.json is force-normalized separately (settings.ts) so it is NOT
	// managed here — we only seed the theme palette.

	writeState(statePath, state);
	return result;
}

/** Placeholder in the bundled persona templates, replaced with the resolved path. */
const LOOM_TOOLS_EXTENSION_PLACEHOLDER = "__LOOM_TOOLS_EXTENSION__";

/**
 * Materialize the bundled Loom subagent personas (cli/assets/home/agents) into the
 * live Pi home (home/agents), injecting the **absolute** path to the loom-tools
 * extension. A subagent runs as a child Pi process; pi-subagents loads a persona's
 * declared `extensions:` with `--no-extensions`, so the path must be absolute and
 * resolved per host (the repo can be cloned anywhere) — hence inject-at-launch
 * rather than a committed path. Hash-tracked: a user edit to a materialized persona
 * is preserved; a template or host-path change re-materializes. No-op when the
 * bundle dir is absent.
 */
export function materializeAgentPersonas(appRoot: string, homeDir: string): BootstrapSyncResult {
	const result: BootstrapSyncResult = { copied: [], updated: [], skipped: [], removed: [] };
	const sourceRoot = resolve(appRoot, "assets", "home", "agents");
	if (!existsSync(sourceRoot)) return result;

	const extensionPath = resolve(appRoot, "extensions", "loom-tools.ts");
	const statePath = resolve(homeDir, ".loom-bootstrap.json");
	const state = readState(statePath);

	syncManagedFiles(sourceRoot, resolve(homeDir, "agents"), "agents", state, result, (text) =>
		text.split(LOOM_TOOLS_EXTENSION_PLACEHOLDER).join(extensionPath),
	);

	writeState(statePath, state);
	return result;
}
