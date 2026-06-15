import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { resolve } from "node:path";

/**
 * `loom update` — run the whole update cycle so what you run matches the latest
 * source.
 *
 * A bare `git pull` is NOT enough: the running `loom` executes the COMPILED
 * `cli/dist`, which is gitignored build output, so a pull refreshes the source but
 * the banner / verbs / tools keep running the old build until it's rebuilt. This:
 *   1. `git pull --ff-only`   (the repo root, one level above the cli package)
 *   2. `npm install` + `npm run build`  (rebuild cli/dist — the part pull can't touch)
 *   3. `pip install -e .`     (refresh the engine's deps; editable so code is live)
 *   4. `loom migrate`         (LLM-guided: reconcile the machine to the new release's
 *                              desired state — re-run cluster setup, env, port-forwards
 *                              as a release needs. A no-op when nothing is newer, so it
 *                              only engages when there is a real migration.)
 *
 * The mechanical steps (1–3) are deterministic; step 4 is where an LLM (claude, else
 * codex) reasons about per-release migration from a `migrations/<version>.yaml`. It is
 * NON-FATAL: the code is already updated, so a migration hiccup never fails the update.
 *
 * The rebuild overwrites the very `dist` this process is running from, which is
 * fine — Node already loaded it; the NEXT `loom` invocation uses the fresh build.
 *
 * @param appRoot   the cli package root (`<repo>/cli`)
 * @param loomPython the resolved engine interpreter (skipped if it doesn't exist)
 * @returns process exit code (0 ok, 1 on the first failing step)
 */
export function runLoomUpdate(appRoot: string, loomPython: string): number {
	const repoRoot = resolve(appRoot, ".."); // <repo>/cli -> <repo>

	const step = (label: string, cmd: string, args: string[]): void => {
		console.log(`\n--> ${label}`);
		execFileSync(cmd, args, { cwd: repoRoot, stdio: "inherit" });
	};

	try {
		step("git pull --ff-only", "git", ["-C", repoRoot, "pull", "--ff-only"]);
		// Rebuild the compiled CLI — the running `loom` is cli/dist (gitignored).
		step("npm install (the loom CLI)", "npm", ["--prefix", appRoot, "install"]);
		step("npm run build (the loom CLI)", "npm", ["--prefix", appRoot, "run", "build"]);
		if (existsSync(loomPython)) {
			step("pip install -e . (the engine)", loomPython, ["-m", "pip", "install", "-e", repoRoot]);
		} else {
			console.log(`\n(!) skipped the engine refresh — LOOM_PYTHON not found at ${loomPython}`);
		}
		console.log("\nUpdated. Re-run `loom` — the rebuilt CLI is live (npm link points at cli/dist).");

		// Step 4 — the LLM-guided migration advisor. NON-FATAL and separate from the
		// mechanical update above: the engine (which hosts `loom migrate`) must exist,
		// and a migration hiccup must not fail an otherwise-successful code update. It
		// no-ops silently when no manifest is newer than the installed version.
		if (existsSync(loomPython)) {
			try {
				console.log("\n--> loom migrate (reconcile to the new release's desired state)");
				execFileSync(loomPython, ["-m", "loom", "migrate"], { cwd: repoRoot, stdio: "inherit" });
			} catch (mErr) {
				// A non-zero exit here is usually a benign end to the interactive
				// migration session (e.g. you quit the assistant), not a real failure —
				// the code update already succeeded. Report it neutrally.
				const mMsg = mErr instanceof Error ? mErr.message : String(mErr);
				console.error(`\n(i) migration advisor exited (${mMsg}).`);
				console.error("The code update succeeded; re-run `loom migrate` anytime for guided migration.");
			}
		}
		return 0;
	} catch (err) {
		const msg = err instanceof Error ? err.message : String(err);
		console.error(`\nUpdate failed: ${msg}`);
		console.error("Fix the error above and re-run `loom update` (or, by hand: `git pull && cd cli && npm run build`).");
		return 1;
	}
}
