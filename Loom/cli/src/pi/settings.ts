import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";

/**
 * Force Loom branding/quiet-startup into the Pi home settings.json before the
 * child boots. We only overwrite the branding-related keys and preserve any
 * user-set model/provider/auth-adjacent keys already present.
 * (Branding details — theme palette, header banner — land in the Persona phase;
 *  this just guarantees the keys exist so the child reads them.)
 */
export function normalizeLoomSettings(settingsPath: string): void {
	let settings: Record<string, unknown> = {};
	if (existsSync(settingsPath)) {
		try {
			settings = JSON.parse(readFileSync(settingsPath, "utf8")) as Record<string, unknown>;
		} catch {
			settings = {};
		}
	}

	settings.theme = "loom";
	settings.quietStartup = true;
	settings.collapseChangelog = true;

	mkdirSync(dirname(settingsPath), { recursive: true });
	writeFileSync(settingsPath, JSON.stringify(settings, null, 2) + "\n", "utf8");
}
