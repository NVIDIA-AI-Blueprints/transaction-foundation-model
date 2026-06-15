import { homedir } from "node:os";

import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";

import { LOOM_ASCII_LOGO, LOOM_TAGLINE } from "./logo.js";

/**
 * Install the LOOM banner as the REPL header on session start. The header is
 * a static-ish card — logo + version + active model + cwd + the registered verb
 * count — recomputed only on resize, so high-churn workflows don't redraw it.
 *
 * `theme.fg(name, text)` / `theme.bold(text)` apply ANSI; the logo strings are
 * plain ASCII so visible width is `.length`.
 */
export function installLoomHeader(pi: ExtensionAPI, ctx: ExtensionContext): void {
	if (!ctx.hasUI) return;

	const version = process.env.LOOM_VERSION?.trim() || "dev";
	const loomVerbToolCount = pi
		.getAllTools()
		.filter((t) => t.name.startsWith("loom_")).length;
	const commandCount = pi.getCommands().length;

	ctx.ui.setHeader((_tui, theme) => ({
		// Static banner — nothing cached, so invalidation is a no-op.
		invalidate(): void {},
		render(width: number): string[] {
			const maxW = Math.max(width - 2, 1);
			const cardW = Math.min(maxW, 96);
			const innerW = cardW - 2;
			const contentW = innerW - 2;
			const outerPad = " ".repeat(Math.max(0, Math.floor((width - cardW) / 2)));
			const lines: string[] = [];
			const push = (line: string) => lines.push(`${outerPad}${line}`);

			const border = (ch: string) => theme.fg("borderMuted", ch);
			const padRight = (text: string, w: number) => {
				const gap = Math.max(0, w - visibleLen(text));
				return `${text}${" ".repeat(gap)}`;
			};
			const row = (content: string): string => `${border("│")} ${padRight(content, contentW)} ${border("│")}`;
			const top = (): string => border(`╭${"─".repeat(innerW)}╮`);
			const bottom = (): string => border(`╰${"─".repeat(innerW)}╯`);

			push(top());

			if (cardW >= 44) {
				const maxLogoW = Math.max(...LOOM_ASCII_LOGO.map((l) => l.length));
				const logoPad = " ".repeat(Math.max(0, Math.floor((contentW - maxLogoW) / 2)));
				for (const logoLine of LOOM_ASCII_LOGO) {
					push(row(theme.fg("accent", theme.bold(`${logoPad}${truncate(logoLine, contentW)}`))));
				}
			} else {
				push(row(theme.fg("accent", theme.bold("LOOM"))));
			}

			push(row(theme.fg("dim", truncate(`  ${LOOM_TAGLINE}`, contentW))));
			push(row(""));

			const modelLabel = ctx.model ? `${ctx.model.provider}/${ctx.model.id}` : "not set (use /model or /login)";
			const dirLabel = formatPath(ctx.cwd);

			const labelW = 13; // widest label is "model-builder"
			const labeled = (label: string, value: string) =>
				row(`${theme.fg("dim", padRight(label, labelW + 1))}${theme.fg("text", truncate(value, contentW - labelW - 1))}`);

			// The active provider stack (the three swappable ports + the agent model),
			// from env with the engine's documented defaults. Cf. Feynman's header,
			// which surfaces the active model/context up front.
			const env = process.env;
			push(labeled("search", env.LOOM_SEARCH_PROVIDER?.trim() || "aide"));
			push(labeled("mlops", env.LOOM_MLOPS_PROVIDER?.trim() || "metaflow"));
			push(labeled("model-builder", env.LOOM_MODEL_BUILDER_PROVIDER?.trim() || "nemo"));
			push(labeled("model", modelLabel));
			push(labeled("cwd", dirLabel));
			push(labeled("verbs", `${loomVerbToolCount} data-science verbs · ${commandCount} commands`));
			push(labeled("version", `v${version}`));
			push(bottom());
			push(theme.fg("dim", "  type /help for the verbs · /plan to plan · /exit to quit"));

			return lines;
		},
	}));
}

function visibleLen(text: string): number {
	// strip the SGR/ANSI escapes the theme injects so width math stays correct.
	// eslint-disable-next-line no-control-regex
	return text.replace(/\[[0-9;]*m/g, "").length;
}

function truncate(text: string, maxVisible: number): string {
	if (visibleLen(text) <= maxVisible) return text;
	if (maxVisible <= 1) return text.slice(0, Math.max(0, maxVisible));
	return `${text.slice(0, Math.max(0, maxVisible - 1))}…`;
}

function formatPath(p: string): string {
	const home = homedir();
	return p.startsWith(home) ? `~${p.slice(home.length)}` : p;
}
