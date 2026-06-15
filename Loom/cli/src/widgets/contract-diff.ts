/**
 * The contract-diff result card — Loom's first widget (PI.md §B.2.2, §B.11 step 4).
 *
 * Wired as `ToolDefinition.renderResult` on each verb tool (with `renderShell:"self"`
 * so Loom draws its own framing). It turns a `VerbResult` envelope into a `Box` card:
 *
 *  - With diagnostics: a colored header line (`verb · STATUS · VERDICT`), a dynamic
 *    border, then for each diagnostic a `[contract/severity] message` row, an
 *    indented `fix:` line when present, and — when a diagnostic carries a textual
 *    delta (an explicit `diff`/`delta`/`diff_text` string, or C1 id-range
 *    `missing`/`out_of_range` lists) — a `renderDiff(deltaText)` block. ⚠ `renderDiff`
 *    returns a STRING, so it is wrapped in a `Text` (it is NOT itself a Component).
 *  - Without diagnostics: a compact card of `summary` + `outputs`.
 *  - Tolerates an empty/missing/garbled `details` (blocked/aborted/missing-tool
 *    results carry `details:{}`): falls back to the tool's text `content`.
 *
 * The card is the value-add over Claude Code's raw-JSON dump (PI.md §B.2.2/§B.2.5):
 * the agent never sees a stack trace, it sees a named-contract diff.
 *
 * Signature LOCK: the parameter/return types are derived from Pi's own
 * `ToolDefinition.renderResult` field type (indexed access) so this stub matches
 * Pi 0.79.0 EXACTLY — `Component`/`ToolRenderContext` are never named here.
 *
 * Resolution note (load-bearing): `Box`/`Text` are runtime VALUES that live only in
 * `@earendil-works/pi-tui`, which is installed nested under `pi-coding-agent` and is
 * NOT hoisted to `cli/node_modules` — so a bare `@earendil-works/pi-tui` specifier
 * resolves at NEITHER tsc-time (from src/) NOR runtime (from dist/). Both this file
 * and its compiled `dist/widgets/contract-diff.js` sit two levels under `cli/`, so a
 * single relative path into the nested package resolves identically in both passes.
 * `renderDiff`/`DynamicBorder` come from the top-level-resolvable `pi-coding-agent`.
 */
import { DynamicBorder, renderDiff } from "@earendil-works/pi-coding-agent";
import type { ToolDefinition } from "@earendil-works/pi-coding-agent";
import type { TSchema } from "@earendil-works/pi-ai";
// pi-tui is nested under pi-coding-agent (not hoisted); this relative path resolves
// from both src/widgets/ (tsc) and dist/widgets/ (runtime). See header note.
import {
	Box,
	Text,
} from "../../node_modules/@earendil-works/pi-coding-agent/node_modules/@earendil-works/pi-tui/dist/index.js";

import type { Diagnostic, LoomSeverity, LoomStatus, LoomVerdict, VerbResult } from "../types.js";

/** Pi's `renderResult` field type, specialized to a `VerbResult` details payload. */
type RenderResult = NonNullable<ToolDefinition<TSchema, VerbResult>["renderResult"]>;

/** The four positional params of `renderResult`: [result, options, theme, ctx]. */
type RenderResultParams = Parameters<RenderResult>;
/** The component `renderResult` returns. */
type RenderResultComponent = ReturnType<RenderResult>;

/** The active theme passed to `renderResult` (Pi's `Theme`). */
type Theme = RenderResultParams[2];
/** The colour names `theme.fg(name, text)` accepts. */
type ThemeColor = Parameters<Theme["fg"]>[0];

/** Map a top-level call status to a theme colour for the header. */
function statusColor(status: LoomStatus): ThemeColor {
	if (status === "OK" || status === "PLAN") return "success";
	if (status === "FAIL") return "error";
	// the whole REFUSED_* family is a structural refusal, not a crash → warn.
	return "warning";
}

/** Map the machine verdict to a theme colour. */
function verdictColor(verdict: LoomVerdict): ThemeColor {
	if (verdict === "PASS") return "success";
	if (verdict === "FAIL") return "error";
	return "warning"; // REVIEW | INCOMPLETE
}

/** Map a diagnostic severity to a theme colour. */
function severityColor(severity: LoomSeverity): ThemeColor {
	if (severity === "error") return "error";
	if (severity === "warning") return "warning";
	return "muted"; // info
}

/**
 * If a diagnostic carries a textual delta, return a unified-diff-style string for
 * `renderDiff`; otherwise null. Handles two shapes:
 *   1. an explicit string field (`diff` | `diff_text` | `delta`);
 *   2. C1 id-range lists (`missing` / `out_of_range`) — rendered as `-`/`+` rows.
 */
function deltaTextFor(data: Record<string, unknown> | undefined): string | null {
	if (!data || typeof data !== "object") return null;

	for (const key of ["diff", "diff_text", "delta"] as const) {
		const v = data[key];
		if (typeof v === "string" && v.length > 0) return v;
	}

	const fmt = (v: unknown): string => (Array.isArray(v) ? v.join(", ") : String(v));
	const missing = data.missing;
	const outOfRange = data.out_of_range;
	const hasMissing = Array.isArray(missing) && missing.length > 0;
	const hasExtra = Array.isArray(outOfRange) && outOfRange.length > 0;
	if (!hasMissing && !hasExtra) return null;

	const lines: string[] = [];
	if (hasMissing) lines.push(`-missing ids: ${fmt(missing)}`);
	if (hasExtra) lines.push(`+out-of-range ids: ${fmt(outOfRange)}`);
	return lines.join("\n");
}

/** Best-effort text fallback from the tool's returned `content` (for empty details). */
function fallbackText(result: RenderResultParams[0]): string {
	const first = result.content?.[0];
	if (first && first.type === "text" && typeof first.text === "string") return first.text;
	return "(no result)";
}

/** True when `result.details` is a usable VerbResult (has at least a verb + status). */
function hasEnvelope(details: unknown): details is VerbResult {
	return (
		!!details &&
		typeof details === "object" &&
		typeof (details as VerbResult).verb === "string" &&
		typeof (details as VerbResult).status === "string"
	);
}

/** One diagnostic → its Text rows (header row, optional fix row, optional diff block). */
function diagnosticLines(diag: Diagnostic, theme: Theme): Text[] {
	const rows: Text[] = [];
	const tag = theme.fg(severityColor(diag.severity), `[${diag.contract}/${diag.severity}]`);
	rows.push(new Text(`${tag} ${diag.message}`, 0, 0));

	if (diag.fix) {
		rows.push(new Text(`  ${theme.fg("dim", `fix: ${diag.fix}`)}`, 0, 0));
	}

	const delta = deltaTextFor(diag.data);
	if (delta) {
		// renderDiff returns a STRING — wrap it in a Text, never use it as a Component.
		rows.push(new Text(renderDiff(delta), 2, 0));
	}
	return rows;
}

/**
 * Render a verb's `VerbResult` as a contract-diff card. Same shape as a Pi
 * `ToolDefinition.renderResult` (`[result, options, theme, ctx]`). `result.details`
 * is the VerbResult; tolerates an empty/missing/garbled `details` (falls back to the
 * tool's text `content`).
 */
export function renderContractResult(
	result: RenderResultParams[0],
	options: RenderResultParams[1],
	theme: RenderResultParams[2],
	ctx: RenderResultParams[3],
): RenderResultComponent {
	void options;
	void ctx;

	const card = new Box(1, 0);
	const details = result.details;

	// Fallback: no usable envelope (blocked/aborted/missing-tool results).
	if (!hasEnvelope(details)) {
		card.addChild(new Text(theme.fg("muted", fallbackText(result)), 0, 0));
		return card;
	}

	const env: VerbResult = details;

	// Header: verb · STATUS · VERDICT, each coloured by outcome.
	const header =
		theme.fg("accent", env.verb) +
		theme.fg("dim", " · ") +
		theme.fg(statusColor(env.status), env.status) +
		theme.fg("dim", " · ") +
		theme.fg(verdictColor(env.verdict), env.verdict);
	card.addChild(new Text(theme.bold(header), 0, 0));

	const diagnostics = Array.isArray(env.diagnostics) ? env.diagnostics : [];

	if (diagnostics.length > 0) {
		// A border separates the header from the named-contract diff body.
		card.addChild(new DynamicBorder((s: string) => theme.fg("borderMuted", s)));
		for (const diag of diagnostics) {
			for (const row of diagnosticLines(diag, theme)) card.addChild(row);
		}
		return card;
	}

	// No diagnostics: a compact summary + outputs card.
	if (env.summary) {
		card.addChild(new Text(theme.fg("text", env.summary), 0, 0));
	}
	const outputs = Array.isArray(env.outputs) ? env.outputs : [];
	if (outputs.length > 0) {
		card.addChild(new Text(theme.fg("success", `outputs: ${outputs.join(", ")}`), 0, 0));
	}
	return card;
}
