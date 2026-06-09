/**
 * The LOOM banner, rendered into the REPL header on session start. Plain ASCII —
 * coloring is applied at render time via the active theme's `fg`/`bold`, so these
 * strings carry no ANSI and their visible width is just `.length`.
 */
export const LOOM_ASCII_LOGO: string[] = [
	"  ██       ██████   ██████  ███    ███",
	"  ██      ██    ██ ██    ██ ████  ████",
	"  ██      ██    ██ ██    ██ ██ ████ ██",
	"  ██      ██    ██ ██    ██ ██  ██  ██",
	"  ███████  ██████   ██████  ██      ██",
];

export const LOOM_TAGLINE = "an agentic CLI for data science";
