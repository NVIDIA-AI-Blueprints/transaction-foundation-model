/**
 * The LOOM banner, rendered into the Pi REPL header on session_start
 * (cf. Feynman's logo.mjs + installFeynmanHeader). Plain ASCII — coloring is
 * applied at render time via the active theme's `fg`/`bold`, so these strings
 * carry no ANSI and their visible width is just `.length`.
 */
export const LOOM_ASCII_LOGO: string[] = [
	"  ██       ██████   ██████  ███    ███",
	"  ██      ██    ██ ██    ██ ████  ████",
	"  ██      ██    ██ ██    ██ ██ ████ ██",
	"  ██      ██    ██ ██    ██ ██  ██  ██",
	"  ███████  ██████   ██████  ██      ██",
];

export const LOOM_TAGLINE = "agentic data-science operator — verbs on the Pi harness";
