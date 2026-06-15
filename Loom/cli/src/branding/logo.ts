/**
 * The LOOM banner (ANSI Shadow font), rendered into the REPL header on session
 * start. These strings carry no ANSI escapes — color is applied at render time via
 * the active theme's `fg`/`bold`. The glyphs are single-width box-drawing/block
 * characters, so visible width is just `.length` (header.ts centers on that).
 * Kept byte-identical to the banner shown in README.md.
 */
export const LOOM_ASCII_LOGO: string[] = [
	"██╗      ██████╗  ██████╗  ███╗   ███╗",
	"██║     ██╔═══██╗██╔═══██╗ ████╗ ████║",
	"██║     ██║   ██║██║   ██║ ██╔████╔██║",
	"██║     ██║   ██║██║   ██║ ██║╚██╔╝██║",
	"███████╗╚██████╔╝╚██████╔╝ ██║ ╚═╝ ██║",
	"╚══════╝ ╚═════╝  ╚═════╝  ╚═╝     ╚═╝",
];

export const LOOM_TAGLINE = "an agent harness for training foundation models";
