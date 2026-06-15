"""The Loom Rich theme: palette, console factory, helpers, and the banner.

A lean, themed render library built on `rich <https://rich.readthedocs.io/>`_.
It provides:

* a fixed **color palette** (six named roles -- INK / STONE / ASH / SAGE / TEAL
  / ROSE), exposed both as a :class:`rich.theme.Theme` of named styles and as
  raw hex constants;
* :func:`get_console` -- the shared console factory, accepting an optional
  ``file`` so the UI is constructable over a ``StringIO`` buffer and unit-tests
  without a TTY;
* the one-line helpers :func:`info` / :func:`success` / :func:`warning` /
  :func:`error` / :func:`section`, plus :func:`panel` (a bordered box);
* :func:`banner` -- a branded ASCII ``LOOM`` block logo + the tagline + the
  version + the active providers (search / mlops / model-builder / model).

``rich`` is imported lazily where convenient so importing this module is cheap,
but the module-level import of :class:`~rich.console.Console` is fine because
this module is only ever reached through the (lazy) UI paths in
:mod:`loom.cli`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, TextIO

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rich.console import RenderableType

    from loom.config import LoomConfig

# ---------------------------------------------------------------------------
# Palette.
#
# A warm, muted, paper-and-sage scheme (rather than generic magenta): six named
# roles in a loom/textile-leaning weave -- warm thread (INK), neutral fibre
# (STONE/ASH), a healthy SAGE, a cool TEAL accent, and a ROSE error.
# ---------------------------------------------------------------------------

#: Warm thread / primary body text.
INK = "#d8c9a3"
#: Neutral fibre / muted secondary text.
STONE = "#9da9a0"
#: Dim structural fibre / info + panel body.
ASH = "#859289"
#: Darker structural fibre / borders + rules.
DARK_ASH = "#5c6a72"
#: Healthy growth / success + PASS.
SAGE = "#a7c080"
#: Cool weave accent / headers + sections + the logo.
TEAL = "#7fbbb3"
#: Warm warning thread.
AMBER = "#dbbc7f"
#: Error thread / FAIL + BLOCK.
ROSE = "#e67e80"

#: The Rich theme of named styles every Loom console is built with. Verb
#: renderers and helpers reference these names (e.g. ``[loom.success]``) so the
#: palette stays in exactly one place.
LOOM_THEME = Theme(
    {
        "loom.ink": INK,
        "loom.stone": STONE,
        "loom.ash": ASH,
        "loom.border": DARK_ASH,
        "loom.success": f"bold {SAGE}",
        "loom.warning": f"bold {AMBER}",
        "loom.error": f"bold {ROSE}",
        "loom.section": f"bold {TEAL}",
        "loom.title": f"bold {TEAL}",
        "loom.logo": f"bold {TEAL}",
        "loom.tagline": ASH,
        # Verdict / gate roles reused by render.py + gate.py so a PASS is always
        # the same green and a BLOCK always the same rose.
        "loom.pass": f"bold {SAGE}",
        "loom.review": f"bold {AMBER}",
        "loom.fail": f"bold {ROSE}",
        "loom.allow": f"bold {SAGE}",
        "loom.block": f"bold {ROSE}",
    }
)

#: The branded ASCII block logo for ``LOOM``.
LOOM_ASCII_LOGO = (
    " ██╗      ██████╗  ██████╗  ███╗   ███╗",
    " ██║     ██╔═══██╗██╔═══██╗ ████╗ ████║",
    " ██║     ██║   ██║██║   ██║ ██╔████╔██║",
    " ██║     ██║   ██║██║   ██║ ██║╚██╔╝██║",
    " ███████╗╚██████╔╝╚██████╔╝ ██║ ╚═╝ ██║",
    " ╚══════╝ ╚═════╝  ╚═════╝  ╚═╝     ╚═╝",
)

#: The product tagline shown under the logo.
TAGLINE = "an agentic CLI for data science"


def get_console(
    file: Optional[TextIO] = None,
    *,
    force_terminal: Optional[bool] = None,
    no_color: bool = False,
    width: Optional[int] = None,
) -> Console:
    """Build a Loom-themed :class:`rich.console.Console`.

    The single console factory the whole UI uses. Passing ``file`` (e.g. a
    :class:`io.StringIO`) makes the console write to a buffer instead of stdout,
    which is exactly how the unit tests render headlessly without a TTY.

    Args:
        file: Optional output stream. ``None`` uses Rich's default (stdout).
        force_terminal: Force terminal control codes on/off. ``None`` lets Rich
            auto-detect; tests pass an explicit value when they want styling in a
            buffer or plain text for assertions.
        no_color: When ``True``, strip color (plain text) -- the ``--no-ui`` /
            ``LOOM_NO_UI`` posture and CI-friendly output.
        width: Optional fixed render width (stabilizes tables/panels in tests).

    Returns:
        A console carrying :data:`LOOM_THEME`, writing to ``file`` when given.
    """
    return Console(
        file=file,
        theme=LOOM_THEME,
        force_terminal=force_terminal,
        no_color=no_color,
        width=width,
        highlight=False,
        soft_wrap=False,
    )


def info(console: Console, text: str) -> None:
    """Print a muted informational line (the ``printInfo`` analogue)."""
    console.print(Text(f"  {text}", style="loom.ash"))


def success(console: Console, text: str) -> None:
    """Print a success line with a check glyph (the ``printSuccess`` analogue)."""
    console.print(Text(f"\N{CHECK MARK} {text}", style="loom.success"))


def warning(console: Console, text: str) -> None:
    """Print a warning line with a warning glyph (the ``printWarning`` analogue)."""
    console.print(Text(f"\N{WARNING SIGN} {text}", style="loom.warning"))


def error(console: Console, text: str) -> None:
    """Print an error line with a cross glyph (the ``printError`` analogue)."""
    console.print(Text(f"\N{BALLOT X} {text}", style="loom.error"))


def section(console: Console, title: str) -> None:
    """Print a blank line then a diamond-prefixed section header.

    The ``printSection`` analogue.
    """
    console.print("")
    console.print(Text(f"\N{BLACK DIAMOND} {title}", style="loom.section"))


def panel(
    console: Console,
    title: str,
    body: "RenderableType",
    *,
    border_style: str = "loom.border",
) -> None:
    """Print a bordered, titled box (the ``printPanel`` analogue).

    Args:
        console: The console to print on.
        title: The panel title (rendered in the title/teal style).
        body: The panel body -- any Rich renderable (a string, a
            :class:`~rich.table.Table`, a :class:`~rich.text.Text`, ...). The
            verb renderers in :mod:`loom.ui.render` build the body and hand it
            here.
        border_style: Style name for the border (defaults to the muted Loom
            border); callers color it by verdict (``loom.pass`` / ``loom.fail``).
    """
    console.print(make_panel(title, body, border_style=border_style))


def make_panel(
    title: str,
    body: "RenderableType",
    *,
    border_style: str = "loom.border",
) -> Panel:
    """Return a Loom-styled :class:`rich.panel.Panel` (without printing it).

    The pure builder behind :func:`panel`; :mod:`loom.ui.render`'s pure verb
    renderers return one of these so a caller can compose or test it before any
    console exists.

    Args:
        title: The panel title.
        body: Any Rich renderable for the panel body.
        border_style: Style name for the border.

    Returns:
        A configured :class:`~rich.panel.Panel`.
    """
    return Panel(
        body,
        title=Text(title, style="loom.title"),
        title_align="left",
        border_style=border_style,
        padding=(0, 1),
    )


def _provider_lines(config: "LoomConfig") -> list[str]:
    """Build the active-provider summary lines shown under the banner tagline.

    Reads only the provider/model NAMES off the config (never any secret
    material): the search brain, the MLOps muscle, the model-builder, and the
    code/feedback model route. Tolerates a partial/duck-typed config so the
    banner never raises in a stripped or test environment.

    Args:
        config: The active :class:`~loom.config.LoomConfig` (or any object with
            the same attribute names; missing attributes degrade to ``"?"``).

    Returns:
        A list of aligned ``label : value`` strings.
    """
    def _get(name: str, default: str = "?") -> str:
        value = getattr(config, name, None)
        return str(value) if value not in (None, "") else default

    code = _get("code_provider")
    feedback = _get("feedback_provider")
    if code == feedback:
        model = code
    else:
        model = f"{code} / {feedback} (feedback)"

    return [
        f"search        : {_get('search_provider')}",
        f"mlops         : {_get('mlops_provider')}",
        f"model-builder : {_get('model_builder_provider')}",
        f"model         : {model}",
    ]


def banner(config: "LoomConfig", *, console: Optional[Console] = None) -> Console:
    """Print the branded launch banner and return the console used.

    The launch header plus a provider summary: the ASCII ``LOOM`` block logo,
    the tagline, the version, and the active providers. Pure-ish and
    headless-safe -- pass a console built over a
    ``StringIO`` (via :func:`get_console`) to render it into a buffer for tests.

    Args:
        config: The active config whose provider/model names are shown.
        console: An optional pre-built console (e.g. headless over a buffer). A
            new default console is created when omitted.

    Returns:
        The console the banner was printed on (so a caller/test can read the
        buffer or keep using it).
    """
    if console is None:
        console = get_console()

    try:
        from loom import __version__ as _version
    except Exception:  # noqa: BLE001 - version is cosmetic; never fail the banner
        _version = "?"

    console.print("")
    for line in LOOM_ASCII_LOGO:
        console.print(Text(line, style="loom.logo"))
    console.print(Text(f"  {TAGLINE}", style="loom.tagline"))
    console.print(Text(f"  v{_version}", style="loom.stone"))
    console.print("")
    for line in _provider_lines(config):
        console.print(Text(f"  {line}", style="loom.ash"))
    console.print("")
    return console


__all__ = [
    "INK",
    "STONE",
    "ASH",
    "DARK_ASH",
    "SAGE",
    "TEAL",
    "AMBER",
    "ROSE",
    "LOOM_THEME",
    "LOOM_ASCII_LOGO",
    "TAGLINE",
    "get_console",
    "info",
    "success",
    "warning",
    "error",
    "section",
    "panel",
    "make_panel",
    "banner",
]
