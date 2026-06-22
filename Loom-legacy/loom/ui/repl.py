"""The Loom interactive REPL: a thin loop over the existing CLI verbs.

``loom`` with no subcommand drops into a branded interactive shell. The crucial
discipline is that **the REPL owns no engine logic** -- it builds the SAME
:func:`loom.cli._build_parser` parser once, tokenizes each line, parses it with
that parser, and invokes the SAME ``args.func(args)`` handler the one-shot
subcommands use. The render layer (:mod:`loom.ui.render`) makes the output
beautiful, the approval layer (:mod:`loom.ui.gate`) gates the costly verbs
interactively, and a Rich spinner wraps a running verb.

Headless-safe by construction: the console is built over an injectable
``file`` (a ``StringIO`` in tests) via :func:`loom.ui.theme.get_console`, the
prompt input is injectable (``read_line``), and the gate confirm is injectable,
so the whole loop unit-tests without a TTY.

Everything heavy (``rich`` / ``prompt_toolkit``) is imported lazily from
:mod:`loom.cli` so a stripped environment still runs the one-shot subcommands;
this module is only ever reached through the (lazy) UI path in ``loom.cli.main``.
"""

from __future__ import annotations

import shlex
from typing import TYPE_CHECKING, Callable, Optional, Sequence

from loom.ui import gate as gate_mod
from loom.ui import theme

if TYPE_CHECKING:  # pragma: no cover - typing only
    import argparse

    from rich.console import Console

    from loom.config import LoomConfig


# ---------------------------------------------------------------------------
# The verb catalog: which verbs exist, and how each gates / renders / streams.
#
# These tables are the ONLY Loom-specific knowledge the REPL adds on top of the
# shared parser; everything else is delegated to the handler the parser already
# wired. The verb set is derived from the live parser at runtime (see
# :func:`verb_names`) so it can never drift from ``loom.cli``.
# ---------------------------------------------------------------------------

#: The meta (REPL-only) slash commands, shown in help + offered by the completer.
META_COMMANDS = ("/help", "/status", "/doctor", "/clear", "/exit", "/quit")

#: Verbs that need an LLM credential to do anything (the search "brain" calls a
#: model to write solutions). A missing key yields an actionable line, never a
#: traceback. Every other verb works WITHOUT a key (read-only / lifecycle).
_LLM_VERBS = frozenset({"run", "pipeline"})

#: Verbs whose costly/irreversible action gates interactively BEFORE the handler
#: runs. Maps the verb -> (the flag that arms the real action, the gate tier, a
#: short action label). When the flag is absent the verb is its safe default
#: (plan/build-only) and does not gate here; the handler stays in charge.
_GATED_VERBS = {
    "deploy": ("apply", gate_mod.IRREVERSIBLE, "deploy to the external registry"),
    "collab": ("send", gate_mod.IRREVERSIBLE, "send the bundle off-box"),
    "train": ("launch", gate_mod.EXPENSIVE, "launch the heavy GPU training run"),
}

#: Verbs that run a (potentially long) search/flow -> wrap in a spinner; ``run``
#: and ``pipeline`` additionally drive the AIDE search.
_SPINNER_VERBS = frozenset(
    {
        "run",
        "pipeline",
        "eda",
        "validate",
        "features",
        "deploy",
        "train",
        "ops",
        "report",
        "viz",
        "collab",
        "ingest",
        "skillopt",
    }
)


def verb_names(parser: "argparse.ArgumentParser") -> list[str]:
    """Return the top-level subcommand (verb) names of the shared parser.

    Read straight off the parser's subparsers action so the REPL's verb set is
    exactly the one-shot CLI's verb set -- it cannot drift from ``loom.cli``.

    Args:
        parser: The parser built by :func:`loom.cli._build_parser`.

    Returns:
        The sorted list of verb tokens (e.g. ``["collab", "datasets", ...]``).
    """
    import argparse

    for action in parser._actions:  # noqa: SLF001 - the public API exposes no getter
        if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
            return sorted(action.choices.keys())
    return []


# ---------------------------------------------------------------------------
# Line routing (pure-ish: no prompt, no loop -- just "what does this line mean").
# ---------------------------------------------------------------------------


def tokenize(line: str) -> list[str]:
    """Split a REPL line into argv tokens, stripping a leading ``/``.

    ``/eda --dataset IngestDataset/1 --target t`` and
    ``eda --dataset IngestDataset/1 --target t`` both tokenize to the argv the
    shared parser expects: ``["eda", "--dataset", "IngestDataset/1",
    "--target", "t"]``. Uses :func:`shlex.split` so quoted free-text args (a
    ``--goal "predict churn"``) survive.

    Args:
        line: The raw input line.

    Returns:
        The argv token list (possibly empty for a blank line).
    """
    text = line.strip()
    if not text:
        return []
    if text.startswith("/"):
        text = text[1:]
    try:
        return shlex.split(text)
    except ValueError:
        # Unbalanced quotes etc. -- fall back to a naive split so the dispatcher
        # can still surface a parser/usage error rather than crash the REPL.
        return text.split()


def _hint_body(verbs: Sequence[str]) -> str:
    """Build the natural-language-line hint body listing the verbs + meta cmds."""
    cols = ", ".join(f"/{v}" for v in verbs)
    metas = ", ".join(META_COMMANDS)
    return (
        "Loom drives data-science verbs, not free-form chat. Type a verb "
        "(optionally with a leading /):\n\n"
        f"{cols}\n\nMeta: {metas}\n\n"
        "e.g.  /eda --dataset IngestDataset/1 --target label\n"
        "      /datasets        (list ingested data objects)\n"
        "      /help            (full command help)"
    )


# ---------------------------------------------------------------------------
# Keyless preflight for the LLM verbs (actionable line, never a traceback).
# ---------------------------------------------------------------------------


def llm_verb_keyless_message(verb: str, config: "LoomConfig") -> Optional[str]:
    """Return an actionable message if an LLM verb has no usable model credential.

    Only the LLM verbs (``run`` / ``pipeline``) need a key; every other verb
    returns ``None`` (works keyless). Delegates the credential probe to
    :func:`loom.cli._llm_preflight` (which asks each resolved model provider what
    it needs) so the message matches the one-shot path. Best-effort: any probe
    failure is treated as "no obvious problem" so the REPL never blocks a verb on
    a flaky import -- the handler's own preflight remains the backstop.

    Args:
        verb: The verb token being dispatched.
        config: The active Loom configuration.

    Returns:
        A single actionable line (set a key / pick a provider), or ``None``.
    """
    if verb not in _LLM_VERBS:
        return None
    try:
        from loom.cli import _llm_preflight

        hints = _llm_preflight(config)
    except Exception:  # noqa: BLE001 - never turn a probe failure into a block
        return None
    if not hints:
        return None
    first = hints[0]
    return (
        f"'{verb}' needs an LLM (the AIDE search brain writes solutions) but no "
        f"model credential was found: {first}\n"
        "Set a key and retry, e.g.  export ANTHROPIC_API_KEY=...  "
        "(or choose a provider: --model-provider <name>). "
        "Read-only/lifecycle verbs (datasets, eda, viz, validate, features, "
        "train, ops, report, doctor) work without a key."
    )


# ---------------------------------------------------------------------------
# The REPL.
# ---------------------------------------------------------------------------


class LoomRepl:
    """The interactive Loom shell: a thin loop over the shared CLI handlers.

    The loop reads a line, routes it (a meta command, a verb dispatched through
    the shared parser+handler, or a bare line -> a hint), wraps a running verb
    in a spinner, gates the costly verbs interactively, and renders a themed
    result. It is fully injectable for headless tests: pass a ``console`` built
    over a ``StringIO``, a ``read_line`` that yields canned input, and a
    ``confirm`` for the gate.
    """

    def __init__(
        self,
        config: "LoomConfig",
        *,
        console: Optional["Console"] = None,
        read_line: Optional[Callable[[str], str]] = None,
        confirm: Optional[Callable[[str], bool]] = None,
        use_rich: bool = True,
    ) -> None:
        """Construct the REPL.

        Args:
            config: The active Loom configuration (drives the banner providers +
                the keyless preflight).
            console: A Loom console (headless over a buffer in tests). Built via
                :func:`loom.ui.theme.get_console` when omitted.
            read_line: Injectable line reader taking the prompt string and
                returning the next line; raising :class:`EOFError` quits. When
                omitted, a ``prompt_toolkit`` session is created lazily.
            confirm: Injectable y/N confirm for the approval gate. Defaults to
                the gate's deny-first Rich prompt.
            use_rich: When ``False`` (the ``--no-ui`` / ``LOOM_NO_UI`` posture),
                the console is built with color stripped for CI/pipes.
        """
        self.config = config
        self.console = console or theme.get_console(no_color=not use_rich)
        self._read_line = read_line
        self._confirm = confirm
        self._use_rich = use_rich
        # Build the shared parser ONCE (the factory) and derive the verb set off
        # it so the REPL can never reimplement or drift from the one-shot CLI.
        from loom.cli import _build_parser

        self.parser = _build_parser()
        # `chat` launches the REPL -- you are already inside it, so drop it from
        # the interactive verb set (no recursive /chat in help or the completer).
        self.verbs = [v for v in verb_names(self.parser) if v != "chat"]
        self._session = None  # lazily created prompt_toolkit PromptSession

    # -- input -------------------------------------------------------------

    def _ensure_session(self) -> None:
        """Lazily build a ``prompt_toolkit`` session (in-memory history + completer).

        Only reached when no ``read_line`` was injected (i.e. a real interactive
        run). The import is lazy so the headless/test path never needs
        ``prompt_toolkit``.
        """
        if self._session is not None or self._read_line is not None:
            return
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import WordCompleter
        from prompt_toolkit.history import InMemoryHistory

        words = [f"/{v}" for v in self.verbs] + list(META_COMMANDS)
        completer = WordCompleter(words, ignore_case=True, sentence=True)
        self._session = PromptSession(
            history=InMemoryHistory(),
            completer=completer,
        )

    def read(self, prompt: str = "loom> ") -> str:
        """Read one input line (injected reader, else a ``prompt_toolkit`` prompt).

        Args:
            prompt: The prompt string to show.

        Returns:
            The next input line.

        Raises:
            EOFError: On Ctrl-D / end of injected input (quits the loop cleanly).
        """
        if self._read_line is not None:
            return self._read_line(prompt)
        self._ensure_session()
        return self._session.prompt(prompt)  # type: ignore[union-attr]

    # -- dispatch ----------------------------------------------------------

    def dispatch(self, tokens: Sequence[str]) -> int:
        """Parse ``tokens`` with the shared parser and run the matching handler.

        This is the load-bearing reuse: it calls
        ``self.parser.parse_args(tokens)`` (the SAME parser the one-shot CLI
        builds) and invokes ``args.func(args)`` (the SAME handler), gating costly
        verbs first and rendering a themed result after. The verb's own plain
        summary still prints (handlers are unchanged); the REPL adds the gate,
        the spinner, and a themed status frame on top.

        Args:
            tokens: The argv token list (verb first).

        Returns:
            The handler's exit code, ``0`` for a clean gate BLOCK, or ``2`` for a
            usage error -- never a raised exception (the loop must survive).
        """
        if not tokens:
            return 0
        verb = tokens[0]

        # Keyless preflight for the LLM verbs: an actionable line, not a deep
        # backend traceback, when no model credential is configured.
        keyless = llm_verb_keyless_message(verb, self.config)
        if keyless is not None:
            theme.warning(self.console, keyless)
            return 2

        # Parse with the SHARED parser. argparse calls sys.exit on a bad line; we
        # catch SystemExit so a usage error reports cleanly instead of killing the
        # REPL. The parser prints its own usage/error to stderr.
        try:
            args = self.parser.parse_args(list(tokens))
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 2
            # A bare `loom` (no verb) would recurse into the REPL via main(); never
            # do that from inside the loop.
            if code != 0:
                theme.error(
                    self.console,
                    f"could not parse '{verb}' -- check the flags "
                    f"(try '/{verb} --help' or '/help').",
                )
            return code

        if not getattr(args, "func", None):
            theme.error(self.console, f"'{verb}' is not a runnable verb.")
            return 2

        # Interactive approval gate for the costly/irreversible verbs, BEFORE the
        # handler runs (deny-first; the model proposes, the user fires).
        decision = self._gate_for(verb, args)
        if decision is not None:
            theme.info(self.console, decision.reason)
            if not decision.allow:
                return 0  # a clean BLOCK is a correct outcome, not an error

        # Run the handler inside a spinner. The handler prints its typed summary;
        # we frame it with a themed status line.
        code = self._run_handler(verb, args)
        self._render_after(verb, code)
        return code

    def _gate_for(self, verb: str, args: "argparse.Namespace"):
        """Return a gate :class:`~loom.ui.gate.Decision` for a costly verb, else None.

        A gated verb only gates when its real-action flag is set (``--apply`` /
        ``--send`` / ``--launch``); otherwise it runs its safe default
        (plan/build-only) and the handler stays in charge. Read-only and
        workspace-write verbs never reach here.

        Args:
            verb: The verb token.
            args: The parsed namespace.

        Returns:
            A :class:`~loom.ui.gate.Decision` when the verb gated, else ``None``.
        """
        spec = _GATED_VERBS.get(verb)
        if spec is None:
            return None
        flag, tier, label = spec
        if not bool(getattr(args, flag, False)):
            return None
        detail = {
            "op": label,
            "source": getattr(args, "validate_run", None)
            or getattr(args, "solution_run", None)
            or getattr(args, "run_pathspec", None)
            or getattr(args, "dataset", None),
        }
        return gate_mod.gate(
            tier,
            label,
            detail,
            confirm=self._confirm,
            console=self.console,
        )

    def _run_handler(self, verb: str, args: "argparse.Namespace") -> int:
        """Invoke the shared handler, optionally under a Rich spinner.

        Ctrl-C cancels the *current action* (not the REPL): a
        :class:`KeyboardInterrupt` raised inside the handler is caught and
        reported. Any other exception is caught and surfaced as an actionable
        error line so one failing verb never kills the loop.

        Args:
            verb: The verb token (selects whether to show a spinner).
            args: The parsed namespace.

        Returns:
            The handler's exit code (``1`` on a caught exception, ``130`` on
            Ctrl-C -- the conventional SIGINT code).
        """
        from rich.text import Text

        run = lambda: args.func(args)  # noqa: E731 - tiny thunk
        try:
            if verb in _SPINNER_VERBS and self._use_rich:
                message = "searching..." if verb in _LLM_VERBS else f"running {verb}..."
                with self.console.status(
                    Text(message, style="loom.section"), spinner="dots"
                ):
                    return int(run() or 0)
            return int(run() or 0)
        except KeyboardInterrupt:
            self.console.print("")
            theme.warning(self.console, f"cancelled '{verb}' (the REPL is still running).")
            return 130
        except SystemExit as exc:  # a handler/proxy may fail fast via SystemExit
            code = exc.code if isinstance(exc.code, int) else 2
            if isinstance(exc.code, str):
                theme.error(self.console, exc.code)
            return code
        except Exception as exc:  # noqa: BLE001 - never let one verb kill the loop
            theme.error(self.console, f"'{verb}' failed: {type(exc).__name__}: {exc}")
            return 1

    def _render_after(self, verb: str, code: int) -> None:
        """Print a themed one-line status frame after a verb finishes.

        The handler already printed its typed summary; this adds a colored
        success/warning footer keyed off the exit code so the result reads
        cleanly in the themed shell.

        Args:
            verb: The verb token.
            code: The handler's exit code.
        """
        if code == 0:
            theme.success(self.console, f"{verb} ok")
        elif code == 130:
            return  # the cancel line was already printed
        else:
            theme.warning(self.console, f"{verb} exited with code {code}")

    # -- meta commands -----------------------------------------------------

    def _meta(self, token: str) -> Optional[bool]:
        """Handle a meta (REPL-only) command. Returns True/False to continue/quit, or None.

        Args:
            token: The first token of the line (e.g. ``"/help"``).

        Returns:
            ``True`` to keep looping, ``False`` to quit, or ``None`` when the
            token is not a meta command (the caller routes it as a verb/line).
        """
        name = token[1:] if token.startswith("/") else token
        if name in ("exit", "quit"):
            theme.info(self.console, "bye.")
            return False
        if name == "clear":
            try:
                self.console.clear()
            except Exception:  # noqa: BLE001 - clear is cosmetic
                pass
            return True
        if name == "help":
            self.print_help()
            return True
        if name == "status":
            theme.banner(self.config, console=self.console)
            return True
        if name == "doctor":
            self.dispatch(["doctor"])
            return True
        return None

    def print_help(self) -> None:
        """Print the themed REPL help: the verb table + the meta commands + UX notes."""
        from rich.table import Table

        theme.section(self.console, "Verbs")
        body = Table.grid(padding=(0, 2))
        body.add_column(style="loom.section", no_wrap=True)
        body.add_column(style="loom.ash")
        for verb in self.verbs:
            body.add_row(f"/{verb}", self._verb_help(verb))
        self.console.print(body)
        theme.section(self.console, "Meta")
        theme.info(self.console, ", ".join(META_COMMANDS))
        theme.info(
            self.console,
            "spinner wraps a running verb; Ctrl-C cancels the action, Ctrl-D / "
            "/exit quits. Costly verbs (deploy --apply, train --launch, collab "
            "--send) prompt for approval; read-only verbs never do.",
        )

    def _verb_help(self, verb: str) -> str:
        """Return the one-line help string for a verb (from the subparser choice)."""
        import argparse

        for action in self.parser._actions:  # noqa: SLF001
            if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
                choice = action.choices.get(verb)
                if choice is not None and choice.description:
                    # First sentence only, kept short for the table.
                    desc = choice.description.strip().split(". ")[0]
                    return (desc[:80] + "...") if len(desc) > 83 else desc
        return ""

    # -- the loop ----------------------------------------------------------

    def route(self, line: str) -> Optional[bool]:
        """Route ONE input line. Returns True/False to continue/quit, or None on blank.

        A meta command is handled here; a verb (with or without a leading ``/``)
        is tokenized and dispatched through the shared parser/handler; any other
        first token is treated as a bare natural-language line and answered with
        the verb hint.

        Args:
            line: The raw input line.

        Returns:
            ``True`` to keep looping, ``False`` to quit, ``None`` for a blank
            line (the loop just re-prompts).
        """
        tokens = tokenize(line)
        if not tokens:
            return None

        # Meta commands are recognized whether or not they carry a leading slash.
        first_raw = line.strip()
        meta_token = first_raw.split()[0] if first_raw else ""
        meta = self._meta(meta_token)
        if meta is not None:
            return meta

        verb = tokens[0]
        if verb in self.verbs:
            self.dispatch(tokens)
            return True

        # A bare natural-language line (or an unknown token): the helpful hint.
        # Keep any LLM-driven routing thin/optional and behind a configured key;
        # in v0.1 we simply guide to the verbs.
        theme.panel(self.console, "Type a verb", _hint_body(self.verbs))
        return True

    def run(self, *, banner: bool = True) -> int:
        """Run the read-eval loop until EOF (Ctrl-D) or ``/exit``.

        Args:
            banner: Whether to print the branded banner first.

        Returns:
            ``0`` on a clean exit.
        """
        if banner:
            theme.banner(self.config, console=self.console)
            theme.info(self.console, "type /help for the verbs, /exit to quit.")
        while True:
            try:
                line = self.read()
            except EOFError:
                self.console.print("")
                theme.info(self.console, "bye.")
                return 0
            except KeyboardInterrupt:
                # Ctrl-C at the prompt cancels the line, not the REPL.
                self.console.print("")
                continue
            cont = self.route(line)
            if cont is False:
                return 0


def run_repl(
    config: Optional["LoomConfig"] = None,
    *,
    use_rich: bool = True,
    console: Optional["Console"] = None,
    read_line: Optional[Callable[[str], str]] = None,
    confirm: Optional[Callable[[str], bool]] = None,
) -> int:
    """Launch the Loom REPL (the no-subcommand / ``loom chat`` entry point).

    Builds the config (env/.env/YAML) when not supplied and runs the loop. This
    is what :func:`loom.cli.main` calls when invoked with no subcommand or
    ``loom chat``.

    Args:
        config: The active config; built from the environment when omitted.
        use_rich: When ``False`` (the ``--no-ui`` posture), color is stripped.
        console: Optional pre-built console (headless over a buffer for tests).
        read_line: Optional injected line reader (canned input for tests).
        confirm: Optional injected gate confirm.

    Returns:
        The loop's exit code (``0`` on a clean exit).
    """
    if config is None:
        from loom.config import LoomConfig

        config = LoomConfig.load()
    repl = LoomRepl(
        config,
        console=console,
        read_line=read_line,
        confirm=confirm,
        use_rich=use_rich,
    )
    return repl.run()


__all__ = [
    "LoomRepl",
    "run_repl",
    "verb_names",
    "tokenize",
    "llm_verb_keyless_message",
    "META_COMMANDS",
]
