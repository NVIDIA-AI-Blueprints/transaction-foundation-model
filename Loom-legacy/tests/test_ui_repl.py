"""Tests for ``loom.ui.repl`` -- the interactive REPL router.

The REPL owns no engine logic: it builds the SAME parser
:func:`loom.cli._build_parser` produces, tokenizes each line, parses it with that
parser, and invokes the SAME ``args.func(args)`` handler the one-shot subcommands
use. These tests pin that reuse + the routing contract, all HEADLESS (the console
is built over a ``StringIO``; the prompt input and the gate confirm are injected,
so no TTY is involved):

* ``/eda --dataset IngestDataset/1 --target t`` parses with the shared parser and
  invokes the ``eda`` handler with the parsed args (the handler is mocked);
* the verb set is read off the live parser (so it can never drift from
  ``loom.cli``) and excludes the recursive ``chat`` verb;
* the slash-completer word list lists every ``/verb`` + the meta commands;
* a bare natural-language line returns the verb hint, not a traceback;
* an LLM verb (``run`` / ``pipeline``) with no model credential yields an
  ACTIONABLE message (not an exception), while a read-only verb never blocks on a
  key;
* a usage error (a bad flag) reports cleanly and keeps the loop alive (no
  ``SystemExit`` escapes);
* a costly verb with its real-action flag set gates through the injected confirm
  (deny BLOCKS before the handler runs);
* ``/exit`` / EOF quit cleanly and ``/help`` renders the verb table;
* :func:`~loom.ui.theme.banner` renders.

To intercept a handler we patch it in :mod:`loom.cli` BEFORE the REPL is
constructed, because :func:`loom.cli._build_parser` captures each handler by
reference at parser-build time (``set_defaults(func=_cmd_eda)``); the REPL builds
that parser once in its constructor.
"""

from __future__ import annotations

from io import StringIO
from unittest import mock

import pytest

import loom.cli as cli
from loom.config import LoomConfig
from loom.ui import repl as repl_mod
from loom.ui import theme
from loom.ui.repl import LoomRepl, llm_verb_keyless_message, tokenize, verb_names


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _console() -> theme.Console:
    return theme.get_console(file=StringIO(), force_terminal=False, width=120)


def _config() -> LoomConfig:
    return LoomConfig.load()


def _eof_reader(_prompt: str) -> str:
    """A line reader that immediately signals EOF (used when input is unused)."""
    raise EOFError


def _make_repl(console=None, read_line=None, confirm=None) -> LoomRepl:
    return LoomRepl(
        _config(),
        console=console or _console(),
        read_line=read_line or _eof_reader,
        confirm=confirm,
    )


# ---------------------------------------------------------------------------
# tokenize.
# ---------------------------------------------------------------------------


def test_tokenize_strips_leading_slash_and_keeps_quotes() -> None:
    assert tokenize("/eda --dataset IngestDataset/1 --target t") == [
        "eda",
        "--dataset",
        "IngestDataset/1",
        "--target",
        "t",
    ]
    # a quoted free-text arg survives the split
    assert tokenize('run --goal "predict churn" --metric auc') == [
        "run",
        "--goal",
        "predict churn",
        "--metric",
        "auc",
    ]
    assert tokenize("   ") == []  # blank line -> no tokens


# ---------------------------------------------------------------------------
# The verb set is the live parser's verb set (no drift) and excludes `chat`.
# ---------------------------------------------------------------------------


def test_verb_set_matches_parser_and_excludes_chat() -> None:
    repl = _make_repl()
    parser_verbs = verb_names(repl.parser)
    assert "eda" in parser_verbs and "deploy" in parser_verbs and "run" in parser_verbs
    # the REPL drops `chat` (you are already inside the REPL) but keeps the rest
    assert "chat" not in repl.verbs
    assert set(repl.verbs) == set(parser_verbs) - {"chat"}


# ---------------------------------------------------------------------------
# The router reuses the shared parser + handler.
# ---------------------------------------------------------------------------


def test_dispatch_eda_invokes_the_shared_handler_with_parsed_args() -> None:
    # Patch the handler BEFORE the parser captures it, then build the REPL.
    with mock.patch.object(cli, "_cmd_eda", return_value=0) as m:
        repl = _make_repl()
        code = repl.dispatch(tokenize("/eda --dataset IngestDataset/1 --target t"))
        # The parser, built in the REPL ctor while the patch was live, wired the
        # mocked handler as the namespace's func (asserted inside the patch scope).
        namespace = m.call_args[0][0]
        assert namespace.func is m
    assert m.called, "the eda handler must be invoked through the shared parser"
    assert code == 0
    assert namespace.dataset == "IngestDataset/1"
    assert namespace.target == "t"


def test_route_dispatches_a_known_verb() -> None:
    with mock.patch.object(cli, "_cmd_datasets", return_value=0) as m:
        repl = _make_repl()
        cont = repl.route("/datasets")
    assert m.called
    assert cont is True  # routing a verb keeps the loop alive


# ---------------------------------------------------------------------------
# The completer word list.
# ---------------------------------------------------------------------------


def test_completer_words_list_verbs_and_meta_commands() -> None:
    repl = _make_repl()
    words = [f"/{v}" for v in repl.verbs] + list(repl_mod.META_COMMANDS)
    assert "/eda" in words and "/deploy" in words and "/datasets" in words
    assert "/help" in words and "/status" in words and "/exit" in words
    assert "/chat" not in words  # no recursive chat in the completer


def test_prompt_toolkit_session_completer_offers_the_words() -> None:
    # When no read_line is injected, the REPL lazily builds a prompt_toolkit
    # session with a WordCompleter over the verb + meta words.
    repl = LoomRepl(_config(), console=_console())  # no read_line -> session path
    repl._ensure_session()
    assert repl._session is not None
    completer_words = list(repl._session.completer.words)  # type: ignore[union-attr]
    assert "/eda" in completer_words and "/help" in completer_words


# ---------------------------------------------------------------------------
# A bare natural-language line -> the verb hint (not a traceback).
# ---------------------------------------------------------------------------


def test_bare_line_returns_the_verb_hint() -> None:
    console = _console()
    repl = _make_repl(console=console)
    cont = repl.route("please build me a model that predicts churn")
    assert cont is True  # the loop survives a bare line
    out = console.file.getvalue()
    assert "Type a verb" in out
    assert "/eda" in out  # the hint lists the verbs


def test_unknown_token_also_returns_the_hint() -> None:
    console = _console()
    repl = _make_repl(console=console)
    repl.route("frobnicate --whatever")
    assert "Type a verb" in console.file.getvalue()


# ---------------------------------------------------------------------------
# An LLM verb with no key -> an actionable message, never an exception.
# ---------------------------------------------------------------------------


def test_llm_verb_keyless_message_is_actionable_for_run() -> None:
    with mock.patch.object(cli, "_llm_preflight", return_value=["no ANTHROPIC_API_KEY found"]):
        message = llm_verb_keyless_message("run", _config())
    assert message is not None
    assert "ANTHROPIC_API_KEY" in message
    # it points the user at the keyless verbs too
    assert "without a key" in message


def test_llm_verb_keyless_message_is_none_for_readonly_verbs() -> None:
    # A read-only / lifecycle verb never needs a key, so the probe returns None
    # even if a credential hint exists.
    with mock.patch.object(cli, "_llm_preflight", return_value=["no key"]):
        assert llm_verb_keyless_message("eda", _config()) is None
        assert llm_verb_keyless_message("datasets", _config()) is None


def test_dispatch_llm_verb_keyless_returns_actionable_message_not_exception() -> None:
    console = _console()
    repl = _make_repl(console=console)
    with mock.patch.object(cli, "_llm_preflight", return_value=["no ANTHROPIC_API_KEY found"]):
        # Must NOT reach the handler and must NOT raise; returns the usage code.
        with mock.patch.object(cli, "_cmd_run") as run_handler:
            code = repl.dispatch(["run", "--goal", "g", "--metric", "m"])
            assert not run_handler.called, "the LLM verb must be short-circuited before the handler"
    assert code == 2
    assert "ANTHROPIC_API_KEY" in console.file.getvalue()


def test_dispatch_llm_verb_with_key_reaches_the_handler() -> None:
    # When the preflight finds no problem, the LLM verb dispatches normally.
    with mock.patch.object(cli, "_llm_preflight", return_value=[]):
        with mock.patch.object(cli, "_cmd_run", return_value=0) as run_handler:
            repl = _make_repl()
            code = repl.dispatch(["run", "--goal", "g", "--metric", "m"])
    assert run_handler.called
    assert code == 0


# ---------------------------------------------------------------------------
# A usage error reports cleanly and keeps the loop alive (no SystemExit escapes).
# ---------------------------------------------------------------------------


def test_bad_flag_reports_cleanly_without_killing_the_loop() -> None:
    console = _console()
    repl = _make_repl(console=console)
    # `eda` requires --dataset; omitting it makes argparse exit(2). The REPL must
    # catch that SystemExit and return the code rather than propagate it.
    code = repl.dispatch(["eda"])  # missing required --dataset
    assert code == 2
    # the loop survives -- a follow-up route still works
    assert repl.route("/help") is True


# ---------------------------------------------------------------------------
# The interactive gate fires for a costly verb's real-action flag.
# ---------------------------------------------------------------------------


def test_deploy_apply_gates_and_a_deny_blocks_before_the_handler() -> None:
    confirm_calls: list[str] = []

    def deny(prompt: str) -> bool:
        confirm_calls.append(prompt)
        return False

    with mock.patch.object(cli, "_cmd_deploy", return_value=0) as deploy_handler:
        repl = _make_repl(confirm=deny)
        code = repl.dispatch(["deploy", "--validate", "ValidateFlow/1", "--apply"])
    assert confirm_calls, "deploy --apply must gate interactively (irreversible)"
    assert not deploy_handler.called, "a denied gate must block before the handler runs"
    assert code == 0  # a clean BLOCK is a correct outcome, not an error


def test_deploy_without_apply_does_not_gate() -> None:
    # The safe default (no --apply) is plan-only; it must NOT prompt, and the
    # handler stays in charge.
    def must_not_prompt(_p: str) -> bool:
        raise AssertionError("a plan-only deploy must not gate")

    with mock.patch.object(cli, "_cmd_deploy", return_value=0) as deploy_handler:
        repl = _make_repl(confirm=must_not_prompt)
        repl.dispatch(["deploy", "--validate", "ValidateFlow/1"])
    assert deploy_handler.called


def test_train_launch_gates_expensive_and_approve_runs_handler() -> None:
    with mock.patch.object(cli, "_cmd_train", return_value=0) as train_handler:
        repl = _make_repl(confirm=lambda _p: True)
        code = repl.dispatch(["train", "--dataset", "IngestDataset/1", "--launch"])
    assert train_handler.called, "an approved gate must run the handler"
    assert code == 0


# ---------------------------------------------------------------------------
# Meta commands.
# ---------------------------------------------------------------------------


def test_exit_quits_cleanly() -> None:
    repl = _make_repl()
    assert repl.route("/exit") is False
    assert repl.route("/quit") is False
    assert repl.route("exit") is False  # recognized without the slash too


def test_help_renders_the_verb_table() -> None:
    console = _console()
    repl = _make_repl(console=console)
    assert repl.route("/help") is True
    out = console.file.getvalue()
    assert "Verbs" in out and "/eda" in out and "Meta" in out


def test_status_renders_the_banner() -> None:
    console = _console()
    repl = _make_repl(console=console)
    repl.route("/status")
    assert theme.TAGLINE in console.file.getvalue()


# ---------------------------------------------------------------------------
# The loop quits on EOF, and the handler-exception path never kills the loop.
# ---------------------------------------------------------------------------


def test_run_quits_on_eof_and_prints_banner() -> None:
    console = _console()
    repl = LoomRepl(_config(), console=console, read_line=_eof_reader)
    code = repl.run()  # the reader EOFs immediately -> clean exit
    assert code == 0
    out = console.file.getvalue()
    assert theme.TAGLINE in out  # the banner printed first
    assert "bye." in out


def test_run_loops_a_line_then_quits() -> None:
    lines = iter(["/help", EOFError])

    def reader(_prompt: str) -> str:
        item = next(lines)
        if item is EOFError:
            raise EOFError
        return item

    console = _console()
    repl = LoomRepl(_config(), console=console, read_line=reader)
    assert repl.run(banner=False) == 0
    assert "Verbs" in console.file.getvalue()  # the /help line was routed


def test_handler_exception_is_caught_and_loop_survives() -> None:
    def boom(_args) -> int:
        raise RuntimeError("kaboom")

    with mock.patch.object(cli, "_cmd_datasets", side_effect=boom):
        console = _console()
        repl = _make_repl(console=console)
        code = repl.dispatch(["datasets"])
    assert code == 1  # caught -> exit code 1, not a propagated exception
    out = console.file.getvalue()
    assert "datasets" in out and "kaboom" in out


# ---------------------------------------------------------------------------
# The no-subcommand launch path (a bare `loom` / `loom chat`).
# ---------------------------------------------------------------------------


def test_main_no_subcommand_launches_the_repl() -> None:
    # A bare `loom` (no argv) must build the config and launch the REPL WITHOUT a
    # traceback. The top-level namespace defines no `--config`, so _build_config
    # must read it defensively (regression: AttributeError on args.config).
    with mock.patch.object(cli, "_launch_repl", return_value=0) as launch:
        code = cli.main([])
    assert launch.called, "no subcommand must launch the REPL"
    assert code == 0


def test_main_chat_alias_launches_the_repl() -> None:
    with mock.patch.object(cli, "_launch_repl", return_value=0) as launch:
        code = cli.main(["chat"])
    assert launch.called
    assert code == 0


def test_launch_repl_builds_config_without_config_attr() -> None:
    # Drives the exact crash: the no-subcommand namespace lacks `config`.
    # _launch_repl must build the config and hand off to run_repl cleanly.
    args = cli._build_parser().parse_args([])  # no subcommand -> no `config` attr
    assert not hasattr(args, "config")
    with mock.patch.object(cli, "_build_config", wraps=cli._build_config) as build:
        with mock.patch("loom.ui.repl.run_repl", return_value=0) as run:
            code = cli._launch_repl(args)
    assert build.called  # built the config from the attr-less namespace, no raise
    assert run.called
    assert code == 0


# ---------------------------------------------------------------------------
# banner() renders (the launch frame).
# ---------------------------------------------------------------------------


def test_banner_renders_logo_and_providers() -> None:
    console = _console()
    theme.banner(_config(), console=console)
    out = console.file.getvalue()
    assert theme.TAGLINE in out
    # the provider summary line labels
    assert "search" in out and "mlops" in out and "model" in out
