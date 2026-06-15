"""Tests for ``loom doctor`` -- the read-only local-stack health check.

``loom doctor`` is factored into small, pure check functions in
:mod:`loom.cli` so they can be unit-tested on a stubbed environment with no
live Metaflow cluster and no real datastore. These tests pin:

* the datastore-env check: all 7 vars set => PASS; any unset => FAIL whose fix
  names the missing vars; an unusual selector value => WARN;
* the endpoint host:port parser (explicit port, default-by-scheme, bare host,
  unparseable);
* the socket reachability probe (PASS on a live local listener, FAIL on a
  refused/closed port, WARN when the endpoint env var is unset);
* the Client-API smoke tolerating zero data objects and degrading to WARN when
  Metaflow is absent / the metadata service errors;
* the verdict roll-up (a FAIL fails the verdict; a WARN does not);
* the ``loom doctor`` argparse wiring (and that the existing subcommands still
  parse).

Everything here is pure-Python: no pandas, no live Metaflow, no real socket to
an external host. The reachability PASS case binds a throwaway localhost socket
so it exercises a real connect against a port we control.
"""

from __future__ import annotations

import socket
import sys
import types

import pytest

from loom.cli import (
    _DOCTOR_DATASTORE_ENV_VARS,
    _DOCTOR_ENDPOINT_ENV_VAR,
    _DOCTOR_FAIL,
    _DOCTOR_PASS,
    _DOCTOR_WARN,
    _build_parser,
    _cmd_doctor,
    _detect_repair_assistant,
    _doctor_check_datastore_env,
    _doctor_check_metaflow,
    _doctor_client_api_smoke,
    _doctor_parse_host_port,
    _doctor_probe_endpoint,
    _doctor_repair_prompt,
    _doctor_try_agentic_fix,
    _doctor_verdict,
)
from loom.config import LoomConfig


# A complete, sane datastore env (the 7 exports the setup script writes). Used as
# the all-set baseline; individual tests drop/override keys from a copy.
def _full_env() -> dict[str, str]:
    return {
        "METAFLOW_DEFAULT_DATASTORE": "s3",
        "METAFLOW_DATASTORE_SYSROOT_S3": "s3" + "://metaflow/metaflow",
        "METAFLOW_S3_ENDPOINT_URL": "http://localhost:9000",
        "AWS_ACCESS_KEY_ID": "minioadmin",
        "AWS_SECRET_ACCESS_KEY": "minioadmin123",
        "METAFLOW_DEFAULT_METADATA": "local",
        "METAFLOW_USER": "tester",
    }


# ---------------------------------------------------------------------------
# (c) datastore env-var check.
# ---------------------------------------------------------------------------


def test_datastore_env_all_set_is_pass() -> None:
    """A fully-populated, sane datastore env yields PASS."""
    check = _doctor_check_datastore_env(_full_env())
    assert check["status"] == _DOCTOR_PASS
    # PASS lines carry no fix.
    assert check["fix"] == ""
    assert "datastore=s3" in check["detail"]
    assert "metadata=local" in check["detail"]


@pytest.mark.parametrize("missing", list(_DOCTOR_DATASTORE_ENV_VARS))
def test_datastore_env_any_missing_is_fail_naming_the_var(missing: str) -> None:
    """Dropping any single required var FAILs and the detail names that var."""
    env = _full_env()
    del env[missing]
    check = _doctor_check_datastore_env(env)
    assert check["status"] == _DOCTOR_FAIL
    assert missing in check["detail"]
    # The fix is actionable: it points at the setup/source-the-env recipe.
    assert "setup_metaflow_minikube.sh" in check["fix"]
    assert "source .env.metaflow" in check["fix"]


def test_datastore_env_blank_value_counts_as_missing() -> None:
    """A present-but-blank var is treated as unset (FAIL)."""
    env = _full_env()
    env["AWS_SECRET_ACCESS_KEY"] = "   "
    check = _doctor_check_datastore_env(env)
    assert check["status"] == _DOCTOR_FAIL
    assert "AWS_SECRET_ACCESS_KEY" in check["detail"]


def test_datastore_env_all_missing_lists_all_seven() -> None:
    """An empty env FAILs and reports all 7 required vars as unset."""
    check = _doctor_check_datastore_env({})
    assert check["status"] == _DOCTOR_FAIL
    for var in _DOCTOR_DATASTORE_ENV_VARS:
        assert var in check["detail"]


def test_datastore_env_unusual_selector_is_warn_not_fail() -> None:
    """All vars set but an unrecognized datastore selector => WARN (still usable)."""
    env = _full_env()
    env["METAFLOW_DEFAULT_DATASTORE"] = "weirdstore"
    check = _doctor_check_datastore_env(env)
    assert check["status"] == _DOCTOR_WARN
    assert "weirdstore" in check["detail"]


# ---------------------------------------------------------------------------
# (d) endpoint host:port parsing + the socket reachability probe.
# ---------------------------------------------------------------------------


def test_parse_host_port_explicit_port() -> None:
    """An explicit host:port URL parses to that pair."""
    assert _doctor_parse_host_port("http://localhost:9000") == ("localhost", 9000)


def test_parse_host_port_defaults_by_scheme() -> None:
    """A scheme with no explicit port falls back to that scheme's default."""
    assert _doctor_parse_host_port("http://example.com") == ("example.com", 80)
    assert _doctor_parse_host_port("https://example.com") == ("example.com", 443)


def test_parse_host_port_bare_host_assumes_http() -> None:
    """A bare host:port (no scheme) is parsed as http."""
    assert _doctor_parse_host_port("127.0.0.1:9000") == ("127.0.0.1", 9000)


@pytest.mark.parametrize("bad", ["", "   ", "http://"])
def test_parse_host_port_unparseable_returns_none(bad: str) -> None:
    """An empty/host-less endpoint string parses to None."""
    assert _doctor_parse_host_port(bad) is None


def test_probe_endpoint_unset_is_warn() -> None:
    """No endpoint to probe is a WARN (the env check already FAILs that case)."""
    check = _doctor_probe_endpoint("")
    assert check["status"] == _DOCTOR_WARN
    assert _DOCTOR_ENDPOINT_ENV_VAR in check["detail"]


def test_probe_endpoint_unparseable_is_fail() -> None:
    """An endpoint with no host FAILs the reachability check."""
    check = _doctor_probe_endpoint("http://")
    assert check["status"] == _DOCTOR_FAIL
    assert _DOCTOR_ENDPOINT_ENV_VAR in check["fix"]


def test_probe_endpoint_live_listener_is_pass() -> None:
    """A real, listening localhost socket connects => PASS (no SDK, just TCP)."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))  # ephemeral free port
    srv.listen(1)
    host, port = srv.getsockname()
    try:
        check = _doctor_probe_endpoint(f"http://{host}:{port}", timeout=2.0)
    finally:
        srv.close()
    assert check["status"] == _DOCTOR_PASS
    assert f"{host}:{port}" in check["detail"]


def test_probe_endpoint_closed_port_is_fail() -> None:
    """A bound-but-closed port (nothing listening) FAILs with an actionable fix."""
    # Grab an ephemeral port then close it so the connect is refused.
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    host, port = srv.getsockname()
    srv.close()
    check = _doctor_probe_endpoint(f"http://{host}:{port}", timeout=1.0)
    assert check["status"] == _DOCTOR_FAIL
    assert "port-forward" in check["fix"]


# ---------------------------------------------------------------------------
# (b) metaflow import + (e) Client-API smoke -- stubbed via sys.modules.
# ---------------------------------------------------------------------------


def test_metaflow_check_present_is_pass() -> None:
    """``import metaflow`` succeeding (real or stub) => PASS."""
    # Metaflow ships with Loom's deps, so this is normally a real import.
    check = _doctor_check_metaflow()
    assert check["status"] == _DOCTOR_PASS


def test_metaflow_check_absent_is_fail(monkeypatch) -> None:
    """When ``import metaflow`` raises, the check FAILs with an install fix."""
    # Force the import inside the check to fail by blocking the module.
    monkeypatch.setitem(sys.modules, "metaflow", None)
    check = _doctor_check_metaflow()
    assert check["status"] == _DOCTOR_FAIL
    assert "install" in check["fix"].lower()


@pytest.fixture
def fake_metaflow_listing(monkeypatch):
    """Install a fake ``metaflow`` exposing ``Flow`` / ``namespace`` for the smoke.

    Returns a setter ``set_runs(runs_or_exc)``: pass a list of fake runs (each
    with a ``.successful`` bool) for ``Flow('IngestDataset').runs(...)``, or an
    Exception instance to make ``Flow(...)`` raise (metadata-down path).
    """
    holder: dict[str, object] = {"runs": []}

    class _FakeFlow:
        def __init__(self, name: str) -> None:
            payload = holder["runs"]
            if isinstance(payload, Exception):
                raise payload

        def runs(self, *_tags):
            return list(holder["runs"])  # type: ignore[arg-type]

    fake = types.ModuleType("metaflow")
    fake.Flow = _FakeFlow  # type: ignore[attr-defined]
    fake.namespace = lambda *_a, **_k: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "metaflow", fake)

    def set_runs(value) -> None:
        holder["runs"] = value

    return set_runs


class _FakeIngestRun:
    """A minimal ingested-run stand-in with a lazily-failing ``.successful``."""

    def __init__(self, successful: object) -> None:
        self._successful = successful

    @property
    def successful(self) -> bool:
        if isinstance(self._successful, Exception):
            raise self._successful
        return bool(self._successful)


def test_client_api_smoke_counts_data_objects(fake_metaflow_listing) -> None:
    """The smoke counts only successful, readable runs and PASSes with the count."""
    fake_metaflow_listing(
        [
            _FakeIngestRun(True),
            _FakeIngestRun(True),
            _FakeIngestRun(False),  # not successful -> not counted
            _FakeIngestRun(RuntimeError("corrupt blob")),  # unreadable -> skipped
        ]
    )
    check = _doctor_client_api_smoke(LoomConfig.load(env={}))
    assert check["status"] == _DOCTOR_PASS
    assert "2 ingested data object" in check["detail"]


def test_client_api_smoke_tolerates_zero(fake_metaflow_listing) -> None:
    """Zero data objects is a PASS (fresh stack), with an ingest hint."""
    fake_metaflow_listing([])
    check = _doctor_client_api_smoke(LoomConfig.load(env={}))
    assert check["status"] == _DOCTOR_PASS
    assert "0 ingested" in check["detail"]


def test_client_api_smoke_metadata_error_is_warn(fake_metaflow_listing) -> None:
    """A Client-API error (no flow yet / metadata down) degrades to WARN."""
    fake_metaflow_listing(RuntimeError("no such flow"))
    check = _doctor_client_api_smoke(LoomConfig.load(env={}))
    assert check["status"] == _DOCTOR_WARN


def test_client_api_smoke_metaflow_absent_is_warn(monkeypatch) -> None:
    """With Metaflow unimportable the smoke is a WARN (the metaflow check FAILs)."""
    monkeypatch.setitem(sys.modules, "metaflow", None)
    check = _doctor_client_api_smoke(LoomConfig.load(env={}))
    assert check["status"] == _DOCTOR_WARN


# ---------------------------------------------------------------------------
# Verdict roll-up.
# ---------------------------------------------------------------------------


def test_verdict_all_pass_is_ok() -> None:
    """All-PASS rolls up to an ok verdict starting with PASS."""
    checks = [
        {"name": "a", "status": _DOCTOR_PASS, "detail": "", "fix": ""},
        {"name": "b", "status": _DOCTOR_PASS, "detail": "", "fix": ""},
    ]
    line, ok = _doctor_verdict(checks)
    assert ok is True
    assert line.startswith("PASS")


def test_verdict_warn_does_not_fail() -> None:
    """A WARN keeps the verdict ok but is surfaced in the line."""
    checks = [
        {"name": "a", "status": _DOCTOR_PASS, "detail": "", "fix": ""},
        {"name": "b", "status": _DOCTOR_WARN, "detail": "", "fix": ""},
    ]
    line, ok = _doctor_verdict(checks)
    assert ok is True
    assert "warning" in line.lower()


def test_verdict_any_fail_is_not_ok() -> None:
    """A single FAIL fails the verdict regardless of warnings."""
    checks = [
        {"name": "a", "status": _DOCTOR_PASS, "detail": "", "fix": ""},
        {"name": "b", "status": _DOCTOR_WARN, "detail": "", "fix": ""},
        {"name": "c", "status": _DOCTOR_FAIL, "detail": "", "fix": ""},
    ]
    line, ok = _doctor_verdict(checks)
    assert ok is False
    assert line.startswith("FAIL")


# ---------------------------------------------------------------------------
# End-to-end command exit code (real env restored around the call).
# ---------------------------------------------------------------------------


def test_cmd_doctor_fails_when_env_unset(monkeypatch, capsys) -> None:
    """`loom doctor` returns non-zero when the datastore env is unset (a FAIL)."""
    for var in _DOCTOR_DATASTORE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    parser = _build_parser()
    args = parser.parse_args(["doctor"])
    code = _cmd_doctor(args)
    out = capsys.readouterr().out
    assert code == 1
    assert "VERDICT:" in out
    assert "datastore env vars" in out


# ---------------------------------------------------------------------------
# CLI arg-parse wiring.
# ---------------------------------------------------------------------------


def test_cli_doctor_parses() -> None:
    """`loom doctor` parses into the doctor handler and takes no required args."""
    parser = _build_parser()
    args = parser.parse_args(["doctor"])
    assert args.command == "doctor"
    assert args.func is _cmd_doctor


def test_cli_doctor_accepts_config() -> None:
    """`loom doctor --config path.yaml` parses (optional config flag)."""
    parser = _build_parser()
    args = parser.parse_args(["doctor", "--config", "loom.yaml"])
    assert args.config == "loom.yaml"


def test_cli_existing_subcommands_still_parse() -> None:
    """Adding `doctor` did not disturb the existing subcommands."""
    parser = _build_parser()
    assert parser.parse_args(["datasets"]).command == "datasets"
    assert (
        parser.parse_args(
            ["run", "--dataset", "IngestDataset/1", "--goal", "g", "--metric", "m"]
        ).command
        == "run"
    )


# ---------------------------------------------------------------------------
# `--fix`: agentic-first repair (claude/codex), scripted fallback.
# ---------------------------------------------------------------------------
_ISSUE = {
    "status": _DOCTOR_FAIL,
    "name": "import metaflow",
    "detail": "metaflow is not importable",
    "fix": "re-run pip install -e .",
}


def test_repair_assistant_prefers_claude_then_codex(monkeypatch) -> None:
    """Detection prefers `claude`, falls back to `codex`, else None."""
    monkeypatch.setattr("loom.cli.shutil.which", lambda n: f"/bin/{n}" if n in ("claude", "codex") else None)
    assert _detect_repair_assistant() == "claude"
    monkeypatch.setattr("loom.cli.shutil.which", lambda n: "/bin/codex" if n == "codex" else None)
    assert _detect_repair_assistant() == "codex"
    monkeypatch.setattr("loom.cli.shutil.which", lambda _n: None)
    assert _detect_repair_assistant() is None


def test_repair_prompt_carries_the_issue_and_guidance() -> None:
    """The prompt hands the assistant the failing check + INSTALL.md + the goal."""
    prompt = _doctor_repair_prompt([_ISSUE])
    assert "import metaflow" in prompt
    assert "INSTALL.md" in prompt
    assert "VERDICT: PASS" in prompt


def test_fix_without_assistant_prints_manual_note(monkeypatch, capsys) -> None:
    """No claude/codex on PATH -> the scripted/manual fallback, never an exec."""
    monkeypatch.setattr("loom.cli._detect_repair_assistant", lambda: None)
    called = {"exec": False}
    monkeypatch.setattr("loom.cli.os.execvp", lambda *a, **k: called.update(exec=True))
    _doctor_try_agentic_fix([_ISSUE])
    out = capsys.readouterr().out
    assert "No `claude` or `codex`" in out and "INSTALL.md" in out
    assert called["exec"] is False


def test_fix_with_assistant_execs_it(monkeypatch) -> None:
    """An available assistant is handed the repair via execvp(assistant, prompt)."""
    monkeypatch.setattr("loom.cli._detect_repair_assistant", lambda: "claude")
    captured = {}

    def _fake_execvp(file, argv):
        captured["file"] = file
        captured["argv"] = argv

    monkeypatch.setattr("loom.cli.os.execvp", _fake_execvp)
    _doctor_try_agentic_fix([_ISSUE])
    assert captured["file"] == "claude"
    assert captured["argv"][0] == "claude"
    assert "import metaflow" in captured["argv"][1]


def test_fix_is_ignored_in_json_mode(monkeypatch) -> None:
    """SAFETY: `--json --fix` must emit the envelope and NEVER exec an assistant
    (the agent tool always runs doctor with --json)."""
    monkeypatch.setattr("loom.cli._detect_repair_assistant", lambda: "claude")

    def _boom(*_a, **_k):  # pragma: no cover - must not be reached
        raise AssertionError("execvp must not run in --json mode")

    monkeypatch.setattr("loom.cli.os.execvp", _boom)
    args = _build_parser().parse_args(["doctor", "--json", "--fix"])
    # Returns a 0/1 exit code (env-dependent) without raising — i.e. no exec.
    assert _cmd_doctor(args) in (0, 1)
