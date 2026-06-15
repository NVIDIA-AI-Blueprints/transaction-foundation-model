"""Global control flags work in POST-verb position (DESIGN.md §2.4).

``--json``/``--experiment``/``-q`` are registered on a shared parent parser, so
argparse accepts them either before or after the verb. The docs' pipe-composable
pattern and every §3.x worked example write ``--experiment <id>`` AFTER the verb
(`loom ingest … --name … -q | xargs loom tokenize --experiment … -q`), so the
post-verb position is the one that must not regress. The existing test_dual_driver
tests call ``fn()`` directly and never exercise argparse positioning — these do, by
driving ``loom.cli.main(argv)`` end to end.
"""

from __future__ import annotations

import json

import pytest

from loom.cli import main


def _run(argv, capsys):
    code = main(argv)
    cap = capsys.readouterr()
    return code, cap.out, cap.err


def test_json_after_verb(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path))
    code, out, _ = _run(["tokenize", "--preset", "financial", "--json"], capsys)
    assert code == 0
    payload = json.loads(out)  # raw envelope on stdout, not a card
    assert payload["verb"] == "tokenize"
    assert payload["data"]["vocab_size"] == 6251


def test_json_before_verb_still_works(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path))
    code, out, _ = _run(["--json", "tokenize", "--preset", "financial"], capsys)
    assert code == 0
    assert json.loads(out)["verb"] == "tokenize"


def test_experiment_after_verb(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path))
    code, out, _ = _run(
        ["tokenize", "--preset", "financial", "--experiment", "exp-after", "--json"],
        capsys,
    )
    assert code == 0
    assert json.loads(out)["experiment"] == "exp-after"


def test_experiment_before_verb_not_clobbered_by_post_verb_default(
    tmp_path, monkeypatch, capsys
):
    """The SUPPRESS default must keep a pre-verb ``--experiment`` from being reset
    by the subparser's (absent) copy of the flag."""
    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path))
    code, out, _ = _run(
        ["--experiment", "exp-before", "tokenize", "--preset", "financial", "--json"],
        capsys,
    )
    assert code == 0
    assert json.loads(out)["experiment"] == "exp-before"


def test_quiet_after_verb_prints_only_pathspec(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path))
    code, out, err = _run(["tokenize", "--preset", "financial", "-q"], capsys)
    assert code == 0
    # Quiet stdout is the machine-pipeable pathspec only — one line, no card glyphs.
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert len(lines) == 1
    assert lines[0].startswith("Corpus/")
    assert "vocab" not in out  # the human card went to stderr (or wasn't printed)


def test_ingest_quiet_after_verb(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LOOM_WORKSPACE", str(tmp_path))
    src = tmp_path / "trades.csv"
    src.write_text("wallet,timestamp,venue,side,item,size_usd\n0xa1,2026-06-01 00:00:00,DEXETH,BUY,WETH,120.0\n")
    code, out, _ = _run(["ingest", str(src), "--name", "dex", "-q"], capsys)
    assert code == 0
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert len(lines) == 1
    assert lines[0].startswith("IngestDataset/")
