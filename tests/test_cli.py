"""Smoke tests for palfitology.cli.

These don't actually run a fit -- they just ensure the CLI parser is
constructed correctly, subcommands exist, and import succeeds.
"""

from __future__ import annotations

import pytest

from palfitology import __version__
from palfitology.cli import build_parser, main


def test_version_is_set():
    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_parser_has_expected_subcommands():
    parser = build_parser()
    # Inspect the subparsers action; one of them holds our subcommand choices.
    subparser_actions = [
        a for a in parser._actions if a.__class__.__name__ == "_SubParsersAction"
    ]
    assert len(subparser_actions) == 1
    commands = set(subparser_actions[0].choices.keys())
    assert {"fit-pa", "download", "consensus", "galfit"}.issubset(commands)


def test_fit_pa_help_does_not_crash(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["fit-pa", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--workers" in out
    assert "--bands" in out


def test_stub_subcommand_returns_nonzero(caplog):
    # `download` is a planned stub; it should fail explicitly rather than
    # silently no-op.
    rc = main(["download"])
    assert rc != 0
