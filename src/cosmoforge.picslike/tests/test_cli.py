"""End-to-end tests for the ``picslike-run`` console script.

The script this replaces, ``main_picslike.py``, called three methods that no
longer existed on ``PICSLike`` and treated ``InputParams`` as a dict. Nothing
ever ran it, so nothing caught that. These tests run the entry point against the
fast nside=4 fixture so the same rot cannot recur.

They run on a ``sandboxed_config``: the shipped fixture config points its
``out*`` keys back into ``tests/data/``, so a CLI run driven by one would write
artifacts into the fixture tree. With those keys stripped the run is hermetic,
which lets "wrote nothing" be an assertion about the filesystem rather than a
mock of the writer.
"""

import sys
from importlib.metadata import entry_points

import pytest

from picslike.cli import main


@pytest.fixture
def cfg(sandboxed_config):
    """The fast 2x2-grid config, with absolute inputs and no ``out*`` keys."""
    return sandboxed_config("tests/data/nside4/TQU/fast_config.yaml")


def test_console_script_is_registered():
    """The pyproject wiring resolves; a typo there breaks the command silently."""
    scripts = {ep.name: ep.value for ep in entry_points(group="console_scripts")}
    assert scripts["picslike-run"] == "picslike.cli:main"


def test_run_writes_nothing_without_out(cfg, monkeypatch, tmp_path):
    """Without --out the grid is evaluated and discarded, not written (ADR-0015)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["picslike-run", cfg])

    main()

    assert list(tmp_path.rglob("*")) == [], "no --out, yet something was written"


def test_run_writes_when_out_is_given(cfg, monkeypatch, tmp_path):
    """--out persists the likelihood results."""
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "results.npz"
    monkeypatch.setattr(sys, "argv", ["picslike-run", cfg, "--out", str(out)])

    main()

    assert list(tmp_path.rglob("*")), "--out was given but nothing was written"
