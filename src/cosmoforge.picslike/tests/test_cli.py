"""End-to-end tests for the ``picslike-run`` console script.

The script this replaces, ``main_picslike.py``, called three methods that no
longer existed on ``PICSLike`` and treated ``InputParams`` as a dict. Nothing
ever ran it, so nothing caught that. These tests run the entry point against the
fast nside=4 fixture so the same rot cannot recur.
"""

import sys
from importlib.metadata import entry_points

import pytest

from picslike import PICSLike
from picslike.cli import main


def test_console_script_is_registered():
    """The pyproject wiring resolves; a typo there breaks the command silently."""
    scripts = {ep.name: ep.value for ep in entry_points(group="console_scripts")}
    assert scripts["picslike-run"] == "picslike.cli:main"


def test_run_writes_when_out_is_given(fast_config_path, monkeypatch, tmp_path):
    """picslike-run evaluates the grid and persists when --out is passed."""
    out = tmp_path / "results.npz"
    monkeypatch.setattr(
        sys, "argv", ["picslike-run", fast_config_path, "--out", str(out)]
    )
    main()

    assert any(tmp_path.iterdir()), "--out was given but nothing was written"


def test_run_writes_nothing_without_out(fast_config_path, monkeypatch):
    """Without --out the grid is evaluated and discarded, not written (ADR-0015).

    Asserted on the writer rather than the filesystem: the config's paths are
    relative to the repo root, so chdir-ing into a tmp_path to watch it would
    break the fixture's own inputs.
    """
    calls = []
    monkeypatch.setattr(PICSLike, "save_results", lambda self, path: calls.append(path))
    monkeypatch.setattr(sys, "argv", ["picslike-run", fast_config_path])
    main()

    assert calls == [], "no --out, yet the writer was called"


def test_config_is_required(monkeypatch):
    """Unlike QUBE there is no packaged default, so the path is mandatory."""
    monkeypatch.setattr(sys, "argv", ["picslike-run"])
    with pytest.raises(SystemExit):
        main()
