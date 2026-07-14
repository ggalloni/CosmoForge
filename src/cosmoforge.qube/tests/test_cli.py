"""End-to-end tests for the QUBE console scripts.

These drive the entry points against the nside=4 fixtures rather than merely
checking that ``--help`` parses. The scripts they replace were never executed by
anything, and drifted until they called methods that no longer existed; only a
test that actually runs the pipeline catches that.
"""

import sys
from importlib.metadata import entry_points

import pytest

from qube import Spectra
from qube.cli import DEFAULT_CONFIG, fisher_main, spectra_main


@pytest.fixture
def config(config_resolver):
    """A cheap single-field configuration."""
    return config_resolver("tests/data/nside4/T/config.yaml")


def test_console_scripts_are_registered():
    """The pyproject wiring resolves; a typo there breaks the command silently."""
    scripts = {ep.name: ep.value for ep in entry_points(group="console_scripts")}
    assert scripts["qube-fisher-run"] == "qube.cli:fisher_main"
    assert scripts["qube-spectra-run"] == "qube.cli:spectra_main"
    assert scripts["qube-run"] == "qube.cli:spectra_main"


def test_packaged_default_config_exists():
    """``qube-run`` with no argument falls back to this file."""
    assert DEFAULT_CONFIG.is_file()


def test_fisher_run(config, monkeypatch):
    """qube-fisher-run drives Fisher to completion."""
    monkeypatch.setattr(sys, "argv", ["qube-fisher-run", config])
    fisher_main()


def test_spectra_run_writes_when_out_is_given(config, monkeypatch, tmp_path):
    """qube-spectra-run persists only when --out is passed (ADR-0015)."""
    out = tmp_path / "spectra.txt"
    monkeypatch.setattr(sys, "argv", ["qube-spectra-run", config, "--out", str(out)])
    spectra_main()

    assert any(tmp_path.iterdir()), "--out was given but nothing was written"


def test_spectra_run_writes_nothing_without_out(config, monkeypatch):
    """Without --out the estimates are computed and discarded, not written.

    Asserted on the writer rather than on the filesystem: the config's paths are
    relative to the repo root, so chdir-ing into a tmp_path to watch it would
    break the fixture's own inputs. `test_no_implicit_io.py` covers the
    filesystem-level invariant.
    """
    calls = []
    monkeypatch.setattr(
        Spectra, "write_power_spectra", lambda self, **kw: calls.append(kw)
    )
    monkeypatch.setattr(sys, "argv", ["qube-spectra-run", config])
    spectra_main()

    assert calls == [], "no --out, yet the writer was called"


@pytest.mark.parametrize("mode", ["deconvolved", "decorrelated", "convolved"])
def test_spectra_run_accepts_every_mode(config, monkeypatch, tmp_path, mode):
    """Each normalisation reaches the writer. Convolved returns a 3-tuple, so it
    exercises the branch that unpacks the estimates out of it."""
    out = tmp_path / f"spectra_{mode}.txt"
    monkeypatch.setattr(
        sys,
        "argv",
        ["qube-spectra-run", config, "--mode", mode, "--out", str(out)],
    )
    spectra_main()

    assert any(tmp_path.iterdir()), f"mode={mode} wrote nothing"


def test_spectra_run_rejects_an_unknown_mode(config, monkeypatch):
    """argparse guards the mode vocabulary."""
    monkeypatch.setattr(sys, "argv", ["qube-spectra-run", config, "--mode", "nonsense"])
    with pytest.raises(SystemExit):
        spectra_main()
