"""End-to-end tests for the QUBE console scripts.

These drive the entry points against the nside=4 fixtures rather than merely
checking that ``--help`` parses. The scripts they replace were never executed by
anything, and drifted until they called methods that no longer existed; only a
test that actually runs the pipeline catches that.

Every test runs on a ``sandboxed_config``: the shipped fixture configs point
their ``out*`` keys back into ``tests/data/``, so a CLI run driven by one would
write artifacts into the fixture tree. With those keys stripped the run is
hermetic, which lets "wrote nothing" be an assertion about the filesystem rather
than a mock of the writer.
"""

import sys
from importlib.metadata import entry_points

import pytest

from qube.cli import DEFAULT_CONFIG, fisher_main, spectra_main


@pytest.fixture
def cfg(sandboxed_config):
    """A cheap single-field config with absolute inputs and no ``out*`` keys."""
    return sandboxed_config("tests/data/nside4/T/config.yaml")


def test_console_scripts_are_registered():
    """The pyproject wiring resolves; a typo there breaks the command silently."""
    scripts = {ep.name: ep.value for ep in entry_points(group="console_scripts")}
    assert scripts["qube-fisher-run"] == "qube.cli:fisher_main"
    assert scripts["qube-spectra-run"] == "qube.cli:spectra_main"
    assert scripts["qube-run"] == "qube.cli:spectra_main"


def test_packaged_default_config_exists():
    """``qube-run`` with no argument falls back to this file."""
    assert DEFAULT_CONFIG.is_file()


def test_fisher_run_writes_nothing_without_an_out_path(cfg, monkeypatch, tmp_path):
    """qube-fisher-run drives Fisher to completion and, with ``outfilefisher``
    unset, leaves the working directory untouched (ADR-0015)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["qube-fisher-run", cfg])

    fisher_main()

    assert list(tmp_path.rglob("*")) == [], "no out* path, yet something was written"


def test_spectra_run_writes_nothing_without_out(cfg, monkeypatch, tmp_path):
    """Without --out the estimates are computed and discarded, not written."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["qube-spectra-run", cfg])

    spectra_main()

    assert list(tmp_path.rglob("*")) == [], "no --out, yet something was written"


@pytest.mark.parametrize("mode", ["deconvolved", "decorrelated", "convolved"])
def test_spectra_run_writes_when_out_is_given(cfg, monkeypatch, tmp_path, mode):
    """--out persists the estimates, in every normalisation."""
    monkeypatch.chdir(tmp_path)
    out = tmp_path / f"spectra_{mode}.txt"
    monkeypatch.setattr(
        sys, "argv", ["qube-spectra-run", cfg, "--mode", mode, "--out", str(out)]
    )

    spectra_main()

    assert list(tmp_path.rglob("*")), f"--out given for mode={mode}, nothing written"
