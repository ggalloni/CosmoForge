"""Acceptance tests for opt-in persistence (Phase B of the in-memory pipeline).

The load-bearing assertion is that a Fisher + Spectra run with no ``out*``
paths set leaves the working directory untouched. It xfails today because the
pipeline still writes implicitly (out* defaults resolve to ``outputs/*``) and
Spectra reads Fisher's noise/inverse covariance back from disk. Slice B1 makes
the write gates real; Slice B2 replaces the file handshake with the live
``fisher=`` seam and un-xfails this test.
"""

import os
import tempfile

import pytest
import yaml

from qube import Fisher, Spectra

# Keys that name output artifacts; stripped so defaults (which, pre-B1, still
# resolve to ``outputs/*``) are what the test exercises.
_OUTPUT_KEYS = {
    "output_geometry_file",
    "outnoisecovmat1",
    "outnoisecovmat2",
    "outinvcovmatfile1",
    "outinvcovmatfile2",
    "outfilefisher",
    "outcovmatfile",
    "outerrfile",
}


@pytest.fixture
def nside8_params(local_path):
    """Path to an nside-8 T config with absolute inputs and no ``out*`` keys.

    Inputs are made absolute so the run is independent of the (chdir'd) working
    directory; every output key is removed so writes fall back to defaults,
    which is exactly what opt-in persistence must silence. The config file lives
    in the system temp dir, not the test's tmp_path, so the workdir stays clean.
    """
    src = os.path.join(local_path, "tests/data/nside8/T/config.yaml")
    with open(src) as f:
        config = yaml.safe_load(f)

    for key, value in list(config.items()):
        if key in _OUTPUT_KEYS:
            del config[key]
        elif isinstance(value, str) and value.startswith("../"):
            config[key] = os.path.join(local_path, value[3:])

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(config, tmp, default_flow_style=False)
    tmp.close()
    yield tmp.name
    os.unlink(tmp.name)


def test_fisher_run_writes_nothing_when_out_paths_unset(
    tmp_path, monkeypatch, nside8_params
):
    """Fisher.run alone with no out* paths set must not create any file (ADR-0015).

    This is the Fisher half of the clean-workdir guarantee; it is not blocked by
    the pixel-basis Fisher->Spectra file handshake (that read is Spectra's, gated
    behind B2), so it can pass at B1.
    """
    monkeypatch.chdir(tmp_path)
    fisher = Fisher(nside8_params)
    fisher.run()
    assert fisher.get_fisher_matrix() is not None
    assert list(tmp_path.rglob("*")) == []


@pytest.mark.xfail(
    strict=True, reason="Phase B not implemented: pipeline still writes implicitly"
)
def test_fisher_spectra_leave_workdir_untouched(tmp_path, monkeypatch, nside8_params):
    """Fisher.run + Spectra.run with no out* paths set must not create any file."""
    monkeypatch.chdir(tmp_path)
    fisher = Fisher(nside8_params)
    fisher.run()
    spectra = Spectra(nside8_params, fisher=fisher)
    spectra.run()
    assert spectra.get_power_spectra(mode="deconvolved") is not None
    # THE acceptance criterion: nothing landed on disk.
    assert list(tmp_path.rglob("*")) == []
