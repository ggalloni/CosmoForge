"""Acceptance tests for opt-in persistence (Phase B of the in-memory pipeline).

The load-bearing assertion is that a Fisher + Spectra run with no ``out*``
paths set leaves the working directory untouched: write gates are opt-in
(ADR-0015) and Spectra takes Fisher's noise/inverse covariance in memory over
the live ``fisher=`` seam rather than reading it back from disk (ADR-0016).
"""

import os
import tempfile

import numpy as np
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


def _nside8_config(local_path):
    """nside-8 T config as a dict with absolute inputs and no out* keys."""
    src = os.path.join(local_path, "tests/data/nside8/T/config.yaml")
    with open(src) as f:
        config = yaml.safe_load(f)
    resolved = {}
    for key, value in config.items():
        if key in _OUTPUT_KEYS:
            continue
        if isinstance(value, str) and value.startswith("../"):
            resolved[key] = os.path.join(local_path, value[3:])
        else:
            resolved[key] = value
    return resolved


def _write_yaml(config, path):
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    return str(path)


def test_live_alias_and_disk_adapter_agree(tmp_path, local_path):
    """ADR-0016: the in-memory alias handoff and the disk read adapter agree.

    Nulling the live Fisher's retained N forces the resolution priority past the
    in-memory alias (i) onto the ``out*`` disk adapter (ii); the estimated
    spectra must be identical either way.
    """
    config = _nside8_config(local_path)

    live_cfg = _write_yaml(config, tmp_path / "live.yaml")
    f_live = Fisher(live_cfg)
    f_live.run()
    s_live = Spectra(live_cfg, fisher=f_live)
    s_live.run()
    ps_live = s_live.get_power_spectra(mode="deconvolved")

    disk = dict(config)
    disk["outnoisecovmat1"] = str(tmp_path / "N1.bin")
    disk["outinvcovmatfile1"] = str(tmp_path / "invC1.bin")
    disk_cfg = _write_yaml(disk, tmp_path / "disk.yaml")
    f_disk = Fisher(disk_cfg)
    f_disk.run()
    f_disk.reduced_noise_cov1 = None
    f_disk.reduced_noise_cov2 = None
    s_disk = Spectra(disk_cfg, fisher=f_disk)
    s_disk.run()
    ps_disk = s_disk.get_power_spectra(mode="deconvolved")

    np.testing.assert_array_equal(ps_live, ps_disk)
