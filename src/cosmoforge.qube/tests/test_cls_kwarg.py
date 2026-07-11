"""Tests for ``cls_data=``/``fiducial_cls=`` injection on Fisher/Spectra (ADR-0017, A4).

Dispatch lives in ``Core.setup_cls`` (forwards ``cls_data`` into ``set_cls``) and the
S_fixed fiducial re-read in ``setup_computation_basis`` (``fiducial_cls`` wins over the
file). Here we assert the orchestration classes surface the kwargs, forward them into
Spectra's internal Fisher, guard against combining with ``fisher=``, and drive a
harmonic-basis run with the spectra as in-memory dicts (no cls/fiducial file).
"""

import inspect
import os
import tempfile

import numpy as np
import pytest

from cosmocore.in_out import readcl
from cosmocore.settings import InputParams
from qube import Fisher, Spectra


def _params(config_resolver, tmpdir, *, nsims=3):
    cfg = config_resolver("tests/data/nside4/T/config.yaml")
    params = InputParams.read_parameter_file(cfg)
    params.nsims = nsims
    rng = np.random.default_rng(0)
    sims = rng.standard_normal((1, 12 * params.nside**2, nsims)).astype(np.float64)
    sims_path = os.path.join(tmpdir, "sims.npy")
    np.save(sims_path, sims)
    params.inputmapfile1 = sims_path
    return params


@pytest.mark.parametrize("cls", [Fisher, Spectra])
@pytest.mark.parametrize("name", ["cls_data", "fiducial_cls"])
def test_cls_kwargs_are_named(cls, name):
    """``cls_data``/``fiducial_cls`` are explicit signature parameters."""
    assert name in inspect.signature(cls.__init__).parameters


def test_spectra_forwards_cls_to_internal_fisher(config_resolver):
    """Injected cls_data/fiducial_cls flow into the Fisher Spectra builds."""
    with tempfile.TemporaryDirectory() as tmpdir:
        params = _params(config_resolver, tmpdir)
        d = readcl(params.inputclfile, params, lmax=4 * params.nside)
        spec = Spectra(params, cls_data=d, fiducial_cls=d)
        assert spec.fisher_instance._injected_cls_data is not None
        assert spec.fisher_instance._injected_fiducial_cls is not None


def test_spectra_rejects_cls_when_reusing_fisher(config_resolver):
    """``cls_data=``/``fiducial_cls=`` cannot be combined with ``fisher=``."""
    with tempfile.TemporaryDirectory() as tmpdir:
        params = _params(config_resolver, tmpdir)
        fisher = Fisher(params)
        fisher.run()
        d = readcl(params.inputclfile, params, lmax=4 * params.nside)
        with pytest.raises(ValueError, match="cls_data= or fiducial_cls= cannot be used"):
            Spectra(params, fisher=fisher, cls_data=d)


def test_harmonic_run_fully_in_memory(config_resolver):
    """Acceptance (A4): harmonic-basis Fisher+Spectra from injected cls/fiducial
    (plus mask/noise_cov/maps), no cls/fiducial/mask/cov/map file (beam still on disk).

    The nside4/T config has lmax=8 < lmax_signal=16, so the S_fixed / SMW path
    triggers and exercises the fiducial injection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        params = _params(config_resolver, tmpdir)
        lmax_signal = 4 * params.nside
        cls = readcl(params.inputclfile, params, lmax=lmax_signal)

        mask = np.ones((12 * params.nside**2, params.nfields), dtype=np.float64)
        probe = Fisher(params, mask=mask, noise_cov1=np.eye(1))
        probe.setup_fields()
        probe.setup_geometry()
        n = int(probe.collection.total_active_pixels)
        noise_cov1 = np.eye(n) * 0.1
        maps1 = np.random.default_rng(1).standard_normal((n, params.nsims))

        params.inputclfile = "/nonexistent/cls.txt"
        params.fiducialfile = "/nonexistent/fiducial.txt"
        params.maskfile = "/nonexistent/mask.fits"
        params.covmatfile1 = "/nonexistent/ncvm1.bin"
        params.inputmapfile1 = "/nonexistent/sims.npy"

        kw = dict(mask=mask, noise_cov1=noise_cov1, cls_data=cls, fiducial_cls=cls)
        fisher = Fisher(params, **kw)
        fisher.run()
        assert fisher.fisher is not None and np.all(np.isfinite(fisher.fisher))

        spec = Spectra(params, maps1=maps1, **kw)
        spec.run()
        assert spec.qml_results is not None
        assert np.all(np.isfinite(spec.qml_results))


def test_fiducial_injection_is_load_bearing(config_resolver):
    """Guard: the S_fixed/SMW path really reads the fiducial. Without
    ``fiducial_cls=`` (fiducialfile missing) the harmonic run must raise,
    proving the injected fiducial is what makes the disk-free run succeed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        params = _params(config_resolver, tmpdir)
        cls = readcl(params.inputclfile, params, lmax=4 * params.nside)
        mask = np.ones((12 * params.nside**2, params.nfields), dtype=np.float64)
        probe = Fisher(params, mask=mask, noise_cov1=np.eye(1))
        probe.setup_fields()
        probe.setup_geometry()
        n = int(probe.collection.total_active_pixels)

        params.inputclfile = "/nonexistent/cls.txt"
        params.fiducialfile = "/nonexistent/fiducial.txt"
        # cls_data injected, but fiducial_cls deliberately omitted.
        fisher = Fisher(params, mask=mask, noise_cov1=np.eye(n) * 0.1, cls_data=cls)
        with pytest.raises(FileNotFoundError):
            fisher.run()
