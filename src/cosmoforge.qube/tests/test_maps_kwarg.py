"""Tests for the ``maps1=``/``maps2=`` injection kwargs on Spectra (ADR-0017, A3).

Maps are a Spectra-only seam (Fisher never reads them), so dispatch + validation
live on ``Spectra._resolve_maps``. The injected object is "exactly what
:func:`read_maps` would have returned": the reduced ``(n_active, n_sims)`` float64
array, *already calibrated* (``read_maps`` applies ``calibration`` internally, so
the injected array is taken as-is — calibration is the file adapter's job only).
"""

import inspect
import os
import tempfile

import numpy as np
import pytest

from cosmocore.settings import InputParams
from qube import Fisher, Spectra


def _write_sims(path, npix, nsims, seed=0):
    """Write an (n_fields=1, npix, nsims) .npy sims stack for the T config."""
    rng = np.random.default_rng(seed)
    sims = rng.standard_normal((1, npix, nsims)).astype(np.float64)
    np.save(path, sims)


def _params(config_resolver, tmpdir, *, nsims=3):
    cfg = config_resolver("tests/data/nside4/T/config.yaml")
    params = InputParams.read_parameter_file(cfg)
    params.nsims = nsims
    sims_path = os.path.join(tmpdir, "sims.npy")
    _write_sims(sims_path, 12 * params.nside**2, nsims)
    params.inputmapfile1 = sims_path
    params.inputmapfile2 = sims_path
    return params


@pytest.mark.parametrize("name", ["maps1", "maps2"])
def test_maps_is_named_kwarg(name):
    """``maps1``/``maps2`` are explicit signature parameters on Spectra."""
    assert name in inspect.signature(Spectra.__init__).parameters


def test_injected_maps1_matches_file_path(config_resolver):
    """Injecting ``maps1=A`` yields the same ``maps1`` as reading A from file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        params = _params(config_resolver, tmpdir)
        fisher = Fisher(params)
        fisher.run()

        # File adapter: read the reduced maps from the .npy on disk.
        ref = Spectra(params, fisher=fisher)
        ref.setup_fields()
        ref.setup_geometry()
        ref.setup_maps()
        ref_maps1 = ref.maps1.copy()

        # Injection adapter: hand the same reduced array in; no file read.
        params.inputmapfile1 = "/nonexistent/sims.npy"
        inj = Spectra(params, fisher=fisher, maps1=ref_maps1.copy())
        inj.setup_fields()
        inj.setup_geometry()
        inj.setup_maps()

        np.testing.assert_array_equal(inj.maps1, ref_maps1)


def test_injected_maps_taken_as_is_no_recalibration(config_resolver):
    """Injected maps are used verbatim; ``calibration`` is NOT re-applied
    (it is the file adapter's job — the injected array is already calibrated)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        params = _params(config_resolver, tmpdir)
        params.calibration = 2.0
        fisher = Fisher(params)
        fisher.run()

        spec = Spectra(params, fisher=fisher)
        spec.setup_fields()
        spec.setup_geometry()
        n = int(spec.collection.total_active_pixels)
        arr = np.arange(n * params.nsims, dtype=np.float64).reshape(n, params.nsims)
        spec._injected_maps1 = arr.copy()
        spec.setup_maps()
        np.testing.assert_array_equal(spec.maps1, arr)  # unchanged, not *2


def test_injected_maps1_wrong_shape_raises(config_resolver):
    """A mis-sized injected maps array fails loudly at resolution."""
    with tempfile.TemporaryDirectory() as tmpdir:
        params = _params(config_resolver, tmpdir)
        fisher = Fisher(params)
        fisher.run()
        spec = Spectra(params, fisher=fisher)
        spec.setup_fields()
        spec.setup_geometry()
        n = int(spec.collection.total_active_pixels)
        spec._injected_maps1 = np.ones((n, params.nsims + 1), dtype=np.float64)
        with pytest.raises(ValueError, match="injected maps have shape"):
            spec.setup_maps()


def test_do_cross_injects_maps2(config_resolver):
    """``maps2`` is resolved independently from injection when ``do_cross`` is set."""
    with tempfile.TemporaryDirectory() as tmpdir:
        params = _params(config_resolver, tmpdir)
        fisher = Fisher(params)
        fisher.run()
        spec = Spectra(params, fisher=fisher)
        spec.setup_fields()
        spec.setup_geometry()
        n = int(spec.collection.total_active_pixels)
        a1 = np.ones((n, params.nsims), dtype=np.float64)
        a2 = np.full((n, params.nsims), 2.0, dtype=np.float64)
        # setup_maps only needs ntot + the injected array for the cross branch,
        # so exercising the maps2 wiring does not require a real cross Fisher.
        spec.params.do_cross = True
        spec._injected_maps1 = a1.copy()
        spec._injected_maps2 = a2.copy()
        spec.setup_maps()
        np.testing.assert_array_equal(spec.maps1, a1)
        np.testing.assert_array_equal(spec.maps2, a2)


def test_spectra_runs_fully_in_memory(config_resolver):
    """Acceptance (A3): Spectra.run() from injected maps + mask + noise_cov,
    with no map/mask/covariance file on disk (cls/beam still file-bound)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        params = _params(config_resolver, tmpdir)
        mask = np.ones((12 * params.nside**2, params.nfields), dtype=np.float64)

        probe = Fisher(params, mask=mask, noise_cov1=np.eye(1))
        probe.setup_fields()
        probe.setup_geometry()
        n = int(probe.collection.total_active_pixels)

        rng = np.random.default_rng(1)
        noise_cov1 = np.eye(n) * 0.1
        maps1 = rng.standard_normal((n, params.nsims))
        params.maskfile = "/nonexistent/mask.fits"
        params.covmatfile1 = "/nonexistent/ncvm1.bin"
        params.inputmapfile1 = "/nonexistent/sims.npy"

        spec = Spectra(params, mask=mask, noise_cov1=noise_cov1, maps1=maps1)
        spec.run()
        assert spec.qml_results is not None
        assert np.all(np.isfinite(spec.qml_results))
