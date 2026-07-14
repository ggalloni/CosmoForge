"""Tests for ``noise_cov1=``/``noise_cov2=`` injection on Fisher/Spectra (ADR-0017, A2).

Dispatch + validation are owned by ``Core`` (tested in cosmocore); here we assert
the orchestration classes surface the seam as *explicit, named* constructor kwargs
and forward them — including into the internal ``Fisher`` that ``Spectra`` builds.
"""

import inspect

import numpy as np
import pytest

from cosmocore import active_pixel_index
from cosmocore.settings import InputParams
from qube import Fisher, Spectra


def _params(config_resolver):
    cfg = config_resolver("tests/data/nside4/T/config.yaml")
    params = InputParams.read_parameter_file(cfg)
    params.maskfile = "/nonexistent/mask.fits"
    params.covmatfile1 = "/nonexistent/ncvm1.bin"
    return params


def _full_sky_mask(params):
    return np.ones(12 * params.nside**2, dtype=np.float64)


@pytest.mark.parametrize("cls", [Fisher, Spectra])
@pytest.mark.parametrize("name", ["noise_cov1", "noise_cov2"])
def test_noise_cov_is_named_kwarg(cls, name):
    """``noise_cov1``/``noise_cov2`` are explicit signature parameters."""
    assert name in inspect.signature(cls.__init__).parameters


def test_injected_noise_cov_reaches_covariance_without_file(config_resolver):
    """Fisher runs covariance setup from an injected array, no file present."""
    params = _params(config_resolver)
    mask = _full_sky_mask(params)
    n = len(active_pixel_index(mask))
    core = Fisher(params, mask=mask, noise_cov1=np.eye(n) * 0.1)
    core.setup_fields()
    core.setup_geometry()
    ncov1, _ = core.setup_covariance_matrices()
    assert ncov1.shape == (n, n)


def test_spectra_forwards_noise_cov_to_internal_fisher(config_resolver):
    """Injected noise_cov1 flows into the Fisher that Spectra builds internally."""
    params = _params(config_resolver)
    mask = _full_sky_mask(params)
    # The active-pixel count comes straight off the mask (ADR-0017).
    n = len(active_pixel_index(mask))

    spec = Spectra(params, mask=mask, noise_cov1=np.eye(n) * 0.1)
    assert spec.fisher_instance._injected_noise_cov1 is not None


def test_fisher_runs_fully_in_memory(config_resolver):
    """Acceptance (A2): Fisher.run() from params + mask array + noise_cov array,
    with neither a mask nor a covariance file on disk."""
    params = _params(config_resolver)
    mask = _full_sky_mask(params)
    n = len(active_pixel_index(mask))

    noise_cov1 = np.eye(n) * 0.1
    fisher = Fisher(params, mask=mask, noise_cov1=noise_cov1)
    fisher.run()
    assert fisher.fisher is not None
    assert np.all(np.isfinite(fisher.fisher))
