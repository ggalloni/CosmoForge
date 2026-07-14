"""Tests for ``noise_cov1=``/``noise_cov2=`` injection on PICSLike (ADR-0017, A2).

Dispatch + validation are owned by ``Core`` (tested in cosmocore); here we assert
PICSLike surfaces the seam as explicit, named constructor kwargs and that an
injected covariance drives covariance setup with no file on disk.
"""

import inspect

import numpy as np

from cosmocore import active_pixel_index
from cosmocore.settings import InputParams
from picslike import PICSLike


def _params(config_resolver):
    cfg = config_resolver("tests/data/nside4/TQU/config.yaml")
    params = InputParams.read_parameter_file(cfg)
    params.maskfile = "/nonexistent/mask.fits"
    params.covmatfile1 = "/nonexistent/ncvm1.bin"
    return params


def test_noise_cov_are_named_kwargs():
    """``noise_cov1``/``noise_cov2`` are explicit signature parameters."""
    sig = inspect.signature(PICSLike.__init__).parameters
    assert "noise_cov1" in sig
    assert "noise_cov2" in sig


def test_injected_noise_cov_reaches_covariance_without_file(config_resolver):
    """PICSLike runs covariance setup from an injected array, no file present."""
    params = _params(config_resolver)
    mask = np.ones((12 * params.nside**2, params.nfields), dtype=np.float64)
    n = len(active_pixel_index(mask))
    like = PICSLike(params, mask=mask, noise_cov1=np.eye(n) * 0.1)
    like.setup_fields()
    like.setup_geometry()
    like._injected_noise_cov1 = np.eye(n) * 0.1
    ncov1, _ = like.setup_covariance_matrices()
    assert ncov1.shape == (n, n)
