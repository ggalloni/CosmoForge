"""Tests for the ``noise_cov1=``/``noise_cov2=`` injection kwargs (ADR-0017, A2).

``Core`` owns dispatch + semantic validation of the noise-covariance seam via
``_resolve_noise_cov``. The injected object is "exactly what the reader would
have returned": the reduced ``(n_active, n_active)`` covariance, pixel-ordered
per ``concatenate_pixact``.
"""

import tempfile
from unittest.mock import patch

import healpy as hp
import numpy as np
import pytest

from cosmocore.settings import InputParams

from .test_core import ConcreteCore


def _params(nside=4, do_cross=False):
    params = InputParams()
    params.nside = nside
    params.lmax = 4 * nside
    params.nfields = 1
    params.spins = [0]
    params.labels = ["T"]
    params.ordering = "RING"
    params.do_cross = do_cross
    params.covmatfile1 = "/nonexistent/ncvm1.bin"
    params.covmatfile2 = "/nonexistent/ncvm2.bin"

    npix = 12 * nside**2
    mask = np.ones(npix, dtype=np.float64)
    mask[: npix // 2] = 0.0
    mask_file = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    hp.write_map(mask_file.name, mask, overwrite=True)
    params.maskfile = mask_file.name
    return params


def _n_active(core):
    return int(core.collection.total_active_pixels)


def test_injected_noise_cov1_matches_file_path():
    """Injecting ``noise_cov1=A`` yields the same ``noise_cov1`` as reading A."""
    params = _params()

    # File path (mocked reader returns the reduced array A).
    ref = ConcreteCore(params)
    ref.setup_fields()
    ref.setup_geometry()
    n = _n_active(ref)
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n))
    A = A @ A.T  # symmetric PD, like a real covariance
    with patch("cosmocore.core.read_covmat", return_value=A.copy()):
        ref.setup_covariance_matrices()

    # Injection path: same A, no file read.
    inj = ConcreteCore(params, noise_cov1=A.copy())
    inj.setup_fields()
    inj.setup_geometry()
    with patch("cosmocore.core.read_covmat", side_effect=AssertionError("read")):
        inj.setup_covariance_matrices()

    np.testing.assert_array_equal(inj.noise_cov1, ref.noise_cov1)


def test_injected_noise_cov_ownership_contract():
    """The injected array is used as-is and never mutated (ADR-0017).

    No defensive copy on the way in, and no scaling on the way out: the seam
    takes "the reduced array" and nothing else, so the resolved covariance is
    the very object the caller handed over.
    """
    params = _params()
    core = ConcreteCore(params)
    core.setup_fields()
    core.setup_geometry()
    n = _n_active(core)
    A = np.eye(n) * 0.1
    A_before = A.copy()
    core._injected_noise_cov1 = A
    core.setup_covariance_matrices()

    assert core.noise_cov1 is A  # no defensive copy
    np.testing.assert_array_equal(A, A_before)  # caller's array untouched


def test_injected_noise_cov1_wrong_shape_raises():
    """A mis-sized injected covariance fails loudly at resolution."""
    params = _params()
    n_bad = 3
    core = ConcreteCore(params, noise_cov1=np.eye(n_bad))
    core.setup_fields()
    core.setup_geometry()
    with pytest.raises(ValueError, match="injected noise covariance has shape"):
        core.setup_covariance_matrices()


def test_do_cross_injects_noise_cov2():
    """``noise_cov2`` is injected independently when ``do_cross`` is set."""
    params = _params(do_cross=True)
    core = ConcreteCore(params)
    core.setup_fields()
    core.setup_geometry()
    n = _n_active(core)
    A1 = np.eye(n) * 0.1
    A2 = np.eye(n) * 0.2
    core._injected_noise_cov1 = A1.copy()
    core._injected_noise_cov2 = A2.copy()
    with patch("cosmocore.core.read_covmat", side_effect=AssertionError("read")):
        ncov1, ncov2 = core.setup_covariance_matrices()
    np.testing.assert_array_equal(ncov1, A1)
    np.testing.assert_array_equal(ncov2, A2)
