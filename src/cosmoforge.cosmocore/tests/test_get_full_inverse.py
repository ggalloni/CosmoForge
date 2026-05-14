"""Both ComputationBasis subclasses must expose get_full_inverse.

Returns the full ``n_pix x n_pix`` inverse covariance regardless of basis.
For HarmonicBasis: exact via SMW. For PixelBasis: derived from compressed
quantities via ``U C_basis^{-1} U^T`` (best available; lossy on a
truncated subspace).
"""

import numpy as np

from cosmocore.basis import create_computation_basis
from cosmocore.basis.base import ComputationBasis


def test_abc_declares_get_full_inverse():
    assert hasattr(ComputationBasis, "get_full_inverse")
    method = ComputationBasis.get_full_inverse
    assert getattr(method, "__isabstractmethod__", False), (
        "get_full_inverse must be declared abstract on the ABC"
    )


def test_harmonic_get_full_inverse_returns_npix_npix(uniform_sky_setup):
    """Harmonic exposes the exact SMW-based full inverse under the
    polymorphic name `get_full_inverse`."""
    setup = uniform_sky_setup
    bm = create_computation_basis(
        method="harmonic",
        N=setup["N"],
        theta=setup["theta"],
        phi=setup["phi"],
        lmax_signal=setup["lmax"],
    )
    bm.setup()
    C_ell = np.ones(setup["lmax"] + 1) * 1e-3

    full_inv = bm.get_full_inverse(C_ell)
    assert full_inv.shape == (setup["n_pix"], setup["n_pix"])
    eigs = np.linalg.eigvalsh(0.5 * (full_inv + full_inv.T))
    assert np.all(eigs > 0), "full inverse must be positive definite"


def test_pixel_get_full_inverse_returns_npix_npix(uniform_sky_setup):
    setup = uniform_sky_setup
    bm = create_computation_basis(
        method="pixel",
        N=setup["N"],
        theta=setup["theta"],
        phi=setup["phi"],
        lmax_signal=setup["lmax"],
        epsilon=1e-6,
    )
    bm.setup()
    C_ell = np.ones(setup["lmax"] + 1) * 1e-3

    full_inv = bm.get_full_inverse(C_ell)
    assert full_inv.shape == (setup["n_pix"], setup["n_pix"])
