"""Both ComputationBasis subclasses must expose get_full_inverse.

Returns the full ``n_pix x n_pix`` inverse covariance.

- HarmonicBasis: exact via the SMW formula.
- PixelBasis in pixel-direct mode: exact (``U`` is the identity).
- PixelBasis on a truncated compressed basis: the restriction
  ``U @ get_inverse(C_ell) @ U.T`` lifted back to ``n_pix`` — not the
  inverse of the full covariance; the kept subspace's inverse, lifted.

(Pixel-direct mode requires a ``FieldCollection`` setup that is heavier
than this regression deserves; it is covered by the broader pixel-direct
test suite. Here we cover the ABC contract, harmonic, and compressed
pixel.)
"""

import numpy as np
from numpy.testing import assert_allclose

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
    assert_allclose(full_inv, full_inv.T, atol=1e-12, rtol=0)
    eigs = np.linalg.eigvalsh(0.5 * (full_inv + full_inv.T))
    assert np.all(eigs > 0), "full inverse must be positive definite"


def test_pixel_compressed_get_full_inverse_is_kept_subspace_lift(uniform_sky_setup):
    """Truncated compressed pixel basis: ``get_full_inverse`` lifts the
    basis-space inverse back to ``n_pix`` via ``U @ . @ U.T``. The result
    is symmetric, has rank equal to ``dim``, and matches the explicit
    reconstruction from public quantities."""
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
    assert_allclose(full_inv, full_inv.T, atol=1e-12, rtol=0)

    # Reproduce via the public projector: ``U C_basis^{-1} U^T``. The
    # projector property returns ``U^T``, so ``U == projector.T``.
    U = bm.projector.T
    basis_inv = bm.get_inverse(C_ell)
    assert_allclose(full_inv, U @ basis_inv @ U.T, atol=1e-12, rtol=1e-10)

    # The lift has rank exactly ``dim`` (kept eigenvectors are
    # orthonormal and ``C_basis`` is positive definite).
    assert np.linalg.matrix_rank(full_inv) == bm.dim
