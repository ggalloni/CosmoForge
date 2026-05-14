"""Verification (a) for PR #2: refactor equivalence.

The rewritten harmonic.get_weighted_compressed_data must match the
two-step composition that qube/spectra.py was doing by hand:

    w = stable_inner_inv.T @ (bm._V_N_inv @ d)

This is bit-equivalent to the algebraic intent of the legacy form
``w = y - M K^{-1} y`` in noise-dominated regimes; the rewrite is a
precision improvement (no large-cancellation subtraction) in the
cosmic-variance-limited regime. The test pins refactor invariance
against the explicit composition.
"""

import numpy as np
import pytest

from cosmocore.basis import HarmonicBasis
from cosmocore.spectrum_key import SpectrumKey, SpectrumKind


@pytest.fixture
def single_field_basis(simple_compression_setup):
    setup = simple_compression_setup
    hc = HarmonicBasis(
        N=setup["N"],
        theta=setup["theta"],
        phi=setup["phi"],
        lmax_signal=setup["lmax"],
    )
    hc.setup()
    return hc


@pytest.fixture
def multi_field_basis():
    np.random.seed(42)
    n_pix_t = 10
    n_pix_p = 8
    theta_t = np.random.uniform(0.1, np.pi - 0.1, n_pix_t)
    phi_t = np.random.uniform(0, 2 * np.pi, n_pix_t)
    theta_p = np.random.uniform(0.1, np.pi - 0.1, n_pix_p)
    phi_p = np.random.uniform(0, 2 * np.pi, n_pix_p)
    lmax = 6
    total_pix = n_pix_t + 2 * n_pix_p
    N = np.eye(total_pix) * 0.01
    hc = HarmonicBasis(N, (theta_t, theta_p), (phi_t, phi_p), lmax, spins=[0, 2])
    hc.setup()
    return hc


def test_single_field_matches_explicit_composition(single_field_basis):
    bm = single_field_basis
    rng = np.random.default_rng(0)
    d = rng.standard_normal((bm.n_pix, 3))
    C_ell = np.ones(bm.lmax_signal + 1) * 1e-3

    inner_inv = bm.prepare_stable_inner_inv(C_ell)
    expected = inner_inv.T @ (bm._V_N_inv @ d)

    actual = bm.get_weighted_compressed_data(d, C_ell, stable_inner_inv=inner_inv)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=0)


def test_builds_inner_inv_when_omitted(single_field_basis):
    bm = single_field_basis
    rng = np.random.default_rng(0)
    d = rng.standard_normal((bm.n_pix, 3))
    C_ell = np.ones(bm.lmax_signal + 1) * 1e-3

    inner_inv = bm.prepare_stable_inner_inv(C_ell)
    expected = inner_inv.T @ (bm._V_N_inv @ d)

    actual = bm.get_weighted_compressed_data(d, C_ell)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=0)


def test_matches_brute_force_reference(single_field_basis, simple_compression_setup):
    """Pin the SMW identity against a brute-force pixel-space reference.

    Reference: ``w_ref = V C^{-1} d`` computed by explicit pixel-space
    solve, with ``C = N + V^T Λ V``. Guards against regressions in the
    algebra itself, not just refactor equivalence with the previous
    composition.
    """
    bm = single_field_basis
    n_pix = bm.n_pix
    rng = np.random.default_rng(7)
    d = rng.standard_normal(n_pix)
    C_ell = np.ones(bm.lmax_signal + 1) * 1.0  # CV-limited regime (S >> N)

    V = bm._V
    Lambda_diag = bm._build_lambda_diagonal(C_ell)
    N = simple_compression_setup["N"]
    C_pix = N + (V.T * Lambda_diag) @ V
    w_ref = V @ np.linalg.solve(C_pix, d)

    w_actual = bm.get_weighted_compressed_data(d, C_ell)
    np.testing.assert_allclose(w_actual, w_ref, rtol=1e-10, atol=1e-12)


def test_multi_field_matches_explicit_composition(multi_field_basis):
    bm = multi_field_basis
    rng = np.random.default_rng(1)
    d = rng.standard_normal((bm.n_pix, 2))

    n_ell = bm.lmax_signal + 1
    C_ell_dict = {
        SpectrumKey(0, 0, SpectrumKind.SS, spins=(0, 2)): np.ones(n_ell) * 1e-3,
        SpectrumKey(1, 1, SpectrumKind.GG, spins=(0, 2)): np.ones(n_ell) * 1e-4,
        SpectrumKey(1, 1, SpectrumKind.CC, spins=(0, 2)): np.ones(n_ell) * 1e-5,
    }

    inner_inv = bm.prepare_stable_inner_inv(C_ell_dict)
    expected = inner_inv.T @ (bm._V_N_inv @ d)

    actual = bm.get_weighted_compressed_data(d, C_ell_dict, stable_inner_inv=inner_inv)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=0)
