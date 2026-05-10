"""Tests for the keyed ComputationBasis API (Slice 1, Task 1.7)."""

import numpy as np
import pytest

from cosmocore.basis import HarmonicBasis
from cosmocore.spectrum_key import SpectrumKey, SpectrumKind


@pytest.fixture
def harmonic_basis_t_qu():
    """HarmonicBasis with T (spin 0) at index 0 and QU (spin 2) at index 1."""
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


def test_keyed_derivative_matches_legacy(harmonic_basis_t_qu):
    """get_derivative_matrix_keyed produces the same matrix as the int API."""
    spins = (0, 2)
    key = SpectrumKey(0, 1, SpectrumKind.SG, spins=spins)
    ell = 5
    legacy = harmonic_basis_t_qu._build_derivative_matrix_with_spins(ell, 0, 1, 0)
    keyed = harmonic_basis_t_qu.get_derivative_matrix_keyed(ell, key)
    np.testing.assert_array_equal(legacy, keyed)
