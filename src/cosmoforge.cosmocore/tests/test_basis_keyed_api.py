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


def test_lambda_matrix_accepts_spectrumkey_keyed_dict(harmonic_basis_t_qu):
    """SpectrumKey-keyed C_ell_dict yields the same Lambda as the 3-tuple form."""
    from cosmocore.spectrum_key import kind_to_legacy_mode

    spins = (0, 2)
    lmax = harmonic_basis_t_qu.lmax_signal
    rng = np.random.default_rng(7)
    cl_template = rng.uniform(0.1, 1.0, lmax + 1).astype(np.float64)

    keys = [
        SpectrumKey(0, 0, SpectrumKind.SS, spins=spins),
        SpectrumKey(0, 1, SpectrumKind.SG, spins=spins),
        SpectrumKey(0, 1, SpectrumKind.SC, spins=spins),
        SpectrumKey(1, 1, SpectrumKind.GG, spins=spins),
        SpectrumKey(1, 1, SpectrumKind.CC, spins=spins),
        SpectrumKey(1, 1, SpectrumKind.GC, spins=spins),
    ]
    keyed = {k: cl_template.copy() for k in keys}
    tupled = {
        (k.comp_i, k.comp_j, kind_to_legacy_mode(k.kind)): cl_template.copy()
        for k in keys
    }

    L_keyed = harmonic_basis_t_qu._build_lambda_matrix(keyed)
    L_tupled = harmonic_basis_t_qu._build_lambda_matrix(tupled)
    np.testing.assert_array_equal(L_keyed, L_tupled)


def test_lambda_blocks_accepts_spectrumkey_keyed_dict(harmonic_basis_t_qu):
    """SpectrumKey-keyed dict (spin-0 components only) yields same Lambda blocks."""
    # _build_lambda_blocks is the single-mode (2-tuple-keyed) path used for
    # spin-0 x spin-0 dicts. We pass only SS kinds so the per-iteration
    # translation maps to mode 0 and the algebra matches the 2-tuple form.
    spins = (0, 2)
    lmax = harmonic_basis_t_qu.lmax_signal
    rng = np.random.default_rng(11)
    cl_auto = rng.uniform(0.1, 1.0, lmax + 1).astype(np.float64)

    keyed = {
        SpectrumKey(0, 0, SpectrumKind.SS, spins=spins): cl_auto.copy(),
    }
    tupled = {
        (0, 0): cl_auto.copy(),
    }

    B_keyed = harmonic_basis_t_qu._build_lambda_blocks(keyed)
    B_tupled = harmonic_basis_t_qu._build_lambda_blocks(tupled)
    assert set(B_keyed.keys()) == set(B_tupled.keys())
    for k in B_keyed:
        np.testing.assert_array_equal(B_keyed[k], B_tupled[k])


def test_detect_field_blocks_accepts_spectrumkey_keyed_dict(harmonic_basis_t_qu):
    """SpectrumKey-keyed dict detects the same field block partition."""
    spins = (0, 2)
    lmax = harmonic_basis_t_qu.lmax_signal
    cl_template = np.ones(lmax + 1, dtype=np.float64)

    # No cross T-QU spectra: T and QU should be separate field blocks.
    keyed = {
        SpectrumKey(0, 0, SpectrumKind.SS, spins=spins): cl_template.copy(),
        SpectrumKey(1, 1, SpectrumKind.GG, spins=spins): cl_template.copy(),
        SpectrumKey(1, 1, SpectrumKind.CC, spins=spins): cl_template.copy(),
    }
    tupled = {
        (0, 0, 0): cl_template.copy(),
        (1, 1, 0): cl_template.copy(),
        (1, 1, 1): cl_template.copy(),
    }
    g_keyed = harmonic_basis_t_qu._detect_field_blocks(keyed)
    g_tupled = harmonic_basis_t_qu._detect_field_blocks(tupled)
    assert g_keyed == g_tupled
