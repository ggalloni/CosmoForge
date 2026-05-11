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


@pytest.fixture
def harmonic_basis_qu_qu():
    """HarmonicBasis with two spin-2 components — cross-pair spin-2×spin-2."""
    np.random.seed(43)
    n_pix_a = 8
    n_pix_b = 8
    theta_a = np.random.uniform(0.1, np.pi - 0.1, n_pix_a)
    phi_a = np.random.uniform(0, 2 * np.pi, n_pix_a)
    theta_b = np.random.uniform(0.1, np.pi - 0.1, n_pix_b)
    phi_b = np.random.uniform(0, 2 * np.pi, n_pix_b)
    lmax = 6

    total_pix = 2 * n_pix_a + 2 * n_pix_b
    N = np.eye(total_pix) * 0.01

    hc = HarmonicBasis(N, (theta_a, theta_b), (phi_a, phi_b), lmax, spins=[2, 2])
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


def test_directional_gc_and_cg_e_matrices_distinct(harmonic_basis_qu_qu):
    """In DIRECTIONAL mode, E_GC and E_CG are distinct matrices.
    SYMMETRIC E_GC equals the legacy symmetrised matrix bit-for-bit."""
    from cosmocore.spectrum_key import (
        SpectrumKey,
        SpectrumKind,
        SymmetryMode,
    )

    spins = (2, 2)
    ell = 5  # any ell within the basis's lmax_signal

    key_gc = SpectrumKey(0, 1, SpectrumKind.GC, spins=spins)
    key_cg = SpectrumKey(0, 1, SpectrumKind.CG, spins=spins)

    e_gc_dir = harmonic_basis_qu_qu.get_derivative_matrix_keyed(
        ell, key_gc, symmetry_mode=SymmetryMode.DIRECTIONAL
    )
    e_cg_dir = harmonic_basis_qu_qu.get_derivative_matrix_keyed(
        ell, key_cg, symmetry_mode=SymmetryMode.DIRECTIONAL
    )
    assert not np.array_equal(e_gc_dir, e_cg_dir)
    np.testing.assert_array_equal(
        e_gc_dir + e_cg_dir,
        harmonic_basis_qu_qu.get_derivative_matrix_keyed(
            ell, key_gc, symmetry_mode=SymmetryMode.SYMMETRIC
        ),
    )

    e_sym = harmonic_basis_qu_qu.get_derivative_matrix_keyed(
        ell, key_gc, symmetry_mode=SymmetryMode.SYMMETRIC
    )
    legacy = harmonic_basis_qu_qu._build_derivative_matrix_with_spins(ell, 0, 1, mode=1)
    np.testing.assert_array_equal(e_sym, legacy)

    # Default (no symmetry_mode kwarg) MUST also equal legacy bit-for-bit.
    e_default = harmonic_basis_qu_qu.get_derivative_matrix_keyed(ell, key_gc)
    np.testing.assert_array_equal(e_default, legacy)


def test_kind_to_legacy_mode_supports_cg_for_cross():
    """CG is valid in cross-component context; raises for auto-pair."""
    from cosmocore.spectrum_key import SpectrumKind, kind_to_legacy_mode

    # Cross-component spin-2 × spin-2 ordering: [GG=0, GC=1, CG=2, CC=3]
    assert kind_to_legacy_mode(SpectrumKind.CG, is_cross=True) == 2
    assert kind_to_legacy_mode(SpectrumKind.GC, is_cross=True) == 1
    assert kind_to_legacy_mode(SpectrumKind.GG, is_cross=True) == 0
    assert kind_to_legacy_mode(SpectrumKind.CC, is_cross=True) == 3

    # Auto-pair (is_cross=False) preserves legacy [GG=0, CC=1, GC=2] mapping.
    assert kind_to_legacy_mode(SpectrumKind.GG, is_cross=False) == 0
    assert kind_to_legacy_mode(SpectrumKind.CC, is_cross=False) == 1
    assert kind_to_legacy_mode(SpectrumKind.GC, is_cross=False) == 2

    # CG has no slot in the auto-pair ordering.
    with pytest.raises((NotImplementedError, KeyError)):
        kind_to_legacy_mode(SpectrumKind.CG, is_cross=False)


def test_directional_lambda_uses_separate_gc_cg(harmonic_basis_qu_qu):
    """In DIRECTIONAL mode the (E_0, B_1) and (B_0, E_1) Lambda blocks
    carry C_GC and C_CG respectively; in SYMMETRIC they both carry C_GC."""
    from cosmocore.spectrum_key import SpectrumKey, SpectrumKind

    spins = (2, 2)
    bm = harmonic_basis_qu_qu
    lmax = bm.lmax_signal
    n = bm._n_modes_base

    c_gc = np.ones(lmax + 1) * 2.0
    c_cg = np.ones(lmax + 1) * 5.0
    c_ee = np.ones(lmax + 1) * 10.0
    c_bb = np.ones(lmax + 1) * 1.0

    cl_dict = {
        SpectrumKey(0, 0, SpectrumKind.GG, spins=spins): c_ee,
        SpectrumKey(0, 0, SpectrumKind.CC, spins=spins): c_bb,
        SpectrumKey(1, 1, SpectrumKind.GG, spins=spins): c_ee,
        SpectrumKey(1, 1, SpectrumKind.CC, spins=spins): c_bb,
        SpectrumKey(0, 1, SpectrumKind.GG, spins=spins): c_ee * 0.9,
        SpectrumKey(0, 1, SpectrumKind.CC, spins=spins): c_bb * 0.9,
        SpectrumKey(0, 1, SpectrumKind.GC, spins=spins): c_gc,
        SpectrumKey(0, 1, SpectrumKind.CG, spins=spins): c_cg,
    }
    L = bm._build_lambda_matrix(cl_dict)

    row0 = bm._mode_offsets[0]
    col1 = bm._mode_offsets[1]

    # Pick any mode index in the basis to probe the (ell, m) entries.
    # The block layout is independent of which ell — we just need a non-trivial mode.
    sample_ell = lmax // 2
    sample_idx = bm._ell_to_modes_local[sample_ell][0]

    # (E_0, B_1) block carries C_GC = 2.0
    assert L[row0 + sample_idx, col1 + n + sample_idx] == pytest.approx(2.0)
    # (B_0, E_1) block carries C_CG = 5.0, NOT 2.0
    assert L[row0 + n + sample_idx, col1 + sample_idx] == pytest.approx(5.0)

    # Sanity: SYMMETRIC (no CG key) reproduces the legacy behaviour where both
    # off-diagonal sub-blocks carry C_GC.
    sym_cl_dict = {k: v for k, v in cl_dict.items() if k.kind is not SpectrumKind.CG}
    L_sym = bm._build_lambda_matrix(sym_cl_dict)
    assert L_sym[row0 + sample_idx, col1 + n + sample_idx] == pytest.approx(2.0)
    assert L_sym[row0 + n + sample_idx, col1 + sample_idx] == pytest.approx(2.0)
