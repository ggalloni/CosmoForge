"""
Tests for spin-2 polarization compression (Phase 2).

Tests for spin-weighted harmonic operators, Lambda matrices, derivatives,
Fisher matrices, and benchmarks comparing compressed vs pixel-space computation
for both HarmonicBasis and PixelBasis with spin-2 support.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


class TestSpin2HarmonicOperator:
    """Tests for spin-2 harmonic operator V construction."""

    def test_spin2_v_matrix_shape(self):
        """V for spin-2 should have shape (2*n_modes, 2*n_pix)."""
        from cosmocore.basis import HarmonicBasis

        np.random.seed(42)
        n_pix = 20
        theta = np.random.uniform(0.1, np.pi - 0.1, n_pix)
        phi = np.random.uniform(0, 2 * np.pi, n_pix)
        lmax = 5

        # Noise for 2*n_pix (Q and U)
        N = np.eye(2 * n_pix) * 0.01
        np.eye(2 * n_pix) * 100.0

        hc = HarmonicBasis(N, theta, phi, lmax, spins=[2])
        hc.setup()

        n_modes_base = (lmax + 1) ** 2 - 4
        expected_rows = 2 * n_modes_base  # E + B modes
        expected_cols = 2 * n_pix  # Q + U pixels

        assert hc._V.shape == (expected_rows, expected_cols), (
            f"V shape {hc._V.shape} != expected ({expected_rows}, {expected_cols})"
        )

    def test_spin2_v_matrix_nonzero(self):
        """V for spin-2 should have non-zero entries."""
        from cosmocore.basis import HarmonicBasis

        np.random.seed(42)
        n_pix = 20
        theta = np.random.uniform(0.1, np.pi - 0.1, n_pix)
        phi = np.random.uniform(0, 2 * np.pi, n_pix)
        lmax = 5

        N = np.eye(2 * n_pix) * 0.01
        np.eye(2 * n_pix) * 100.0

        hc = HarmonicBasis(N, theta, phi, lmax, spins=[2])
        hc.setup()

        # Both E and B blocks should have non-zero entries
        n_modes_base = (lmax + 1) ** 2 - 4
        V_E = hc._V[:n_modes_base, :]
        V_B = hc._V[n_modes_base:, :]

        assert np.any(V_E != 0), "E-mode block of V should have non-zero entries"
        assert np.any(V_B != 0), "B-mode block of V should have non-zero entries"

    def test_spin2_dimensions_tracking(self):
        """Check that spin-2 component dimensions are tracked correctly."""
        from cosmocore.basis import HarmonicBasis

        np.random.seed(42)
        n_pix = 20
        theta = np.random.uniform(0.1, np.pi - 0.1, n_pix)
        phi = np.random.uniform(0, 2 * np.pi, n_pix)
        lmax = 5

        N = np.eye(2 * n_pix) * 0.01
        np.eye(2 * n_pix) * 100.0

        hc = HarmonicBasis(N, theta, phi, lmax, spins=[2])

        n_modes_base = (lmax + 1) ** 2 - 4

        # Spin-2 doubles pixel count and mode count
        assert hc.n_pix == 2 * n_pix
        assert hc._n_pix_per_component == [2 * n_pix]
        assert hc._n_modes_per_component_list == [2 * n_modes_base]
        assert hc.n_modes_total == 2 * n_modes_base

    def test_spins_validation(self):
        """Test that invalid spins are rejected."""
        from cosmocore.basis import HarmonicBasis

        np.random.seed(42)
        n_pix = 10
        theta = np.random.uniform(0.1, np.pi - 0.1, n_pix)
        phi = np.random.uniform(0, 2 * np.pi, n_pix)
        lmax = 4

        N = np.eye(n_pix) * 0.01
        np.eye(n_pix) * 100.0

        with pytest.raises(ValueError, match="Spin must be 0"):
            HarmonicBasis(N, theta, phi, lmax, spins=[1])

        with pytest.raises(ValueError, match="spins list length"):
            HarmonicBasis(N, theta, phi, lmax, spins=[0, 2])


class TestSpin2Lambda:
    """Tests for spin-2 Lambda matrix construction."""

    def test_lambda_block_spin2_shape(self):
        """Lambda for spin-2 auto-correlation should be (2*n_modes, 2*n_modes)."""
        from cosmocore.basis import HarmonicBasis

        np.random.seed(42)
        n_pix = 20
        theta = np.random.uniform(0.1, np.pi - 0.1, n_pix)
        phi = np.random.uniform(0, 2 * np.pi, n_pix)
        lmax = 5

        N = np.eye(2 * n_pix) * 0.01
        np.eye(2 * n_pix) * 100.0

        hc = HarmonicBasis(N, theta, phi, lmax, spins=[2])

        n_modes_base = (lmax + 1) ** 2 - 4
        C_ell = np.ones(lmax + 1) * 1e-3

        Lambda = hc._build_lambda_block_spin2(C_ell, C_ell * 0.5, C_ell * 0.1)

        assert Lambda.shape == (2 * n_modes_base, 2 * n_modes_base)

    def test_lambda_block_spin2_structure(self):
        """Lambda EE/BB diagonals and EB off-diagonals should be correctly placed."""
        from cosmocore.basis import HarmonicBasis

        np.random.seed(42)
        n_pix = 20
        theta = np.random.uniform(0.1, np.pi - 0.1, n_pix)
        phi = np.random.uniform(0, 2 * np.pi, n_pix)
        lmax = 4

        N = np.eye(2 * n_pix) * 0.01
        np.eye(2 * n_pix) * 100.0

        hc = HarmonicBasis(N, theta, phi, lmax, spins=[2])

        n = hc._n_modes_base
        C_EE = np.ones(lmax + 1) * 2.0
        C_BB = np.ones(lmax + 1) * 1.0
        C_EB = np.ones(lmax + 1) * 0.5

        Lambda = hc._build_lambda_block_spin2(C_EE, C_BB, C_EB)

        # EE block: top-left n×n diagonal
        for k in range(n):
            assert Lambda[k, k] == pytest.approx(2.0)
        # BB block: bottom-right n×n diagonal
        for k in range(n):
            assert Lambda[n + k, n + k] == pytest.approx(1.0)
        # EB block: off-diagonal
        for k in range(n):
            assert Lambda[k, n + k] == pytest.approx(0.5)
            assert Lambda[n + k, k] == pytest.approx(0.5)

    def test_lambda_full_with_spins_teb(self):
        """Test full Lambda assembly for T+E/B (spin-0 + spin-2)."""
        from cosmocore.basis import HarmonicBasis

        np.random.seed(42)
        n_pix_t = 15
        n_pix_p = 12
        theta_t = np.random.uniform(0.1, np.pi - 0.1, n_pix_t)
        phi_t = np.random.uniform(0, 2 * np.pi, n_pix_t)
        theta_p = np.random.uniform(0.1, np.pi - 0.1, n_pix_p)
        phi_p = np.random.uniform(0, 2 * np.pi, n_pix_p)
        lmax = 4

        # Total pixels: n_pix_t + 2*n_pix_p
        total_pix = n_pix_t + 2 * n_pix_p
        N = np.eye(total_pix) * 0.01
        np.eye(total_pix) * 100.0

        hc = HarmonicBasis(N, (theta_t, theta_p), (phi_t, phi_p), lmax, spins=[0, 2])
        hc.setup()

        n_base = hc._n_modes_base

        # Build C_ell_dict with 3-tuple keys
        C_ell_dict = {
            (0, 0, 0): np.ones(lmax + 1) * 1e-3,  # TT
            (1, 1, 0): np.ones(lmax + 1) * 5e-4,  # EE
            (1, 1, 1): np.ones(lmax + 1) * 1e-4,  # BB
            (1, 1, 2): np.ones(lmax + 1) * 1e-5,  # EB
            (0, 1, 0): np.ones(lmax + 1) * 2e-4,  # TE
            (0, 1, 1): np.zeros(lmax + 1),  # TB
        }

        Lambda = hc._build_lambda_matrix_3tuple(C_ell_dict)

        # Total size: n_base (T) + 2*n_base (E+B) = 3*n_base
        assert Lambda.shape == (3 * n_base, 3 * n_base)

        # TT block: top-left
        for k in range(n_base):
            assert Lambda[k, k] == pytest.approx(1e-3), f"TT diagonal at {k}"

        # EE block: diagonal at [n_base:2*n_base, n_base:2*n_base]
        for k in range(n_base):
            assert Lambda[n_base + k, n_base + k] == pytest.approx(5e-4)

        # BB block: diagonal at [2*n_base:3*n_base, 2*n_base:3*n_base]
        for k in range(n_base):
            assert Lambda[2 * n_base + k, 2 * n_base + k] == pytest.approx(1e-4)


class TestSpin2Derivatives:
    """Tests for spin-2 derivative matrices."""

    def test_derivative_ee_structure(self):
        """EE derivative should only fill E×E sub-block."""
        from cosmocore.basis import HarmonicBasis

        np.random.seed(42)
        n_pix_t = 10
        n_pix_p = 8
        theta_t = np.random.uniform(0.1, np.pi - 0.1, n_pix_t)
        phi_t = np.random.uniform(0, 2 * np.pi, n_pix_t)
        theta_p = np.random.uniform(0.1, np.pi - 0.1, n_pix_p)
        phi_p = np.random.uniform(0, 2 * np.pi, n_pix_p)
        lmax = 4

        total_pix = n_pix_t + 2 * n_pix_p
        N = np.eye(total_pix) * 0.01
        np.eye(total_pix) * 100.0

        hc = HarmonicBasis(N, (theta_t, theta_p), (phi_t, phi_p), lmax, spins=[0, 2])
        hc.setup()

        n_base = hc._n_modes_base
        ell = 3

        # EE derivative
        E_ee = hc.get_derivative_matrix(ell, 1, 1, mode=0)
        # BB derivative
        E_bb = hc.get_derivative_matrix(ell, 1, 1, mode=1)
        # EB derivative
        E_eb = hc.get_derivative_matrix(ell, 1, 1, mode=2)

        # EE should only have entries in [n_base:2*n_base, n_base:2*n_base]
        assert np.all(E_ee[:n_base, :] == 0), "EE should not touch T block"
        assert np.any(E_ee[n_base : 2 * n_base, n_base : 2 * n_base] != 0), (
            "EE should have entries in E×E block"
        )
        assert np.all(E_ee[2 * n_base :, 2 * n_base :] == 0), (
            "EE should not touch B×B block"
        )

        # BB should only have entries in [2*n_base:3*n_base, 2*n_base:3*n_base]
        assert np.all(E_bb[: 2 * n_base, : 2 * n_base] == 0)
        assert np.any(E_bb[2 * n_base :, 2 * n_base :] != 0)

        # EB should have entries in E×B and B×E off-diagonals
        assert np.any(E_eb[n_base : 2 * n_base, 2 * n_base :] != 0)
        assert np.any(E_eb[2 * n_base :, n_base : 2 * n_base] != 0)

    def test_derivative_te_structure(self):
        """TE derivative should fill T×E and E×T sub-blocks."""
        from cosmocore.basis import HarmonicBasis

        np.random.seed(42)
        n_pix_t = 10
        n_pix_p = 8
        theta_t = np.random.uniform(0.1, np.pi - 0.1, n_pix_t)
        phi_t = np.random.uniform(0, 2 * np.pi, n_pix_t)
        theta_p = np.random.uniform(0.1, np.pi - 0.1, n_pix_p)
        phi_p = np.random.uniform(0, 2 * np.pi, n_pix_p)
        lmax = 4

        total_pix = n_pix_t + 2 * n_pix_p
        N = np.eye(total_pix) * 0.01
        np.eye(total_pix) * 100.0

        hc = HarmonicBasis(N, (theta_t, theta_p), (phi_t, phi_p), lmax, spins=[0, 2])
        hc.setup()

        n_base = hc._n_modes_base
        ell = 3

        # TE derivative
        E_te = hc.get_derivative_matrix(ell, 0, 1, mode=0)

        # Should have entries in T×E block and E×T block (symmetric)
        assert np.any(E_te[:n_base, n_base : 2 * n_base] != 0)
        assert np.any(E_te[n_base : 2 * n_base, :n_base] != 0)
        # Should NOT touch B blocks
        assert np.all(E_te[:n_base, 2 * n_base :] == 0)
        assert np.all(E_te[2 * n_base :, :n_base] == 0)


class TestSpin2Fisher:
    """Tests for spin-2 Fisher matrix computation."""

    def test_fisher_with_spins_shape(self):
        """Fisher matrix should have correct shape for TEB spectra."""
        from cosmocore.basis import HarmonicBasis

        np.random.seed(42)
        n_pix_t = 15
        n_pix_p = 12
        theta_t = np.random.uniform(0.1, np.pi - 0.1, n_pix_t)
        phi_t = np.random.uniform(0, 2 * np.pi, n_pix_t)
        theta_p = np.random.uniform(0.1, np.pi - 0.1, n_pix_p)
        phi_p = np.random.uniform(0, 2 * np.pi, n_pix_p)
        lmax = 4

        total_pix = n_pix_t + 2 * n_pix_p
        N = np.eye(total_pix) * 0.01
        np.eye(total_pix) * 100.0

        hc = HarmonicBasis(N, (theta_t, theta_p), (phi_t, phi_p), lmax, spins=[0, 2])
        hc.setup()

        C_ell_dict = {
            (0, 0, 0): np.ones(lmax + 1) * 1e-3,
            (1, 1, 0): np.ones(lmax + 1) * 5e-4,
            (1, 1, 1): np.ones(lmax + 1) * 1e-4,
            (1, 1, 2): np.zeros(lmax + 1),
            (0, 1, 0): np.ones(lmax + 1) * 2e-4,
            (0, 1, 1): np.zeros(lmax + 1),
        }

        spectra_list = [
            (0, 0, 0),  # TT
            (1, 1, 0),  # EE
            (1, 1, 1),  # BB
            (0, 1, 0),  # TE
        ]

        fisher = hc.compute_fisher_matrix(
            C_ell_dict, spectra_list, ell_min=2, ell_max=lmax
        )

        n_ell = lmax - 1
        n_spec = len(spectra_list)
        assert fisher.shape == (n_spec * n_ell, n_spec * n_ell)

    def test_fisher_with_spins_symmetric(self):
        """Fisher matrix should be symmetric."""
        from cosmocore.basis import HarmonicBasis

        np.random.seed(42)
        n_pix_t = 15
        n_pix_p = 12
        theta_t = np.random.uniform(0.1, np.pi - 0.1, n_pix_t)
        phi_t = np.random.uniform(0, 2 * np.pi, n_pix_t)
        theta_p = np.random.uniform(0.1, np.pi - 0.1, n_pix_p)
        phi_p = np.random.uniform(0, 2 * np.pi, n_pix_p)
        lmax = 4

        total_pix = n_pix_t + 2 * n_pix_p
        N = np.eye(total_pix) * 0.01
        np.eye(total_pix) * 100.0

        hc = HarmonicBasis(N, (theta_t, theta_p), (phi_t, phi_p), lmax, spins=[0, 2])
        hc.setup()

        C_ell_dict = {
            (0, 0, 0): np.ones(lmax + 1) * 1e-3,
            (1, 1, 0): np.ones(lmax + 1) * 5e-4,
            (1, 1, 1): np.ones(lmax + 1) * 1e-4,
        }

        spectra_list = [(0, 0, 0), (1, 1, 0), (1, 1, 1)]

        fisher = hc.compute_fisher_matrix(
            C_ell_dict, spectra_list, ell_min=2, ell_max=lmax
        )

        assert_allclose(
            fisher, fisher.T, atol=1e-12, err_msg="Fisher matrix should be symmetric"
        )

    def test_fisher_with_spins_positive_diagonal(self):
        """Fisher diagonal elements should be positive."""
        from cosmocore.basis import HarmonicBasis

        np.random.seed(42)
        n_pix = 20
        theta = np.random.uniform(0.1, np.pi - 0.1, n_pix)
        phi = np.random.uniform(0, 2 * np.pi, n_pix)
        lmax = 5

        N = np.eye(2 * n_pix) * 0.01
        np.eye(2 * n_pix) * 100.0

        hc = HarmonicBasis(N, theta, phi, lmax, spins=[2])
        hc.setup()

        C_ell_dict = {
            (0, 0, 0): np.ones(lmax + 1) * 5e-4,  # EE
            (0, 0, 1): np.ones(lmax + 1) * 1e-4,  # BB
        }

        spectra_list = [(0, 0, 0), (0, 0, 1)]

        fisher = hc.compute_fisher_matrix(
            C_ell_dict, spectra_list, ell_min=2, ell_max=lmax
        )

        # Diagonal elements should be non-negative
        assert np.all(np.diag(fisher) >= -1e-15), (
            f"Fisher diagonal has negative entries: {np.diag(fisher).min()}"
        )


# =============================================================================
# Spin-2 Benchmark: Compressed vs Traditional Pixel-Space Computation
# =============================================================================


def _pixel_space_fisher_with_spins(C_inv, V, hc, lmax, spectra_list):
    """
    Compute Fisher matrix using traditional pixel-space method for spin-2.

    F_ij = 0.5 * Tr[C^{-1} dS_i C^{-1} dS_j]

    where dS_i = V^T E_i V is the derivative of the signal matrix.
    """
    from cosmocore.basics import matrix_mult, matrix_trace

    n_ell = lmax - 1
    n_spectra = len(spectra_list)
    fisher = np.zeros((n_spectra * n_ell, n_spectra * n_ell))

    for si, (ci, cj, mode_i) in enumerate(spectra_list):
        for ell_i in range(2, lmax + 1):
            E_i = hc.get_derivative_matrix(ell_i, ci, cj, mode_i)
            dS_i = V.T @ E_i @ V

            for sj, (ck, cl, mode_j) in enumerate(spectra_list):
                for ell_j in range(2, lmax + 1):
                    E_j = hc.get_derivative_matrix(ell_j, ck, cl, mode_j)
                    dS_j = V.T @ E_j @ V

                    temp1 = matrix_mult(C_inv, dS_i)
                    temp2 = matrix_mult(C_inv, dS_j)
                    f_val = 0.5 * matrix_trace(temp1, temp2)

                    row = si * n_ell + (ell_i - 2)
                    col = sj * n_ell + (ell_j - 2)
                    fisher[row, col] = f_val

    return fisher


class TestSpin2Benchmark:
    """
    Benchmark: compressed spin-2 Fisher vs traditional pixel-space computation.

    These tests verify that the spin-weighted harmonic compression produces
    numerically equivalent results to the direct pixel-space Fisher computation.
    """

    def test_qu_compressed_vs_pixel_space(self):
        """
        QU-only: compressed EE/BB Fisher matches pixel-space computation.

        Single spin-2 field (Q, U) → (E, B) decomposition.
        Sizes chosen so n_pix_total >> n_modes for compression speedup:
        - n_pix=120 physical → 240 total (Q+U)
        - lmax_signal=6 → n_modes_base=45 → 90 total (E+B)
        """
        import time

        from cosmocore.basics import matrix_inverse_symm
        from cosmocore.basis import HarmonicBasis

        np.random.seed(42)
        n_pix = 120  # 240 total (Q+U) >> 90 modes
        golden_ratio = (1 + np.sqrt(5)) / 2
        indices = np.arange(n_pix)
        theta = np.arccos(1 - 2 * (indices + 0.5) / n_pix)
        phi = (2 * np.pi * indices / golden_ratio) % (2 * np.pi)
        lmax = 6

        # Noise for Q, U pixels
        noise_level = 1e-2
        N = np.eye(2 * n_pix) * noise_level
        np.eye(2 * n_pix) / noise_level

        hc = HarmonicBasis(N, theta, phi, lmax, spins=[2])
        hc.setup()

        n_modes_total = hc.n_modes_total
        n_pix_total = hc.n_pix

        # Power spectra with normalization
        ells = np.arange(2, lmax + 1)
        norm = (2 * ells + 1) / (4 * np.pi)

        C_EE = norm * 1e-4 / ells**2
        C_BB = norm * 1e-5 / ells**2

        C_ell_dict = {
            (0, 0, 0): C_EE,
            (0, 0, 1): C_BB,
        }
        spectra_list = [(0, 0, 0), (0, 0, 1)]  # EE, BB

        # --- Compressed Fisher ---
        t0 = time.perf_counter()
        fisher_compressed = hc.compute_fisher_matrix(
            C_ell_dict, spectra_list, ell_min=2, ell_max=lmax
        )
        t_compressed = time.perf_counter() - t0

        # --- Pixel-space Fisher ---
        V = hc._V
        lambda_matrix = hc._build_lambda_matrix_3tuple(C_ell_dict)
        S = V.T @ lambda_matrix @ V
        C = N + S
        C_inv = matrix_inverse_symm(C.copy())

        t0 = time.perf_counter()
        fisher_pixel = _pixel_space_fisher_with_spins(C_inv, V, hc, lmax, spectra_list)
        t_pixel = time.perf_counter() - t0

        # --- Compare ---
        mask = np.abs(fisher_pixel) > 1e-20
        rel_diff = np.abs(fisher_compressed - fisher_pixel) / (
            np.abs(fisher_pixel) + 1e-30
        )

        max_rel_diff = rel_diff[mask].max() if mask.any() else 0.0
        speedup = t_pixel / t_compressed

        print(
            f"\n  QU-only benchmark (n_pix_total={n_pix_total}, "
            f"n_modes={n_modes_total}, lmax={lmax}):"
        )
        print(f"    Compressed time: {t_compressed:.4f}s")
        print(f"    Pixel-space time: {t_pixel:.4f}s")
        print(f"    Speedup: {speedup:.1f}x")
        print(f"    Max relative difference: {max_rel_diff:.2e}")

        assert_allclose(
            fisher_compressed,
            fisher_pixel,
            rtol=1e-6,
            atol=1e-15,
            err_msg="QU compressed Fisher should match pixel-space computation",
        )
        print(f"Speedup: {speedup:.1f}x")

    def test_tqu_compressed_vs_pixel_space(self):
        """
        TQU: compressed TT/EE/BB/TE Fisher matches pixel-space computation.

        Mixed spin-0 (T) + spin-2 (Q, U) fields → TT, EE, BB, TE spectra.
        Sizes chosen so n_pix_total >> n_modes for compression speedup:
        - n_pix_t=50, n_pix_p=45 → total_pix = 50 + 2*45 = 140
        - lmax_signal=6 → n_modes_base=45 → 45 (T) + 90 (E+B) = 135 modes
        """
        import time

        from cosmocore.basics import matrix_inverse_symm
        from cosmocore.basis import HarmonicBasis

        np.random.seed(42)
        n_pix_t = 50
        n_pix_p = 45
        golden_ratio = (1 + np.sqrt(5)) / 2

        def _spiral(n, offset=0):
            idx = np.arange(n)
            th = np.arccos(1 - 2 * (idx + 0.5) / n)
            ph = (2 * np.pi * (idx + offset) / golden_ratio) % (2 * np.pi)
            return th, ph

        theta_t, phi_t = _spiral(n_pix_t, offset=0)
        theta_p, phi_p = _spiral(n_pix_p, offset=1000)
        lmax = 6

        # Total pixels: n_pix_t + 2*n_pix_p
        total_pix = n_pix_t + 2 * n_pix_p
        N = np.eye(total_pix)
        N[:n_pix_t, :n_pix_t] *= 1e-3
        N[n_pix_t:, n_pix_t:] *= 5e-3
        np.linalg.inv(N)

        hc = HarmonicBasis(N, (theta_t, theta_p), (phi_t, phi_p), lmax, spins=[0, 2])
        hc.setup()

        n_modes_total = hc.n_modes_total
        n_pix_total = hc.n_pix

        # Power spectra
        ells = np.arange(2, lmax + 1)
        norm = (2 * ells + 1) / (4 * np.pi)

        C_TT = norm * 1e-4 / ells**2
        C_EE = norm * 1e-5 / ells**2
        C_BB = norm * 1e-6 / ells**2
        C_TE = norm * 3e-5 / ells**2

        C_ell_dict = {
            (0, 0, 0): C_TT,
            (1, 1, 0): C_EE,
            (1, 1, 1): C_BB,
            (0, 1, 0): C_TE,
        }
        spectra_list = [
            (0, 0, 0),  # TT
            (1, 1, 0),  # EE
            (1, 1, 1),  # BB
            (0, 1, 0),  # TE
        ]

        # --- Compressed Fisher ---
        t0 = time.perf_counter()
        fisher_compressed = hc.compute_fisher_matrix(
            C_ell_dict, spectra_list, ell_min=2, ell_max=lmax
        )
        t_compressed = time.perf_counter() - t0

        # --- Pixel-space Fisher ---
        V = hc._V
        lambda_matrix = hc._build_lambda_matrix_3tuple(C_ell_dict)
        S = V.T @ lambda_matrix @ V
        C = N + S
        C_inv = matrix_inverse_symm(C.copy())

        t0 = time.perf_counter()
        fisher_pixel = _pixel_space_fisher_with_spins(C_inv, V, hc, lmax, spectra_list)
        t_pixel = time.perf_counter() - t0

        # --- Compare ---
        mask = np.abs(fisher_pixel) > 1e-20
        rel_diff = np.abs(fisher_compressed - fisher_pixel) / (
            np.abs(fisher_pixel) + 1e-30
        )
        max_rel_diff = rel_diff[mask].max() if mask.any() else 0.0
        speedup = t_pixel / t_compressed

        print(
            f"\n  TQU benchmark (n_pix_total={n_pix_total}, "
            f"n_modes={n_modes_total}, lmax={lmax}):"
        )
        print(f"    Compressed time: {t_compressed:.4f}s")
        print(f"    Pixel-space time: {t_pixel:.4f}s")
        print(f"    Speedup: {speedup:.1f}x")
        print(f"    Max relative difference: {max_rel_diff:.2e}")
        print(f"    Fisher matrix shape: {fisher_compressed.shape}")

        assert_allclose(
            fisher_compressed,
            fisher_pixel,
            rtol=1e-6,
            atol=1e-15,
            err_msg="TQU compressed Fisher should match pixel-space computation",
        )


# =============================================================================
# Phase 2: PixelProjected Spin-2 Compression Tests
# =============================================================================


class TestPixelProjectedSpin2:
    """Tests for PixelBasis with spin-2 support."""

    def test_spin2_v_matrix_shape(self):
        """V for spin-2 PixelProjected should have correct shape."""
        from cosmocore.basis import PixelBasis

        np.random.seed(42)
        n_pix = 20
        theta = np.random.uniform(0.1, np.pi - 0.1, n_pix)
        phi = np.random.uniform(0, 2 * np.pi, n_pix)
        lmax = 5

        N = np.eye(2 * n_pix) * 0.01
        np.eye(2 * n_pix) * 100.0

        ppc = PixelBasis(
            N,
            theta,
            phi,
            lmax,
            spins=[2],
            epsilon=1e-6,
        )
        ppc.setup()

        n_modes_base = (lmax + 1) ** 2 - 4
        expected_rows = 2 * n_modes_base
        expected_cols = 2 * n_pix

        assert ppc._V.shape == (expected_rows, expected_cols)

    def test_spin2_compressed_covariance_shape(self):
        """Compressed covariance should have shape (n_kept, n_kept)."""
        from cosmocore.basis import PixelBasis

        np.random.seed(42)
        n_pix = 20
        theta = np.random.uniform(0.1, np.pi - 0.1, n_pix)
        phi = np.random.uniform(0, 2 * np.pi, n_pix)
        lmax = 5

        N = np.eye(2 * n_pix) * 0.01
        np.eye(2 * n_pix) * 100.0

        ppc = PixelBasis(
            N,
            theta,
            phi,
            lmax,
            spins=[2],
            epsilon=1e-6,
        )
        ppc.setup()

        C_ell_dict = {
            (0, 0, 0): np.ones(lmax + 1) * 5e-4,  # EE
            (0, 0, 1): np.ones(lmax + 1) * 1e-4,  # BB
        }

        C_c = ppc.get_compressed_covariance(C_ell_dict)
        assert C_c.shape == (ppc.n_kept, ppc.n_kept)
        # Should be symmetric
        assert_allclose(C_c, C_c.T, atol=1e-12)

    def test_spin2_fisher_shape_and_symmetry(self):
        """Fisher matrix should have correct shape and be symmetric."""
        from cosmocore.basis import PixelBasis

        np.random.seed(42)
        n_pix = 20
        theta = np.random.uniform(0.1, np.pi - 0.1, n_pix)
        phi = np.random.uniform(0, 2 * np.pi, n_pix)
        lmax = 5

        N = np.eye(2 * n_pix) * 0.01
        np.eye(2 * n_pix) * 100.0

        ppc = PixelBasis(
            N,
            theta,
            phi,
            lmax,
            spins=[2],
            epsilon=1e-6,
        )
        ppc.setup()

        C_ell_dict = {
            (0, 0, 0): np.ones(lmax + 1) * 5e-4,  # EE
            (0, 0, 1): np.ones(lmax + 1) * 1e-4,  # BB
        }
        spectra_list = [(0, 0, 0), (0, 0, 1)]

        fisher = ppc.compute_fisher_matrix(
            C_ell_dict, spectra_list, ell_min=2, ell_max=lmax
        )

        n_ell = lmax - 1
        n_spec = len(spectra_list)
        assert fisher.shape == (n_spec * n_ell, n_spec * n_ell)
        assert_allclose(fisher, fisher.T, atol=1e-12)

    def test_spin2_tqu_fisher_shape(self):
        """TQU PixelProjected Fisher should have correct shape."""
        from cosmocore.basis import PixelBasis

        np.random.seed(42)
        n_pix_t = 15
        n_pix_p = 12
        theta_t = np.random.uniform(0.1, np.pi - 0.1, n_pix_t)
        phi_t = np.random.uniform(0, 2 * np.pi, n_pix_t)
        theta_p = np.random.uniform(0.1, np.pi - 0.1, n_pix_p)
        phi_p = np.random.uniform(0, 2 * np.pi, n_pix_p)
        lmax = 4

        total_pix = n_pix_t + 2 * n_pix_p
        N = np.eye(total_pix) * 0.01
        np.eye(total_pix) * 100.0

        ppc = PixelBasis(
            N,
            (theta_t, theta_p),
            (phi_t, phi_p),
            lmax,
            spins=[0, 2],
            epsilon=1e-6,
        )
        ppc.setup()

        C_ell_dict = {
            (0, 0, 0): np.ones(lmax + 1) * 1e-3,  # TT
            (1, 1, 0): np.ones(lmax + 1) * 5e-4,  # EE
            (1, 1, 1): np.ones(lmax + 1) * 1e-4,  # BB
            (0, 1, 0): np.ones(lmax + 1) * 2e-4,  # TE
        }
        spectra_list = [
            (0, 0, 0),  # TT
            (1, 1, 0),  # EE
            (1, 1, 1),  # BB
            (0, 1, 0),  # TE
        ]

        fisher = ppc.compute_fisher_matrix(
            C_ell_dict, spectra_list, ell_min=2, ell_max=lmax
        )

        n_ell = lmax - 1
        n_spec = len(spectra_list)
        assert fisher.shape == (n_spec * n_ell, n_spec * n_ell)
        assert_allclose(fisher, fisher.T, atol=1e-12)

    def test_spin2_weighted_data_shape(self):
        """Weighted compressed data should have shape (n_kept,)."""
        from cosmocore.basis import PixelBasis

        np.random.seed(42)
        n_pix = 20
        theta = np.random.uniform(0.1, np.pi - 0.1, n_pix)
        phi = np.random.uniform(0, 2 * np.pi, n_pix)
        lmax = 5

        N = np.eye(2 * n_pix) * 0.01
        np.eye(2 * n_pix) * 100.0

        ppc = PixelBasis(
            N,
            theta,
            phi,
            lmax,
            spins=[2],
            epsilon=1e-6,
        )
        ppc.setup()

        C_ell_dict = {
            (0, 0, 0): np.ones(lmax + 1) * 5e-4,
            (0, 0, 1): np.ones(lmax + 1) * 1e-4,
        }

        data = np.random.randn(2 * n_pix)
        w = ppc.get_weighted_compressed_data(data, C_ell_dict)
        assert w.shape == (ppc.n_kept,)

    def test_spin2_manager_delegates(self):
        """create_computation_basis should produce working spin-2 PixelProjected."""
        from cosmocore.basis import create_computation_basis

        np.random.seed(42)
        n_pix = 20
        theta = np.random.uniform(0.1, np.pi - 0.1, n_pix)
        phi = np.random.uniform(0, 2 * np.pi, n_pix)
        lmax = 5

        N = np.eye(2 * n_pix) * 0.01
        np.eye(2 * n_pix) * 100.0

        cm = create_computation_basis(
            method="pixel",
            N=N,
            theta=theta,
            phi=phi,
            lmax_signal=lmax,
            spins=[2],
            epsilon=1e-6,
        )
        cm.setup()

        C_ell_dict = {
            (0, 0, 0): np.ones(lmax + 1) * 5e-4,
            (0, 0, 1): np.ones(lmax + 1) * 1e-4,
        }
        spectra_list = [(0, 0, 0), (0, 0, 1)]

        fisher = cm.compute_fisher_matrix(
            C_ell_dict, spectra_list, ell_min=2, ell_max=lmax
        )

        n_ell = lmax - 1
        n_spec = len(spectra_list)
        assert fisher.shape == (n_spec * n_ell, n_spec * n_ell)
        assert_allclose(fisher, fisher.T, atol=1e-12)


def _pixel_space_fisher_with_spins_raw(
    C_inv,
    V,
    n_modes_total,
    lmax,
    spectra_list,
    spins,
    mode_offsets,
    n_modes_base,
    ell_to_modes_local,
):
    """
    Compute Fisher matrix using traditional pixel-space method for spin-2.

    Uses raw harmonic-space E matrices (not compressed) for pixel-space reference.
    F_ij = 0.5 * Tr[C^{-1} dS_i C^{-1} dS_j]
    where dS_i = V^T E_i V is the derivative of the signal matrix.
    """
    from cosmocore.basics import matrix_mult, matrix_trace

    n_ell = lmax - 1
    n_spectra = len(spectra_list)
    fisher = np.zeros((n_spectra * n_ell, n_spectra * n_ell))

    def _build_E(ell, comp_i, comp_j, mode):
        """Build E matrix in harmonic space."""
        spin_i = spins[comp_i]
        spin_j = spins[comp_j]
        E = np.zeros((n_modes_total, n_modes_total), dtype=np.float64)
        local_mode_indices = ell_to_modes_local[ell]
        n_base = n_modes_base

        if spin_i == 0 and spin_j == 0:
            row_offset = mode_offsets[comp_i]
            col_offset = mode_offsets[comp_j]
            for idx in local_mode_indices:
                E[row_offset + idx, col_offset + idx] = 1.0
            if comp_i != comp_j:
                for idx in local_mode_indices:
                    E[col_offset + idx, row_offset + idx] = 1.0
        elif spin_i == 2 and spin_j == 2:
            deriv_val = 1.0
            row_start = mode_offsets[comp_i]
            col_start = mode_offsets[comp_j]
            if mode == 0:
                for idx in local_mode_indices:
                    E[row_start + idx, col_start + idx] = deriv_val
            elif mode == 1:
                for idx in local_mode_indices:
                    E[row_start + n_base + idx, col_start + n_base + idx] = deriv_val
            elif mode == 2:
                for idx in local_mode_indices:
                    E[row_start + idx, col_start + n_base + idx] = deriv_val
                    E[col_start + n_base + idx, row_start + idx] = deriv_val
            if comp_i != comp_j:
                if mode == 0:
                    for idx in local_mode_indices:
                        E[col_start + idx, row_start + idx] = deriv_val
                elif mode == 1:
                    for idx in local_mode_indices:
                        E[col_start + n_base + idx, row_start + n_base + idx] = deriv_val
                elif mode == 2:
                    for idx in local_mode_indices:
                        E[col_start + idx, row_start + n_base + idx] = deriv_val
                        E[row_start + n_base + idx, col_start + idx] = deriv_val
        elif spin_i == 0 and spin_j == 2:
            deriv_val = -1.0
            row_start = mode_offsets[comp_i]
            col_start = mode_offsets[comp_j]
            col_sub = col_start + mode * n_base
            for idx in local_mode_indices:
                E[row_start + idx, col_sub + idx] = deriv_val
                E[col_sub + idx, row_start + idx] = deriv_val
        elif spin_i == 2 and spin_j == 0:
            deriv_val = -1.0
            row_start = mode_offsets[comp_i]
            col_start = mode_offsets[comp_j]
            row_sub = row_start + mode * n_base
            for idx in local_mode_indices:
                E[row_sub + idx, col_start + idx] = deriv_val
                E[col_start + idx, row_sub + idx] = deriv_val
        return E

    for si, (ci, cj, mode_i) in enumerate(spectra_list):
        for ell_i in range(2, lmax + 1):
            E_i = _build_E(ell_i, ci, cj, mode_i)
            dS_i = V.T @ E_i @ V

            for sj, (ck, cl, mode_j) in enumerate(spectra_list):
                for ell_j in range(2, lmax + 1):
                    E_j = _build_E(ell_j, ck, cl, mode_j)
                    dS_j = V.T @ E_j @ V

                    temp1 = matrix_mult(C_inv, dS_i)
                    temp2 = matrix_mult(C_inv, dS_j)
                    f_val = 0.5 * matrix_trace(temp1, temp2)

                    row = si * n_ell + (ell_i - 2)
                    col = sj * n_ell + (ell_j - 2)
                    fisher[row, col] = f_val

    return fisher


class TestPixelProjectedSpin2Benchmark:
    """
    Benchmark: PixelProjected spin-2 compressed Fisher vs pixel-space computation.

    Verifies that PixelBasis with eigenvalue compression produces
    results close to the traditional pixel-space Fisher computation.
    Note: PixelProjected is an approximation (unlike Harmonic which is exact),
    so tolerances are relaxed.
    """

    def test_qu_compressed_vs_pixel_space(self):
        """
        QU-only: PixelProjected compressed EE/BB Fisher vs pixel-space.
        """
        from cosmocore.basics import matrix_inverse_symm
        from cosmocore.basis import PixelBasis

        np.random.seed(42)
        n_pix = 60
        golden_ratio = (1 + np.sqrt(5)) / 2
        indices = np.arange(n_pix)
        theta = np.arccos(1 - 2 * (indices + 0.5) / n_pix)
        phi = (2 * np.pi * indices / golden_ratio) % (2 * np.pi)
        lmax = 5

        noise_level = 1e-2
        N = np.eye(2 * n_pix) * noise_level
        np.eye(2 * n_pix) / noise_level

        # Keep all modes (no truncation) for most accurate comparison
        ppc = PixelBasis(
            N,
            theta,
            phi,
            lmax,
            spins=[2],
            epsilon=1e-12,
        )
        ppc.setup()

        ells = np.arange(2, lmax + 1)
        norm = (2 * ells + 1) / (4 * np.pi)
        C_EE = norm * 1e-4 / ells**2
        C_BB = norm * 1e-5 / ells**2

        C_ell_dict = {
            (0, 0, 0): C_EE,
            (0, 0, 1): C_BB,
        }
        spectra_list = [(0, 0, 0), (0, 0, 1)]

        fisher_compressed = ppc.compute_fisher_matrix(
            C_ell_dict, spectra_list, ell_min=2, ell_max=lmax
        )

        # Pixel-space Fisher
        V = ppc._V
        lambda_matrix = ppc._build_lambda_matrix_3tuple(C_ell_dict)
        S = V.T @ lambda_matrix @ V
        C = N + S
        C_inv = matrix_inverse_symm(C.copy())

        fisher_pixel = _pixel_space_fisher_with_spins_raw(
            C_inv,
            V,
            ppc.n_modes_total,
            lmax,
            spectra_list,
            ppc._spins,
            ppc._mode_offsets,
            ppc._n_modes_base,
            ppc._ell_to_modes_local,
        )

        mask = np.abs(fisher_pixel) > 1e-20
        rel_diff = np.abs(fisher_compressed - fisher_pixel) / (
            np.abs(fisher_pixel) + 1e-30
        )
        max_rel_diff = rel_diff[mask].max() if mask.any() else 0.0

        print(f"\n  PixelProjected QU benchmark (n_kept={ppc.n_kept}, lmax={lmax}):")
        print(f"    Max relative difference: {max_rel_diff:.2e}")

        # PixelProjected with nearly all modes should be close to pixel-space
        assert_allclose(
            fisher_compressed,
            fisher_pixel,
            rtol=1e-3,
            atol=1e-12,
            err_msg="PixelProjected QU Fisher should approximate pixel-space",
        )

    def test_tqu_compressed_vs_pixel_space(self):
        """
        TQU: PixelProjected compressed Fisher vs pixel-space.
        """
        from cosmocore.basics import matrix_inverse_symm
        from cosmocore.basis import PixelBasis

        np.random.seed(42)
        n_pix_t = 30
        n_pix_p = 25
        golden_ratio = (1 + np.sqrt(5)) / 2

        def _spiral(n, offset=0):
            idx = np.arange(n)
            th = np.arccos(1 - 2 * (idx + 0.5) / n)
            ph = (2 * np.pi * (idx + offset) / golden_ratio) % (2 * np.pi)
            return th, ph

        theta_t, phi_t = _spiral(n_pix_t, offset=0)
        theta_p, phi_p = _spiral(n_pix_p, offset=1000)
        lmax = 5

        total_pix = n_pix_t + 2 * n_pix_p
        N = np.eye(total_pix)
        N[:n_pix_t, :n_pix_t] *= 1e-3
        N[n_pix_t:, n_pix_t:] *= 5e-3
        np.linalg.inv(N)

        ppc = PixelBasis(
            N,
            (theta_t, theta_p),
            (phi_t, phi_p),
            lmax,
            spins=[0, 2],
            epsilon=1e-12,
        )
        ppc.setup()

        ells = np.arange(2, lmax + 1)
        norm = (2 * ells + 1) / (4 * np.pi)
        C_TT = norm * 1e-4 / ells**2
        C_EE = norm * 1e-5 / ells**2
        C_BB = norm * 1e-6 / ells**2
        C_TE = norm * 3e-5 / ells**2

        C_ell_dict = {
            (0, 0, 0): C_TT,
            (1, 1, 0): C_EE,
            (1, 1, 1): C_BB,
            (0, 1, 0): C_TE,
        }
        spectra_list = [
            (0, 0, 0),  # TT
            (1, 1, 0),  # EE
            (1, 1, 1),  # BB
            (0, 1, 0),  # TE
        ]

        fisher_compressed = ppc.compute_fisher_matrix(
            C_ell_dict, spectra_list, ell_min=2, ell_max=lmax
        )

        # Pixel-space Fisher
        V = ppc._V
        lambda_matrix = ppc._build_lambda_matrix_3tuple(C_ell_dict)
        S = V.T @ lambda_matrix @ V
        C = N + S
        C_inv = matrix_inverse_symm(C.copy())

        fisher_pixel = _pixel_space_fisher_with_spins_raw(
            C_inv,
            V,
            ppc.n_modes_total,
            lmax,
            spectra_list,
            ppc._spins,
            ppc._mode_offsets,
            ppc._n_modes_base,
            ppc._ell_to_modes_local,
        )

        mask = np.abs(fisher_pixel) > 1e-20
        rel_diff = np.abs(fisher_compressed - fisher_pixel) / (
            np.abs(fisher_pixel) + 1e-30
        )
        max_rel_diff = rel_diff[mask].max() if mask.any() else 0.0

        print(f"\n  PixelProjected TQU benchmark (n_kept={ppc.n_kept}, lmax={lmax}):")
        print(f"    Max relative difference: {max_rel_diff:.2e}")

        assert_allclose(
            fisher_compressed,
            fisher_pixel,
            rtol=1e-3,
            atol=1e-12,
            err_msg="PixelProjected TQU Fisher should approximate pixel-space",
        )
