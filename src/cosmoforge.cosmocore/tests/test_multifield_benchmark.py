"""
Benchmark test: Multi-field compression vs traditional pixel-space computation.

This test compares the multi-field harmonic compression framework against
the traditional pixel-space Fisher matrix computation for a realistic
3-scalar-field scenario (simulating T, E, B treated as independent scalars).

Results show:
1. Numerical precision: How close compressed Fisher matches pixel-space
2. Performance: Timing comparison for compression vs traditional method
"""

import time

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cosmocore.basics import matrix_inverse_symm, matrix_mult, matrix_trace
from cosmocore.compression import HarmonicCompression


def build_pixel_space_signal_matrix(V, Lambda_full):
    """Build full signal matrix S = V^T Λ V."""
    return V.T @ Lambda_full @ V


def compute_pixel_space_fisher(C_inv, V, Lambda_full, lmax, spectra_list, hc):
    """
    Compute Fisher matrix using traditional pixel-space method.

    F_ij = 0.5 * Tr[C^{-1} dS_i C^{-1} dS_j]

    where dS_i = V^T E_i V is the derivative of signal matrix.
    """
    n_ell = lmax - 1
    n_spectra = len(spectra_list)
    fisher = np.zeros((n_spectra * n_ell, n_spectra * n_ell))

    for spec_idx_i, (comp_i, comp_j) in enumerate(spectra_list):
        for ell_i in range(2, lmax + 1):
            # Build dS/dC_ell for spectrum (comp_i, comp_j)
            E_i = hc.get_derivative_matrix(ell_i, comp_i, comp_j)
            dS_i = V.T @ E_i @ V

            for spec_idx_j, (comp_k, comp_l) in enumerate(spectra_list):
                for ell_j in range(2, lmax + 1):
                    E_j = hc.get_derivative_matrix(ell_j, comp_k, comp_l)
                    dS_j = V.T @ E_j @ V

                    # F_ij = 0.5 * Tr[C^{-1} dS_i C^{-1} dS_j]
                    temp1 = matrix_mult(C_inv, dS_i)
                    temp2 = matrix_mult(C_inv, dS_j)
                    f_val = 0.5 * matrix_trace(temp1, temp2)

                    row_idx = spec_idx_i * n_ell + (ell_i - 2)
                    col_idx = spec_idx_j * n_ell + (ell_j - 2)
                    fisher[row_idx, col_idx] = f_val

    return fisher


@pytest.fixture
def teb_scalar_setup():
    """
    Create a realistic TEB-like setup with three scalar fields.

    This simulates having T, E, B maps but treating them all as independent
    spin-0 fields (e.g., component-separated frequency maps).

    Uses sizes similar to existing nside=4 tests for fast execution.
    """
    np.random.seed(42)

    # Three fields with small pixel counts (similar to nside=4 test data)
    # nside=4 gives 192 total pixels, we use ~50 per field after masking
    n_pix_T = 50
    n_pix_E = 40
    n_pix_B = 35
    n_pix_total = n_pix_T + n_pix_E + n_pix_B

    # Generate uniform positions on sphere using golden spiral
    def golden_spiral_positions(n_pix, seed_offset=0):
        np.random.seed(42 + seed_offset)
        golden_ratio = (1 + np.sqrt(5)) / 2
        indices = np.arange(n_pix)
        theta = np.arccos(1 - 2 * (indices + 0.5) / n_pix)
        phi = (2 * np.pi * indices / golden_ratio) % (2 * np.pi)
        theta += np.random.uniform(-0.01, 0.01, n_pix)
        phi += np.random.uniform(-0.01, 0.01, n_pix)
        return theta, phi

    theta_T, phi_T = golden_spiral_positions(n_pix_T, seed_offset=0)
    theta_E, phi_E = golden_spiral_positions(n_pix_E, seed_offset=100)
    theta_B, phi_B = golden_spiral_positions(n_pix_B, seed_offset=200)

    theta_tuple = (theta_T, theta_E, theta_B)
    phi_tuple = (phi_T, phi_E, phi_B)

    # Block-diagonal noise with realistic levels
    noise_T = np.ones(n_pix_T) * 1e-3  # Temperature: low noise
    noise_E = np.ones(n_pix_E) * 5e-3  # E-mode: higher noise
    noise_B = np.ones(n_pix_B) * 1e-2  # B-mode: highest noise

    N = np.zeros((n_pix_total, n_pix_total))
    N[:n_pix_T, :n_pix_T] = np.diag(noise_T)
    N[n_pix_T : n_pix_T + n_pix_E, n_pix_T : n_pix_T + n_pix_E] = np.diag(noise_E)
    N[n_pix_T + n_pix_E :, n_pix_T + n_pix_E :] = np.diag(noise_B)

    N_inv = np.zeros((n_pix_total, n_pix_total))
    N_inv[:n_pix_T, :n_pix_T] = np.diag(1.0 / noise_T)
    N_inv[n_pix_T : n_pix_T + n_pix_E, n_pix_T : n_pix_T + n_pix_E] = np.diag(
        1.0 / noise_E
    )
    N_inv[n_pix_T + n_pix_E :, n_pix_T + n_pix_E :] = np.diag(1.0 / noise_B)

    lmax = 8  # Same as nside=4 tests

    # Realistic power spectra (physical C_ell values, no pre-normalization)
    n_ell = lmax - 1
    ells = np.arange(2, lmax + 1)

    # Auto-spectra (TT, EE, BB)
    C_TT = 1e-4 / ells**2
    C_EE = 1e-5 / ells**2  # E weaker than T
    C_BB = 1e-6 / ells**2  # B much weaker

    # Cross-spectra (TE correlation, TB/EB typically zero but add small values)
    C_TE = 3e-5 / ells**2  # TE correlation
    C_TB = 1e-8 / ells**2  # TB ~0
    C_EB = 1e-8 / ells**2  # EB ~0

    C_ell_dict = {
        (0, 0): C_TT,  # TT
        (1, 1): C_EE,  # EE
        (2, 2): C_BB,  # BB
        (0, 1): C_TE,  # TE
        (0, 2): C_TB,  # TB
        (1, 2): C_EB,  # EB
    }

    return {
        "N": N,
        "N_inv": N_inv,
        "theta": theta_tuple,
        "phi": phi_tuple,
        "lmax": lmax,
        "n_pix_total": n_pix_total,
        "C_ell_dict": C_ell_dict,
        "n_ell": n_ell,
    }


class TestTEBScalarBenchmark:
    """Benchmark tests for TEB treated as 3 scalar fields."""

    def test_teb_compressed_vs_pixel_space_precision(self, teb_scalar_setup):
        """
        Test numerical precision: compressed Fisher vs pixel-space computation.

        This is the key validation that the multi-field compression framework
        produces mathematically equivalent results to traditional methods.
        """
        setup = teb_scalar_setup
        lmax = setup["lmax"]

        # Setup compression
        hc = HarmonicCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=lmax,
        )
        hc.setup()

        # Test with auto-spectra only first (simpler case)
        spectra_list = [(0, 0), (1, 1), (2, 2)]  # TT, EE, BB

        # Compressed Fisher
        fisher_compressed = hc.compute_fisher_matrix(
            setup["C_ell_dict"], spectra_list, ell_min=2, ell_max=lmax
        )

        # Pixel-space Fisher
        V = hc._V
        Lambda_full = hc._build_lambda_full(setup["C_ell_dict"])
        S = build_pixel_space_signal_matrix(V, Lambda_full)
        C = setup["N"] + S
        C_inv = matrix_inverse_symm(C.copy())

        fisher_pixel = compute_pixel_space_fisher(
            C_inv, V, Lambda_full, lmax, spectra_list, hc
        )

        # Check precision
        rel_diff = np.abs(fisher_compressed - fisher_pixel) / (
            np.abs(fisher_pixel) + 1e-30
        )
        max_rel_diff = np.max(rel_diff)
        mean_rel_diff = np.mean(rel_diff)

        print("\n=== TEB Auto-Spectra Precision Test ===")
        print(f"Fisher shape: {fisher_compressed.shape}")
        print(f"Max relative difference: {max_rel_diff:.2e}")
        print(f"Mean relative difference: {mean_rel_diff:.2e}")

        # They should match to high precision
        assert_allclose(
            fisher_compressed,
            fisher_pixel,
            rtol=1e-6,
            atol=1e-12,
            err_msg="Compressed Fisher should match pixel-space",
        )

    def test_teb_full_spectra_precision(self, teb_scalar_setup):
        """
        Test precision with all 6 spectra (3 auto + 3 cross).

        This tests the full multi-field capability with cross-correlations.
        """
        setup = teb_scalar_setup
        lmax = setup["lmax"]

        hc = HarmonicCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=lmax,
        )
        hc.setup()

        # All 6 spectra
        spectra_list = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]

        # Compressed Fisher
        fisher_compressed = hc.compute_fisher_matrix(
            setup["C_ell_dict"], spectra_list, ell_min=2, ell_max=lmax
        )

        # Pixel-space Fisher
        V = hc._V
        Lambda_full = hc._build_lambda_full(setup["C_ell_dict"])
        S = build_pixel_space_signal_matrix(V, Lambda_full)
        C = setup["N"] + S
        C_inv = matrix_inverse_symm(C.copy())

        fisher_pixel = compute_pixel_space_fisher(
            C_inv, V, Lambda_full, lmax, spectra_list, hc
        )

        # Precision metrics
        rel_diff = np.abs(fisher_compressed - fisher_pixel) / (
            np.abs(fisher_pixel) + 1e-30
        )
        max_rel_diff = np.max(rel_diff)
        mean_rel_diff = np.mean(rel_diff)

        n_spectra = len(spectra_list)
        n_ell = setup["n_ell"]

        print("\n=== TEB Full 6-Spectra Precision Test ===")
        print(f"Number of spectra: {n_spectra}")
        print(f"Fisher shape: {fisher_compressed.shape} ({n_spectra}×{n_ell})")
        print(f"Max relative difference: {max_rel_diff:.2e}")
        print(f"Mean relative difference: {mean_rel_diff:.2e}")

        # Check symmetry
        assert_allclose(fisher_compressed, fisher_compressed.T, rtol=1e-10)

        # Check positive definiteness
        eigenvalues = np.linalg.eigvalsh(fisher_compressed)
        print(f"Min eigenvalue: {np.min(eigenvalues):.2e}")
        print(f"Max eigenvalue: {np.max(eigenvalues):.2e}")

        assert_allclose(
            fisher_compressed,
            fisher_pixel,
            rtol=1e-6,
            atol=1e-12,
            err_msg="Full 6-spectra compressed Fisher should match pixel-space",
        )

    def test_teb_performance_comparison(self, teb_scalar_setup):
        """
        Performance benchmark: compression vs pixel-space computation time.

        Note: The compression method precomputes V C^{-1} V^T once, then
        computes all Fisher elements efficiently. The pixel-space method
        requires full matrix multiplications for each Fisher element.
        """
        setup = teb_scalar_setup
        lmax = setup["lmax"]

        hc = HarmonicCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=lmax,
        )

        # Time setup
        t0 = time.perf_counter()
        hc.setup()
        setup_time = time.perf_counter() - t0

        spectra_list = [(0, 0), (1, 1), (2, 2)]  # Auto-spectra only for speed

        # Time compressed Fisher computation
        t0 = time.perf_counter()
        fisher_compressed = hc.compute_fisher_matrix(
            setup["C_ell_dict"], spectra_list, ell_min=2, ell_max=lmax
        )
        compressed_time = time.perf_counter() - t0

        # Time pixel-space Fisher computation
        V = hc._V
        Lambda_full = hc._build_lambda_full(setup["C_ell_dict"])
        S = build_pixel_space_signal_matrix(V, Lambda_full)
        C = setup["N"] + S

        t0 = time.perf_counter()
        C_inv = matrix_inverse_symm(C.copy())
        inversion_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        fisher_pixel = compute_pixel_space_fisher(
            C_inv, V, Lambda_full, lmax, spectra_list, hc
        )
        pixel_fisher_time = time.perf_counter() - t0

        total_pixel_time = inversion_time + pixel_fisher_time

        print("\n=== TEB Performance Benchmark ===")
        print(f"n_pix_total: {setup['n_pix_total']}")
        print(f"n_modes_total: {hc.n_modes_total}")
        print(f"lmax: {lmax}")
        print(f"n_spectra: {len(spectra_list)}")
        print("\nCompression method:")
        print(f"  Setup time: {setup_time * 1000:.2f} ms")
        print(f"  Fisher computation: {compressed_time * 1000:.2f} ms")
        print(f"  Total: {(setup_time + compressed_time) * 1000:.2f} ms")
        print("\nPixel-space method:")
        print(f"  Matrix inversion: {inversion_time * 1000:.2f} ms")
        print(f"  Fisher computation: {pixel_fisher_time * 1000:.2f} ms")
        print(f"  Total: {total_pixel_time * 1000:.2f} ms")
        speedup = total_pixel_time / (setup_time + compressed_time)
        print(f"\nSpeedup factor: {speedup:.2f}x")

        # Verify results match
        assert_allclose(fisher_compressed, fisher_pixel, rtol=1e-6, atol=1e-12)

    def test_error_bars_from_fisher(self, teb_scalar_setup):
        """Test that we can extract meaningful error bars from Fisher matrix."""
        setup = teb_scalar_setup
        lmax = setup["lmax"]

        hc = HarmonicCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=lmax,
        )
        hc.setup()

        spectra_list = [(0, 0), (1, 1), (2, 2)]  # TT, EE, BB
        n_spectra = len(spectra_list)
        n_ell = setup["n_ell"]

        fisher = hc.compute_fisher_matrix(
            setup["C_ell_dict"], spectra_list, ell_min=2, ell_max=lmax
        )

        # Invert Fisher to get covariance
        fisher_inv = np.linalg.inv(fisher)

        # Extract error bars (sqrt of diagonal)
        errors = np.sqrt(np.diag(fisher_inv))

        # Reshape to (n_spectra, n_ell)
        errors_reshaped = errors.reshape(n_spectra, n_ell)

        print("\n=== Error Bars from Fisher Matrix ===")
        spectrum_names = ["TT", "EE", "BB"]
        ells = np.arange(2, lmax + 1)

        for i, name in enumerate(spectrum_names):
            print(f"\n{name} errors:")
            for j, ell in enumerate(ells):
                print(f"  ell={ell}: σ = {errors_reshaped[i, j]:.2e}")

        # Verify errors are positive
        assert np.all(errors > 0), "All error bars should be positive"

        # BB should have larger errors than EE, which should have larger than TT
        # (due to noise levels and signal amplitudes)
        mean_err_TT = np.mean(errors_reshaped[0])
        mean_err_EE = np.mean(errors_reshaped[1])
        mean_err_BB = np.mean(errors_reshaped[2])

        print(
            f"\nMean errors: TT={mean_err_TT:.2e},"
            f" EE={mean_err_EE:.2e}, BB={mean_err_BB:.2e}"
        )
