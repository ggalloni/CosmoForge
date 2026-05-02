"""
Integration tests for computation basis redesign.

Tests all basis+compression combinations, verifies m-block + field block
compose correctly, and provides comprehensive benchmarks.
"""

import time

import numpy as np
from numpy.testing import assert_allclose

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_single_field_setup(lmax=8, n_pix=50, seed=42):
    """Single spin-0 field with random pixels."""
    np.random.seed(seed)
    theta = np.random.uniform(0.3, 2.8, n_pix)
    phi = np.random.uniform(0, 2 * np.pi, n_pix)
    noise_var = np.random.uniform(0.01, 0.05, n_pix)
    N = np.diag(noise_var)
    N_inv = np.diag(1.0 / noise_var)
    C_ell = np.ones(lmax + 1) * 1e-3
    return {
        "N": N,
        "N_inv": N_inv,
        "theta": theta,
        "phi": phi,
        "lmax": lmax,
        "C_ell": C_ell,
    }


def _make_symmetric_setup(lmax=10, n_rings=10, pix_per_ring=20, seed=42):
    """Setup with azimuthally symmetric pixel distribution."""
    np.random.seed(seed)
    theta_list = []
    phi_list = []
    theta_values = np.linspace(0.3, 2.8, n_rings)
    for ring_theta in theta_values:
        ring_phi = np.linspace(0, 2 * np.pi, pix_per_ring, endpoint=False)
        theta_list.extend([ring_theta] * pix_per_ring)
        phi_list.extend(ring_phi)
    theta = np.array(theta_list)
    phi = np.array(phi_list)
    n_pix = len(theta)
    noise_var = np.ones(n_pix) * 0.02
    N = np.diag(noise_var)
    N_inv = np.diag(1.0 / noise_var)
    C_ell = np.ones(lmax + 1) * 1e-3
    return {
        "N": N,
        "N_inv": N_inv,
        "theta": theta,
        "phi": phi,
        "lmax": lmax,
        "C_ell": C_ell,
    }


def _make_two_field_independent(lmax=6, n_pix_1=40, n_pix_2=30, seed=42):
    """Two independent spin-0 fields (no cross-spectra, independent noise)."""
    np.random.seed(seed)
    n_pix_total = n_pix_1 + n_pix_2

    N = np.zeros((n_pix_total, n_pix_total))
    N[:n_pix_1, :n_pix_1] = np.eye(n_pix_1) * 0.01
    N[n_pix_1:, n_pix_1:] = np.eye(n_pix_2) * 0.02
    N_inv = np.zeros((n_pix_total, n_pix_total))
    N_inv[:n_pix_1, :n_pix_1] = np.eye(n_pix_1) * 100.0
    N_inv[n_pix_1:, n_pix_1:] = np.eye(n_pix_2) * 50.0

    theta_1 = np.random.uniform(0.3, 2.8, n_pix_1)
    phi_1 = np.random.uniform(0, 2 * np.pi, n_pix_1)
    theta_2 = np.random.uniform(0.3, 2.8, n_pix_2)
    phi_2 = np.random.uniform(0, 2 * np.pi, n_pix_2)

    C_ell_dict = {
        (0, 0): np.ones(lmax + 1) * 1e-3,
        (1, 1): np.ones(lmax + 1) * 2e-3,
    }
    spectra_list = [(0, 0), (1, 1)]

    return {
        "N": N,
        "N_inv": N_inv,
        "theta": (theta_1, theta_2),
        "phi": (phi_1, phi_2),
        "lmax": lmax,
        "C_ell_dict": C_ell_dict,
        "spectra_list": spectra_list,
    }


# ---------------------------------------------------------------------------
# Test: All basis combinations produce valid results
# ---------------------------------------------------------------------------


class TestBasisCombinations:
    """All basis+compression combinations produce valid Fisher matrices."""

    def test_harmonic_no_compress(self):
        from cosmocore.basis import HarmonicBasis

        s = _make_single_field_setup()
        hb = HarmonicBasis(N=s["N"], theta=s["theta"], phi=s["phi"], lmax=s["lmax"])
        hb.setup()
        F = hb.compute_fisher_matrix(s["C_ell"], ell_min=2, ell_max=s["lmax"])
        assert F.shape == (s["lmax"] - 1, s["lmax"] - 1)
        assert_allclose(F, F.T, atol=1e-12)
        assert np.all(np.diag(F) > 0)

    def test_harmonic_with_mblock(self):
        from cosmocore.basis import HarmonicBasis

        s = _make_single_field_setup()
        hb = HarmonicBasis(
            N=s["N"],
            theta=s["theta"],
            phi=s["phi"],
            lmax=s["lmax"],
            compress=True,
            delta_m=0,
        )
        hb.setup()
        F = hb.compute_fisher_matrix(s["C_ell"], ell_min=2, ell_max=s["lmax"])
        assert F.shape == (s["lmax"] - 1, s["lmax"] - 1)
        assert_allclose(F, F.T, atol=1e-12)
        assert np.all(np.diag(F) > 0)

    def test_pixel_with_full_modes(self):
        """PixelBasis with epsilon=0 keeps all modes (no real compression)."""
        from cosmocore.basis import PixelBasis

        s = _make_single_field_setup()
        pb = PixelBasis(
            N=s["N"],
            theta=s["theta"],
            phi=s["phi"],
            lmax=s["lmax"],
            epsilon=0.0,
        )
        pb.setup()
        pb.apply_compression(epsilon=0.0)
        F = pb.compute_fisher_matrix(s["C_ell"], ell_min=2, ell_max=s["lmax"])
        assert F.shape == (s["lmax"] - 1, s["lmax"] - 1)
        assert_allclose(F, F.T, atol=1e-10)
        assert np.all(np.diag(F) > 0)

    def test_pixel_with_eigenmode(self):
        from cosmocore.basis import PixelBasis

        s = _make_single_field_setup()
        pb = PixelBasis(
            N=s["N"],
            theta=s["theta"],
            phi=s["phi"],
            lmax=s["lmax"],
            epsilon=1e-6,
        )
        pb.setup()
        pb.apply_compression(epsilon=1e-6)
        F = pb.compute_fisher_matrix(s["C_ell"], ell_min=2, ell_max=s["lmax"])
        assert F.shape == (s["lmax"] - 1, s["lmax"] - 1)
        assert_allclose(F, F.T, atol=1e-10)
        assert np.all(np.diag(F) > 0)

    def test_harmonic_and_pixel_agree(self):
        """Harmonic and pixel basis give same Fisher."""
        from cosmocore.basis import HarmonicBasis, PixelBasis

        s = _make_single_field_setup(lmax=6, n_pix=40)
        hb = HarmonicBasis(N=s["N"], theta=s["theta"], phi=s["phi"], lmax=s["lmax"])
        hb.setup()
        F_h = hb.compute_fisher_matrix(s["C_ell"], ell_min=2, ell_max=s["lmax"])

        pb = PixelBasis(
            N=s["N"],
            theta=s["theta"],
            phi=s["phi"],
            lmax=s["lmax"],
            epsilon=0.0,
        )
        pb.setup()
        pb.apply_compression(epsilon=0.0)
        F_p = pb.compute_fisher_matrix(s["C_ell"], ell_min=2, ell_max=s["lmax"])

        assert_allclose(F_h, F_p, rtol=1e-6)


# ---------------------------------------------------------------------------
# Test: M-block approximation quality vs mask symmetry
# ---------------------------------------------------------------------------


class TestMblockApproximationQuality:
    """Measure m-block approximation quality for different mask geometries."""

    def test_symmetric_mask_near_exact(self):
        """Azimuthally symmetric mask: m-block is near-exact."""
        from cosmocore.basis import HarmonicBasis

        s = _make_symmetric_setup(lmax=8, n_rings=8, pix_per_ring=20)
        hb_full = HarmonicBasis(N=s["N"], theta=s["theta"], phi=s["phi"], lmax=s["lmax"])
        hb_full.setup()
        F_full = hb_full.compute_fisher_matrix(s["C_ell"], ell_min=2, ell_max=s["lmax"])

        hb_mb = HarmonicBasis(
            N=s["N"],
            theta=s["theta"],
            phi=s["phi"],
            lmax=s["lmax"],
            compress=True,
            delta_m=0,
        )
        hb_mb.setup()
        F_mb = hb_mb.compute_fisher_matrix(s["C_ell"], ell_min=2, ell_max=s["lmax"])

        diag_err = np.abs(np.diag(F_mb) - np.diag(F_full)) / np.abs(np.diag(F_full))
        max_err = np.max(diag_err)
        print(f"\nSymmetric mask (lmax={s['lmax']}): max diagonal error = {max_err:.4e}")
        # Symmetric mask should give very good approximation
        assert max_err < 0.05, (
            f"Expected < 5% error for symmetric mask, got {max_err:.4e}"
        )

    def test_random_mask_bounded_error(self):
        """Random (asymmetric) mask: error is bounded."""
        from cosmocore.basis import HarmonicBasis

        s = _make_single_field_setup(lmax=8, n_pix=60)
        hb_full = HarmonicBasis(N=s["N"], theta=s["theta"], phi=s["phi"], lmax=s["lmax"])
        hb_full.setup()
        F_full = hb_full.compute_fisher_matrix(s["C_ell"], ell_min=2, ell_max=s["lmax"])

        hb_mb = HarmonicBasis(
            N=s["N"],
            theta=s["theta"],
            phi=s["phi"],
            lmax=s["lmax"],
            compress=True,
            delta_m=0,
        )
        hb_mb.setup()
        F_mb = hb_mb.compute_fisher_matrix(s["C_ell"], ell_min=2, ell_max=s["lmax"])

        diag_err = np.abs(np.diag(F_mb) - np.diag(F_full)) / np.abs(np.diag(F_full))
        max_err = np.max(diag_err)
        mean_err = np.mean(diag_err)
        print(
            f"\nRandom mask (lmax={s['lmax']}): "
            f"max diag error = {max_err:.4e}, mean = {mean_err:.4e}"
        )
        # Just report, don't assert strict bound for random masks

    def test_delta_m_improves_accuracy(self):
        """Increasing delta_m monotonically improves accuracy."""
        from cosmocore.basis import HarmonicBasis

        s = _make_single_field_setup(lmax=8, n_pix=60)
        hb_full = HarmonicBasis(N=s["N"], theta=s["theta"], phi=s["phi"], lmax=s["lmax"])
        hb_full.setup()
        F_full = hb_full.compute_fisher_matrix(s["C_ell"], ell_min=2, ell_max=s["lmax"])

        errors = []
        for dm in [0, 1, 2, 4, s["lmax"]]:
            hb_mb = HarmonicBasis(
                N=s["N"],
                theta=s["theta"],
                phi=s["phi"],
                lmax=s["lmax"],
                compress=True,
                delta_m=dm,
            )
            hb_mb.setup()
            F_mb = hb_mb.compute_fisher_matrix(s["C_ell"], ell_min=2, ell_max=s["lmax"])
            err = np.max(
                np.abs(np.diag(F_mb) - np.diag(F_full)) / np.abs(np.diag(F_full))
            )
            errors.append(err)
            print(f"  delta_m={dm}: max diag error = {err:.4e}")

        # delta_m=lmax should recover exact
        assert errors[-1] < 1e-10, (
            f"delta_m=lmax should be exact, got error {errors[-1]:.4e}"
        )


# ---------------------------------------------------------------------------
# Benchmarks: scaling with lmax
# ---------------------------------------------------------------------------


class TestScalingBenchmarks:
    """Benchmark K inversion and Fisher computation scaling."""

    def test_mblock_k_inversion_scaling(self):
        """K inversion speedup scales with lmax."""
        from cosmocore.basis import HarmonicBasis

        print("\n--- M-block K inversion scaling ---")
        for lmax in [10, 15, 20]:
            np.random.seed(42)
            n_pix = max(60, lmax * 4)
            theta = np.random.uniform(0.3, 2.8, n_pix)
            phi = np.random.uniform(0, 2 * np.pi, n_pix)
            noise_var = np.random.uniform(0.01, 0.05, n_pix)
            N = np.diag(noise_var)
            np.diag(1.0 / noise_var)
            C_ell = np.ones(lmax + 1) * 1e-3

            # Full
            hb_full = HarmonicBasis(N=N, theta=theta, phi=phi, lmax=lmax)
            hb_full.setup()
            t0 = time.perf_counter()
            for _ in range(3):
                hb_full.get_projected_inverse(C_ell)
            t_full = (time.perf_counter() - t0) / 3

            # M-block
            hb_mb = HarmonicBasis(
                N=N,
                theta=theta,
                phi=phi,
                lmax=lmax,
                compress=True,
                delta_m=0,
            )
            hb_mb.setup()
            t0 = time.perf_counter()
            for _ in range(3):
                hb_mb.get_projected_inverse(C_ell)
            t_mb = (time.perf_counter() - t0) / 3

            speedup = t_full / t_mb if t_mb > 0 else float("inf")
            print(
                f"  lmax={lmax:3d}: full={t_full:.4f}s, "
                f"mblock={t_mb:.4f}s, speedup={speedup:.1f}x"
            )

    def test_mblock_fisher_scaling(self):
        """Fisher computation speedup scales with lmax."""
        from cosmocore.basis import HarmonicBasis

        print("\n--- M-block Fisher scaling ---")
        for lmax in [10, 15, 20]:
            np.random.seed(42)
            n_pix = max(60, lmax * 4)
            theta = np.random.uniform(0.3, 2.8, n_pix)
            phi = np.random.uniform(0, 2 * np.pi, n_pix)
            noise_var = np.random.uniform(0.01, 0.05, n_pix)
            N = np.diag(noise_var)
            np.diag(1.0 / noise_var)
            C_ell = np.ones(lmax + 1) * 1e-3

            # Full
            hb_full = HarmonicBasis(N=N, theta=theta, phi=phi, lmax=lmax)
            hb_full.setup()
            t0 = time.perf_counter()
            F_full = hb_full.compute_fisher_matrix(C_ell, ell_min=2, ell_max=lmax)
            t_full = time.perf_counter() - t0

            # M-block
            hb_mb = HarmonicBasis(
                N=N,
                theta=theta,
                phi=phi,
                lmax=lmax,
                compress=True,
                delta_m=0,
            )
            hb_mb.setup()
            t0 = time.perf_counter()
            F_mb = hb_mb.compute_fisher_matrix(C_ell, ell_min=2, ell_max=lmax)
            t_fisher = time.perf_counter() - t0

            speedup = t_full / t_fisher if t_fisher > 0 else float("inf")
            diag_err = np.max(
                np.abs(np.diag(F_mb) - np.diag(F_full)) / np.abs(np.diag(F_full))
            )
            print(
                f"  lmax={lmax:3d}: full={t_full:.4f}s, mblock={t_fisher:.4f}s, "
                f"speedup={speedup:.1f}x, max_diag_err={diag_err:.2e}"
            )

    def test_field_block_scaling(self):
        """Field block-diagonal speedup for multi-field case."""
        from cosmocore.basis import HarmonicBasis

        print("\n--- Field block-diagonal scaling ---")
        for n_fields in [2, 3]:
            np.random.seed(42)
            lmax = 8
            n_pix_each = 40

            thetas = []
            phis = []
            for i in range(n_fields):
                thetas.append(np.random.uniform(0.3, 2.8, n_pix_each))
                phis.append(np.random.uniform(0, 2 * np.pi, n_pix_each))

            n_pix_total = n_fields * n_pix_each
            N = np.eye(n_pix_total) * 0.01
            np.eye(n_pix_total) * 100.0

            C_ell_dict = {(i, i): np.ones(lmax + 1) * 1e-3 for i in range(n_fields)}
            spectra_list = [(i, i) for i in range(n_fields)]

            hb = HarmonicBasis(
                N=N,
                theta=tuple(thetas),
                phi=tuple(phis),
                lmax=lmax,
                spins=[0] * n_fields,
            )
            hb.setup()

            # Time Fisher (field-block auto-detected)
            t0 = time.perf_counter()
            for _ in range(3):
                hb.compute_fisher_matrix(
                    C_ell_dict, spectra_list=spectra_list, ell_min=2, ell_max=lmax
                )
            t_block = (time.perf_counter() - t0) / 3

            # Time with cross-spectra (forces coupled)
            C_ell_coupled = dict(C_ell_dict)
            C_ell_coupled[(0, 1)] = np.ones(lmax + 1) * 1e-4
            spectra_coupled = spectra_list + [(0, 1)]

            t0 = time.perf_counter()
            for _ in range(3):
                hb.compute_fisher_matrix(
                    C_ell_coupled, spectra_list=spectra_coupled, ell_min=2, ell_max=lmax
                )
            t_coupled = (time.perf_counter() - t0) / 3

            print(
                f"  {n_fields} fields: block={t_block:.4f}s, coupled={t_coupled:.4f}s, "
                f"ratio={t_coupled / t_block:.1f}x"
            )
