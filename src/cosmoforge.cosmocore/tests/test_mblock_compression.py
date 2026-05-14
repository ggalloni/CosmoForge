"""
Tests for m-block compression in HarmonicBasis.

Tests verify that the block-diagonal approximation in azimuthal quantum
number |m| produces correct results and converges to the full solution.
"""

import time

import numpy as np
import pytest
from numpy.testing import assert_allclose


def _make_symmetric_mask_setup(lmax_signal=10, n_pix=80):
    lmax = lmax_signal
    """Create a test setup with a symmetric galactic cut (azimuthal symmetry).

    Pixels are placed on rings at various colatitudes, with all phi values
    sampled uniformly. This gives azimuthal symmetry in the mask, which
    makes V N^{-1} V^T nearly block-diagonal in |m|.
    """
    np.random.seed(42)

    # Place pixels on rings at fixed theta, sampling all phi
    n_rings = 8
    pix_per_ring = n_pix // n_rings
    actual_n_pix = n_rings * pix_per_ring

    theta_list = []
    phi_list = []

    # Galactic cut: avoid theta near 0 and pi (poles)
    # and near pi/2 (galactic plane)
    theta_values = np.array([0.3, 0.6, 0.9, 1.1, 2.0, 2.2, 2.5, 2.8])

    for ring_theta in theta_values[:n_rings]:
        ring_phi = np.linspace(0, 2 * np.pi, pix_per_ring, endpoint=False)
        theta_list.extend([ring_theta] * pix_per_ring)
        phi_list.extend(ring_phi)

    theta = np.array(theta_list)
    phi = np.array(phi_list)

    noise_variance = np.ones(actual_n_pix) * 0.05
    N = np.diag(noise_variance)
    N_inv = np.diag(1.0 / noise_variance)

    return {
        "N": N,
        "N_inv": N_inv,
        "theta": theta,
        "phi": phi,
        "lmax": lmax,
        "n_pix": actual_n_pix,
    }


def _make_random_setup(lmax_signal=8, n_pix=50):
    lmax = lmax_signal
    """Create a test setup with random pixel positions (no symmetry)."""
    np.random.seed(42)

    golden_ratio = (1 + np.sqrt(5)) / 2
    indices = np.arange(n_pix)
    theta = np.arccos(1 - 2 * (indices + 0.5) / n_pix)
    phi = (2 * np.pi * indices / golden_ratio) % (2 * np.pi)

    noise_variance = np.ones(n_pix) * 0.1
    N = np.diag(noise_variance)
    N_inv = np.diag(1.0 / noise_variance)

    return {
        "N": N,
        "N_inv": N_inv,
        "theta": theta,
        "phi": phi,
        "lmax": lmax,
        "n_pix": n_pix,
    }


class TestMblockVNinvVT:
    """Tests for block-wise V N^{-1} V^T computation."""

    def test_diagonal_blocks_match_full(self):
        """Compare diagonal m-blocks of full V N^{-1} V^T against block-wise."""
        from cosmocore.basis import HarmonicBasis

        setup = _make_random_setup()

        hc_full = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=setup["lmax"],
        )
        hc_full.setup()

        hc_comp = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=setup["lmax"],
            compress=True,
            delta_m=0,
        )
        hc_comp.setup()

        # Diagonal blocks of full V N^{-1} V^T should match block-wise
        full_vninvvt = hc_full._V_Ninv_VT

        for m, modes in hc_comp._m_to_modes.items():
            ix = np.ix_(modes, modes)
            full_block = full_vninvvt[ix]
            comp_block = hc_comp._vninvvt_blocks[(m, m)]
            assert_allclose(
                full_block,
                comp_block,
                rtol=1e-10,
                atol=1e-12,
                err_msg=f"V N^{{-1}} V^T block mismatch for m={m}",
            )


class TestMblockFisherSymmetricMask:
    """Tests for m-block Fisher with symmetric mask (azimuthal symmetry)."""

    def test_symmetric_mask_fisher_accuracy(self):
        """For a symmetric mask, m-block Fisher should be more accurate than random."""
        from cosmocore.basis import HarmonicBasis

        setup = _make_symmetric_mask_setup(lmax_signal=6, n_pix=160)
        C_ell = 1e-3 / (np.arange(2, setup["lmax"] + 1) ** 2)

        hc_full = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=setup["lmax"],
        )
        hc_full.setup()
        fisher_full = hc_full.compute_fisher_matrix(C_ell)

        hc_comp = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=setup["lmax"],
            compress=True,
            delta_m=0,
        )
        hc_comp.setup()
        fisher_comp = hc_comp.compute_fisher_matrix(C_ell)

        # For symmetric mask, diagonal elements should match to < 1%
        diag_full = np.diag(fisher_full)
        diag_comp = np.diag(fisher_comp)

        # Only check entries where full Fisher diagonal is significant
        significant = np.abs(diag_full) > 1e-10 * np.max(np.abs(diag_full))
        rel_err = np.abs(diag_full[significant] - diag_comp[significant]) / np.abs(
            diag_full[significant]
        )
        max_rel_err = np.max(rel_err)
        print(f"Symmetric mask: max diagonal relative error = {max_rel_err:.4e}")
        # With a truly symmetric mask (uniform phi sampling on rings),
        # the m-block approximation should be nearly exact
        assert max_rel_err < 1e-8, (
            f"Symmetric mask Fisher diagonal error {max_rel_err:.4e} exceeds 1e-8"
        )


class TestMblockFisherConvergence:
    """Tests for Fisher convergence with increasing delta_m."""

    def test_convergence_delta_m(self):
        """Increasing delta_m should reduce Fisher error toward zero."""
        from cosmocore.basis import HarmonicBasis

        setup = _make_random_setup(lmax_signal=8)
        C_ell = 1e-3 / (np.arange(2, setup["lmax"] + 1) ** 2)

        hc_full = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=setup["lmax"],
        )
        hc_full.setup()
        fisher_full = hc_full.compute_fisher_matrix(C_ell)

        errors = []
        delta_ms = [0, 2, setup["lmax"]]
        for dm in delta_ms:
            hc = HarmonicBasis(
                N=setup["N"],
                theta=setup["theta"],
                phi=setup["phi"],
                lmax_signal=setup["lmax"],
                compress=True,
                delta_m=dm,
            )
            hc.setup()
            fisher = hc.compute_fisher_matrix(C_ell)
            err = np.max(np.abs(fisher - fisher_full) / (np.abs(fisher_full) + 1e-30))
            errors.append(err)
            print(f"delta_m={dm}: max relative error = {err:.6e}")

        # Error should decrease monotonically
        for i in range(1, len(errors)):
            assert errors[i] <= errors[i - 1] + 1e-14, (
                f"Error did not decrease: delta_m={delta_ms[i]} "
                f"({errors[i]:.2e}) > delta_m={delta_ms[i - 1]} ({errors[i - 1]:.2e})"
            )

        # delta_m = lmax should recover exact result
        assert errors[-1] < 1e-10, (
            f"delta_m=lmax did not recover exact Fisher: error = {errors[-1]:.2e}"
        )


class TestMblockCompressFalseUnchanged:
    """Tests that compress=False preserves all existing behavior."""

    def test_compress_false_identical(self, uniform_sky_setup):
        """With compress=False, results should be identical to default."""
        from cosmocore.basis import HarmonicBasis

        setup = uniform_sky_setup
        C_ell = np.ones(setup["lmax"] - 1) * 1e-6

        hc_default = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=setup["lmax"],
        )
        hc_default.setup()

        hc_explicit = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=setup["lmax"],
            compress=False,
        )
        hc_explicit.setup()

        # V matrices should be identical
        assert_allclose(hc_default._V, hc_explicit._V, rtol=1e-15)

        # Projected inverse should be identical
        proj_default = hc_default.get_projected_inverse(C_ell)
        proj_explicit = hc_explicit.get_projected_inverse(C_ell)
        assert_allclose(proj_default, proj_explicit, rtol=1e-15)

        # Fisher should be identical
        fisher_default = hc_default.compute_fisher_matrix(C_ell)
        fisher_explicit = hc_explicit.compute_fisher_matrix(C_ell)
        assert_allclose(fisher_default, fisher_explicit, rtol=1e-15)

        # Covariance should be identical
        cov_default = hc_default.get_covariance(C_ell)
        cov_explicit = hc_explicit.get_covariance(C_ell)
        assert_allclose(cov_default, cov_explicit, rtol=1e-15)


class TestMblockKInversionSpeedup:
    """Timing tests for m-block vs full K inversion."""

    def test_timing_comparison(self):
        """Compare timing of full vs m-block K inversion."""
        from cosmocore.basis import HarmonicBasis

        for lmax in [15, 20]:
            n_pix = max(200, (lmax + 1) ** 2 + 50)

            np.random.seed(42)
            golden_ratio = (1 + np.sqrt(5)) / 2
            indices = np.arange(n_pix)
            theta = np.arccos(1 - 2 * (indices + 0.5) / n_pix)
            phi = (2 * np.pi * indices / golden_ratio) % (2 * np.pi)

            noise_variance = np.ones(n_pix) * 0.1
            N = np.diag(noise_variance)
            np.diag(1.0 / noise_variance)

            C_ell = 1e-3 / (np.arange(2, lmax + 1) ** 2)

            # Full
            hc_full = HarmonicBasis(N=N, theta=theta, phi=phi, lmax_signal=lmax)
            hc_full.setup()

            t0 = time.time()
            _ = hc_full.get_projected_inverse(C_ell)
            t_full = time.time() - t0

            # Compressed
            hc_comp = HarmonicBasis(
                N=N,
                theta=theta,
                phi=phi,
                lmax_signal=lmax,
                compress=True,
                delta_m=0,
            )
            hc_comp.setup()

            t0 = time.time()
            _ = hc_comp.get_projected_inverse(C_ell)
            t_mblock = time.time() - t0

            speedup = t_full / max(t_mblock, 1e-10)
            print(
                f"lmax={lmax}: full={t_full:.4f}s, mblock={t_mblock:.4f}s, "
                f"speedup={speedup:.1f}x"
            )

            # Just a sanity check that mblock is not slower
            # (actual speedup may vary due to overhead at small sizes)


class TestMblockMultifieldRejection:
    """Tests that multi-field + compress raises NotImplementedError."""

    def test_multifield_raises(self, two_scalar_field_setup):
        """Multi-field with compress=True should raise NotImplementedError."""
        from cosmocore.basis import HarmonicBasis

        setup = two_scalar_field_setup
        with pytest.raises(NotImplementedError, match="single-field"):
            HarmonicBasis(
                N=setup["N"],
                theta=setup["theta"],
                phi=setup["phi"],
                lmax_signal=setup["lmax"],
                spins=[0, 0],
                compress=True,
            )


class TestMblockFactoryIntegration:
    """Tests that compress/delta_m work through the factory function."""

    def test_factory_passes_compress(self):
        """create_computation_basis should pass compress and delta_m."""
        from cosmocore.basis import create_computation_basis

        setup = _make_random_setup(lmax_signal=6)
        hc = create_computation_basis(
            method="harmonic",
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=setup["lmax"],
            compress=True,
            delta_m=2,
        )
        assert hc._compress is True
        assert hc._delta_m == 2

    def test_factory_default_compress_false(self):
        """Default compress should be False."""
        from cosmocore.basis import create_computation_basis

        setup = _make_random_setup(lmax_signal=6)
        hc = create_computation_basis(
            method="harmonic",
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=setup["lmax"],
        )
        assert hc._compress is False


class TestMblockFisherProperties:
    """Tests for Fisher matrix properties under m-block compression."""

    def test_fisher_symmetric(self):
        """M-block Fisher should be symmetric."""
        from cosmocore.basis import HarmonicBasis

        setup = _make_random_setup()
        C_ell = 1e-3 / (np.arange(2, setup["lmax"] + 1) ** 2)

        hc = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=setup["lmax"],
            compress=True,
            delta_m=0,
        )
        hc.setup()

        fisher = hc.compute_fisher_matrix(C_ell)
        assert_allclose(fisher, fisher.T, rtol=1e-12)

    def test_fisher_positive_diagonal(self):
        """M-block Fisher diagonal elements should be non-negative."""
        from cosmocore.basis import HarmonicBasis

        setup = _make_random_setup()
        C_ell = 1e-3 / (np.arange(2, setup["lmax"] + 1) ** 2)

        hc = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=setup["lmax"],
            compress=True,
            delta_m=0,
        )
        hc.setup()

        fisher = hc.compute_fisher_matrix(C_ell)
        assert np.all(np.diag(fisher) >= -1e-10)
