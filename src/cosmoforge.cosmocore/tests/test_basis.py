"""
Tests for create_computation_basis, compute_fisher_matrix, per-field thresholds,
and PICSLike-compatible methods on PixelBasis.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cosmocore.spectrum_key import SpectrumKey, SpectrumKind


def _to_key(entry, *, spins):
    """Test-only adapter: legacy (comp_i, comp_j[, mode]) tuple -> SpectrumKey."""
    if len(entry) == 2:
        comp_i, comp_j = entry
        mode = 0
    else:
        comp_i, comp_j, mode = entry
    spin_i, spin_j = spins[comp_i], spins[comp_j]
    if (spin_i, spin_j) == (0, 0):
        kind = SpectrumKind.SS
    elif (spin_i, spin_j) == (2, 2):
        if comp_i != comp_j:
            kind = {
                0: SpectrumKind.GG,
                1: SpectrumKind.GC,
                2: SpectrumKind.CG,
                3: SpectrumKind.CC,
            }[mode]
        else:
            kind = {0: SpectrumKind.GG, 1: SpectrumKind.CC, 2: SpectrumKind.GC}[mode]
    elif (spin_i, spin_j) == (0, 2):
        kind = {0: SpectrumKind.SG, 1: SpectrumKind.SC}[mode]
    elif (spin_i, spin_j) == (2, 0):
        kind = {0: SpectrumKind.GS, 1: SpectrumKind.CS}[mode]
    else:
        raise ValueError(f"unsupported spin pair ({spin_i}, {spin_j})")
    return SpectrumKey(comp_i, comp_j, kind, spins=spins)


class TestCreateCompression:
    """Tests for the create_computation_basis factory function."""

    def test_harmonic_method(self, uniform_sky_setup):
        """Test create_computation_basis with harmonic method."""
        from cosmocore.basis import create_computation_basis

        setup = uniform_sky_setup
        cm = create_computation_basis(
            method="harmonic",
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=setup["lmax"],
        )

        cm.setup()

        assert cm.method == "harmonic"
        assert cm.n_modes > 0
        assert cm.dim == cm.n_modes  # Harmonic keeps all modes

    def test_pixel_method(self, uniform_sky_setup):
        """Test create_computation_basis with pixel method."""
        from cosmocore.basis import create_computation_basis

        setup = uniform_sky_setup
        cm = create_computation_basis(
            method="pixel",
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=setup["lmax"],
            epsilon=1e-6,
        )

        cm.setup()

        assert cm.method == "pixel"
        assert cm.dim > 0
        assert cm.dim <= setup["n_pix"]

    def test_unknown_method_raises(self, uniform_sky_setup):
        """Test that unknown method raises error."""
        from cosmocore.basis import create_computation_basis

        setup = uniform_sky_setup

        with pytest.raises(ValueError, match="Unknown computation basis method"):
            create_computation_basis(
                method="invalid",
                N=setup["N"],
                theta=setup["theta"],
                phi=setup["phi"],
                lmax_signal=setup["lmax"],
            )

    def test_facade_delegates_correctly(self, uniform_sky_setup):
        """Test that create_computation_basis returns correct implementation."""
        from cosmocore.basis import HarmonicBasis, create_computation_basis

        setup = uniform_sky_setup
        # ℓ-indexed C_ell of length lmax+1; ℓ<2 entries zero (spin-0 floor).
        C_ell = np.zeros(setup["lmax"] + 1, dtype=np.float64)
        C_ell[2:] = 1e-6

        # Create both directly and via factory
        cm = create_computation_basis(
            method="harmonic",
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=setup["lmax"],
        )
        cm.setup()

        hc = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=setup["lmax"],
        )
        hc.setup()

        # Results should match
        assert_allclose(
            cm.get_covariance(C_ell),
            hc.get_covariance(C_ell),
            rtol=1e-10,
        )

    def test_fisher_matrix_via_facade(self, uniform_sky_setup):
        """Test Fisher matrix computation via factory."""
        from cosmocore.basis import create_computation_basis

        setup = uniform_sky_setup
        # ℓ-indexed C_ell of length lmax+1; ℓ<2 entries zero (spin-0 floor).
        C_ell = np.zeros(setup["lmax"] + 1, dtype=np.float64)
        C_ell[2:] = 1e-6

        cm = create_computation_basis(
            method="harmonic",
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=setup["lmax"],
        )
        cm.setup()

        # Compute Fisher matrix
        fisher = cm.compute_fisher_matrix(C_ell)
        F_33 = fisher[3 - 2, 3 - 2]  # ell=3 is index 1
        assert F_33 > 0  # Diagonal should be positive

    def test_pixel_with_basis(self, uniform_sky_setup):
        """Test pixel with different basis."""
        from cosmocore.basis import create_computation_basis

        setup = uniform_sky_setup
        # ℓ-indexed C_ell of length lmax+1; ℓ<2 entries zero (spin-0 floor).
        C_ell = np.zeros(setup["lmax"] + 1, dtype=np.float64)
        C_ell[2:] = 1e-6

        cm = create_computation_basis(
            method="pixel",
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=setup["lmax"],
            basis="snr",
            C_ell=C_ell,
            epsilon=1e-4,
        )

        cm.setup()

        assert cm.dim > 0


class TestComputeFisherMatrix:
    """Tests for the optimized compute_fisher_matrix method."""

    def test_fisher_matrix_shape(self, uniform_sky_setup):
        """Test that compute_fisher_matrix returns correct shape."""
        from cosmocore.basis import HarmonicBasis

        setup = uniform_sky_setup
        lmax = setup["lmax"]
        n_ell = lmax - 1
        # ℓ-indexed C_ell of length lmax+1; ℓ<2 entries zero (spin-0 floor).
        C_ell = np.zeros(lmax + 1, dtype=np.float64)
        C_ell[2:] = 1e-6

        hc = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=lmax,
        )
        hc.setup()

        fisher = hc.compute_fisher_matrix(C_ell, ell_min=2, ell_max=lmax)

        assert fisher.shape == (n_ell, n_ell)

    def test_fisher_matrix_symmetric(self, uniform_sky_setup):
        """Test that compute_fisher_matrix produces symmetric result."""
        from cosmocore.basis import HarmonicBasis

        setup = uniform_sky_setup
        lmax = setup["lmax"]
        # ℓ-indexed C_ell of length lmax+1; ℓ<2 entries zero (spin-0 floor).
        C_ell = np.zeros(lmax + 1, dtype=np.float64)
        C_ell[2:] = 1e-6

        hc = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=lmax,
        )
        hc.setup()

        fisher = hc.compute_fisher_matrix(C_ell)

        assert_allclose(fisher, fisher.T, rtol=1e-10)

    def test_fisher_matrix_positive_semidefinite(self, uniform_sky_setup):
        """Test that compute_fisher_matrix produces positive semi-definite matrix."""
        from cosmocore.basis import HarmonicBasis

        setup = uniform_sky_setup
        lmax = setup["lmax"]
        # ℓ-indexed C_ell of length lmax+1; ℓ<2 entries zero (spin-0 floor).
        C_ell = np.zeros(lmax + 1, dtype=np.float64)
        C_ell[2:] = 1e-6

        hc = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=lmax,
        )
        hc.setup()

        fisher = hc.compute_fisher_matrix(C_ell)

        # All eigenvalues should be non-negative
        eigenvalues = np.linalg.eigvalsh(fisher)
        assert np.all(eigenvalues >= -1e-10)

    def test_fisher_matrix_via_manager(self, uniform_sky_setup):
        """Test compute_fisher_matrix via create_computation_basis factory."""
        from cosmocore.basis import create_computation_basis

        setup = uniform_sky_setup
        lmax = setup["lmax"]
        n_ell = lmax - 1
        # ℓ-indexed C_ell of length lmax+1; ℓ<2 entries zero (spin-0 floor).
        C_ell = np.zeros(lmax + 1, dtype=np.float64)
        C_ell[2:] = 1e-6

        cm = create_computation_basis(
            method="harmonic",
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=lmax,
        )
        cm.setup()

        fisher = cm.compute_fisher_matrix(C_ell)

        assert fisher.shape == (n_ell, n_ell)
        assert_allclose(fisher, fisher.T, rtol=1e-10)

    def test_pixel_fisher_matrix(self, uniform_sky_setup):
        """Test compute_fisher_matrix with PixelBasis."""
        from cosmocore.basis import PixelBasis

        setup = uniform_sky_setup
        lmax = setup["lmax"]
        n_ell = lmax - 1
        # ℓ-indexed C_ell of length lmax+1; ℓ<2 entries zero (spin-0 floor).
        C_ell = np.zeros(lmax + 1, dtype=np.float64)
        C_ell[2:] = 1e-6

        ppc = PixelBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=lmax,
            epsilon=1e-6,
        )
        ppc.setup()

        # Compute via optimized method
        fisher = ppc.compute_fisher_matrix(C_ell, ell_min=2, ell_max=lmax)

        # Verify shape and symmetry
        assert fisher.shape == (n_ell, n_ell)
        assert_allclose(fisher, fisher.T, rtol=1e-10)

        # Verify positive semi-definite
        eigenvalues = np.linalg.eigvalsh(fisher)
        assert np.all(eigenvalues >= -1e-10)


class TestPerFieldThreshold:
    """Tests for per-field and E/B split threshold parsing and behavior."""

    golden_ratio = (1 + np.sqrt(5)) / 2

    @staticmethod
    def _spiral(n, offset=0):
        gr = (1 + np.sqrt(5)) / 2
        idx = np.arange(n)
        th = np.arccos(1 - 2 * (idx + 0.5) / n)
        ph = (2 * np.pi * (idx + offset) / gr) % (2 * np.pi)
        return th, ph

    def test_single_epsilon_backward_compatible(self):
        """Single float epsilon produces same results as per-field broadcast."""
        from cosmocore.basis import PixelBasis

        np.random.seed(42)
        n_pix = 30
        theta, phi = self._spiral(n_pix)
        lmax = 5

        N = np.eye(n_pix) * 0.01
        np.eye(n_pix) * 100.0

        # Scalar epsilon
        ppc1 = PixelBasis(
            N,
            theta,
            phi,
            lmax,
            spins=[0],
            epsilon=1e-6,
        )
        ppc1.setup()

        # List epsilon (1-element)
        ppc2 = PixelBasis(
            N,
            theta,
            phi,
            lmax,
            spins=[0],
            epsilon=[1e-6],
        )
        ppc2.setup()

        assert ppc1.dim == ppc2.dim

        # Fisher must match
        C_ell = np.ones(lmax + 1) * 1e-4
        f1 = ppc1.compute_fisher_matrix(C_ell, ell_min=2, ell_max=lmax)
        f2 = ppc2.compute_fisher_matrix(C_ell, ell_min=2, ell_max=lmax)
        assert_allclose(f1, f2, rtol=1e-12)

    def test_per_field_list_epsilon(self):
        """Two spin-0 fields with different epsilons → different dim per field."""
        from cosmocore.basis import PixelBasis

        np.random.seed(42)
        n_pix_a = 25
        n_pix_b = 25
        theta_a, phi_a = self._spiral(n_pix_a, offset=0)
        theta_b, phi_b = self._spiral(n_pix_b, offset=500)
        lmax = 5

        total_pix = n_pix_a + n_pix_b
        N = np.eye(total_pix) * 0.01
        np.eye(total_pix) * 100.0

        # Tight threshold for field 0, loose for field 1
        ppc_split = PixelBasis(
            N,
            (theta_a, theta_b),
            (phi_a, phi_b),
            lmax,
            spins=[0, 0],
            epsilon=[1e-8, 1e-2],
        )
        ppc_split.setup()

        # Uniform threshold
        ppc_uniform = PixelBasis(
            N,
            (theta_a, theta_b),
            (phi_a, phi_b),
            lmax,
            spins=[0, 0],
            epsilon=1e-8,
        )
        ppc_uniform.setup()

        # Split should keep fewer modes (field 1 is aggressive)
        assert ppc_split.dim < ppc_uniform.dim

    def test_spin2_tuple_epsilon_eb_split(self):
        """Spin-2 field with tuple epsilon uses E/B split thresholding."""
        from cosmocore.basis import PixelBasis

        np.random.seed(42)
        n_pix = 20
        theta, phi = self._spiral(n_pix)
        lmax = 5

        N = np.eye(2 * n_pix) * 0.01
        np.eye(2 * n_pix) * 100.0

        # Tight E threshold, loose B threshold → keep more B modes
        ppc_split = PixelBasis(
            N,
            theta,
            phi,
            lmax,
            spins=[2],
            epsilon=[(1e-2, 1e-8)],
        )
        ppc_split.setup()

        # Uniform scalar → same threshold for both E and B
        ppc_uniform = PixelBasis(
            N,
            theta,
            phi,
            lmax,
            spins=[2],
            epsilon=1e-2,
        )
        ppc_uniform.setup()

        # Split should keep >= uniform modes since B gets looser threshold
        assert ppc_split.dim >= ppc_uniform.dim

        # Fisher should still be PSD and symmetric
        C_ell_dict = {
            (0, 0, 0): np.ones(lmax + 1) * 5e-4,
            (0, 0, 1): np.ones(lmax + 1) * 1e-4,
        }
        spins = (2,)
        spectra_list = [_to_key(t, spins=spins) for t in [(0, 0, 0), (0, 0, 1)]]
        fisher = ppc_split.compute_fisher_matrix(
            C_ell_dict, spectra_list, ell_min=2, ell_max=lmax
        )
        assert_allclose(fisher, fisher.T, atol=1e-12)
        eigvals = np.linalg.eigvalsh(fisher)
        assert np.all(eigvals > -1e-10)

    def test_spin2_eb_split_vs_single(self):
        """E/B split with tight thresholds converges to single-threshold result."""
        from cosmocore.basis import PixelBasis

        np.random.seed(42)
        n_pix = 25
        theta, phi = self._spiral(n_pix)
        lmax = 5

        N = np.eye(2 * n_pix) * 0.01
        np.eye(2 * n_pix) * 100.0

        eps = 1e-10  # Very tight → keeps essentially all modes

        # Scalar
        ppc_scalar = PixelBasis(
            N,
            theta,
            phi,
            lmax,
            spins=[2],
            epsilon=eps,
        )
        ppc_scalar.setup()

        # Tuple with same value for both E and B
        ppc_tuple = PixelBasis(
            N,
            theta,
            phi,
            lmax,
            spins=[2],
            epsilon=[(eps, eps)],
        )
        ppc_tuple.setup()

        # Fisher matrices should match closely
        C_ell_dict = {
            (0, 0, 0): np.ones(lmax + 1) * 5e-4,
            (0, 0, 1): np.ones(lmax + 1) * 1e-4,
        }
        spins = (2,)
        spectra_list = [_to_key(t, spins=spins) for t in [(0, 0, 0), (0, 0, 1)]]

        f_scalar = ppc_scalar.compute_fisher_matrix(
            C_ell_dict, spectra_list, ell_min=2, ell_max=lmax
        )
        f_tuple = ppc_tuple.compute_fisher_matrix(
            C_ell_dict, spectra_list, ell_min=2, ell_max=lmax
        )

        # At tight thresholds, the results should be very close
        # (not exact because SVD orthogonalization in E/B split differs)
        diag = np.abs(np.diag(f_scalar))
        scale = np.sqrt(np.outer(diag, diag)) + 1e-30
        norm_diff = np.abs(f_tuple - f_scalar) / scale
        assert norm_diff.max() < 0.01, (
            f"E/B split with equal thresholds should match scalar, "
            f"got max norm diff {norm_diff.max():.2e}"
        )

    def test_tqu_per_field_mixed(self):
        """T(spin-0) + QU(spin-2) with per-field list including tuple."""
        from cosmocore.basis import PixelBasis

        np.random.seed(42)
        n_pix_t = 15
        n_pix_p = 12
        theta_t, phi_t = self._spiral(n_pix_t, offset=0)
        theta_p, phi_p = self._spiral(n_pix_p, offset=500)
        lmax = 4

        total_pix = n_pix_t + 2 * n_pix_p
        N = np.eye(total_pix) * 0.01
        np.eye(total_pix) * 100.0

        # T: scalar epsilon, QU: E/B split tuple
        ppc = PixelBasis(
            N,
            (theta_t, theta_p),
            (phi_t, phi_p),
            lmax,
            spins=[0, 2],
            epsilon=[1e-6, (1e-4, 1e-8)],
        )
        ppc.setup()

        assert ppc.dim > 0
        assert ppc.dim <= total_pix

        # Fisher should work and be valid
        C_ell_dict = {
            (0, 0, 0): np.ones(lmax + 1) * 1e-3,
            (1, 1, 0): np.ones(lmax + 1) * 5e-4,
            (1, 1, 1): np.ones(lmax + 1) * 1e-4,
            (0, 1, 0): np.ones(lmax + 1) * 2e-4,
        }
        spins = (0, 2)
        spectra_list = [
            _to_key(t, spins=spins) for t in [(0, 0, 0), (1, 1, 0), (1, 1, 1), (0, 1, 0)]
        ]
        fisher = ppc.compute_fisher_matrix(
            C_ell_dict, spectra_list, ell_min=2, ell_max=lmax
        )
        assert_allclose(fisher, fisher.T, atol=1e-12)

    def test_invalid_tuple_for_spin0_raises(self):
        """Tuple epsilon for spin-0 field should raise ValueError."""
        from cosmocore.basis import PixelBasis

        np.random.seed(42)
        n_pix = 20
        theta, phi = self._spiral(n_pix)
        lmax = 5

        N = np.eye(n_pix) * 0.01
        np.eye(n_pix) * 100.0

        with pytest.raises(ValueError, match="tuple.*E/B split.*spin.*not 2"):
            PixelBasis(
                N,
                theta,
                phi,
                lmax,
                spins=[0],
                epsilon=[(1e-4, 1e-8)],
            )

    def test_invalid_list_length_raises(self):
        """Epsilon list with wrong length should raise ValueError."""
        from cosmocore.basis import PixelBasis

        np.random.seed(42)
        n_pix = 20
        theta, phi = self._spiral(n_pix)
        lmax = 5

        N = np.eye(n_pix) * 0.01
        np.eye(n_pix) * 100.0

        with pytest.raises(ValueError, match="list length.*must match"):
            PixelBasis(
                N,
                theta,
                phi,
                lmax,
                spins=[0],
                epsilon=[1e-4, 1e-8],  # 2 values for 1 field
            )


# =============================================================================
# PICSLike methods tests (prepare_for_basis, quadratic_form, logdet)
# =============================================================================


class TestPixelProjectedPICSLikeMethods:
    """Tests for PICSLike-compatible methods on PixelBasis."""

    @staticmethod
    def _spiral(n, offset=0):
        gr = (1 + np.sqrt(5)) / 2
        idx = np.arange(n)
        th = np.arccos(1 - 2 * (idx + 0.5) / n)
        ph = (2 * np.pi * (idx + offset) / gr) % (2 * np.pi)
        return th, ph

    def _make_spin2_setup(self, n_pix=20, lmax=5, epsilon=1e-6):
        """Create a spin-2 PixelProjected setup and return (ppc, C_ell_dict)."""
        from cosmocore.basis import PixelBasis

        theta, phi = self._spiral(n_pix)
        N = np.eye(2 * n_pix) * 0.01
        np.eye(2 * n_pix) * 100.0

        ppc = PixelBasis(
            N,
            theta,
            phi,
            lmax,
            spins=[2],
            epsilon=epsilon,
        )
        ppc.setup()

        C_ell_dict = {
            (0, 0, 0): np.ones(lmax + 1) * 5e-4,
            (0, 0, 1): np.ones(lmax + 1) * 1e-4,
        }
        return ppc, C_ell_dict, N

    def test_quadratic_form(self):
        """Quadratic form matches brute-force d^T C^{-1} d."""
        from cosmocore.basics import matrix_inverse_symm

        np.random.seed(42)
        ppc, C_ell_dict, N = self._make_spin2_setup()

        # Generate random data
        data = np.random.randn(ppc.n_pix)

        # Compressed quadratic form
        qf_compressed = ppc.quadratic_form(data, C_ell_dict)

        # Brute-force in full pixel space
        lambda_matrix = ppc._build_lambda_matrix_3tuple(C_ell_dict)
        S = ppc._V.T @ lambda_matrix @ ppc._V
        C = N + S
        C_inv = matrix_inverse_symm(C.copy())
        qf_brute = float(data @ C_inv @ data)

        assert_allclose(
            qf_compressed,
            qf_brute,
            rtol=1e-3,
            err_msg="Compressed quadratic form should match brute-force",
        )

    def test_prepare_and_quadratic_form_from_prepared(self):
        """prepare_for_basis → quadratic_form_from_prepared matches direct compute."""
        np.random.seed(42)
        ppc, C_ell_dict, _ = self._make_spin2_setup()

        data = np.random.randn(ppc.n_pix)

        # Direct computation
        qf_direct = ppc.quadratic_form(data, C_ell_dict)

        # Two-step: prepare then compute
        C_c_inv, logdet = ppc.prepare_for_basis(C_ell_dict)
        assert C_c_inv.shape == (ppc.dim, ppc.dim)
        assert isinstance(logdet, float)

        qf_prepared = ppc.quadratic_form_from_prepared(data, C_c_inv)

        assert_allclose(
            qf_prepared,
            qf_direct,
            rtol=1e-12,
            err_msg="Prepared quadratic form should match direct",
        )

    def test_get_logdet(self):
        """get_logdet matches slogdet of compressed covariance."""
        np.random.seed(42)
        ppc, C_ell_dict, _ = self._make_spin2_setup()

        logdet = ppc.get_logdet(C_ell_dict)

        # Directly compute
        C_c = ppc.get_covariance(C_ell_dict)
        _, expected_logdet = np.linalg.slogdet(C_c)

        assert_allclose(
            logdet,
            expected_logdet,
            rtol=1e-10,
            err_msg="get_logdet should match slogdet(C_compressed)",
        )

    def test_eb_split_preserves_b_mode_fisher(self):
        """
        Aggressive single threshold kills B modes;
        E/B split with loose B threshold retains them.
        """
        from cosmocore.basis import PixelBasis

        np.random.seed(42)
        n_pix = 25
        theta, phi = self._spiral(n_pix)
        lmax = 5

        N = np.eye(2 * n_pix) * 0.01
        np.eye(2 * n_pix) * 100.0

        # Aggressive single threshold (E signal >> B → kills B modes)
        ppc_aggressive = PixelBasis(
            N,
            theta,
            phi,
            lmax,
            spins=[2],
            epsilon=1e-1,
        )
        ppc_aggressive.setup()

        # E/B split: same aggressive E but loose B
        ppc_split = PixelBasis(
            N,
            theta,
            phi,
            lmax,
            spins=[2],
            epsilon=[(1e-1, 1e-10)],
        )
        ppc_split.setup()

        # Split should keep more modes (B modes preserved)
        assert ppc_split.dim >= ppc_aggressive.dim

        C_ell_dict = {
            (0, 0, 0): np.ones(lmax + 1) * 5e-4,  # EE
            (0, 0, 1): np.ones(lmax + 1) * 1e-4,  # BB
        }
        spins = (2,)
        spectra_list = [_to_key(t, spins=spins) for t in [(0, 0, 0), (0, 0, 1)]]  # EE, BB

        fisher_agg = ppc_aggressive.compute_fisher_matrix(
            C_ell_dict, spectra_list, ell_min=2, ell_max=lmax
        )
        fisher_split = ppc_split.compute_fisher_matrix(
            C_ell_dict, spectra_list, ell_min=2, ell_max=lmax
        )

        # BB block of Fisher: last n_ell rows/cols
        n_ell = lmax - 1
        bb_diag_agg = np.diag(fisher_agg[n_ell:, n_ell:])
        bb_diag_split = np.diag(fisher_split[n_ell:, n_ell:])

        # Split should have larger (or equal) BB diagonal → more BB info
        assert np.sum(bb_diag_split) >= np.sum(bb_diag_agg) * 0.99, (
            f"E/B split BB Fisher {np.sum(bb_diag_split):.2e} should be >= "
            f"aggressive {np.sum(bb_diag_agg):.2e}"
        )
