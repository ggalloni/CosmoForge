"""
Tests for HarmonicBasis class.

Tests for the Tegmark-like harmonic compression approach.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cosmocore.spectrum_key import SpectrumKey, SpectrumKind


class TestHarmonicBasisInitialization:
    """Tests for HarmonicBasis initialization."""

    def test_basic_initialization(self, simple_compression_setup):
        """Test that HarmonicBasis initializes correctly."""
        from cosmocore.basis import HarmonicBasis

        setup = simple_compression_setup
        hc = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=setup["lmax"],
        )

        assert hc.n_pix == setup["n_pix"]
        assert hc.lmax_signal == setup["lmax"]

    def test_mode_count(self, simple_compression_setup):
        """Test that n_modes is computed correctly."""
        from cosmocore.basis import HarmonicBasis

        setup = simple_compression_setup
        hc = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=setup["lmax"],
        )

        # n_modes = (lmax+1)^2 - 4 for ell >= 2
        expected_n_modes = (setup["lmax"] + 1) ** 2 - 4
        assert hc.n_modes == expected_n_modes


class TestHarmonicBasisSetup:
    """Tests for HarmonicBasis setup operations."""

    def test_setup_creates_required_matrices(self, simple_compression_setup):
        """Test that setup() creates all required matrices."""
        from cosmocore.basis import HarmonicBasis

        setup = simple_compression_setup
        hc = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=setup["lmax"],
        )

        hc.setup()

        # Check V matrix exists and has correct shape
        assert hc._V is not None
        assert hc._V.shape == (hc.n_modes, setup["n_pix"])

        # Check SMW components exist (V_N_VT is lazy-computed)
        assert hc._V_N_inv is not None
        assert hc._V_Ninv_VT is not None
        assert hc._log_det_N is not None
        assert hc._V_N_VT is not None

        # Check ell-to-mode mapping exists
        assert hc._ell_to_modes is not None
        assert 2 in hc._ell_to_modes  # ell=2 should be present

    def test_ell_to_mode_mapping_correct(self, simple_compression_setup):
        """Test that ell-to-mode mapping has correct counts."""
        from cosmocore.basis import HarmonicBasis

        setup = simple_compression_setup
        hc = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=setup["lmax"],
        )

        hc.setup()

        # Each ell should have (2*ell + 1) modes
        for ell in range(2, setup["lmax"] + 1):
            assert len(hc._ell_to_modes[ell]) == 2 * ell + 1

    def test_smw_matrices_symmetric(self, simple_compression_setup):
        """Test that SMW matrices are symmetric."""
        from cosmocore.basis import HarmonicBasis

        setup = simple_compression_setup
        hc = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=setup["lmax"],
        )

        hc.setup()

        assert_allclose(hc._V_Ninv_VT, hc._V_Ninv_VT.T, rtol=1e-10)
        assert_allclose(hc._V_N_VT, hc._V_N_VT.T, rtol=1e-10)


class TestHarmonicBasisOperations:
    """Tests for HarmonicBasis compressed-space operations."""

    def test_compress_data(self, uniform_sky_setup):
        """Test data compression to harmonic space."""
        from cosmocore.basis import HarmonicBasis

        setup = uniform_sky_setup
        hc = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=setup["lmax"],
        )

        hc.setup()

        data = np.random.randn(setup["n_pix"])
        d_compressed = hc.compress_data(data)

        assert d_compressed.shape == (hc.n_modes,)

    def test_compressed_covariance_shape(self, uniform_sky_setup):
        """Test compressed covariance has correct shape."""
        from cosmocore.basis import HarmonicBasis

        setup = uniform_sky_setup
        hc = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=setup["lmax"],
        )

        hc.setup()

        n_ell = setup["lmax"] + 1
        C_ell = np.ones(n_ell) * 1e-6

        C_bar = hc.get_compressed_covariance(C_ell)

        assert C_bar.shape == (hc.n_modes, hc.n_modes)

    def test_compressed_covariance_symmetric(self, uniform_sky_setup):
        """Test compressed covariance is symmetric."""
        from cosmocore.basis import HarmonicBasis

        setup = uniform_sky_setup
        hc = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=setup["lmax"],
        )

        hc.setup()

        n_ell = setup["lmax"] + 1
        C_ell = np.ones(n_ell) * 1e-6

        C_bar = hc.get_compressed_covariance(C_ell)

        assert_allclose(C_bar, C_bar.T, rtol=1e-10)

    def test_compressed_inverse_is_inverse(self, uniform_sky_setup):
        """Test that compressed inverse is actually the inverse."""
        from cosmocore.basis import HarmonicBasis

        setup = uniform_sky_setup
        hc = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=setup["lmax"],
        )

        hc.setup()

        n_ell = setup["lmax"] + 1
        C_ell = np.ones(n_ell) * 1e-6

        C_bar = hc.get_compressed_covariance(C_ell)
        C_bar_inv = hc.get_compressed_inverse(C_ell)

        product = C_bar @ C_bar_inv
        identity = np.eye(hc.n_modes)

        # Slightly relaxed tolerance due to matrix inversion precision limits
        assert_allclose(product, identity, rtol=1e-8, atol=1e-9)

    def test_compressed_logdet_positive_definite(self, uniform_sky_setup):
        """Test log determinant computation for positive definite matrix."""
        from cosmocore.basis import HarmonicBasis

        setup = uniform_sky_setup
        hc = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=setup["lmax"],
        )

        hc.setup()

        n_ell = setup["lmax"] + 1
        C_ell = np.ones(n_ell) * 1e-6

        logdet = hc.get_compressed_logdet(C_ell)

        assert np.isfinite(logdet)

        # Cross-check with numpy
        C_bar = hc.get_compressed_covariance(C_ell)
        _, logdet_np = np.linalg.slogdet(C_bar)

        assert_allclose(logdet, logdet_np, rtol=1e-8)


class TestHarmonicBasisFisher:
    """Tests for HarmonicBasis Fisher matrix computation."""

    def test_fisher_matrix_positive_diagonal(self, uniform_sky_setup):
        """Test Fisher diagonal elements are positive using compute_fisher_matrix."""
        from cosmocore.basis import HarmonicBasis

        setup = uniform_sky_setup
        hc = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=setup["lmax"],
        )

        hc.setup()

        n_ell = setup["lmax"] + 1
        C_ell = np.ones(n_ell) * 1e-6

        fisher = hc.compute_fisher_matrix(C_ell)
        F_55 = fisher[5 - 2, 5 - 2]  # ell=5 corresponds to index 3

        assert F_55 >= 0

    def test_fisher_matrix_symmetric(self, uniform_sky_setup):
        """Test Fisher matrix is symmetric using compute_fisher_matrix."""
        from cosmocore.basis import HarmonicBasis

        setup = uniform_sky_setup
        hc = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=setup["lmax"],
        )

        hc.setup()

        n_ell = setup["lmax"] + 1
        C_ell = np.ones(n_ell) * 1e-6

        fisher = hc.compute_fisher_matrix(C_ell)

        assert_allclose(fisher, fisher.T, rtol=1e-10)


class TestHarmonicBasisPixelSpace:
    """Tests for HarmonicBasis full pixel-space SMW operations."""

    def test_smw_inverse_shape(self, simple_compression_setup):
        """Test SMW inverse has correct shape."""
        from cosmocore.basis import HarmonicBasis

        setup = simple_compression_setup
        hc = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=setup["lmax"],
        )

        hc.setup()

        n_ell = setup["lmax"] + 1
        C_ell = np.ones(n_ell) * 1e-6

        C_inv = hc.get_inverse(C_ell)

        assert C_inv.shape == (setup["n_pix"], setup["n_pix"])

    def test_smw_inverse_symmetric(self, simple_compression_setup):
        """Test SMW inverse is symmetric."""
        from cosmocore.basis import HarmonicBasis

        setup = simple_compression_setup
        hc = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=setup["lmax"],
        )

        hc.setup()

        n_ell = setup["lmax"] + 1
        C_ell = np.ones(n_ell) * 1e-6

        C_inv = hc.get_inverse(C_ell)

        assert_allclose(C_inv, C_inv.T, rtol=1e-8)

    def test_smw_logdet_finite(self, simple_compression_setup):
        """Test SMW log determinant is finite."""
        from cosmocore.basis import HarmonicBasis

        setup = simple_compression_setup
        hc = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=setup["lmax"],
        )

        hc.setup()

        n_ell = setup["lmax"] + 1
        C_ell = np.ones(n_ell) * 1e-6

        logdet = hc.get_logdet(C_ell)

        assert np.isfinite(logdet)


class TestHarmonicBasisValidation:
    """Validation tests for HarmonicBasis against direct computation."""

    @pytest.fixture
    def validation_setup(self):
        """Create a well-conditioned test setup for validation."""
        np.random.seed(42)

        n_pix = 50
        lmax = 6

        noise_variance = np.random.uniform(0.01, 0.1, n_pix)
        N = np.diag(noise_variance)
        N_inv = np.diag(1.0 / noise_variance)

        golden_ratio = (1 + np.sqrt(5)) / 2
        indices = np.arange(n_pix)
        theta = np.arccos(1 - 2 * (indices + 0.5) / n_pix)
        phi = (2 * np.pi * indices / golden_ratio) % (2 * np.pi)

        C_ell = np.zeros(lmax + 1, dtype=np.float64)
        C_ell[2:] = 1e-3 / (np.arange(2, lmax + 1) ** 2)

        return {
            "N": N,
            "N_inv": N_inv,
            "theta": theta,
            "phi": phi,
            "lmax": lmax,
            "n_pix": n_pix,
            "C_ell": C_ell,
        }

    def _build_signal_covariance_direct(self, V, C_ell, lmax, ell_to_modes):
        """Build signal covariance S = V^T Λ V directly."""
        n_modes = V.shape[0]
        Lambda_diag = np.zeros(n_modes)
        for ell in range(2, lmax + 1):
            c_ell_value = C_ell[ell] if ell < len(C_ell) else 0.0
            for idx in ell_to_modes[ell]:
                Lambda_diag[idx] = c_ell_value
        Lambda = np.diag(Lambda_diag)
        return V.T @ Lambda @ V

    def test_smw_inverse_matches_direct(self, validation_setup):
        """Test that SMW inverse matches direct matrix inverse."""
        from cosmocore.basis import HarmonicBasis

        setup = validation_setup
        hc = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=setup["lmax"],
        )
        hc.setup()

        V = hc._V.copy()
        C_ell = setup["C_ell"]

        S = self._build_signal_covariance_direct(
            V, C_ell, setup["lmax"], hc._ell_to_modes
        )
        C_direct = setup["N"] + S
        C_inv_direct = np.linalg.inv(C_direct)

        C_inv_smw = hc.get_inverse(C_ell)

        assert_allclose(C_inv_smw, C_inv_direct, rtol=1e-8, atol=1e-12)

    def test_smw_logdet_matches_direct(self, validation_setup):
        """Test that SMW log-determinant matches direct computation."""
        from cosmocore.basis import HarmonicBasis

        setup = validation_setup
        hc = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=setup["lmax"],
        )
        hc.setup()

        V = hc._V.copy()
        C_ell = setup["C_ell"]

        S = self._build_signal_covariance_direct(
            V, C_ell, setup["lmax"], hc._ell_to_modes
        )
        C_direct = setup["N"] + S
        _, logdet_direct = np.linalg.slogdet(C_direct)

        logdet_smw = hc.get_logdet(C_ell)

        assert_allclose(logdet_smw, logdet_direct, rtol=1e-8)

    def test_smw_inverse_is_actual_inverse(self, validation_setup):
        """Test that SMW inverse times full covariance gives identity."""
        from cosmocore.basis import HarmonicBasis

        setup = validation_setup
        hc = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=setup["lmax"],
        )
        hc.setup()

        V = hc._V.copy()
        C_ell = setup["C_ell"]

        S = self._build_signal_covariance_direct(
            V, C_ell, setup["lmax"], hc._ell_to_modes
        )
        C_full = setup["N"] + S

        C_inv_smw = hc.get_inverse(C_ell)

        product = C_full @ C_inv_smw
        identity = np.eye(setup["n_pix"])

        assert_allclose(product, identity, rtol=1e-8, atol=1e-10)

    def test_smw_quadratic_form_matches_direct(self, validation_setup):
        """Test that SMW quadratic form matches direct computation."""
        from cosmocore.basis import HarmonicBasis

        setup = validation_setup
        hc = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=setup["lmax"],
        )
        hc.setup()

        V = hc._V.copy()
        C_ell = setup["C_ell"]

        # Generate random data vector
        np.random.seed(42)
        data = np.random.normal(0, 1, setup["n_pix"])

        # Direct computation: d^T @ C^{-1} @ d
        S = self._build_signal_covariance_direct(
            V, C_ell, setup["lmax"], hc._ell_to_modes
        )
        C_direct = setup["N"] + S
        C_inv_direct = np.linalg.inv(C_direct)
        quad_form_direct = float(data.T @ C_inv_direct @ data)

        # SMW computation
        quad_form_smw = hc.compute_quadratic_form(data, C_ell)

        assert_allclose(quad_form_smw, quad_form_direct, rtol=1e-8)


class TestHarmonicBasisBeam:
    """Tests for beam window function support in HarmonicBasis."""

    def test_beam_initialization(self, simple_compression_setup):
        """Test that beam can be provided during initialization."""
        from cosmocore.basis import HarmonicBasis

        setup = simple_compression_setup
        lmax = setup["lmax"]

        # ℓ-indexed beam of length lmax+1. ℓ=0,1 hold the gaussian values
        # (≈1.0 for sigma≪1); they are computed but never consumed by the
        # inference path since the spin-0 floor is ℓ≥2.
        ell = np.arange(lmax + 1)
        fwhm_rad = np.deg2rad(30.0 / 60.0)
        sigma = fwhm_rad / np.sqrt(8 * np.log(2))
        beam = np.exp(-0.5 * ell * (ell + 1) * sigma**2)

        hc = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=lmax,
            beam=beam,
        )

        assert hc._beam is not None
        assert len(hc._beam) == lmax + 1
        assert_allclose(hc._beam, beam)

    def test_beam_validation_wrong_length(self, simple_compression_setup):
        """Test that incorrect beam length raises an error."""
        from cosmocore.basis import HarmonicBasis

        setup = simple_compression_setup
        lmax = setup["lmax"]

        wrong_beam = np.ones(lmax + 5)

        with pytest.raises(ValueError, match="Beam must have length"):
            HarmonicBasis(
                N=setup["N"],
                theta=setup["theta"],
                phi=setup["phi"],
                lmax_signal=lmax,
                beam=wrong_beam,
            )

    def test_unit_beam_no_effect(self, simple_compression_setup):
        """Test that unit beam produces same result as no beam."""
        from cosmocore.basis import HarmonicBasis

        setup = simple_compression_setup
        lmax = setup["lmax"]

        beam = np.ones(lmax + 1)
        C_ell = np.ones(lmax + 1) * 1e-5

        hc_no_beam = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=lmax,
        )
        hc_no_beam.setup()

        hc_unit_beam = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=lmax,
            beam=beam,
        )
        hc_unit_beam.setup()

        assert_allclose(hc_no_beam._V, hc_unit_beam._V, rtol=1e-12)

        C_bar_no_beam = hc_no_beam.get_compressed_covariance(C_ell)
        C_bar_unit_beam = hc_unit_beam.get_compressed_covariance(C_ell)
        assert_allclose(C_bar_no_beam, C_bar_unit_beam, rtol=1e-10)


# =============================================================================
# Coverage-focused tests for untested HarmonicBasis operations
# =============================================================================


class TestHarmonicDictOperations:
    """Cover dict-path operations: weighted data, quadratic form, SMW, logdet."""

    def test_single_field_weighted_data_and_qf(self, uniform_sky_setup):
        """Cover get_weighted_compressed_data and compute_quadratic_form (array)."""
        from cosmocore.basis import HarmonicBasis

        setup = uniform_sky_setup
        hc = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=setup["lmax"],
        )
        hc.setup()

        C_ell = np.ones(setup["lmax"] - 1) * 1e-6
        np.random.seed(42)
        data = np.random.randn(setup["n_pix"])

        w = hc.get_weighted_compressed_data(data, C_ell)
        assert w.shape == (hc.n_modes,)

        qf = hc.compute_quadratic_form(data, C_ell)
        assert qf > 0

        ss_key = SpectrumKey(0, 0, SpectrumKind.SS, spins=(0,))
        dC = hc.get_derivative_matrix(5, ss_key)
        assert dC.shape == (hc.n_modes, hc.n_modes)

    def test_multi_field_weighted_data_qf_logdet(self, two_scalar_field_setup):
        """Cover dict paths: weighted data, quadratic form, logdet, prepare_smw."""
        from cosmocore.basis import HarmonicBasis

        setup = two_scalar_field_setup
        lmax = setup["lmax"]
        hc = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=lmax,
            spins=[0, 0],
        )
        hc.setup()

        C_ell_dict = {
            (0, 0, 0): np.ones(lmax + 1) * 1e-5,
            (1, 1, 0): np.ones(lmax + 1) * 1e-5,
            (0, 1, 0): np.ones(lmax + 1) * 5e-6,
        }

        np.random.seed(42)
        data = np.random.randn(hc.n_pix)

        # Weighted compressed data (dict path)
        w = hc.get_weighted_compressed_data(data, C_ell_dict)
        assert w.shape == (hc.n_modes_total,)

        # Quadratic form (dict path)
        qf = hc.compute_quadratic_form(data, C_ell_dict)
        assert qf > 0

        # Log determinant (dict path)
        logdet = hc.get_logdet(C_ell_dict)
        assert isinstance(logdet, float)

        # get_full_logdet alias
        logdet2 = hc.get_full_logdet(C_ell_dict)
        assert_allclose(logdet, logdet2)

        # prepare_smw and quadratic_form_from_prepared
        K_chol, logdet_smw = hc.prepare_smw(C_ell_dict)
        assert isinstance(logdet_smw, float)
        qf2 = hc.quadratic_form_from_prepared(data, K_chol)
        assert_allclose(qf, qf2, rtol=1e-8)

    def test_single_entry_dict_fast_paths(self, uniform_sky_setup):
        """Cover single-entry dict fast paths in projected_inverse, covariance, Fisher."""
        from cosmocore.basis import HarmonicBasis

        setup = uniform_sky_setup
        lmax = setup["lmax"]
        hc = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=lmax,
        )
        hc.setup()

        C_ell_arr = np.ones(lmax + 1) * 1e-6
        C_ell_dict = {(0, 0, 0): C_ell_arr}

        # These should hit the single-entry dict fast paths
        # and produce results identical to the array path
        cov_arr = hc.get_compressed_covariance(C_ell_arr)
        cov_dict = hc.get_compressed_covariance(C_ell_dict)
        assert_allclose(cov_arr, cov_dict, rtol=1e-10)

        inv_arr = hc.get_projected_inverse(C_ell_arr)
        inv_dict = hc.get_projected_inverse(C_ell_dict)
        assert_allclose(inv_arr, inv_dict, rtol=1e-10)

        # Single-entry dict Fisher fast path
        fisher_arr = hc.compute_fisher_matrix(C_ell_arr)
        fisher_dict = hc.compute_fisher_matrix(C_ell_dict, spectra_list=[(0, 0, 0)])
        assert_allclose(fisher_arr, fisher_dict, rtol=1e-10)

        # Weighted data single-entry dict fast path
        np.random.seed(42)
        data = np.random.randn(setup["n_pix"])
        w_arr = hc.get_weighted_compressed_data(data, C_ell_arr)
        w_dict = hc.get_weighted_compressed_data(data, C_ell_dict)
        assert_allclose(w_arr, w_dict, rtol=1e-10)


class TestNoiseCovWithNonDiagonalN:
    """Cov(w|noise) must be correct when N has off-diagonal structure."""

    def _build_basis(self, N, lmax=8):
        from cosmocore.basics import matrix_inverse_symm
        from cosmocore.basis import HarmonicBasis

        n_pix = N.shape[0]
        rng = np.random.default_rng(7)
        theta = rng.uniform(0, np.pi, n_pix)
        phi = rng.uniform(0, 2 * np.pi, n_pix)
        matrix_inverse_symm(np.asfortranarray(N))
        hc = HarmonicBasis(N=N, theta=theta, phi=phi, lmax_signal=lmax)
        hc.setup()
        return hc

    def _direct_noise_cov(self, hc, kernel_inv, N_orig=None):
        # With switch optimisation the basis overwrites N in-place with N_eff
        # and never holds a raw-N copy, so the caller must supply N_orig.
        n = kernel_inv.shape[0]
        V_Cinv = (np.eye(n) - hc._V_Ninv_VT @ kernel_inv) @ hc._V_N_inv
        if N_orig is None:
            N_orig = hc._N_symmetric
        return V_Cinv @ N_orig @ V_Cinv.T

    def _compressed_noise_cov(self, hc, kernel_inv):
        # Compressed: A T A^T (the path used in production).
        n = kernel_inv.shape[0]
        A = np.eye(n) - hc._V_Ninv_VT @ kernel_inv
        return A @ hc._noise_cov_T @ A.T

    def _make_kernel_inv(self, hc, C_ell):
        from cosmocore.basics import matrix_inverse_symm, smw_kernel

        Lambda_diag = hc._build_lambda_diagonal(C_ell)
        K = smw_kernel(hc._V_Ninv_VT, Lambda_diag)
        return matrix_inverse_symm(np.asfortranarray(K))

    def _random_psd(self, n_pix, off_scale, seed):
        rng = np.random.default_rng(seed)
        diag = rng.uniform(0.5, 1.5, n_pix)
        offdiag = off_scale * rng.standard_normal((n_pix, n_pix))
        N = np.diag(diag) + 0.5 * (offdiag + offdiag.T)
        # Ensure positive-definiteness by shifting eigenvalues if needed.
        eigvals = np.linalg.eigvalsh(N)
        if eigvals.min() <= 0:
            N += (abs(eigvals.min()) + 0.1) * np.eye(n_pix)
        return N

    def test_diagonal_n_matches_reference(self):
        n_pix = 60
        rng = np.random.default_rng(1)
        N = np.diag(rng.uniform(0.5, 1.5, n_pix))
        hc = self._build_basis(N)
        C_ell = np.full(hc.lmax_signal + 1, 0.05)
        kernel_inv = self._make_kernel_inv(hc, C_ell)
        ref = self._direct_noise_cov(hc, kernel_inv)
        got = self._compressed_noise_cov(hc, kernel_inv)
        assert_allclose(got, ref, rtol=1e-10, atol=1e-12)

    def test_nondiagonal_n_matches_reference(self):
        n_pix = 60
        N = self._random_psd(n_pix, off_scale=0.05, seed=2)
        # Sanity: N is genuinely non-diagonal
        assert np.max(np.abs(N - np.diag(np.diag(N)))) > 1e-3
        hc = self._build_basis(N)
        C_ell = np.full(hc.lmax_signal + 1, 0.05)
        kernel_inv = self._make_kernel_inv(hc, C_ell)
        ref = self._direct_noise_cov(hc, kernel_inv)
        got = self._compressed_noise_cov(hc, kernel_inv)
        assert_allclose(got, ref, rtol=1e-10, atol=1e-12)

    def test_nondiagonal_n_with_switch_optimization(self):
        """Lock the Tegmark noise-bias convention with ``S_fixed`` active.

        QUBE follows the Tegmark form
        ``bias = ½ Tr[E_b · V C⁻¹ N C⁻¹ V^T]`` with the *raw* noise N,
        not ``N_eff = N + S_fixed``. The basis materialises this through
        the algebraic identity
        ``T = V N_eff⁻¹ N N_eff⁻¹ V^T = V_Ninv_VT − V_N_inv·S_fixed·V_N_inv^T``
        so it never has to carry a second ``n_pix²`` buffer for raw N.

        This test exercises the switch path (``lmax < lmax_signal`` →
        ``S_fixed`` non-zero) with a non-diagonal noise covariance and
        compares ``A·_noise_cov_T·A^T`` (production) against
        ``V C⁻¹ N_orig C⁻¹ V^T`` evaluated *directly* with ``N_orig``
        passed in. They must agree to machine precision.

        If the convention is ever switched (e.g. someone "fixes" the
        residual ``S_fixed`` leakage by replacing ``T`` with
        ``V_Ninv_VT``), this test fails with ~100 % mismatched elements.
        That's not a bug in the test; the residual is the documented
        Tegmark behaviour (see ADR 0009 and the project memory). To
        change it, every consumer of ``_noise_cov_T`` plus the
        traditional pixel-space path must be migrated together.
        """
        from cosmocore.basis import HarmonicBasis

        n_pix = 60
        lmax = 10
        N = self._random_psd(n_pix, off_scale=0.05, seed=3)

        rng = np.random.default_rng(4)
        theta = rng.uniform(0, np.pi, n_pix)
        phi = rng.uniform(0, 2 * np.pi, n_pix)
        fiducial_C_ell = np.full(lmax + 1, 0.05)

        # Switch optimization active: N_eff = N + S_fixed for ℓ > inference upper.
        hc = HarmonicBasis(
            N=N,
            theta=theta,
            phi=phi,
            lmax_signal=lmax,
            lmin=2,
            lmax=6,
            fiducial_C_ell=fiducial_C_ell,
        )
        hc.setup()
        assert hc.lmax < hc.lmax_signal

        C_ell = fiducial_C_ell.copy()
        kernel_inv = self._make_kernel_inv(hc, C_ell)
        ref = self._direct_noise_cov(hc, kernel_inv, N_orig=N)
        got = self._compressed_noise_cov(hc, kernel_inv)
        assert_allclose(got, ref, rtol=1e-10, atol=1e-12)
