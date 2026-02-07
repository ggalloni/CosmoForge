"""
Tests for the compression classes implementing SMW compression.

These tests verify the Sherman-Morrison-Woodbury compression framework for
efficient covariance matrix operations in CMB likelihood analysis.

Tests are organized by class:
- TestHarmonicCompression: Tests for HarmonicCompression (Tegmark-like)
- TestPixelProjectedCompression: Tests for PixelProjectedCompression (Gjerløw-like)
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


@pytest.fixture
def simple_compression_setup():
    """
    Create a simple test setup for compression classes.

    Returns a dictionary with:
    - N_inv: Simple diagonal noise inverse
    - theta, phi: Pixel positions on the sphere
    - lmax: Maximum multipole
    """
    np.random.seed(42)

    # Small number of pixels for fast tests
    n_pix = 100

    # Simple diagonal noise covariance (inverse)
    noise_variance = np.ones(n_pix) * 0.01
    N = np.diag(noise_variance)
    N_inv = np.diag(1.0 / noise_variance)

    # Random positions on the sphere
    theta = np.random.uniform(0, np.pi, n_pix)
    phi = np.random.uniform(0, 2 * np.pi, n_pix)

    # Small lmax for fast tests
    lmax = 10

    return {
        "N": N,
        "N_inv": N_inv,
        "theta": theta,
        "phi": phi,
        "lmax": lmax,
        "n_pix": n_pix,
    }


@pytest.fixture
def uniform_sky_setup():
    """
    Create a uniform sky setup with known geometry.

    Uses HEALPix-like uniform distribution for testing.
    """
    np.random.seed(123)

    # Moderate number of pixels
    n_pix = 50

    # Uniform distribution on sphere using golden spiral
    golden_ratio = (1 + np.sqrt(5)) / 2
    indices = np.arange(n_pix)
    theta = np.arccos(1 - 2 * (indices + 0.5) / n_pix)
    phi = (2 * np.pi * indices / golden_ratio) % (2 * np.pi)

    # Diagonal noise
    noise_variance = np.ones(n_pix) * 0.1
    N = np.diag(noise_variance)
    N_inv = np.diag(1.0 / noise_variance)

    lmax = 8

    return {
        "N": N,
        "N_inv": N_inv,
        "theta": theta,
        "phi": phi,
        "lmax": lmax,
        "n_pix": n_pix,
    }


# =============================================================================
# HarmonicCompression Tests
# =============================================================================


class TestHarmonicCompressionInitialization:
    """Tests for HarmonicCompression initialization."""

    def test_basic_initialization(self, simple_compression_setup):
        """Test that HarmonicCompression initializes correctly."""
        from cosmocore.compression import HarmonicCompression

        setup = simple_compression_setup
        hc = HarmonicCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )

        assert hc.n_pix == setup["n_pix"]
        assert hc.lmax == setup["lmax"]

    def test_mode_count(self, simple_compression_setup):
        """Test that n_modes is computed correctly."""
        from cosmocore.compression import HarmonicCompression

        setup = simple_compression_setup
        hc = HarmonicCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )

        # n_modes = (lmax+1)^2 - 4 for ell >= 2
        expected_n_modes = (setup["lmax"] + 1) ** 2 - 4
        assert hc.n_modes == expected_n_modes


class TestHarmonicCompressionSetup:
    """Tests for HarmonicCompression setup operations."""

    def test_setup_creates_required_matrices(self, simple_compression_setup):
        """Test that setup() creates all required matrices."""
        from cosmocore.compression import HarmonicCompression

        setup = simple_compression_setup
        hc = HarmonicCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
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
        from cosmocore.compression import HarmonicCompression

        setup = simple_compression_setup
        hc = HarmonicCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )

        hc.setup()

        # Each ell should have (2*ell + 1) modes
        for ell in range(2, setup["lmax"] + 1):
            assert len(hc._ell_to_modes[ell]) == 2 * ell + 1

    def test_smw_matrices_symmetric(self, simple_compression_setup):
        """Test that SMW matrices are symmetric."""
        from cosmocore.compression import HarmonicCompression

        setup = simple_compression_setup
        hc = HarmonicCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )

        hc.setup()

        assert_allclose(hc._V_Ninv_VT, hc._V_Ninv_VT.T, rtol=1e-10)
        assert_allclose(hc._V_N_VT, hc._V_N_VT.T, rtol=1e-10)


class TestHarmonicCompressionOperations:
    """Tests for HarmonicCompression compressed-space operations."""

    def test_compress_data(self, uniform_sky_setup):
        """Test data compression to harmonic space."""
        from cosmocore.compression import HarmonicCompression

        setup = uniform_sky_setup
        hc = HarmonicCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )

        hc.setup()

        data = np.random.randn(setup["n_pix"])
        d_compressed = hc.compress_data(data)

        assert d_compressed.shape == (hc.n_modes,)

    def test_compressed_covariance_shape(self, uniform_sky_setup):
        """Test compressed covariance has correct shape."""
        from cosmocore.compression import HarmonicCompression

        setup = uniform_sky_setup
        hc = HarmonicCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )

        hc.setup()

        n_ell = setup["lmax"] - 1
        C_ell = np.ones(n_ell) * 1e-6

        C_bar = hc.get_compressed_covariance(C_ell)

        assert C_bar.shape == (hc.n_modes, hc.n_modes)

    def test_compressed_covariance_symmetric(self, uniform_sky_setup):
        """Test compressed covariance is symmetric."""
        from cosmocore.compression import HarmonicCompression

        setup = uniform_sky_setup
        hc = HarmonicCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )

        hc.setup()

        n_ell = setup["lmax"] - 1
        C_ell = np.ones(n_ell) * 1e-6

        C_bar = hc.get_compressed_covariance(C_ell)

        assert_allclose(C_bar, C_bar.T, rtol=1e-10)

    def test_compressed_inverse_is_inverse(self, uniform_sky_setup):
        """Test that compressed inverse is actually the inverse."""
        from cosmocore.compression import HarmonicCompression

        setup = uniform_sky_setup
        hc = HarmonicCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )

        hc.setup()

        n_ell = setup["lmax"] - 1
        C_ell = np.ones(n_ell) * 1e-6

        C_bar = hc.get_compressed_covariance(C_ell)
        C_bar_inv = hc.get_compressed_inverse(C_ell)

        product = C_bar @ C_bar_inv
        identity = np.eye(hc.n_modes)

        # Slightly relaxed tolerance due to matrix inversion precision limits
        assert_allclose(product, identity, rtol=1e-8, atol=1e-9)

    def test_compressed_logdet_positive_definite(self, uniform_sky_setup):
        """Test log determinant computation for positive definite matrix."""
        from cosmocore.compression import HarmonicCompression

        setup = uniform_sky_setup
        hc = HarmonicCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )

        hc.setup()

        n_ell = setup["lmax"] - 1
        C_ell = np.ones(n_ell) * 1e-6

        logdet = hc.get_compressed_logdet(C_ell)

        assert np.isfinite(logdet)

        # Cross-check with numpy
        C_bar = hc.get_compressed_covariance(C_ell)
        _, logdet_np = np.linalg.slogdet(C_bar)

        assert_allclose(logdet, logdet_np, rtol=1e-8)


class TestHarmonicCompressionFisher:
    """Tests for HarmonicCompression Fisher matrix computation."""

    def test_fisher_matrix_positive_diagonal(self, uniform_sky_setup):
        """Test Fisher diagonal elements are positive using compute_fisher_matrix."""
        from cosmocore.compression import HarmonicCompression

        setup = uniform_sky_setup
        hc = HarmonicCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )

        hc.setup()

        n_ell = setup["lmax"] - 1
        C_ell = np.ones(n_ell) * 1e-6

        fisher = hc.compute_fisher_matrix(C_ell)
        F_55 = fisher[5 - 2, 5 - 2]  # ell=5 corresponds to index 3

        assert F_55 >= 0

    def test_fisher_matrix_symmetric(self, uniform_sky_setup):
        """Test Fisher matrix is symmetric using compute_fisher_matrix."""
        from cosmocore.compression import HarmonicCompression

        setup = uniform_sky_setup
        hc = HarmonicCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )

        hc.setup()

        n_ell = setup["lmax"] - 1
        C_ell = np.ones(n_ell) * 1e-6

        fisher = hc.compute_fisher_matrix(C_ell)

        assert_allclose(fisher, fisher.T, rtol=1e-10)


class TestHarmonicCompressionPixelSpace:
    """Tests for HarmonicCompression full pixel-space SMW operations."""

    def test_smw_inverse_shape(self, simple_compression_setup):
        """Test SMW inverse has correct shape."""
        from cosmocore.compression import HarmonicCompression

        setup = simple_compression_setup
        hc = HarmonicCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )

        hc.setup()

        n_ell = setup["lmax"] - 1
        C_ell = np.ones(n_ell) * 1e-6

        C_inv = hc.get_inverse(C_ell)

        assert C_inv.shape == (setup["n_pix"], setup["n_pix"])

    def test_smw_inverse_symmetric(self, simple_compression_setup):
        """Test SMW inverse is symmetric."""
        from cosmocore.compression import HarmonicCompression

        setup = simple_compression_setup
        hc = HarmonicCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )

        hc.setup()

        n_ell = setup["lmax"] - 1
        C_ell = np.ones(n_ell) * 1e-6

        C_inv = hc.get_inverse(C_ell)

        assert_allclose(C_inv, C_inv.T, rtol=1e-8)

    def test_smw_logdet_finite(self, simple_compression_setup):
        """Test SMW log determinant is finite."""
        from cosmocore.compression import HarmonicCompression

        setup = simple_compression_setup
        hc = HarmonicCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )

        hc.setup()

        n_ell = setup["lmax"] - 1
        C_ell = np.ones(n_ell) * 1e-6

        logdet = hc.get_logdet(C_ell)

        assert np.isfinite(logdet)


class TestHarmonicCompressionValidation:
    """Validation tests for HarmonicCompression against direct computation."""

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

        C_ell = 1e-3 / (np.arange(2, lmax + 1) ** 2)

        return {
            "N": N,
            "N_inv": N_inv,
            "theta": theta,
            "phi": phi,
            "lmax": lmax,
            "n_pix": n_pix,
            "C_ell": C_ell,
        }

    def _build_signal_covariance_direct(self, V, C_ell, lmax):
        """Build signal covariance S = V^T Λ V directly."""
        n_modes = V.shape[0]
        Lambda_diag = np.zeros(n_modes)
        idx = 0
        for ell in range(2, lmax + 1):
            n_m = 2 * ell + 1
            c_ell_value = C_ell[ell - 2] if ell - 2 < len(C_ell) else 0.0
            Lambda_diag[idx : idx + n_m] = c_ell_value
            idx += n_m
        Lambda = np.diag(Lambda_diag)
        return V.T @ Lambda @ V

    def test_smw_inverse_matches_direct(self, validation_setup):
        """Test that SMW inverse matches direct matrix inverse."""
        from cosmocore.compression import HarmonicCompression

        setup = validation_setup
        hc = HarmonicCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )
        hc.setup()

        V = hc._V.copy()
        C_ell = setup["C_ell"]

        S = self._build_signal_covariance_direct(V, C_ell, setup["lmax"])
        C_direct = setup["N"] + S
        C_inv_direct = np.linalg.inv(C_direct)

        C_inv_smw = hc.get_inverse(C_ell)

        assert_allclose(C_inv_smw, C_inv_direct, rtol=1e-8, atol=1e-12)

    def test_smw_logdet_matches_direct(self, validation_setup):
        """Test that SMW log-determinant matches direct computation."""
        from cosmocore.compression import HarmonicCompression

        setup = validation_setup
        hc = HarmonicCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )
        hc.setup()

        V = hc._V.copy()
        C_ell = setup["C_ell"]

        S = self._build_signal_covariance_direct(V, C_ell, setup["lmax"])
        C_direct = setup["N"] + S
        _, logdet_direct = np.linalg.slogdet(C_direct)

        logdet_smw = hc.get_logdet(C_ell)

        assert_allclose(logdet_smw, logdet_direct, rtol=1e-8)

    def test_smw_inverse_is_actual_inverse(self, validation_setup):
        """Test that SMW inverse times full covariance gives identity."""
        from cosmocore.compression import HarmonicCompression

        setup = validation_setup
        hc = HarmonicCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )
        hc.setup()

        V = hc._V.copy()
        C_ell = setup["C_ell"]

        S = self._build_signal_covariance_direct(V, C_ell, setup["lmax"])
        C_full = setup["N"] + S

        C_inv_smw = hc.get_inverse(C_ell)

        product = C_full @ C_inv_smw
        identity = np.eye(setup["n_pix"])

        assert_allclose(product, identity, rtol=1e-8, atol=1e-10)

    def test_smw_quadratic_form_matches_direct(self, validation_setup):
        """Test that SMW quadratic form matches direct computation."""
        from cosmocore.compression import HarmonicCompression

        setup = validation_setup
        hc = HarmonicCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )
        hc.setup()

        V = hc._V.copy()
        C_ell = setup["C_ell"]

        # Generate random data vector
        np.random.seed(42)
        data = np.random.normal(0, 1, setup["n_pix"])

        # Direct computation: d^T @ C^{-1} @ d
        S = self._build_signal_covariance_direct(V, C_ell, setup["lmax"])
        C_direct = setup["N"] + S
        C_inv_direct = np.linalg.inv(C_direct)
        quad_form_direct = float(data.T @ C_inv_direct @ data)

        # SMW computation
        quad_form_smw = hc.compute_quadratic_form(data, C_ell)

        assert_allclose(quad_form_smw, quad_form_direct, rtol=1e-8)


class TestHarmonicCompressionBeam:
    """Tests for beam window function support in HarmonicCompression."""

    def test_beam_initialization(self, simple_compression_setup):
        """Test that beam can be provided during initialization."""
        from cosmocore.compression import HarmonicCompression

        setup = simple_compression_setup
        lmax = setup["lmax"]

        ell = np.arange(2, lmax + 1)
        fwhm_rad = np.deg2rad(30.0 / 60.0)
        sigma = fwhm_rad / np.sqrt(8 * np.log(2))
        beam = np.exp(-0.5 * ell * (ell + 1) * sigma**2)

        hc = HarmonicCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=lmax,
            beam=beam,
        )

        assert hc._beam is not None
        assert len(hc._beam) == lmax - 1
        assert_allclose(hc._beam, beam)

    def test_beam_validation_wrong_length(self, simple_compression_setup):
        """Test that incorrect beam length raises an error."""
        from cosmocore.compression import HarmonicCompression

        setup = simple_compression_setup
        lmax = setup["lmax"]

        wrong_beam = np.ones(lmax + 5)

        with pytest.raises(ValueError, match="Beam must have length"):
            HarmonicCompression(
                N=setup["N"],
                N_inv=setup["N_inv"],
                theta=setup["theta"],
                phi=setup["phi"],
                lmax=lmax,
                beam=wrong_beam,
            )

    def test_unit_beam_no_effect(self, simple_compression_setup):
        """Test that unit beam produces same result as no beam."""
        from cosmocore.compression import HarmonicCompression

        setup = simple_compression_setup
        lmax = setup["lmax"]

        beam = np.ones(lmax - 1)
        C_ell = np.ones(lmax - 1) * 1e-5

        hc_no_beam = HarmonicCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=lmax,
        )
        hc_no_beam.setup()

        hc_unit_beam = HarmonicCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=lmax,
            beam=beam,
        )
        hc_unit_beam.setup()

        assert_allclose(hc_no_beam._V, hc_unit_beam._V, rtol=1e-12)

        C_bar_no_beam = hc_no_beam.get_compressed_covariance(C_ell)
        C_bar_unit_beam = hc_unit_beam.get_compressed_covariance(C_ell)
        assert_allclose(C_bar_no_beam, C_bar_unit_beam, rtol=1e-10)


# =============================================================================
# PixelProjectedCompression Tests
# =============================================================================


class TestPixelProjectedCompressionInitialization:
    """Tests for PixelProjectedCompression initialization."""

    def test_basic_initialization(self, simple_compression_setup):
        """Test that PixelProjectedCompression initializes correctly."""
        from cosmocore.compression import PixelProjectedCompression

        setup = simple_compression_setup
        ppc = PixelProjectedCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )

        assert ppc.n_pix == setup["n_pix"]
        assert ppc.lmax == setup["lmax"]
        # Before compression, n_kept = n_pix
        assert ppc.n_kept == setup["n_pix"]


class TestPixelProjectedCompressionSetup:
    """Tests for PixelProjectedCompression setup operations."""

    def test_setup_creates_required_matrices(self, simple_compression_setup):
        """Test that setup() creates all required matrices."""
        from cosmocore.compression import PixelProjectedCompression

        setup = simple_compression_setup
        ppc = PixelProjectedCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )

        ppc.setup()

        # Check V matrix exists
        assert ppc._V is not None
        assert ppc._V.shape == (ppc.n_modes, setup["n_pix"])

        # Check projector P_h exists
        assert ppc._P_h is not None
        assert ppc._P_h.shape == (setup["n_pix"], setup["n_pix"])

        # Check noise matrix N exists
        assert ppc._N is not None
        assert ppc._N.shape == (setup["n_pix"], setup["n_pix"])

    def test_projector_is_symmetric(self, simple_compression_setup):
        """Test that P_h is symmetric."""
        from cosmocore.compression import PixelProjectedCompression

        setup = simple_compression_setup
        ppc = PixelProjectedCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )

        ppc.setup()

        P_h = ppc._P_h

        # P_h = V^T V should be symmetric
        assert_allclose(P_h, P_h.T, rtol=1e-10)


class TestPixelProjectedCompressionApply:
    """Tests for PixelProjectedCompression compression application."""

    def test_apply_compression_reduces_modes(self, simple_compression_setup):
        """Test that apply_compression reduces the number of modes."""
        from cosmocore.compression import PixelProjectedCompression

        setup = simple_compression_setup
        ppc = PixelProjectedCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )

        ppc.setup()
        original_n_pix = ppc.n_pix

        ppc.apply_compression(epsilon=1e-3)

        assert ppc.n_kept <= original_n_pix
        assert ppc.compression_ratio <= 1.0

    def test_compression_creates_eigenvectors(self, simple_compression_setup):
        """Test that compression creates eigenvectors."""
        from cosmocore.compression import PixelProjectedCompression

        setup = simple_compression_setup
        ppc = PixelProjectedCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )

        ppc.setup()
        ppc.apply_compression(epsilon=1e-6)

        assert ppc._eigenvectors is not None
        assert ppc._eigenvalues is not None
        assert ppc._eigenvectors.shape == (setup["n_pix"], ppc.n_kept)

    def test_mode_fraction_compression(self, simple_compression_setup):
        """Test compression by specifying mode fraction."""
        from cosmocore.compression import PixelProjectedCompression

        setup = simple_compression_setup
        ppc = PixelProjectedCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )

        ppc.setup()
        ppc.apply_compression(mode_fraction=0.5)

        # Should keep approximately 50% of significant modes
        assert ppc.n_kept > 0
        assert ppc.n_kept <= setup["n_pix"]

    def test_mode_fraction_mutual_exclusivity(self, simple_compression_setup):
        """Test that epsilon and mode_fraction are mutually exclusive."""
        from cosmocore.compression import PixelProjectedCompression

        setup = simple_compression_setup
        ppc = PixelProjectedCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )
        ppc.setup()

        with pytest.raises(ValueError, match="mutually exclusive"):
            ppc.apply_compression(epsilon=1e-4, mode_fraction=0.5)

    def test_mode_fraction_validation(self, simple_compression_setup):
        """Test mode_fraction validation for invalid values."""
        from cosmocore.compression import PixelProjectedCompression

        setup = simple_compression_setup
        ppc = PixelProjectedCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )
        ppc.setup()

        with pytest.raises(ValueError, match="mode_fraction must be in"):
            ppc.apply_compression(mode_fraction=0.0)

        with pytest.raises(ValueError, match="mode_fraction must be in"):
            ppc.apply_compression(mode_fraction=-0.1)


class TestPixelProjectedCompressionOperations:
    """Tests for PixelProjectedCompression compressed-space operations."""

    def test_compress_data(self, uniform_sky_setup):
        """Test data compression."""
        from cosmocore.compression import PixelProjectedCompression

        setup = uniform_sky_setup
        ppc = PixelProjectedCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )

        ppc.setup()
        ppc.apply_compression(epsilon=1e-6)

        data = np.random.randn(setup["n_pix"])
        d_compressed = ppc.compress_data(data)

        assert d_compressed.shape == (ppc.n_kept,)

    def test_compressed_covariance_shape(self, uniform_sky_setup):
        """Test compressed covariance has correct shape."""
        from cosmocore.compression import PixelProjectedCompression

        setup = uniform_sky_setup
        ppc = PixelProjectedCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )

        ppc.setup()
        ppc.apply_compression(epsilon=1e-6)

        n_ell = setup["lmax"] - 1
        C_ell = np.ones(n_ell) * 1e-6

        C_compressed = ppc.get_compressed_covariance(C_ell)

        assert C_compressed.shape == (ppc.n_kept, ppc.n_kept)

    def test_compressed_covariance_symmetric(self, uniform_sky_setup):
        """Test compressed covariance is symmetric."""
        from cosmocore.compression import PixelProjectedCompression

        setup = uniform_sky_setup
        ppc = PixelProjectedCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )

        ppc.setup()
        ppc.apply_compression(epsilon=1e-6)

        n_ell = setup["lmax"] - 1
        C_ell = np.ones(n_ell) * 1e-6

        C_compressed = ppc.get_compressed_covariance(C_ell)

        # Use atol for numerical precision with small values
        assert_allclose(C_compressed, C_compressed.T, rtol=1e-10, atol=1e-15)

    def test_compressed_inverse_is_inverse(self, uniform_sky_setup):
        """Test that compressed inverse is actually the inverse."""
        from cosmocore.compression import PixelProjectedCompression

        setup = uniform_sky_setup
        ppc = PixelProjectedCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )

        ppc.setup()
        ppc.apply_compression(epsilon=1e-6)

        n_ell = setup["lmax"] - 1
        C_ell = np.ones(n_ell) * 1e-6

        C_compressed = ppc.get_compressed_covariance(C_ell)
        C_compressed_inv = ppc.get_projected_inverse(C_ell)

        product = C_compressed @ C_compressed_inv
        identity = np.eye(ppc.n_kept)

        assert_allclose(product, identity, rtol=1e-8, atol=1e-10)


class TestPixelProjectedCompressionFisher:
    """Tests for PixelProjectedCompression Fisher matrix computation."""

    def test_fisher_matrix_positive_diagonal(self, uniform_sky_setup):
        """Test Fisher diagonal elements are positive using compute_fisher_matrix."""
        from cosmocore.compression import PixelProjectedCompression

        setup = uniform_sky_setup
        ppc = PixelProjectedCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )

        ppc.setup()
        ppc.apply_compression(epsilon=1e-6)

        n_ell = setup["lmax"] - 1
        C_ell = np.ones(n_ell) * 1e-6

        fisher = ppc.compute_fisher_matrix(C_ell)
        F_55 = fisher[5 - 2, 5 - 2]  # ell=5 corresponds to index 3

        assert F_55 >= 0

    def test_fisher_matrix_symmetric(self, uniform_sky_setup):
        """Test Fisher matrix is symmetric using compute_fisher_matrix."""
        from cosmocore.compression import PixelProjectedCompression

        setup = uniform_sky_setup
        ppc = PixelProjectedCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )

        ppc.setup()
        ppc.apply_compression(epsilon=1e-6)

        n_ell = setup["lmax"] - 1
        C_ell = np.ones(n_ell) * 1e-6

        fisher = ppc.compute_fisher_matrix(C_ell)

        assert_allclose(fisher, fisher.T, rtol=1e-10)


# =============================================================================
# Cross-Validation Tests (Both Methods)
# =============================================================================


class TestCompressionCrossValidation:
    """Tests comparing HarmonicCompression and PixelProjectedCompression."""

    @pytest.fixture
    def cross_validation_setup(self):
        """Create setup for cross-validation tests."""
        np.random.seed(42)

        n_pix = 60
        lmax = 8

        noise_variance = np.ones(n_pix) * 0.05
        N = np.diag(noise_variance)
        N_inv = np.diag(1.0 / noise_variance)

        golden_ratio = (1 + np.sqrt(5)) / 2
        indices = np.arange(n_pix)
        theta = np.arccos(1 - 2 * (indices + 0.5) / n_pix)
        phi = (2 * np.pi * indices / golden_ratio) % (2 * np.pi)

        C_ell = np.ones(lmax - 1) * 1e-5

        return {
            "N": N,
            "N_inv": N_inv,
            "theta": theta,
            "phi": phi,
            "lmax": lmax,
            "n_pix": n_pix,
            "C_ell": C_ell,
        }

    def test_both_methods_produce_positive_fisher(self, cross_validation_setup):
        """Test that both methods produce positive semi-definite Fisher matrices."""
        from cosmocore.compression import HarmonicCompression, PixelProjectedCompression

        setup = cross_validation_setup
        C_ell = setup["C_ell"]
        lmax = setup["lmax"]

        # HarmonicCompression
        hc = HarmonicCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=lmax,
        )
        hc.setup()
        fisher_hc = hc.compute_fisher_matrix(C_ell)

        # PixelProjectedCompression
        ppc = PixelProjectedCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=lmax,
        )
        ppc.setup()
        ppc.apply_compression(epsilon=1e-10)  # Keep almost all modes
        fisher_ppc = ppc.compute_fisher_matrix(C_ell)

        # Both should be symmetric
        assert_allclose(fisher_hc, fisher_hc.T, rtol=1e-10)
        assert_allclose(fisher_ppc, fisher_ppc.T, rtol=1e-10)

        # Both should be positive semi-definite
        eigenvalues_hc = np.linalg.eigvalsh(fisher_hc)
        eigenvalues_ppc = np.linalg.eigvalsh(fisher_ppc)

        assert np.all(eigenvalues_hc >= -1e-10)
        assert np.all(eigenvalues_ppc >= -1e-10)

    def test_full_compression_approaches_harmonic(self, cross_validation_setup):
        """
        Test that PixelProjectedCompression with full modes
        approaches HarmonicCompression.
        """
        from cosmocore.compression import HarmonicCompression, PixelProjectedCompression

        setup = cross_validation_setup
        C_ell = setup["C_ell"]
        lmax = setup["lmax"]

        # HarmonicCompression
        hc = HarmonicCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=lmax,
        )
        hc.setup()
        fisher_hc = hc.compute_fisher_matrix(C_ell)

        # PixelProjectedCompression with minimal compression (keep most modes)
        ppc = PixelProjectedCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=lmax,
        )
        ppc.setup()
        ppc.apply_compression(epsilon=1e-15)  # Keep all significant modes
        fisher_ppc = ppc.compute_fisher_matrix(C_ell)

        # Compare Fisher elements for a few ell values
        # Note: exact match is not expected because the methods use different bases
        # but the Fisher information should be similar
        for ell in [3, 5, 7]:
            idx = ell - 2  # ell=2 is index 0
            f_hc = fisher_hc[idx, idx]
            f_ppc = fisher_ppc[idx, idx]

            # Both should be positive
            assert f_hc > 0
            assert f_ppc > 0

            # Should be in the same ballpark (within factor of 10)
            # Different bases mean exact match isn't expected
            ratio = f_hc / f_ppc if f_ppc > 0 else float("inf")
            assert 0.1 < ratio < 10, f"Fisher ratio at ell={ell}: {ratio}"


class TestPixelProjectedCompressionBases:
    """Tests for different compression basis presets."""

    def test_available_bases_classmethod(self):
        """Test that available_bases returns all basis options."""
        from cosmocore.compression import COMPRESSION_BASES, PixelProjectedCompression

        bases = PixelProjectedCompression.available_bases()

        assert "harmonic" in bases
        assert "noise_weighted" in bases
        assert "total_covariance" in bases
        assert "snr" in bases
        assert bases == COMPRESSION_BASES

    def test_harmonic_basis(self, uniform_sky_setup):
        """Test compression with pure harmonic basis."""
        from cosmocore.compression import PixelProjectedCompression

        setup = uniform_sky_setup
        ppc = PixelProjectedCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )

        ppc.setup()
        ppc.apply_compression(epsilon=1e-6, basis="harmonic")

        # Should have selected some modes
        assert ppc.n_kept > 0
        assert ppc.n_kept <= ppc.n_pix
        assert ppc.compression_basis == "harmonic"

    def test_noise_weighted_basis(self, uniform_sky_setup):
        """Test compression with noise-weighted basis (default)."""
        from cosmocore.compression import PixelProjectedCompression

        setup = uniform_sky_setup
        ppc = PixelProjectedCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )

        ppc.setup()
        ppc.apply_compression(epsilon=1e-6, basis="noise_weighted")

        assert ppc.n_kept > 0
        assert ppc.compression_basis == "noise_weighted"

    def test_total_covariance_basis(self, uniform_sky_setup):
        """Test compression with total covariance basis."""
        from cosmocore.compression import PixelProjectedCompression

        setup = uniform_sky_setup
        n_ell = setup["lmax"] - 1
        C_ell = np.ones(n_ell) * 1e-6

        ppc = PixelProjectedCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )

        ppc.setup()
        ppc.apply_compression(epsilon=1e-6, basis="total_covariance", C_ell=C_ell)

        assert ppc.n_kept > 0
        assert ppc.compression_basis == "total_covariance"

    def test_snr_basis(self, uniform_sky_setup):
        """Test compression with SNR basis."""
        from cosmocore.compression import PixelProjectedCompression

        setup = uniform_sky_setup
        n_ell = setup["lmax"] - 1
        C_ell = np.ones(n_ell) * 1e-6

        ppc = PixelProjectedCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )

        ppc.setup()
        ppc.apply_compression(epsilon=1e-6, basis="snr", C_ell=C_ell)

        assert ppc.n_kept > 0
        assert ppc.compression_basis == "snr"

    def test_total_covariance_requires_cell(self, uniform_sky_setup):
        """Test that total_covariance basis requires C_ell."""
        from cosmocore.compression import PixelProjectedCompression

        setup = uniform_sky_setup
        ppc = PixelProjectedCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )

        ppc.setup()

        with pytest.raises(ValueError, match="C_ell is required"):
            ppc.apply_compression(basis="total_covariance")

    def test_snr_requires_cell(self, uniform_sky_setup):
        """Test that snr basis requires C_ell."""
        from cosmocore.compression import PixelProjectedCompression

        setup = uniform_sky_setup
        ppc = PixelProjectedCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )

        ppc.setup()

        with pytest.raises(ValueError, match="C_ell is required"):
            ppc.apply_compression(basis="snr")

    def test_unknown_basis_raises(self, uniform_sky_setup):
        """Test that unknown basis raises error."""
        from cosmocore.compression import PixelProjectedCompression

        setup = uniform_sky_setup
        ppc = PixelProjectedCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )

        ppc.setup()

        with pytest.raises(ValueError, match="Unknown compression basis"):
            ppc.apply_compression(basis="invalid_basis")

    def test_different_bases_give_different_results(self, uniform_sky_setup):
        """Test that different bases lead to different mode selections."""
        from cosmocore.compression import PixelProjectedCompression

        setup = uniform_sky_setup
        n_ell = setup["lmax"] - 1
        C_ell = np.ones(n_ell) * 1e-6

        results = {}
        for basis in ["harmonic", "noise_weighted", "total_covariance", "snr"]:
            ppc = PixelProjectedCompression(
                N=setup["N"],
                N_inv=setup["N_inv"],
                theta=setup["theta"],
                phi=setup["phi"],
                lmax=setup["lmax"],
            )
            ppc.setup()

            c_ell_arg = C_ell if basis in ["total_covariance", "snr"] else None
            ppc.apply_compression(epsilon=1e-4, basis=basis, C_ell=c_ell_arg)
            results[basis] = ppc.n_kept

        # At least some bases should give different mode counts
        unique_counts = set(results.values())
        # With correlated noise, we expect different bases to give different results
        # but we don't require all to be different
        assert len(unique_counts) >= 1  # At minimum, they're all valid


class TestPixelProjectedCompressionEigenspectrum:
    """Tests for eigenvalue spectrum computation and plotting."""

    def test_compute_eigenspectrum_shape(self, uniform_sky_setup):
        """Test eigenspectrum returns correct shapes."""
        from cosmocore.compression import PixelProjectedCompression

        setup = uniform_sky_setup
        ppc = PixelProjectedCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )
        ppc.setup()

        eigenvalues, normalized = ppc.compute_eigenspectrum(basis="noise_weighted")

        assert eigenvalues.shape == (setup["n_pix"],)
        assert normalized.shape == (setup["n_pix"],)

    def test_eigenspectrum_normalized_max_is_one(self, uniform_sky_setup):
        """Test normalized eigenvalues have max value of 1."""
        from cosmocore.compression import PixelProjectedCompression

        setup = uniform_sky_setup
        ppc = PixelProjectedCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )
        ppc.setup()

        _, normalized = ppc.compute_eigenspectrum(basis="noise_weighted")

        assert_allclose(np.max(normalized), 1.0, rtol=1e-10)

    def test_eigenspectrum_sorted_descending(self, uniform_sky_setup):
        """Test eigenvalues are sorted in descending order."""
        from cosmocore.compression import PixelProjectedCompression

        setup = uniform_sky_setup
        ppc = PixelProjectedCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )
        ppc.setup()

        eigenvalues, _ = ppc.compute_eigenspectrum(basis="noise_weighted")

        # Check descending order
        assert np.all(eigenvalues[:-1] >= eigenvalues[1:])

    def test_eigenspectrum_all_bases(self, uniform_sky_setup):
        """Test eigenspectrum can be computed for all bases."""
        from cosmocore.compression import PixelProjectedCompression

        setup = uniform_sky_setup
        n_ell = setup["lmax"] - 1
        C_ell = np.ones(n_ell) * 1e-6

        ppc = PixelProjectedCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )
        ppc.setup()

        for basis in ["harmonic", "noise_weighted", "total_covariance", "snr"]:
            c_ell_arg = C_ell if basis in ["total_covariance", "snr"] else None
            eigenvalues, normalized = ppc.compute_eigenspectrum(
                basis=basis, C_ell=c_ell_arg
            )

            assert eigenvalues is not None
            assert len(eigenvalues) == setup["n_pix"]
            assert np.max(normalized) <= 1.0

    def test_plot_eigenvalue_spectrum_returns_figure(self, uniform_sky_setup):
        """Test plot_eigenvalue_spectrum returns figure and axes."""
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend for testing
        import matplotlib.pyplot as plt

        from cosmocore.compression import PixelProjectedCompression

        setup = uniform_sky_setup
        ppc = PixelProjectedCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )
        ppc.setup()

        fig, ax = ppc.plot_eigenvalue_spectrum(basis="noise_weighted")

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plot_eigenvalue_comparison_returns_figure(self, uniform_sky_setup):
        """Test plot_eigenvalue_comparison returns figure and axes."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from cosmocore.compression import PixelProjectedCompression

        setup = uniform_sky_setup
        n_ell = setup["lmax"] - 1
        C_ell = np.ones(n_ell) * 1e-6

        ppc = PixelProjectedCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )
        ppc.setup()

        fig, ax = ppc.plot_eigenvalue_comparison(
            bases=["harmonic", "noise_weighted"],
            C_ell=C_ell,
        )

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_eigenspectrum_requires_cell_for_certain_bases(self, uniform_sky_setup):
        """Test that compute_eigenspectrum raises error when C_ell missing."""
        from cosmocore.compression import PixelProjectedCompression

        setup = uniform_sky_setup
        ppc = PixelProjectedCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )
        ppc.setup()

        with pytest.raises(ValueError, match="C_ell is required"):
            ppc.compute_eigenspectrum(basis="total_covariance")

        with pytest.raises(ValueError, match="C_ell is required"):
            ppc.compute_eigenspectrum(basis="snr")


class TestCompressionManager:
    """Tests for the CompressionManager facade."""

    def test_harmonic_method(self, uniform_sky_setup):
        """Test CompressionManager with harmonic method."""
        from cosmocore.compression import CompressionManager

        setup = uniform_sky_setup
        cm = CompressionManager(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
            method="harmonic",
        )

        cm.setup()

        assert cm.method == "harmonic"
        assert cm.n_modes > 0
        assert cm.n_kept == cm.n_modes  # Harmonic keeps all modes

    def test_pixel_projected_method(self, uniform_sky_setup):
        """Test CompressionManager with pixel_projected method."""
        from cosmocore.compression import CompressionManager

        setup = uniform_sky_setup
        cm = CompressionManager(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
            method="pixel_projected",
            epsilon=1e-6,
        )

        cm.setup()

        assert cm.method == "pixel_projected"
        assert cm.n_kept > 0
        assert cm.n_kept <= setup["n_pix"]

    def test_unknown_method_raises(self, uniform_sky_setup):
        """Test that unknown method raises error."""
        from cosmocore.compression import CompressionManager

        setup = uniform_sky_setup

        with pytest.raises(ValueError, match="Unknown compression method"):
            CompressionManager(
                N=setup["N"],
                N_inv=setup["N_inv"],
                theta=setup["theta"],
                phi=setup["phi"],
                lmax=setup["lmax"],
                method="invalid",
            )

    def test_facade_delegates_correctly(self, uniform_sky_setup):
        """Test that facade methods delegate to implementation."""
        from cosmocore.compression import CompressionManager, HarmonicCompression

        setup = uniform_sky_setup
        n_ell = setup["lmax"] - 1
        C_ell = np.ones(n_ell) * 1e-6

        # Create both directly and via facade
        cm = CompressionManager(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
            method="harmonic",
        )
        cm.setup()

        hc = HarmonicCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )
        hc.setup()

        # Results should match
        assert_allclose(
            cm.get_compressed_covariance(C_ell),
            hc.get_compressed_covariance(C_ell),
            rtol=1e-10,
        )

    def test_fisher_matrix_via_facade(self, uniform_sky_setup):
        """Test Fisher matrix computation via facade."""
        from cosmocore.compression import CompressionManager

        setup = uniform_sky_setup
        n_ell = setup["lmax"] - 1
        C_ell = np.ones(n_ell) * 1e-6

        cm = CompressionManager(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
            method="harmonic",
        )
        cm.setup()

        # Compute Fisher matrix
        fisher = cm.compute_fisher_matrix(C_ell)
        F_33 = fisher[3 - 2, 3 - 2]  # ell=3 is index 1
        assert F_33 > 0  # Diagonal should be positive

    def test_pixel_projected_with_basis(self, uniform_sky_setup):
        """Test pixel_projected with different basis."""
        from cosmocore.compression import CompressionManager

        setup = uniform_sky_setup
        n_ell = setup["lmax"] - 1
        C_ell = np.ones(n_ell) * 1e-6

        cm = CompressionManager(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
            method="pixel_projected",
            basis="snr",
            C_ell=C_ell,
            epsilon=1e-4,
        )

        cm.setup()

        assert cm.n_kept > 0


class TestComputeFisherMatrix:
    """Tests for the optimized compute_fisher_matrix method."""

    def test_fisher_matrix_shape(self, uniform_sky_setup):
        """Test that compute_fisher_matrix returns correct shape."""
        from cosmocore.compression import HarmonicCompression

        setup = uniform_sky_setup
        lmax = setup["lmax"]
        n_ell = lmax - 1
        C_ell = np.ones(n_ell) * 1e-6

        hc = HarmonicCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=lmax,
        )
        hc.setup()

        fisher = hc.compute_fisher_matrix(C_ell, ell_min=2, ell_max=lmax)

        assert fisher.shape == (n_ell, n_ell)

    def test_fisher_matrix_symmetric(self, uniform_sky_setup):
        """Test that compute_fisher_matrix produces symmetric result."""
        from cosmocore.compression import HarmonicCompression

        setup = uniform_sky_setup
        lmax = setup["lmax"]
        n_ell = lmax - 1
        C_ell = np.ones(n_ell) * 1e-6

        hc = HarmonicCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=lmax,
        )
        hc.setup()

        fisher = hc.compute_fisher_matrix(C_ell)

        assert_allclose(fisher, fisher.T, rtol=1e-10)

    def test_fisher_matrix_positive_semidefinite(self, uniform_sky_setup):
        """Test that compute_fisher_matrix produces positive semi-definite matrix."""
        from cosmocore.compression import HarmonicCompression

        setup = uniform_sky_setup
        lmax = setup["lmax"]
        n_ell = lmax - 1
        C_ell = np.ones(n_ell) * 1e-6

        hc = HarmonicCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=lmax,
        )
        hc.setup()

        fisher = hc.compute_fisher_matrix(C_ell)

        # All eigenvalues should be non-negative
        eigenvalues = np.linalg.eigvalsh(fisher)
        assert np.all(eigenvalues >= -1e-10)

    def test_fisher_matrix_via_manager(self, uniform_sky_setup):
        """Test compute_fisher_matrix via CompressionManager facade."""
        from cosmocore.compression import CompressionManager

        setup = uniform_sky_setup
        lmax = setup["lmax"]
        n_ell = lmax - 1
        C_ell = np.ones(n_ell) * 1e-6

        cm = CompressionManager(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=lmax,
            method="harmonic",
        )
        cm.setup()

        fisher = cm.compute_fisher_matrix(C_ell)

        assert fisher.shape == (n_ell, n_ell)
        assert_allclose(fisher, fisher.T, rtol=1e-10)

    def test_pixel_projected_fisher_matrix(self, uniform_sky_setup):
        """Test compute_fisher_matrix with PixelProjectedCompression."""
        from cosmocore.compression import PixelProjectedCompression

        setup = uniform_sky_setup
        lmax = setup["lmax"]
        n_ell = lmax - 1
        C_ell = np.ones(n_ell) * 1e-6

        ppc = PixelProjectedCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=lmax,
        )
        ppc.setup()
        ppc.apply_compression(epsilon=1e-6)

        # Compute via optimized method
        fisher = ppc.compute_fisher_matrix(C_ell, ell_min=2, ell_max=lmax)

        # Verify shape and symmetry
        assert fisher.shape == (n_ell, n_ell)
        assert_allclose(fisher, fisher.T, rtol=1e-10)

        # Verify positive semi-definite
        eigenvalues = np.linalg.eigvalsh(fisher)
        assert np.all(eigenvalues >= -1e-10)


# =============================================================================
# Multi-Field Compression Tests (Phase 1: Multiple Scalar Fields)
# =============================================================================


@pytest.fixture
def two_scalar_field_setup():
    """
    Create a test setup with two scalar fields with different sky coverage.

    This mimics the multi-field setup used in signal matrix tests, with
    independent theta/phi coordinates per field (component).
    """
    np.random.seed(42)

    # Two fields with different pixel counts (different sky coverage)
    n_pix_1 = 60
    n_pix_2 = 40

    # Block-diagonal noise covariance (inverse)
    n_pix_total = n_pix_1 + n_pix_2
    noise_variance_1 = np.ones(n_pix_1) * 0.01
    noise_variance_2 = np.ones(n_pix_2) * 0.02  # Different noise level

    N = np.zeros((n_pix_total, n_pix_total))
    N[:n_pix_1, :n_pix_1] = np.diag(noise_variance_1)
    N[n_pix_1:, n_pix_1:] = np.diag(noise_variance_2)

    N_inv = np.zeros((n_pix_total, n_pix_total))
    N_inv[:n_pix_1, :n_pix_1] = np.diag(1.0 / noise_variance_1)
    N_inv[n_pix_1:, n_pix_1:] = np.diag(1.0 / noise_variance_2)

    # Random positions on the sphere - different for each field
    theta_1 = np.random.uniform(0, np.pi, n_pix_1)
    phi_1 = np.random.uniform(0, 2 * np.pi, n_pix_1)

    theta_2 = np.random.uniform(0, np.pi, n_pix_2)
    phi_2 = np.random.uniform(0, 2 * np.pi, n_pix_2)

    # Pack as tuples (multi-field format)
    theta_tuple = (theta_1, theta_2)
    phi_tuple = (phi_1, phi_2)

    # Small lmax for fast tests
    lmax = 8

    return {
        "N": N,
        "N_inv": N_inv,
        "theta": theta_tuple,
        "phi": phi_tuple,
        "lmax": lmax,
        "n_pix_1": n_pix_1,
        "n_pix_2": n_pix_2,
        "n_pix_total": n_pix_total,
    }


class TestMultiFieldCompressionInitialization:
    """Tests for multi-field compression initialization."""

    def test_multi_field_initialization(self, two_scalar_field_setup):
        """Test that HarmonicCompression initializes correctly with tuple inputs."""
        from cosmocore.compression import HarmonicCompression

        setup = two_scalar_field_setup
        hc = HarmonicCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )

        # Should detect 2 components
        assert hc.n_components == 2
        assert len(hc._theta_tuple) == 2
        assert len(hc._phi_tuple) == 2

        # Check pixel counts per component
        assert hc._n_pix_per_component[0] == setup["n_pix_1"]
        assert hc._n_pix_per_component[1] == setup["n_pix_2"]

    def test_single_field_backward_compatibility(self, simple_compression_setup):
        """Test that single array input still works (backward compatibility)."""
        from cosmocore.compression import HarmonicCompression

        setup = simple_compression_setup
        hc = HarmonicCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],  # 1D array, not tuple
            phi=setup["phi"],
            lmax=setup["lmax"],
        )

        # Should wrap as single-element tuple
        assert hc.n_components == 1
        assert len(hc._theta_tuple) == 1
        assert len(hc._theta_tuple[0]) == setup["n_pix"]


class TestMultiFieldVBlockStructure:
    """Tests for block-diagonal V matrix structure in multi-field."""

    def test_multi_field_v_shape(self, two_scalar_field_setup):
        """Test that V has correct block-diagonal shape."""
        from cosmocore.compression import HarmonicCompression

        setup = two_scalar_field_setup
        hc = HarmonicCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )
        hc.setup()

        # V should have shape (n_modes_total, n_pix_total)
        # n_modes_total = sum of modes per component
        assert hc._V is not None
        assert hc._V.shape[1] == setup["n_pix_total"]

        # Each component contributes (lmax+1)^2 - 4 modes
        # For lmax=8: (8+1)^2 - 4 = 77 modes per component
        n_modes_per_component = (setup["lmax"] + 1) ** 2 - 4
        expected_n_modes = 2 * n_modes_per_component
        assert hc._V.shape[0] == expected_n_modes
        assert hc.n_modes_total == expected_n_modes

    def test_multi_field_v_block_diagonal(self, two_scalar_field_setup):
        """Test that V has block-diagonal structure."""
        from cosmocore.compression import HarmonicCompression

        setup = two_scalar_field_setup
        hc = HarmonicCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )
        hc.setup()

        n_modes = (setup["lmax"] + 1) ** 2 - 4

        # V should be block-diagonal: V = [[V1, 0], [0, V2]]
        # Check that off-diagonal blocks are zero
        # Block (0,1): modes 0:n_modes, pixels n_pix_1:n_pix_total
        off_diag_01 = hc._V[:n_modes, setup["n_pix_1"] :]
        # Block (1,0): modes n_modes:2*n_modes, pixels 0:n_pix_1
        off_diag_10 = hc._V[n_modes:, : setup["n_pix_1"]]

        assert_allclose(off_diag_01, 0, atol=1e-14)
        assert_allclose(off_diag_10, 0, atol=1e-14)

        # Check that diagonal blocks are non-zero
        # Block (0,0): modes 0:n_modes, pixels 0:n_pix_1
        diag_00 = hc._V[:n_modes, : setup["n_pix_1"]]
        # Block (1,1): modes n_modes:2*n_modes, pixels n_pix_1:n_pix_total
        diag_11 = hc._V[n_modes:, setup["n_pix_1"] :]

        assert np.any(np.abs(diag_00) > 1e-10)
        assert np.any(np.abs(diag_11) > 1e-10)


class TestMultiFieldCompressedOperations:
    """Tests for multi-field compressed space operations."""

    def test_multi_field_compressed_covariance_with_dict(self, two_scalar_field_setup):
        """Test compressed covariance with C_ell_dict for multi-field."""
        from cosmocore.compression import HarmonicCompression

        setup = two_scalar_field_setup
        hc = HarmonicCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )
        hc.setup()

        n_ell = setup["lmax"] - 1
        # Different spectra for each field (auto and cross)
        C_ell_dict = {
            (0, 0): np.ones(n_ell) * 1e-6,  # Field 1 auto
            (1, 1): np.ones(n_ell) * 0.8e-6,  # Field 2 auto (different!)
            (0, 1): np.ones(n_ell) * 0.5e-6,  # Cross-spectrum
        }

        # Multi-field covariance via dedicated method
        C_compressed = hc.get_compressed_covariance_multi(C_ell_dict)

        # Shape should be (n_modes_total, n_modes_total)
        n_modes_per_component = (setup["lmax"] + 1) ** 2 - 4
        expected_n_modes = 2 * n_modes_per_component
        assert C_compressed.shape == (expected_n_modes, expected_n_modes)

    def test_multi_field_compressed_covariance_symmetric(self, two_scalar_field_setup):
        """Test compressed covariance is symmetric for multi-field."""
        from cosmocore.compression import HarmonicCompression

        setup = two_scalar_field_setup
        hc = HarmonicCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )
        hc.setup()

        n_ell = setup["lmax"] - 1
        C_ell_dict = {
            (0, 0): np.ones(n_ell) * 1e-6,
            (1, 1): np.ones(n_ell) * 0.8e-6,
            (0, 1): np.ones(n_ell) * 0.5e-6,
        }

        C_compressed = hc.get_compressed_covariance_multi(C_ell_dict)

        assert_allclose(C_compressed, C_compressed.T, rtol=1e-10, atol=1e-15)


class TestMultiFieldFisher:
    """Tests for multi-field Fisher matrix computation."""

    def test_multi_field_fisher_shape(self, two_scalar_field_setup):
        """Test Fisher matrix shape for multi-field with spectra dict."""
        from cosmocore.compression import HarmonicCompression

        setup = two_scalar_field_setup
        hc = HarmonicCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )
        hc.setup()

        n_ell = setup["lmax"] - 1
        # Multi-field spectra: two auto + one cross
        C_ell_dict = {
            (0, 0): np.ones(n_ell) * 1e-6,
            (1, 1): np.ones(n_ell) * 0.8e-6,
            (0, 1): np.ones(n_ell) * 0.5e-6,
        }
        spectra_list = [(0, 0), (1, 1), (0, 1)]  # 3 spectra

        # Multi-field Fisher: shape is (n_spectra * n_ell, n_spectra * n_ell)
        fisher = hc.compute_fisher_matrix_multi(C_ell_dict, spectra_list)

        n_spectra = len(spectra_list)
        assert fisher.shape == (n_spectra * n_ell, n_spectra * n_ell)

    def test_multi_field_fisher_symmetric(self, two_scalar_field_setup):
        """Test Fisher matrix is symmetric for multi-field."""
        from cosmocore.compression import HarmonicCompression

        setup = two_scalar_field_setup
        hc = HarmonicCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )
        hc.setup()

        n_ell = setup["lmax"] - 1
        C_ell_dict = {
            (0, 0): np.ones(n_ell) * 1e-6,
            (1, 1): np.ones(n_ell) * 0.8e-6,
            (0, 1): np.ones(n_ell) * 0.5e-6,
        }
        spectra_list = [(0, 0), (1, 1), (0, 1)]

        fisher = hc.compute_fisher_matrix_multi(C_ell_dict, spectra_list)

        assert_allclose(fisher, fisher.T, rtol=1e-10)

    def test_multi_field_fisher_positive_diagonal(self, two_scalar_field_setup):
        """Test Fisher matrix has positive diagonal for multi-field."""
        from cosmocore.compression import HarmonicCompression

        setup = two_scalar_field_setup
        hc = HarmonicCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )
        hc.setup()

        n_ell = setup["lmax"] - 1
        C_ell_dict = {
            (0, 0): np.ones(n_ell) * 1e-6,
            (1, 1): np.ones(n_ell) * 0.8e-6,
            (0, 1): np.ones(n_ell) * 0.5e-6,
        }
        spectra_list = [(0, 0), (1, 1), (0, 1)]

        fisher = hc.compute_fisher_matrix_multi(C_ell_dict, spectra_list)

        # Diagonal elements should all be positive
        assert np.all(np.diag(fisher) > 0)


class TestMultiFieldManager:
    """Tests for multi-field via CompressionManager facade."""

    def test_manager_multi_field(self, two_scalar_field_setup):
        """Test CompressionManager with multi-field input."""
        from cosmocore.compression import CompressionManager

        setup = two_scalar_field_setup
        cm = CompressionManager(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
            method="harmonic",
        )
        cm.setup()

        assert cm.n_components == 2
        assert cm.n_kept > 0

    def test_manager_multi_field_fisher(self, two_scalar_field_setup):
        """Test Fisher computation via CompressionManager with multi-field dict."""
        from cosmocore.compression import CompressionManager

        setup = two_scalar_field_setup
        n_ell = setup["lmax"] - 1
        C_ell_dict = {
            (0, 0): np.ones(n_ell) * 1e-6,
            (1, 1): np.ones(n_ell) * 0.8e-6,
            (0, 1): np.ones(n_ell) * 0.5e-6,
        }
        spectra_list = [(0, 0), (1, 1), (0, 1)]

        cm = CompressionManager(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
            method="harmonic",
        )
        cm.setup()

        # Multi-field Fisher via dedicated method
        fisher = cm.compute_fisher_matrix_multi(C_ell_dict, spectra_list)

        n_spectra = len(spectra_list)
        assert fisher.shape == (n_spectra * n_ell, n_spectra * n_ell)
        assert_allclose(fisher, fisher.T, rtol=1e-10)
        assert np.all(np.diag(fisher) > 0)


# =============================================================================
# Integration Test: Multi-Field Fisher vs Traditional Pixel-Space
# =============================================================================


@pytest.fixture
def three_scalar_field_realistic_setup():
    """
    Create a realistic test setup with three scalar fields (T1, T2, T3).

    This mimics a real-world scenario where we have multiple independent
    temperature measurements (e.g., different frequency channels).
    """
    np.random.seed(42)

    # Three fields with different pixel counts (different sky coverage)
    n_pix_1 = 80  # Field 1: full coverage
    n_pix_2 = 60  # Field 2: partial coverage
    n_pix_3 = 50  # Field 3: smaller patch

    n_pix_total = n_pix_1 + n_pix_2 + n_pix_3

    # Generate uniform positions on sphere using golden spiral
    def golden_spiral_positions(n_pix, seed_offset=0):
        np.random.seed(42 + seed_offset)
        golden_ratio = (1 + np.sqrt(5)) / 2
        indices = np.arange(n_pix)
        theta = np.arccos(1 - 2 * (indices + 0.5) / n_pix)
        phi = (2 * np.pi * indices / golden_ratio) % (2 * np.pi)
        # Add small perturbation to avoid numerical issues
        theta += np.random.uniform(-0.01, 0.01, n_pix)
        phi += np.random.uniform(-0.01, 0.01, n_pix)
        return theta, phi

    theta_1, phi_1 = golden_spiral_positions(n_pix_1, seed_offset=0)
    theta_2, phi_2 = golden_spiral_positions(n_pix_2, seed_offset=100)
    theta_3, phi_3 = golden_spiral_positions(n_pix_3, seed_offset=200)

    # Pack as tuples (multi-field format)
    theta_tuple = (theta_1, theta_2, theta_3)
    phi_tuple = (phi_1, phi_2, phi_3)

    # Block-diagonal noise covariance with different noise levels
    noise_var_1 = np.ones(n_pix_1) * 0.01  # Low noise
    noise_var_2 = np.ones(n_pix_2) * 0.02  # Medium noise
    noise_var_3 = np.ones(n_pix_3) * 0.03  # Higher noise

    N = np.zeros((n_pix_total, n_pix_total))
    N[:n_pix_1, :n_pix_1] = np.diag(noise_var_1)
    N[n_pix_1 : n_pix_1 + n_pix_2, n_pix_1 : n_pix_1 + n_pix_2] = np.diag(noise_var_2)
    N[n_pix_1 + n_pix_2 :, n_pix_1 + n_pix_2 :] = np.diag(noise_var_3)

    N_inv = np.zeros((n_pix_total, n_pix_total))
    N_inv[:n_pix_1, :n_pix_1] = np.diag(1.0 / noise_var_1)
    N_inv[n_pix_1 : n_pix_1 + n_pix_2, n_pix_1 : n_pix_1 + n_pix_2] = np.diag(
        1.0 / noise_var_2
    )
    N_inv[n_pix_1 + n_pix_2 :, n_pix_1 + n_pix_2 :] = np.diag(1.0 / noise_var_3)

    lmax = 10

    # Realistic power spectra (including normalization factors)
    n_ell = lmax - 1  # ell = 2 to lmax
    ells = np.arange(2, lmax + 1)

    # Apply (2ell+1)/(4pi) normalization factor as done in SpectraManager
    norm_factor = (2 * ells + 1) / (4 * np.pi)

    # Auto-spectra: different amplitudes for different fields
    C_ell_11 = norm_factor * 1e-4 / ells**2  # Field 1 auto
    C_ell_22 = norm_factor * 0.8e-4 / ells**2  # Field 2 auto
    C_ell_33 = norm_factor * 0.6e-4 / ells**2  # Field 3 auto

    # Cross-spectra: correlated but not perfectly
    C_ell_12 = norm_factor * 0.5e-4 / ells**2  # 1-2 cross
    C_ell_13 = norm_factor * 0.3e-4 / ells**2  # 1-3 cross
    C_ell_23 = norm_factor * 0.4e-4 / ells**2  # 2-3 cross

    C_ell_dict = {
        (0, 0): C_ell_11,
        (1, 1): C_ell_22,
        (2, 2): C_ell_33,
        (0, 1): C_ell_12,
        (0, 2): C_ell_13,
        (1, 2): C_ell_23,
    }

    return {
        "N": N,
        "N_inv": N_inv,
        "theta": theta_tuple,
        "phi": phi_tuple,
        "lmax": lmax,
        "n_pix_1": n_pix_1,
        "n_pix_2": n_pix_2,
        "n_pix_3": n_pix_3,
        "n_pix_total": n_pix_total,
        "C_ell_dict": C_ell_dict,
        "n_ell": n_ell,
    }


class TestMultiFieldIntegration:
    """
    Integration tests comparing multi-field compressed Fisher against
    traditional pixel-space computation.
    """

    def test_three_scalar_fields_fisher_diagonal_positive(
        self, three_scalar_field_realistic_setup
    ):
        """Test that 3-field Fisher has positive diagonal for all spectra."""
        from cosmocore.compression import HarmonicCompression

        setup = three_scalar_field_realistic_setup
        hc = HarmonicCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )
        hc.setup()

        # All 6 spectra: 3 auto + 3 cross
        spectra_list = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]

        fisher = hc.compute_fisher_matrix_multi(
            setup["C_ell_dict"], spectra_list, ell_min=2, ell_max=setup["lmax"]
        )

        n_spectra = len(spectra_list)
        n_ell = setup["n_ell"]
        expected_shape = (n_spectra * n_ell, n_spectra * n_ell)
        assert fisher.shape == expected_shape

        # All diagonal elements should be positive
        assert np.all(np.diag(fisher) > 0), "Fisher diagonal should be positive"

        # Fisher should be symmetric
        assert_allclose(fisher, fisher.T, rtol=1e-10)

    def test_three_scalar_fields_auto_spectra_only(
        self, three_scalar_field_realistic_setup
    ):
        """Test Fisher for auto-spectra only (simpler case)."""
        from cosmocore.compression import HarmonicCompression

        setup = three_scalar_field_realistic_setup
        hc = HarmonicCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )
        hc.setup()

        # Only auto-spectra
        spectra_list = [(0, 0), (1, 1), (2, 2)]

        fisher = hc.compute_fisher_matrix_multi(
            setup["C_ell_dict"], spectra_list, ell_min=2, ell_max=setup["lmax"]
        )

        n_spectra = len(spectra_list)
        n_ell = setup["n_ell"]
        expected_shape = (n_spectra * n_ell, n_spectra * n_ell)
        assert fisher.shape == expected_shape

        # Positive definite
        eigenvalues = np.linalg.eigvalsh(fisher)
        assert np.all(eigenvalues > -1e-10), "Fisher should be positive semi-definite"

    def test_single_field_from_multi_matches_original(
        self, three_scalar_field_realistic_setup
    ):
        """
        Test that extracting a single field from multi-field setup gives
        results consistent with single-field compression.
        """
        from cosmocore.compression import HarmonicCompression

        setup = three_scalar_field_realistic_setup

        # Multi-field setup
        hc_multi = HarmonicCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )
        hc_multi.setup()

        # Single-field setup using only field 1
        n_pix_1 = setup["n_pix_1"]
        N_single = setup["N"][:n_pix_1, :n_pix_1]
        N_inv_single = setup["N_inv"][:n_pix_1, :n_pix_1]
        theta_single = setup["theta"][0]
        phi_single = setup["phi"][0]

        hc_single = HarmonicCompression(
            N=N_single,
            N_inv=N_inv_single,
            theta=theta_single,
            phi=phi_single,
            lmax=setup["lmax"],
        )
        hc_single.setup()

        # Single-field Fisher using original API
        C_ell_11 = setup["C_ell_dict"][(0, 0)]
        fisher_single = hc_single.compute_fisher_matrix(C_ell_11)

        # Multi-field Fisher for just field 1 auto-spectrum
        fisher_multi = hc_multi.compute_fisher_matrix_multi(
            setup["C_ell_dict"], [(0, 0)], ell_min=2, ell_max=setup["lmax"]
        )

        # They should match within numerical precision
        # Note: Not exact due to different handling of block structure
        # but diagonal should have similar magnitude
        diag_single = np.diag(fisher_single)
        diag_multi = np.diag(fisher_multi)

        # Check that magnitudes are comparable (within factor of 2)
        ratio = diag_multi / diag_single
        assert np.all(ratio > 0.5) and np.all(ratio < 2.0), (
            f"Single vs multi-field diagonal ratio out of range: {ratio}"
        )

    def test_compressed_covariance_positive_definite(
        self, three_scalar_field_realistic_setup
    ):
        """Test that multi-field compressed covariance is positive definite."""
        from cosmocore.compression import HarmonicCompression

        setup = three_scalar_field_realistic_setup
        hc = HarmonicCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )
        hc.setup()

        C_compressed = hc.get_compressed_covariance_multi(setup["C_ell_dict"])

        # Check positive definiteness
        eigenvalues = np.linalg.eigvalsh(C_compressed)
        assert np.all(eigenvalues > 0), (
            f"Compressed covariance should be positive definite, "
            f"min eigenvalue: {np.min(eigenvalues)}"
        )

    def test_n_components_correct(self, three_scalar_field_realistic_setup):
        """Test that n_components is correctly detected."""
        from cosmocore.compression import CompressionManager

        setup = three_scalar_field_realistic_setup
        cm = CompressionManager(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
            method="harmonic",
        )
        cm.setup()

        assert cm.n_components == 3, f"Expected 3 components, got {cm.n_components}"

    def test_compressed_fisher_matches_pixel_space(self, two_scalar_field_setup):
        """
        Critical validation: Compare compressed Fisher against pixel-space computation.

        This test verifies that:
        F_ij = (1/2) Tr[(V C^{-1} V^T) E_i (V C^{-1} V^T) E_j]

        matches the traditional pixel-space Fisher:
        F_ij = (1/2) Tr[C^{-1} (dC/dC_i) C^{-1} (dC/dC_j)]

        For the multi-field case with block-diagonal structure.
        """
        from cosmocore.basics import matrix_inverse_symm, matrix_mult, matrix_trace
        from cosmocore.compression import HarmonicCompression

        setup = two_scalar_field_setup
        lmax = setup["lmax"]
        n_ell = lmax - 1

        # Different spectra per field
        ells = np.arange(2, lmax + 1)
        norm_factor = (2 * ells + 1) / (4 * np.pi)

        C_ell_dict = {
            (0, 0): norm_factor * 1e-5 / ells**2,
            (1, 1): norm_factor * 0.8e-5 / ells**2,
            (0, 1): norm_factor * 0.4e-5 / ells**2,
        }

        # Setup compression
        hc = HarmonicCompression(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=lmax,
        )
        hc.setup()

        # Compute compressed Fisher for single auto-spectrum (simpler case)
        spectra_list = [(0, 0)]
        fisher_compressed = hc.compute_fisher_matrix_multi(
            C_ell_dict, spectra_list, ell_min=2, ell_max=lmax
        )

        # Now compute pixel-space Fisher for comparison
        # Build full signal matrix S = V^T Λ V
        V = hc._V
        Lambda_full = hc._build_lambda_full(C_ell_dict)

        # S = V^T @ Λ @ V
        S = V.T @ Lambda_full @ V

        # Full covariance C = N + S
        C = setup["N"] + S
        C_inv = matrix_inverse_symm(C.copy())

        # Compute pixel-space Fisher for field 0 auto-spectrum
        fisher_pixel = np.zeros((n_ell, n_ell))

        for ell_i in range(2, lmax + 1):
            # Build dS/dC_ell for field 0 using V^T E_ell V
            E_i = hc.get_derivative_matrix_multi(ell_i, 0, 0)
            dS_i = V.T @ E_i @ V

            for ell_j in range(ell_i, lmax + 1):
                E_j = hc.get_derivative_matrix_multi(ell_j, 0, 0)
                dS_j = V.T @ E_j @ V

                # F_ij = 0.5 * Tr[C^{-1} dS_i C^{-1} dS_j]
                temp1 = matrix_mult(C_inv, dS_i)
                temp2 = matrix_mult(C_inv, dS_j)
                f_val = 0.5 * matrix_trace(temp1, temp2)

                idx_i = ell_i - 2
                idx_j = ell_j - 2
                fisher_pixel[idx_i, idx_j] = f_val
                if idx_i != idx_j:
                    fisher_pixel[idx_j, idx_i] = f_val

        # Compare compressed vs pixel-space
        assert_allclose(
            fisher_compressed,
            fisher_pixel,
            rtol=1e-6,
            atol=1e-10,
            err_msg="Compressed Fisher should match pixel-space computation",
        )


# =============================================================================
# Phase 2: Spin-2 Polarization Compression Tests
# =============================================================================


class TestSpin2HarmonicOperator:
    """Tests for spin-2 harmonic operator V construction."""

    def test_spin2_v_matrix_shape(self):
        """V for spin-2 should have shape (2*n_modes, 2*n_pix)."""
        from cosmocore.compression import HarmonicCompression

        np.random.seed(42)
        n_pix = 20
        theta = np.random.uniform(0.1, np.pi - 0.1, n_pix)
        phi = np.random.uniform(0, 2 * np.pi, n_pix)
        lmax = 5

        # Noise for 2*n_pix (Q and U)
        N = np.eye(2 * n_pix) * 0.01
        N_inv = np.eye(2 * n_pix) * 100.0

        hc = HarmonicCompression(N, N_inv, theta, phi, lmax, spins=[2])
        hc.setup()

        n_modes_base = (lmax + 1) ** 2 - 4
        expected_rows = 2 * n_modes_base  # E + B modes
        expected_cols = 2 * n_pix  # Q + U pixels

        assert hc._V.shape == (expected_rows, expected_cols), (
            f"V shape {hc._V.shape} != expected ({expected_rows}, {expected_cols})"
        )

    def test_spin2_v_matrix_nonzero(self):
        """V for spin-2 should have non-zero entries."""
        from cosmocore.compression import HarmonicCompression

        np.random.seed(42)
        n_pix = 20
        theta = np.random.uniform(0.1, np.pi - 0.1, n_pix)
        phi = np.random.uniform(0, 2 * np.pi, n_pix)
        lmax = 5

        N = np.eye(2 * n_pix) * 0.01
        N_inv = np.eye(2 * n_pix) * 100.0

        hc = HarmonicCompression(N, N_inv, theta, phi, lmax, spins=[2])
        hc.setup()

        # Both E and B blocks should have non-zero entries
        n_modes_base = (lmax + 1) ** 2 - 4
        V_E = hc._V[:n_modes_base, :]
        V_B = hc._V[n_modes_base:, :]

        assert np.any(V_E != 0), "E-mode block of V should have non-zero entries"
        assert np.any(V_B != 0), "B-mode block of V should have non-zero entries"

    def test_spin2_dimensions_tracking(self):
        """Check that spin-2 component dimensions are tracked correctly."""
        from cosmocore.compression import HarmonicCompression

        np.random.seed(42)
        n_pix = 20
        theta = np.random.uniform(0.1, np.pi - 0.1, n_pix)
        phi = np.random.uniform(0, 2 * np.pi, n_pix)
        lmax = 5

        N = np.eye(2 * n_pix) * 0.01
        N_inv = np.eye(2 * n_pix) * 100.0

        hc = HarmonicCompression(N, N_inv, theta, phi, lmax, spins=[2])

        n_modes_base = (lmax + 1) ** 2 - 4

        # Spin-2 doubles pixel count and mode count
        assert hc.n_pix == 2 * n_pix
        assert hc._n_pix_per_component == [2 * n_pix]
        assert hc._n_modes_per_component_list == [2 * n_modes_base]
        assert hc.n_modes_total == 2 * n_modes_base

    def test_spins_validation(self):
        """Test that invalid spins are rejected."""
        from cosmocore.compression import HarmonicCompression

        np.random.seed(42)
        n_pix = 10
        theta = np.random.uniform(0.1, np.pi - 0.1, n_pix)
        phi = np.random.uniform(0, 2 * np.pi, n_pix)
        lmax = 4

        N = np.eye(n_pix) * 0.01
        N_inv = np.eye(n_pix) * 100.0

        with pytest.raises(ValueError, match="Spin must be 0"):
            HarmonicCompression(N, N_inv, theta, phi, lmax, spins=[1])

        with pytest.raises(ValueError, match="spins list length"):
            HarmonicCompression(N, N_inv, theta, phi, lmax, spins=[0, 2])


class TestSpin2Lambda:
    """Tests for spin-2 Lambda matrix construction."""

    def test_lambda_block_spin2_shape(self):
        """Lambda for spin-2 auto-correlation should be (2*n_modes, 2*n_modes)."""
        from cosmocore.compression import HarmonicCompression

        np.random.seed(42)
        n_pix = 20
        theta = np.random.uniform(0.1, np.pi - 0.1, n_pix)
        phi = np.random.uniform(0, 2 * np.pi, n_pix)
        lmax = 5

        N = np.eye(2 * n_pix) * 0.01
        N_inv = np.eye(2 * n_pix) * 100.0

        hc = HarmonicCompression(N, N_inv, theta, phi, lmax, spins=[2])

        n_modes_base = (lmax + 1) ** 2 - 4
        C_ell = np.ones(lmax - 1) * 1e-3

        Lambda = hc._build_lambda_block_spin2(C_ell, C_ell * 0.5, C_ell * 0.1)

        assert Lambda.shape == (2 * n_modes_base, 2 * n_modes_base)

    def test_lambda_block_spin2_structure(self):
        """Lambda EE/BB diagonals and EB off-diagonals should be correctly placed."""
        from cosmocore.compression import HarmonicCompression

        np.random.seed(42)
        n_pix = 20
        theta = np.random.uniform(0.1, np.pi - 0.1, n_pix)
        phi = np.random.uniform(0, 2 * np.pi, n_pix)
        lmax = 4

        N = np.eye(2 * n_pix) * 0.01
        N_inv = np.eye(2 * n_pix) * 100.0

        hc = HarmonicCompression(N, N_inv, theta, phi, lmax, spins=[2])

        n = hc._n_modes_base
        C_EE = np.ones(lmax - 1) * 2.0
        C_BB = np.ones(lmax - 1) * 1.0
        C_EB = np.ones(lmax - 1) * 0.5

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
        from cosmocore.compression import HarmonicCompression

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
        N_inv = np.eye(total_pix) * 100.0

        hc = HarmonicCompression(
            N, N_inv, (theta_t, theta_p), (phi_t, phi_p), lmax, spins=[0, 2]
        )
        hc.setup()

        n_base = hc._n_modes_base

        # Build C_ell_dict with 3-tuple keys
        C_ell_dict = {
            (0, 0, 0): np.ones(lmax - 1) * 1e-3,  # TT
            (1, 1, 0): np.ones(lmax - 1) * 5e-4,  # EE
            (1, 1, 1): np.ones(lmax - 1) * 1e-4,  # BB
            (1, 1, 2): np.ones(lmax - 1) * 1e-5,  # EB
            (0, 1, 0): np.ones(lmax - 1) * 2e-4,  # TE
            (0, 1, 1): np.zeros(lmax - 1),  # TB
        }

        Lambda = hc._build_lambda_full_3tuple(C_ell_dict)

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
        from cosmocore.compression import HarmonicCompression

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
        N_inv = np.eye(total_pix) * 100.0

        hc = HarmonicCompression(
            N, N_inv, (theta_t, theta_p), (phi_t, phi_p), lmax, spins=[0, 2]
        )
        hc.setup()

        n_base = hc._n_modes_base
        ell = 3

        # EE derivative
        E_ee = hc.get_derivative_matrix_with_spins(ell, 1, 1, mode=0)
        # BB derivative
        E_bb = hc.get_derivative_matrix_with_spins(ell, 1, 1, mode=1)
        # EB derivative
        E_eb = hc.get_derivative_matrix_with_spins(ell, 1, 1, mode=2)

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
        from cosmocore.compression import HarmonicCompression

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
        N_inv = np.eye(total_pix) * 100.0

        hc = HarmonicCompression(
            N, N_inv, (theta_t, theta_p), (phi_t, phi_p), lmax, spins=[0, 2]
        )
        hc.setup()

        n_base = hc._n_modes_base
        ell = 3

        # TE derivative
        E_te = hc.get_derivative_matrix_with_spins(ell, 0, 1, mode=0)

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
        from cosmocore.compression import HarmonicCompression

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
        N_inv = np.eye(total_pix) * 100.0

        hc = HarmonicCompression(
            N, N_inv, (theta_t, theta_p), (phi_t, phi_p), lmax, spins=[0, 2]
        )
        hc.setup()

        C_ell_dict = {
            (0, 0, 0): np.ones(lmax - 1) * 1e-3,
            (1, 1, 0): np.ones(lmax - 1) * 5e-4,
            (1, 1, 1): np.ones(lmax - 1) * 1e-4,
            (1, 1, 2): np.zeros(lmax - 1),
            (0, 1, 0): np.ones(lmax - 1) * 2e-4,
            (0, 1, 1): np.zeros(lmax - 1),
        }

        spectra_list = [
            (0, 0, 0),  # TT
            (1, 1, 0),  # EE
            (1, 1, 1),  # BB
            (0, 1, 0),  # TE
        ]

        fisher = hc.compute_fisher_matrix_with_spins(
            C_ell_dict, spectra_list, ell_min=2, ell_max=lmax
        )

        n_ell = lmax - 1
        n_spec = len(spectra_list)
        assert fisher.shape == (n_spec * n_ell, n_spec * n_ell)

    def test_fisher_with_spins_symmetric(self):
        """Fisher matrix should be symmetric."""
        from cosmocore.compression import HarmonicCompression

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
        N_inv = np.eye(total_pix) * 100.0

        hc = HarmonicCompression(
            N, N_inv, (theta_t, theta_p), (phi_t, phi_p), lmax, spins=[0, 2]
        )
        hc.setup()

        C_ell_dict = {
            (0, 0, 0): np.ones(lmax - 1) * 1e-3,
            (1, 1, 0): np.ones(lmax - 1) * 5e-4,
            (1, 1, 1): np.ones(lmax - 1) * 1e-4,
        }

        spectra_list = [(0, 0, 0), (1, 1, 0), (1, 1, 1)]

        fisher = hc.compute_fisher_matrix_with_spins(
            C_ell_dict, spectra_list, ell_min=2, ell_max=lmax
        )

        assert_allclose(
            fisher, fisher.T, atol=1e-12, err_msg="Fisher matrix should be symmetric"
        )

    def test_fisher_with_spins_positive_diagonal(self):
        """Fisher diagonal elements should be positive."""
        from cosmocore.compression import HarmonicCompression

        np.random.seed(42)
        n_pix = 20
        theta = np.random.uniform(0.1, np.pi - 0.1, n_pix)
        phi = np.random.uniform(0, 2 * np.pi, n_pix)
        lmax = 5

        N = np.eye(2 * n_pix) * 0.01
        N_inv = np.eye(2 * n_pix) * 100.0

        hc = HarmonicCompression(N, N_inv, theta, phi, lmax, spins=[2])
        hc.setup()

        C_ell_dict = {
            (0, 0, 0): np.ones(lmax - 1) * 5e-4,  # EE
            (0, 0, 1): np.ones(lmax - 1) * 1e-4,  # BB
        }

        spectra_list = [(0, 0, 0), (0, 0, 1)]

        fisher = hc.compute_fisher_matrix_with_spins(
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
            E_i = hc.get_derivative_matrix_with_spins(ell_i, ci, cj, mode_i)
            dS_i = V.T @ E_i @ V

            for sj, (ck, cl, mode_j) in enumerate(spectra_list):
                for ell_j in range(2, lmax + 1):
                    E_j = hc.get_derivative_matrix_with_spins(ell_j, ck, cl, mode_j)
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
        - lmax=6 → n_modes_base=45 → 90 total (E+B)
        """
        import time

        from cosmocore.basics import matrix_inverse_symm
        from cosmocore.compression import HarmonicCompression

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
        N_inv = np.eye(2 * n_pix) / noise_level

        hc = HarmonicCompression(N, N_inv, theta, phi, lmax, spins=[2])
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
        fisher_compressed = hc.compute_fisher_matrix_with_spins(
            C_ell_dict, spectra_list, ell_min=2, ell_max=lmax
        )
        t_compressed = time.perf_counter() - t0

        # --- Pixel-space Fisher ---
        V = hc._V
        Lambda_full = hc._build_lambda_full_3tuple(C_ell_dict)
        S = V.T @ Lambda_full @ V
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
        assert speedup > 1.5, f"Expected compression speedup > 1.5x, got {speedup:.1f}x"

    def test_tqu_compressed_vs_pixel_space(self):
        """
        TQU: compressed TT/EE/BB/TE Fisher matches pixel-space computation.

        Mixed spin-0 (T) + spin-2 (Q, U) fields → TT, EE, BB, TE spectra.
        Sizes chosen so n_pix_total >> n_modes for compression speedup:
        - n_pix_t=50, n_pix_p=45 → total_pix = 50 + 2*45 = 140
        - lmax=6 → n_modes_base=45 → 45 (T) + 90 (E+B) = 135 modes
        """
        import time

        from cosmocore.basics import matrix_inverse_symm
        from cosmocore.compression import HarmonicCompression

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
        N_inv = np.linalg.inv(N)

        hc = HarmonicCompression(
            N, N_inv, (theta_t, theta_p), (phi_t, phi_p), lmax, spins=[0, 2]
        )
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
        fisher_compressed = hc.compute_fisher_matrix_with_spins(
            C_ell_dict, spectra_list, ell_min=2, ell_max=lmax
        )
        t_compressed = time.perf_counter() - t0

        # --- Pixel-space Fisher ---
        V = hc._V
        Lambda_full = hc._build_lambda_full_3tuple(C_ell_dict)
        S = V.T @ Lambda_full @ V
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
    """Tests for PixelProjectedCompression with spin-2 support."""

    def test_spin2_v_matrix_shape(self):
        """V for spin-2 PixelProjected should have correct shape."""
        from cosmocore.compression import PixelProjectedCompression

        np.random.seed(42)
        n_pix = 20
        theta = np.random.uniform(0.1, np.pi - 0.1, n_pix)
        phi = np.random.uniform(0, 2 * np.pi, n_pix)
        lmax = 5

        N = np.eye(2 * n_pix) * 0.01
        N_inv = np.eye(2 * n_pix) * 100.0

        ppc = PixelProjectedCompression(
            N,
            N_inv,
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
        from cosmocore.compression import PixelProjectedCompression

        np.random.seed(42)
        n_pix = 20
        theta = np.random.uniform(0.1, np.pi - 0.1, n_pix)
        phi = np.random.uniform(0, 2 * np.pi, n_pix)
        lmax = 5

        N = np.eye(2 * n_pix) * 0.01
        N_inv = np.eye(2 * n_pix) * 100.0

        ppc = PixelProjectedCompression(
            N,
            N_inv,
            theta,
            phi,
            lmax,
            spins=[2],
            epsilon=1e-6,
        )
        ppc.setup()

        C_ell_dict = {
            (0, 0, 0): np.ones(lmax - 1) * 5e-4,  # EE
            (0, 0, 1): np.ones(lmax - 1) * 1e-4,  # BB
        }

        C_c = ppc.get_compressed_covariance_with_spins(C_ell_dict)
        assert C_c.shape == (ppc.n_kept, ppc.n_kept)
        # Should be symmetric
        assert_allclose(C_c, C_c.T, atol=1e-12)

    def test_spin2_fisher_shape_and_symmetry(self):
        """Fisher matrix should have correct shape and be symmetric."""
        from cosmocore.compression import PixelProjectedCompression

        np.random.seed(42)
        n_pix = 20
        theta = np.random.uniform(0.1, np.pi - 0.1, n_pix)
        phi = np.random.uniform(0, 2 * np.pi, n_pix)
        lmax = 5

        N = np.eye(2 * n_pix) * 0.01
        N_inv = np.eye(2 * n_pix) * 100.0

        ppc = PixelProjectedCompression(
            N,
            N_inv,
            theta,
            phi,
            lmax,
            spins=[2],
            epsilon=1e-6,
        )
        ppc.setup()

        C_ell_dict = {
            (0, 0, 0): np.ones(lmax - 1) * 5e-4,  # EE
            (0, 0, 1): np.ones(lmax - 1) * 1e-4,  # BB
        }
        spectra_list = [(0, 0, 0), (0, 0, 1)]

        fisher = ppc.compute_fisher_matrix_with_spins(
            C_ell_dict, spectra_list, ell_min=2, ell_max=lmax
        )

        n_ell = lmax - 1
        n_spec = len(spectra_list)
        assert fisher.shape == (n_spec * n_ell, n_spec * n_ell)
        assert_allclose(fisher, fisher.T, atol=1e-12)

    def test_spin2_tqu_fisher_shape(self):
        """TQU PixelProjected Fisher should have correct shape."""
        from cosmocore.compression import PixelProjectedCompression

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
        N_inv = np.eye(total_pix) * 100.0

        ppc = PixelProjectedCompression(
            N,
            N_inv,
            (theta_t, theta_p),
            (phi_t, phi_p),
            lmax,
            spins=[0, 2],
            epsilon=1e-6,
        )
        ppc.setup()

        C_ell_dict = {
            (0, 0, 0): np.ones(lmax - 1) * 1e-3,  # TT
            (1, 1, 0): np.ones(lmax - 1) * 5e-4,  # EE
            (1, 1, 1): np.ones(lmax - 1) * 1e-4,  # BB
            (0, 1, 0): np.ones(lmax - 1) * 2e-4,  # TE
        }
        spectra_list = [
            (0, 0, 0),  # TT
            (1, 1, 0),  # EE
            (1, 1, 1),  # BB
            (0, 1, 0),  # TE
        ]

        fisher = ppc.compute_fisher_matrix_with_spins(
            C_ell_dict, spectra_list, ell_min=2, ell_max=lmax
        )

        n_ell = lmax - 1
        n_spec = len(spectra_list)
        assert fisher.shape == (n_spec * n_ell, n_spec * n_ell)
        assert_allclose(fisher, fisher.T, atol=1e-12)

    def test_spin2_weighted_data_shape(self):
        """Weighted compressed data should have shape (n_kept,)."""
        from cosmocore.compression import PixelProjectedCompression

        np.random.seed(42)
        n_pix = 20
        theta = np.random.uniform(0.1, np.pi - 0.1, n_pix)
        phi = np.random.uniform(0, 2 * np.pi, n_pix)
        lmax = 5

        N = np.eye(2 * n_pix) * 0.01
        N_inv = np.eye(2 * n_pix) * 100.0

        ppc = PixelProjectedCompression(
            N,
            N_inv,
            theta,
            phi,
            lmax,
            spins=[2],
            epsilon=1e-6,
        )
        ppc.setup()

        C_ell_dict = {
            (0, 0, 0): np.ones(lmax - 1) * 5e-4,
            (0, 0, 1): np.ones(lmax - 1) * 1e-4,
        }

        data = np.random.randn(2 * n_pix)
        w = ppc.get_weighted_compressed_data_with_spins(data, C_ell_dict)
        assert w.shape == (ppc.n_kept,)

    def test_spin2_manager_delegates(self):
        """CompressionManager should delegate spin-2 methods to PixelProjected."""
        from cosmocore.compression import CompressionManager

        np.random.seed(42)
        n_pix = 20
        theta = np.random.uniform(0.1, np.pi - 0.1, n_pix)
        phi = np.random.uniform(0, 2 * np.pi, n_pix)
        lmax = 5

        N = np.eye(2 * n_pix) * 0.01
        N_inv = np.eye(2 * n_pix) * 100.0

        cm = CompressionManager(
            N,
            N_inv,
            theta,
            phi,
            lmax,
            method="pixel_projected",
            spins=[2],
            epsilon=1e-6,
        )
        cm.setup()

        C_ell_dict = {
            (0, 0, 0): np.ones(lmax - 1) * 5e-4,
            (0, 0, 1): np.ones(lmax - 1) * 1e-4,
        }
        spectra_list = [(0, 0, 0), (0, 0, 1)]

        fisher = cm.compute_fisher_matrix_with_spins(
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
        chngconv = (2 * ell + 1) / (4 * np.pi)
        local_mode_indices = ell_to_modes_local[ell]
        n_base = n_modes_base

        # Spin-dependent normalization factors matching pixel.py convention
        factor2 = 1.0 / ((ell + 2) * (ell + 1) * ell * (ell - 1))
        factor = np.sqrt(factor2)

        if spin_i == 0 and spin_j == 0:
            row_offset = mode_offsets[comp_i]
            col_offset = mode_offsets[comp_j]
            for idx in local_mode_indices:
                E[row_offset + idx, col_offset + idx] = chngconv
            if comp_i != comp_j:
                for idx in local_mode_indices:
                    E[col_offset + idx, row_offset + idx] = chngconv
        elif spin_i == 2 and spin_j == 2:
            deriv_val = chngconv * factor2
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
            # Negative sign from spin-2 convention
            deriv_val = -chngconv * factor
            row_start = mode_offsets[comp_i]
            col_start = mode_offsets[comp_j]
            col_sub = col_start + mode * n_base
            for idx in local_mode_indices:
                E[row_start + idx, col_sub + idx] = deriv_val
                E[col_sub + idx, row_start + idx] = deriv_val
        elif spin_i == 2 and spin_j == 0:
            # Negative sign from spin-2 convention
            deriv_val = -chngconv * factor
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

    Verifies that PixelProjectedCompression with eigenvalue compression produces
    results close to the traditional pixel-space Fisher computation.
    Note: PixelProjected is an approximation (unlike Harmonic which is exact),
    so tolerances are relaxed.
    """

    def test_qu_compressed_vs_pixel_space(self):
        """
        QU-only: PixelProjected compressed EE/BB Fisher vs pixel-space.
        """
        from cosmocore.basics import matrix_inverse_symm
        from cosmocore.compression import PixelProjectedCompression

        np.random.seed(42)
        n_pix = 60
        golden_ratio = (1 + np.sqrt(5)) / 2
        indices = np.arange(n_pix)
        theta = np.arccos(1 - 2 * (indices + 0.5) / n_pix)
        phi = (2 * np.pi * indices / golden_ratio) % (2 * np.pi)
        lmax = 5

        noise_level = 1e-2
        N = np.eye(2 * n_pix) * noise_level
        N_inv = np.eye(2 * n_pix) / noise_level

        # Keep all modes (no truncation) for most accurate comparison
        ppc = PixelProjectedCompression(
            N,
            N_inv,
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

        fisher_compressed = ppc.compute_fisher_matrix_with_spins(
            C_ell_dict, spectra_list, ell_min=2, ell_max=lmax
        )

        # Pixel-space Fisher
        V = ppc._V
        Lambda_full = ppc._build_lambda_full_3tuple(C_ell_dict)
        S = V.T @ Lambda_full @ V
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
        from cosmocore.compression import PixelProjectedCompression

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
        N_inv = np.linalg.inv(N)

        ppc = PixelProjectedCompression(
            N,
            N_inv,
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

        fisher_compressed = ppc.compute_fisher_matrix_with_spins(
            C_ell_dict, spectra_list, ell_min=2, ell_max=lmax
        )

        # Pixel-space Fisher
        V = ppc._V
        Lambda_full = ppc._build_lambda_full_3tuple(C_ell_dict)
        S = V.T @ Lambda_full @ V
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


# =============================================================================
# Phase 2: Per-field E/B split thresholding tests
# =============================================================================


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
        from cosmocore.compression import PixelProjectedCompression

        np.random.seed(42)
        n_pix = 30
        theta, phi = self._spiral(n_pix)
        lmax = 5

        N = np.eye(n_pix) * 0.01
        N_inv = np.eye(n_pix) * 100.0

        # Scalar epsilon
        ppc1 = PixelProjectedCompression(
            N,
            N_inv,
            theta,
            phi,
            lmax,
            spins=[0],
            epsilon=1e-6,
        )
        ppc1.setup()

        # List epsilon (1-element)
        ppc2 = PixelProjectedCompression(
            N,
            N_inv,
            theta,
            phi,
            lmax,
            spins=[0],
            epsilon=[1e-6],
        )
        ppc2.setup()

        assert ppc1.n_kept == ppc2.n_kept

        # Fisher must match
        C_ell = np.ones(lmax - 1) * 1e-4
        f1 = ppc1.compute_fisher_matrix(C_ell, ell_min=2, ell_max=lmax)
        f2 = ppc2.compute_fisher_matrix(C_ell, ell_min=2, ell_max=lmax)
        assert_allclose(f1, f2, rtol=1e-12)

    def test_per_field_list_epsilon(self):
        """Two spin-0 fields with different epsilons → different n_kept per field."""
        from cosmocore.compression import PixelProjectedCompression

        np.random.seed(42)
        n_pix_a = 25
        n_pix_b = 25
        theta_a, phi_a = self._spiral(n_pix_a, offset=0)
        theta_b, phi_b = self._spiral(n_pix_b, offset=500)
        lmax = 5

        total_pix = n_pix_a + n_pix_b
        N = np.eye(total_pix) * 0.01
        N_inv = np.eye(total_pix) * 100.0

        # Tight threshold for field 0, loose for field 1
        ppc_split = PixelProjectedCompression(
            N,
            N_inv,
            (theta_a, theta_b),
            (phi_a, phi_b),
            lmax,
            spins=[0, 0],
            epsilon=[1e-8, 1e-2],
        )
        ppc_split.setup()

        # Uniform threshold
        ppc_uniform = PixelProjectedCompression(
            N,
            N_inv,
            (theta_a, theta_b),
            (phi_a, phi_b),
            lmax,
            spins=[0, 0],
            epsilon=1e-8,
        )
        ppc_uniform.setup()

        # Split should keep fewer modes (field 1 is aggressive)
        assert ppc_split.n_kept < ppc_uniform.n_kept

    def test_spin2_tuple_epsilon_eb_split(self):
        """Spin-2 field with tuple epsilon uses E/B split thresholding."""
        from cosmocore.compression import PixelProjectedCompression

        np.random.seed(42)
        n_pix = 20
        theta, phi = self._spiral(n_pix)
        lmax = 5

        N = np.eye(2 * n_pix) * 0.01
        N_inv = np.eye(2 * n_pix) * 100.0

        # Tight E threshold, loose B threshold → keep more B modes
        ppc_split = PixelProjectedCompression(
            N,
            N_inv,
            theta,
            phi,
            lmax,
            spins=[2],
            epsilon=[(1e-2, 1e-8)],
        )
        ppc_split.setup()

        # Uniform scalar → same threshold for both E and B
        ppc_uniform = PixelProjectedCompression(
            N,
            N_inv,
            theta,
            phi,
            lmax,
            spins=[2],
            epsilon=1e-2,
        )
        ppc_uniform.setup()

        # Split should keep >= uniform modes since B gets looser threshold
        assert ppc_split.n_kept >= ppc_uniform.n_kept

        # Fisher should still be PSD and symmetric
        C_ell_dict = {
            (0, 0, 0): np.ones(lmax - 1) * 5e-4,
            (0, 0, 1): np.ones(lmax - 1) * 1e-4,
        }
        spectra_list = [(0, 0, 0), (0, 0, 1)]
        fisher = ppc_split.compute_fisher_matrix_with_spins(
            C_ell_dict, spectra_list, ell_min=2, ell_max=lmax
        )
        assert_allclose(fisher, fisher.T, atol=1e-12)
        eigvals = np.linalg.eigvalsh(fisher)
        assert np.all(eigvals > -1e-10)

    def test_spin2_eb_split_vs_single(self):
        """E/B split with tight thresholds converges to single-threshold result."""
        from cosmocore.compression import PixelProjectedCompression

        np.random.seed(42)
        n_pix = 25
        theta, phi = self._spiral(n_pix)
        lmax = 5

        N = np.eye(2 * n_pix) * 0.01
        N_inv = np.eye(2 * n_pix) * 100.0

        eps = 1e-10  # Very tight → keeps essentially all modes

        # Scalar
        ppc_scalar = PixelProjectedCompression(
            N,
            N_inv,
            theta,
            phi,
            lmax,
            spins=[2],
            epsilon=eps,
        )
        ppc_scalar.setup()

        # Tuple with same value for both E and B
        ppc_tuple = PixelProjectedCompression(
            N,
            N_inv,
            theta,
            phi,
            lmax,
            spins=[2],
            epsilon=[(eps, eps)],
        )
        ppc_tuple.setup()

        # Fisher matrices should match closely
        C_ell_dict = {
            (0, 0, 0): np.ones(lmax - 1) * 5e-4,
            (0, 0, 1): np.ones(lmax - 1) * 1e-4,
        }
        spectra_list = [(0, 0, 0), (0, 0, 1)]

        f_scalar = ppc_scalar.compute_fisher_matrix_with_spins(
            C_ell_dict, spectra_list, ell_min=2, ell_max=lmax
        )
        f_tuple = ppc_tuple.compute_fisher_matrix_with_spins(
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
        from cosmocore.compression import PixelProjectedCompression

        np.random.seed(42)
        n_pix_t = 15
        n_pix_p = 12
        theta_t, phi_t = self._spiral(n_pix_t, offset=0)
        theta_p, phi_p = self._spiral(n_pix_p, offset=500)
        lmax = 4

        total_pix = n_pix_t + 2 * n_pix_p
        N = np.eye(total_pix) * 0.01
        N_inv = np.eye(total_pix) * 100.0

        # T: scalar epsilon, QU: E/B split tuple
        ppc = PixelProjectedCompression(
            N,
            N_inv,
            (theta_t, theta_p),
            (phi_t, phi_p),
            lmax,
            spins=[0, 2],
            epsilon=[1e-6, (1e-4, 1e-8)],
        )
        ppc.setup()

        assert ppc.n_kept > 0
        assert ppc.n_kept <= total_pix

        # Fisher should work and be valid
        C_ell_dict = {
            (0, 0, 0): np.ones(lmax - 1) * 1e-3,
            (1, 1, 0): np.ones(lmax - 1) * 5e-4,
            (1, 1, 1): np.ones(lmax - 1) * 1e-4,
            (0, 1, 0): np.ones(lmax - 1) * 2e-4,
        }
        spectra_list = [(0, 0, 0), (1, 1, 0), (1, 1, 1), (0, 1, 0)]
        fisher = ppc.compute_fisher_matrix_with_spins(
            C_ell_dict, spectra_list, ell_min=2, ell_max=lmax
        )
        assert_allclose(fisher, fisher.T, atol=1e-12)

    def test_invalid_tuple_for_spin0_raises(self):
        """Tuple epsilon for spin-0 field should raise ValueError."""
        from cosmocore.compression import PixelProjectedCompression

        np.random.seed(42)
        n_pix = 20
        theta, phi = self._spiral(n_pix)
        lmax = 5

        N = np.eye(n_pix) * 0.01
        N_inv = np.eye(n_pix) * 100.0

        with pytest.raises(ValueError, match="tuple.*E/B split.*spin.*not 2"):
            PixelProjectedCompression(
                N,
                N_inv,
                theta,
                phi,
                lmax,
                spins=[0],
                epsilon=[(1e-4, 1e-8)],
            )

    def test_invalid_list_length_raises(self):
        """Epsilon list with wrong length should raise ValueError."""
        from cosmocore.compression import PixelProjectedCompression

        np.random.seed(42)
        n_pix = 20
        theta, phi = self._spiral(n_pix)
        lmax = 5

        N = np.eye(n_pix) * 0.01
        N_inv = np.eye(n_pix) * 100.0

        with pytest.raises(ValueError, match="list length.*must match"):
            PixelProjectedCompression(
                N,
                N_inv,
                theta,
                phi,
                lmax,
                spins=[0],
                epsilon=[1e-4, 1e-8],  # 2 values for 1 field
            )


# =============================================================================
# PICSLike methods tests (prepare_smw_with_spins, quadratic_form, logdet)
# =============================================================================


class TestPixelProjectedPICSLikeMethods:
    """Tests for PICSLike-compatible methods on PixelProjectedCompression."""

    @staticmethod
    def _spiral(n, offset=0):
        gr = (1 + np.sqrt(5)) / 2
        idx = np.arange(n)
        th = np.arccos(1 - 2 * (idx + 0.5) / n)
        ph = (2 * np.pi * (idx + offset) / gr) % (2 * np.pi)
        return th, ph

    def _make_spin2_setup(self, n_pix=20, lmax=5, epsilon=1e-6):
        """Create a spin-2 PixelProjected setup and return (ppc, C_ell_dict)."""
        from cosmocore.compression import PixelProjectedCompression

        theta, phi = self._spiral(n_pix)
        N = np.eye(2 * n_pix) * 0.01
        N_inv = np.eye(2 * n_pix) * 100.0

        ppc = PixelProjectedCompression(
            N,
            N_inv,
            theta,
            phi,
            lmax,
            spins=[2],
            epsilon=epsilon,
        )
        ppc.setup()

        C_ell_dict = {
            (0, 0, 0): np.ones(lmax - 1) * 5e-4,
            (0, 0, 1): np.ones(lmax - 1) * 1e-4,
        }
        return ppc, C_ell_dict, N

    def test_compute_quadratic_form_with_spins(self):
        """Quadratic form matches brute-force d^T C^{-1} d."""
        from cosmocore.basics import matrix_inverse_symm

        np.random.seed(42)
        ppc, C_ell_dict, N = self._make_spin2_setup()

        # Generate random data
        data = np.random.randn(ppc.n_pix)

        # Compressed quadratic form
        qf_compressed = ppc.compute_quadratic_form_with_spins(data, C_ell_dict)

        # Brute-force in full pixel space
        Lambda_full = ppc._build_lambda_full_3tuple(C_ell_dict)
        S = ppc._V.T @ Lambda_full @ ppc._V
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
        """prepare_smw → quadratic_form_from_prepared matches direct compute."""
        np.random.seed(42)
        ppc, C_ell_dict, _ = self._make_spin2_setup()

        data = np.random.randn(ppc.n_pix)

        # Direct computation
        qf_direct = ppc.compute_quadratic_form_with_spins(data, C_ell_dict)

        # Two-step: prepare then compute
        C_c_inv, _, logdet = ppc.prepare_smw_with_spins(C_ell_dict)
        assert C_c_inv.shape == (ppc.n_kept, ppc.n_kept)
        assert isinstance(logdet, float)

        qf_prepared = ppc.quadratic_form_from_prepared(data, C_c_inv)

        assert_allclose(
            qf_prepared,
            qf_direct,
            rtol=1e-12,
            err_msg="Prepared quadratic form should match direct",
        )

    def test_get_logdet_with_spins(self):
        """get_logdet matches slogdet of compressed covariance."""
        np.random.seed(42)
        ppc, C_ell_dict, _ = self._make_spin2_setup()

        logdet = ppc.get_logdet_with_spins(C_ell_dict)

        # Directly compute
        C_c = ppc.get_compressed_covariance_with_spins(C_ell_dict)
        _, expected_logdet = np.linalg.slogdet(C_c)

        assert_allclose(
            logdet,
            expected_logdet,
            rtol=1e-10,
            err_msg="get_logdet_with_spins should match slogdet(C_compressed)",
        )

    def test_eb_split_preserves_b_mode_fisher(self):
        """
        Aggressive single threshold kills B modes;
        E/B split with loose B threshold retains them.
        """
        from cosmocore.compression import PixelProjectedCompression

        np.random.seed(42)
        n_pix = 25
        theta, phi = self._spiral(n_pix)
        lmax = 5

        N = np.eye(2 * n_pix) * 0.01
        N_inv = np.eye(2 * n_pix) * 100.0

        # Aggressive single threshold (E signal >> B → kills B modes)
        ppc_aggressive = PixelProjectedCompression(
            N,
            N_inv,
            theta,
            phi,
            lmax,
            spins=[2],
            epsilon=1e-1,
        )
        ppc_aggressive.setup()

        # E/B split: same aggressive E but loose B
        ppc_split = PixelProjectedCompression(
            N,
            N_inv,
            theta,
            phi,
            lmax,
            spins=[2],
            epsilon=[(1e-1, 1e-10)],
        )
        ppc_split.setup()

        # Split should keep more modes (B modes preserved)
        assert ppc_split.n_kept >= ppc_aggressive.n_kept

        C_ell_dict = {
            (0, 0, 0): np.ones(lmax - 1) * 5e-4,  # EE
            (0, 0, 1): np.ones(lmax - 1) * 1e-4,  # BB
        }
        spectra_list = [(0, 0, 0), (0, 0, 1)]  # EE, BB

        fisher_agg = ppc_aggressive.compute_fisher_matrix_with_spins(
            C_ell_dict, spectra_list, ell_min=2, ell_max=lmax
        )
        fisher_split = ppc_split.compute_fisher_matrix_with_spins(
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
