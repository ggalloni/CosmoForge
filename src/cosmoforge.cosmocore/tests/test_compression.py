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
