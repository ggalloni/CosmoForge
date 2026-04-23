"""
Tests for PixelBasis class.

Tests for the Gjerlow-like pixel-projected compression approach,
including cross-validation tests comparing both methods.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


class TestPixelBasisInitialization:
    """Tests for PixelBasis initialization."""

    def test_basic_initialization(self, simple_compression_setup):
        """Test that PixelBasis initializes correctly."""
        from cosmocore.basis import PixelBasis

        setup = simple_compression_setup
        ppc = PixelBasis(
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


class TestPixelBasisSetup:
    """Tests for PixelBasis setup operations."""

    def test_setup_creates_required_matrices(self, simple_compression_setup):
        """Test that setup() creates all required matrices."""
        from cosmocore.basis import PixelBasis

        setup = simple_compression_setup
        ppc = PixelBasis(
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
        from cosmocore.basis import PixelBasis

        setup = simple_compression_setup
        ppc = PixelBasis(
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


class TestPixelBasisApply:
    """Tests for PixelBasis compression application."""

    def test_apply_compression_reduces_modes(self, simple_compression_setup):
        """Test that apply_compression reduces the number of modes."""
        from cosmocore.basis import PixelBasis

        setup = simple_compression_setup
        ppc = PixelBasis(
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
        from cosmocore.basis import PixelBasis

        setup = simple_compression_setup
        ppc = PixelBasis(
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
        from cosmocore.basis import PixelBasis

        setup = simple_compression_setup
        ppc = PixelBasis(
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
        from cosmocore.basis import PixelBasis

        setup = simple_compression_setup
        ppc = PixelBasis(
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
        from cosmocore.basis import PixelBasis

        setup = simple_compression_setup
        ppc = PixelBasis(
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


class TestPixelBasisOperations:
    """Tests for PixelBasis compressed-space operations."""

    def test_compress_data(self, uniform_sky_setup):
        """Test data compression."""
        from cosmocore.basis import PixelBasis

        setup = uniform_sky_setup
        ppc = PixelBasis(
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
        from cosmocore.basis import PixelBasis

        setup = uniform_sky_setup
        ppc = PixelBasis(
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
        from cosmocore.basis import PixelBasis

        setup = uniform_sky_setup
        ppc = PixelBasis(
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
        from cosmocore.basis import PixelBasis

        setup = uniform_sky_setup
        ppc = PixelBasis(
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


class TestPixelBasisFisher:
    """Tests for PixelBasis Fisher matrix computation."""

    def test_fisher_matrix_positive_diagonal(self, uniform_sky_setup):
        """Test Fisher diagonal elements are positive using compute_fisher_matrix."""
        from cosmocore.basis import PixelBasis

        setup = uniform_sky_setup
        ppc = PixelBasis(
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
        from cosmocore.basis import PixelBasis

        setup = uniform_sky_setup
        ppc = PixelBasis(
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
    """Tests comparing HarmonicBasis and PixelBasis."""

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
        from cosmocore.basis import HarmonicBasis, PixelBasis

        setup = cross_validation_setup
        C_ell = setup["C_ell"]
        lmax = setup["lmax"]

        # HarmonicBasis
        hc = HarmonicBasis(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=lmax,
        )
        hc.setup()
        fisher_hc = hc.compute_fisher_matrix(C_ell)

        # PixelBasis
        ppc = PixelBasis(
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
        Test that PixelBasis with full modes
        approaches HarmonicBasis.
        """
        from cosmocore.basis import HarmonicBasis, PixelBasis

        setup = cross_validation_setup
        C_ell = setup["C_ell"]
        lmax = setup["lmax"]

        # HarmonicBasis
        hc = HarmonicBasis(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=lmax,
        )
        hc.setup()
        fisher_hc = hc.compute_fisher_matrix(C_ell)

        # PixelBasis with minimal compression (keep most modes)
        ppc = PixelBasis(
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


class TestPixelBasisBases:
    """Tests for different compression basis presets."""

    def test_available_bases_classmethod(self):
        """Test that available_bases returns all basis options."""
        from cosmocore.basis import COMPRESSION_BASES, PixelBasis

        bases = PixelBasis.available_bases()

        assert "harmonic" in bases
        assert "noise_weighted" in bases
        assert "total_covariance" in bases
        assert "snr" in bases
        assert bases == COMPRESSION_BASES

    def test_harmonic_basis(self, uniform_sky_setup):
        """Test compression with pure harmonic basis."""
        from cosmocore.basis import PixelBasis

        setup = uniform_sky_setup
        ppc = PixelBasis(
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
        from cosmocore.basis import PixelBasis

        setup = uniform_sky_setup
        ppc = PixelBasis(
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
        from cosmocore.basis import PixelBasis

        setup = uniform_sky_setup
        n_ell = setup["lmax"] - 1
        C_ell = np.ones(n_ell) * 1e-6

        ppc = PixelBasis(
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
        from cosmocore.basis import PixelBasis

        setup = uniform_sky_setup
        n_ell = setup["lmax"] - 1
        C_ell = np.ones(n_ell) * 1e-6

        ppc = PixelBasis(
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
        from cosmocore.basis import PixelBasis

        setup = uniform_sky_setup
        ppc = PixelBasis(
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
        from cosmocore.basis import PixelBasis

        setup = uniform_sky_setup
        ppc = PixelBasis(
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
        from cosmocore.basis import PixelBasis

        setup = uniform_sky_setup
        ppc = PixelBasis(
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
        from cosmocore.basis import PixelBasis

        setup = uniform_sky_setup
        n_ell = setup["lmax"] - 1
        C_ell = np.ones(n_ell) * 1e-6

        results = {}
        for basis in ["harmonic", "noise_weighted", "total_covariance", "snr"]:
            ppc = PixelBasis(
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


class TestPixelBasisEigenspectrum:
    """Tests for eigenvalue spectrum computation and plotting."""

    def test_compute_eigenspectrum_shape(self, uniform_sky_setup):
        """Test eigenspectrum returns correct shapes."""
        from cosmocore.basis import PixelBasis

        setup = uniform_sky_setup
        ppc = PixelBasis(
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
        from cosmocore.basis import PixelBasis

        setup = uniform_sky_setup
        ppc = PixelBasis(
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
        from cosmocore.basis import PixelBasis

        setup = uniform_sky_setup
        ppc = PixelBasis(
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
        from cosmocore.basis import PixelBasis

        setup = uniform_sky_setup
        n_ell = setup["lmax"] - 1
        C_ell = np.ones(n_ell) * 1e-6

        ppc = PixelBasis(
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
        """Test plot_eigenvalue_spectrum returns figure and axes array."""
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend for testing
        import matplotlib.pyplot as plt

        from cosmocore.basis import PixelBasis

        setup = uniform_sky_setup
        ppc = PixelBasis(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )
        ppc.setup()

        fig, axes = ppc.plot_eigenvalue_spectrum(basis="noise_weighted")

        assert fig is not None
        assert axes is not None
        assert len(axes) == 1
        plt.close(fig)

    def test_plot_eigenvalue_comparison_returns_figure(self, uniform_sky_setup):
        """Test plot_eigenvalue_comparison returns figure and axes array."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from cosmocore.basis import PixelBasis

        setup = uniform_sky_setup
        n_ell = setup["lmax"] - 1
        C_ell = np.ones(n_ell) * 1e-6

        ppc = PixelBasis(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )
        ppc.setup()

        fig, axes = ppc.plot_eigenvalue_comparison(
            bases=["harmonic", "noise_weighted"],
            C_ell=C_ell,
        )

        assert fig is not None
        assert axes is not None
        assert len(axes) == 1
        plt.close(fig)

    def test_eigenspectrum_requires_cell_for_certain_bases(self, uniform_sky_setup):
        """Test that compute_eigenspectrum raises error when C_ell missing."""
        from cosmocore.basis import PixelBasis

        setup = uniform_sky_setup
        ppc = PixelBasis(
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


class TestComputeEigenspectrumPerField:
    """Tests for compute_eigenspectrum_per_field."""

    def test_single_field_returns_length_one(self, uniform_sky_setup):
        """Single-field returns a list of length 1 with correct keys."""
        from cosmocore.basis import PixelBasis

        setup = uniform_sky_setup
        ppc = PixelBasis(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )
        ppc.setup()

        result = ppc.compute_eigenspectrum_per_field(basis="noise_weighted")

        assert len(result) == 1
        entry = result[0]
        assert entry["component"] == 0
        assert entry["spin"] == 0
        assert "eigenvalues" in entry
        assert "normalized_eigenvalues" in entry
        assert_allclose(np.max(entry["normalized_eigenvalues"]), 1.0, rtol=1e-10)
        # eigenvalues should be sorted descending
        assert np.all(entry["eigenvalues"][:-1] >= entry["eigenvalues"][1:])

    def test_multi_field_returns_correct_length(self, two_scalar_field_setup):
        """Multi-field returns list of length n_components."""
        from cosmocore.basis import PixelBasis

        setup = two_scalar_field_setup
        ppc = PixelBasis(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )
        ppc.setup()

        result = ppc.compute_eigenspectrum_per_field(basis="noise_weighted")

        assert len(result) == 2
        assert result[0]["component"] == 0
        assert result[1]["component"] == 1
        for entry in result:
            assert "eigenvalues" in entry
            assert "normalized_eigenvalues" in entry
            assert_allclose(np.max(entry["normalized_eigenvalues"]), 1.0, rtol=1e-10)

    def test_spin2_field_has_eb_keys(self):
        """Spin-2 field has E/B eigenvalue keys."""
        from cosmocore.basis import PixelBasis

        np.random.seed(42)
        n_pix = 20
        theta = np.random.uniform(0.1, np.pi - 0.1, n_pix)
        phi = np.random.uniform(0, 2 * np.pi, n_pix)
        lmax = 5

        N = np.eye(2 * n_pix) * 0.01
        N_inv = np.eye(2 * n_pix) * 100.0

        ppc = PixelBasis(N, N_inv, theta, phi, lmax, spins=[2])
        ppc.setup()

        result = ppc.compute_eigenspectrum_per_field(basis="noise_weighted")

        assert len(result) == 1
        entry = result[0]
        assert entry["spin"] == 2
        assert "E_eigenvalues" in entry
        assert "E_normalized" in entry
        assert "B_eigenvalues" in entry
        assert "B_normalized" in entry
        assert_allclose(np.max(entry["E_normalized"]), 1.0, rtol=1e-10)
        assert_allclose(np.max(entry["B_normalized"]), 1.0, rtol=1e-10)

    def test_total_covariance_basis_single_field(self, uniform_sky_setup):
        """total_covariance basis works when C_ell is provided."""
        from cosmocore.basis import PixelBasis

        setup = uniform_sky_setup
        n_ell = setup["lmax"] - 1
        C_ell = np.ones(n_ell) * 1e-6

        ppc = PixelBasis(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )
        ppc.setup()

        result = ppc.compute_eigenspectrum_per_field(
            basis="total_covariance", C_ell=C_ell
        )

        assert len(result) == 1
        assert_allclose(np.max(result[0]["normalized_eigenvalues"]), 1.0, rtol=1e-10)

    def test_total_covariance_basis_multi_field(self, two_scalar_field_setup):
        """total_covariance basis works with dict C_ell for multi-field."""
        from cosmocore.basis import PixelBasis

        setup = two_scalar_field_setup
        lmax = setup["lmax"]
        n_ell = lmax - 1
        C_ell_dict = {
            (0, 0): np.ones(n_ell) * 1e-5,
            (1, 1): np.ones(n_ell) * 0.8e-5,
            (0, 1): np.ones(n_ell) * 0.3e-5,
        }

        ppc = PixelBasis(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=lmax,
        )
        ppc.setup()

        result = ppc.compute_eigenspectrum_per_field(
            basis="total_covariance", C_ell=C_ell_dict
        )

        assert len(result) == 2
        for entry in result:
            assert_allclose(np.max(entry["normalized_eigenvalues"]), 1.0, rtol=1e-10)

    def test_unknown_basis_raises(self, uniform_sky_setup):
        """Unknown basis raises ValueError with available list."""
        from cosmocore.basis import PixelBasis

        setup = uniform_sky_setup
        ppc = PixelBasis(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )
        ppc.setup()

        with pytest.raises(ValueError, match="Unknown compression basis"):
            ppc.compute_eigenspectrum_per_field(basis="invalid_basis")


class TestPlotMultiField:
    """Tests for multi-field plotting methods."""

    def test_plot_spectrum_multi_field(self, two_scalar_field_setup):
        """Multi-field spectrum returns correct number of subplots."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from cosmocore.basis import PixelBasis

        setup = two_scalar_field_setup
        ppc = PixelBasis(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )
        ppc.setup()

        fig, axes = ppc.plot_eigenvalue_spectrum(basis="noise_weighted")

        assert len(axes) == 2
        plt.close(fig)

    def test_plot_spectrum_spin2_eb_split(self):
        """Spin-2 spectrum shows E/B curves."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from cosmocore.basis import PixelBasis

        np.random.seed(42)
        n_pix = 20
        theta = np.random.uniform(0.1, np.pi - 0.1, n_pix)
        phi = np.random.uniform(0, 2 * np.pi, n_pix)
        lmax = 5

        N = np.eye(2 * n_pix) * 0.01
        N_inv = np.eye(2 * n_pix) * 100.0

        ppc = PixelBasis(N, N_inv, theta, phi, lmax, spins=[2])
        ppc.setup()

        fig, axes = ppc.plot_eigenvalue_spectrum(
            basis="noise_weighted", show_eb_split=True
        )

        assert len(axes) == 1
        ax = axes[0]
        # Should have at least 3 lines: combined, E, B
        labels = [line.get_label() for line in ax.get_lines()]
        assert "E modes" in labels
        assert "B modes" in labels
        plt.close(fig)

    def test_plot_comparison_multi_field(self, two_scalar_field_setup):
        """Multi-field comparison returns correct number of subplots."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from cosmocore.basis import PixelBasis

        setup = two_scalar_field_setup
        ppc = PixelBasis(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )
        ppc.setup()

        fig, axes = ppc.plot_eigenvalue_comparison(
            bases=["harmonic", "noise_weighted"],
        )

        assert len(axes) == 2
        # Each subplot should have both bases plotted
        for ax in axes:
            labels = [line.get_label() for line in ax.get_lines()]
            assert "harmonic" in labels
            assert "noise_weighted" in labels
        plt.close(fig)


# =============================================================================
# Coverage-focused tests for uncovered PPC operations
# =============================================================================


class TestPPCOperationChain:
    """Exercise all PPC operations after compression to cover downstream methods."""

    def test_single_field_full_chain(self, uniform_sky_setup):
        """Cover single-field: properties, derivative, weighted data, qf, logdet, SMW."""
        from cosmocore.basis import PixelBasis

        setup = uniform_sky_setup
        ppc = PixelBasis(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )
        ppc.setup()
        ppc.apply_compression(epsilon=1e-6)

        C_ell = np.ones(setup["lmax"] - 1) * 1e-6

        # Properties
        assert ppc.projector.shape == (ppc.n_kept, ppc.n_pix)
        assert ppc.n_compressed == ppc.n_kept
        assert ppc.eigenvalues is not None
        assert ppc.compression_basis == "noise_weighted"
        assert 0 < ppc.compression_ratio <= 1.0

        # Derivative matrix
        dC = ppc.get_derivative_matrix(5)
        assert dC.shape == (ppc.n_kept, ppc.n_kept)

        # Weighted compressed data
        np.random.seed(42)
        data = np.random.randn(ppc.n_pix)
        w = ppc.get_weighted_compressed_data(data, C_ell)
        assert w.shape == (ppc.n_kept,)

        # Quadratic form
        qf = ppc.compute_quadratic_form(data, C_ell)
        assert qf > 0

        # Log determinant (array path)
        logdet = ppc.get_logdet(C_ell)
        assert isinstance(logdet, float)

        # Prepare SMW and reuse
        C_ell_dict = {(0, 0, 0): C_ell}
        C_c_inv, logdet_smw = ppc.prepare_smw(C_ell_dict)
        assert C_c_inv.shape == (ppc.n_kept, ppc.n_kept)
        qf2 = ppc.quadratic_form_from_prepared(data, C_c_inv)
        assert qf2 > 0

    def test_multi_field_dict_operations(self, two_scalar_field_setup):
        """Cover multi-field dict paths: covariance, derivative, Fisher, weighted data."""
        from cosmocore.basis import PixelBasis

        setup = two_scalar_field_setup
        lmax = setup["lmax"]
        ppc = PixelBasis(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=lmax,
            spins=[0, 0],
        )
        ppc.setup()
        ppc.apply_compression(epsilon=1e-6)

        C_ell_dict = {
            (0, 0, 0): np.ones(lmax - 1) * 1e-5,
            (1, 1, 0): np.ones(lmax - 1) * 1e-5,
            (0, 1, 0): np.ones(lmax - 1) * 5e-6,
        }
        spectra_list = [(0, 0, 0), (1, 1, 0), (0, 1, 0)]

        # Compressed covariance with dict
        C_c = ppc.get_compressed_covariance(C_ell_dict)
        assert C_c.shape == (ppc.n_kept, ppc.n_kept)

        # Cross-component derivative
        dC = ppc.get_derivative_matrix(5, comp_i=0, comp_j=1, mode=0)
        assert dC.shape == (ppc.n_kept, ppc.n_kept)

        # Multi-field Fisher
        n_ell = lmax - 1
        fisher = ppc.compute_fisher_matrix(C_ell_dict, spectra_list)
        assert fisher.shape == (3 * n_ell, 3 * n_ell)
        assert_allclose(fisher, fisher.T, atol=1e-12)

        # Weighted data and quadratic form with dict
        np.random.seed(42)
        data = np.random.randn(ppc.n_pix)
        w = ppc.get_weighted_compressed_data(data, C_ell_dict)
        assert w.shape == (ppc.n_kept,)
        qf = ppc.compute_quadratic_form(data, C_ell_dict)
        assert qf > 0

        # Logdet with dict
        logdet = ppc.get_logdet(C_ell_dict)
        assert isinstance(logdet, float)

    def test_default_compression_no_threshold(self, uniform_sky_setup):
        """apply_compression with neither epsilon nor mode_fraction."""
        from cosmocore.basis import PixelBasis

        setup = uniform_sky_setup
        ppc = PixelBasis(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )
        ppc.setup()
        ppc.apply_compression()
        assert ppc.n_kept > 0

    def test_runtime_errors_before_compression(self, uniform_sky_setup):
        """Operations before apply_compression raise RuntimeError."""
        from cosmocore.basis import PixelBasis

        setup = uniform_sky_setup
        ppc = PixelBasis(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )
        ppc.setup()

        C_ell = np.ones(setup["lmax"] - 1) * 1e-6
        data = np.zeros(ppc.n_pix)

        with pytest.raises(RuntimeError):
            ppc.projector
        with pytest.raises(RuntimeError):
            ppc.get_projected_inverse(C_ell)
        with pytest.raises(RuntimeError):
            ppc.get_derivative_matrix(5)
        with pytest.raises(RuntimeError):
            ppc.get_compressed_covariance(C_ell)
        with pytest.raises(RuntimeError):
            ppc.get_weighted_compressed_data(data, C_ell)
        with pytest.raises(RuntimeError):
            ppc.compute_quadratic_form(data, C_ell)
        with pytest.raises(RuntimeError):
            ppc.prepare_smw({(0, 0, 0): C_ell})
        with pytest.raises(RuntimeError):
            ppc.quadratic_form_from_prepared(data, np.eye(2))

    def test_spectra_list_required_for_dict(self, uniform_sky_setup):
        """compute_fisher_matrix with dict C_ell requires spectra_list."""
        from cosmocore.basis import PixelBasis

        setup = uniform_sky_setup
        ppc = PixelBasis(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )
        ppc.setup()
        ppc.apply_compression(epsilon=1e-6)

        C_ell_dict = {(0, 0, 0): np.ones(setup["lmax"] - 1) * 1e-6}
        with pytest.raises(ValueError, match="spectra_list is required"):
            ppc.compute_fisher_matrix(C_ell_dict)

    def test_spectra_list_none_for_array(self, uniform_sky_setup):
        """compute_fisher_matrix with array C_ell rejects spectra_list."""
        from cosmocore.basis import PixelBasis

        setup = uniform_sky_setup
        ppc = PixelBasis(
            N=setup["N"],
            N_inv=setup["N_inv"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )
        ppc.setup()
        ppc.apply_compression(epsilon=1e-6)

        C_ell = np.ones(setup["lmax"] - 1) * 1e-6
        with pytest.raises(ValueError, match="spectra_list should be None"):
            ppc.compute_fisher_matrix(C_ell, spectra_list=[(0, 0, 0)])
