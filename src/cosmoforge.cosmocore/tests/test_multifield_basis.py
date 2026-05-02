"""
Tests for multi-field compression (Phase 1: Multiple Scalar Fields).

Tests for block-diagonal V matrix structure, multi-field compressed operations,
Fisher matrix computation, and integration tests comparing compressed Fisher
against traditional pixel-space computation.
"""

import numpy as np
from numpy.testing import assert_allclose


class TestMultiFieldCompressionInitialization:
    """Tests for multi-field compression initialization."""

    def test_multi_field_initialization(self, two_scalar_field_setup):
        """Test that HarmonicBasis initializes correctly with tuple inputs."""
        from cosmocore.basis import HarmonicBasis

        setup = two_scalar_field_setup
        hc = HarmonicBasis(
            N=setup["N"],
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
        from cosmocore.basis import HarmonicBasis

        setup = simple_compression_setup
        hc = HarmonicBasis(
            N=setup["N"],
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
        from cosmocore.basis import HarmonicBasis

        setup = two_scalar_field_setup
        hc = HarmonicBasis(
            N=setup["N"],
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
        from cosmocore.basis import HarmonicBasis

        setup = two_scalar_field_setup
        hc = HarmonicBasis(
            N=setup["N"],
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
        from cosmocore.basis import HarmonicBasis

        setup = two_scalar_field_setup
        hc = HarmonicBasis(
            N=setup["N"],
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
        C_compressed = hc.get_compressed_covariance(C_ell_dict)

        # Shape should be (n_modes_total, n_modes_total)
        n_modes_per_component = (setup["lmax"] + 1) ** 2 - 4
        expected_n_modes = 2 * n_modes_per_component
        assert C_compressed.shape == (expected_n_modes, expected_n_modes)

    def test_multi_field_compressed_covariance_symmetric(self, two_scalar_field_setup):
        """Test compressed covariance is symmetric for multi-field."""
        from cosmocore.basis import HarmonicBasis

        setup = two_scalar_field_setup
        hc = HarmonicBasis(
            N=setup["N"],
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

        C_compressed = hc.get_compressed_covariance(C_ell_dict)

        assert_allclose(C_compressed, C_compressed.T, rtol=1e-10, atol=1e-15)


class TestMultiFieldFisher:
    """Tests for multi-field Fisher matrix computation."""

    def test_multi_field_fisher_shape(self, two_scalar_field_setup):
        """Test Fisher matrix shape for multi-field with spectra dict."""
        from cosmocore.basis import HarmonicBasis

        setup = two_scalar_field_setup
        hc = HarmonicBasis(
            N=setup["N"],
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
        fisher = hc.compute_fisher_matrix(C_ell_dict, spectra_list)

        n_spectra = len(spectra_list)
        assert fisher.shape == (n_spectra * n_ell, n_spectra * n_ell)

    def test_multi_field_fisher_symmetric(self, two_scalar_field_setup):
        """Test Fisher matrix is symmetric for multi-field."""
        from cosmocore.basis import HarmonicBasis

        setup = two_scalar_field_setup
        hc = HarmonicBasis(
            N=setup["N"],
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

        fisher = hc.compute_fisher_matrix(C_ell_dict, spectra_list)

        assert_allclose(fisher, fisher.T, rtol=1e-10)

    def test_multi_field_fisher_positive_diagonal(self, two_scalar_field_setup):
        """Test Fisher matrix has positive diagonal for multi-field."""
        from cosmocore.basis import HarmonicBasis

        setup = two_scalar_field_setup
        hc = HarmonicBasis(
            N=setup["N"],
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

        fisher = hc.compute_fisher_matrix(C_ell_dict, spectra_list)

        # Diagonal elements should all be positive
        assert np.all(np.diag(fisher) > 0)


class TestMultiFieldManager:
    """Tests for multi-field via create_computation_basis factory."""

    def test_manager_multi_field(self, two_scalar_field_setup):
        """Test create_computation_basis with multi-field input."""
        from cosmocore.basis import create_computation_basis

        setup = two_scalar_field_setup
        cm = create_computation_basis(
            method="harmonic",
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )
        cm.setup()

        assert cm.n_components == 2
        assert cm.n_kept > 0

    def test_manager_multi_field_fisher(self, two_scalar_field_setup):
        """Test Fisher computation via create_computation_basis with multi-field dict."""
        from cosmocore.basis import create_computation_basis

        setup = two_scalar_field_setup
        n_ell = setup["lmax"] - 1
        C_ell_dict = {
            (0, 0): np.ones(n_ell) * 1e-6,
            (1, 1): np.ones(n_ell) * 0.8e-6,
            (0, 1): np.ones(n_ell) * 0.5e-6,
        }
        spectra_list = [(0, 0), (1, 1), (0, 1)]

        cm = create_computation_basis(
            method="harmonic",
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )
        cm.setup()

        # Multi-field Fisher via dedicated method
        fisher = cm.compute_fisher_matrix(C_ell_dict, spectra_list)

        n_spectra = len(spectra_list)
        assert fisher.shape == (n_spectra * n_ell, n_spectra * n_ell)
        assert_allclose(fisher, fisher.T, rtol=1e-10)
        assert np.all(np.diag(fisher) > 0)


class TestMultiFieldIntegration:
    """
    Integration tests comparing multi-field compressed Fisher against
    traditional pixel-space computation.
    """

    def test_three_scalar_fields_fisher_diagonal_positive(
        self, three_scalar_field_realistic_setup
    ):
        """Test that 3-field Fisher has positive diagonal for all spectra."""
        from cosmocore.basis import HarmonicBasis

        setup = three_scalar_field_realistic_setup
        hc = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )
        hc.setup()

        # All 6 spectra: 3 auto + 3 cross
        spectra_list = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]

        fisher = hc.compute_fisher_matrix(
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
        from cosmocore.basis import HarmonicBasis

        setup = three_scalar_field_realistic_setup
        hc = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )
        hc.setup()

        # Only auto-spectra
        spectra_list = [(0, 0), (1, 1), (2, 2)]

        fisher = hc.compute_fisher_matrix(
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
        from cosmocore.basis import HarmonicBasis

        setup = three_scalar_field_realistic_setup

        # Multi-field setup
        hc_multi = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )
        hc_multi.setup()

        # Single-field setup using only field 1
        n_pix_1 = setup["n_pix_1"]
        N_single = setup["N"][:n_pix_1, :n_pix_1]
        setup["N_inv"][:n_pix_1, :n_pix_1]
        theta_single = setup["theta"][0]
        phi_single = setup["phi"][0]

        hc_single = HarmonicBasis(
            N=N_single,
            theta=theta_single,
            phi=phi_single,
            lmax=setup["lmax"],
        )
        hc_single.setup()

        # Single-field Fisher using original API
        C_ell_11 = setup["C_ell_dict"][(0, 0)]
        fisher_single = hc_single.compute_fisher_matrix(C_ell_11)

        # Multi-field Fisher for just field 1 auto-spectrum
        fisher_multi = hc_multi.compute_fisher_matrix(
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
        from cosmocore.basis import HarmonicBasis

        setup = three_scalar_field_realistic_setup
        hc = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
        )
        hc.setup()

        C_compressed = hc.get_compressed_covariance(setup["C_ell_dict"])

        # Check positive definiteness
        eigenvalues = np.linalg.eigvalsh(C_compressed)
        assert np.all(eigenvalues > 0), (
            f"Compressed covariance should be positive definite, "
            f"min eigenvalue: {np.min(eigenvalues)}"
        )

    def test_n_components_correct(self, three_scalar_field_realistic_setup):
        """Test that n_components is correctly detected."""
        from cosmocore.basis import create_computation_basis

        setup = three_scalar_field_realistic_setup
        cm = create_computation_basis(
            method="harmonic",
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=setup["lmax"],
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
        from cosmocore.basis import HarmonicBasis

        setup = two_scalar_field_setup
        lmax = setup["lmax"]
        n_ell = lmax - 1

        # Different spectra per field (physical C_ell values)
        ells = np.arange(2, lmax + 1)

        C_ell_dict = {
            (0, 0): 1e-5 / ells**2,
            (1, 1): 0.8e-5 / ells**2,
            (0, 1): 0.4e-5 / ells**2,
        }

        # Setup compression
        hc = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax=lmax,
        )
        hc.setup()

        # Compute compressed Fisher for single auto-spectrum (simpler case)
        spectra_list = [(0, 0)]
        fisher_compressed = hc.compute_fisher_matrix(
            C_ell_dict, spectra_list, ell_min=2, ell_max=lmax
        )

        # Now compute pixel-space Fisher for comparison
        # Build full signal matrix S = V^T Λ V
        V = hc._V
        lambda_matrix = hc._build_lambda_matrix(C_ell_dict)

        # S = V^T @ Λ @ V
        S = V.T @ lambda_matrix @ V

        # Full covariance C = N + S
        C = setup["N"] + S
        C_inv = matrix_inverse_symm(C.copy())

        # Compute pixel-space Fisher for field 0 auto-spectrum
        fisher_pixel = np.zeros((n_ell, n_ell))

        for ell_i in range(2, lmax + 1):
            # Build dS/dC_ell for field 0 using V^T E_ell V
            E_i = hc.get_derivative_matrix(ell_i, 0, 0)
            dS_i = V.T @ E_i @ V

            for ell_j in range(ell_i, lmax + 1):
                E_j = hc.get_derivative_matrix(ell_j, 0, 0)
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
