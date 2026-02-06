"""
Harmonic compression (Tegmark-like) for CMB Fisher matrix computation.

This module implements direct harmonic space compression where data is
transformed directly to n_modes dimensions via the harmonic operator V.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_solve

from ..basics import (
    add_diagonal,
    cholesky_decomposition,
    matrix_inverse_symm,
    matrix_mult,
    matrix_slogdet_symm,
    matrix_trace,
    smw_inverse,
    smw_kernel,
    smw_logdet,
    smw_quadratic_form,
)
from .base import BaseCompression


class HarmonicCompression(BaseCompression):
    """
    Direct harmonic space compression (Tegmark-like).

    Transforms directly to n_modes dimensions via V. Fast and efficient
    when n_modes << n_pix. No additional eigenvalue compression.

    The key operations are:
    - Data compression: d̄ = P @ d where P = V
    - Compressed covariance: C̄ = V N V^T + Λ
    - Projected inverse: V C^{-1} V^T = M - M K^{-1} M (via SMW)

    Parameters
    ----------
    N : numpy.ndarray
        Noise covariance matrix of shape (n_pix, n_pix).
    N_inv : numpy.ndarray
        Precomputed noise inverse matrix of shape (n_pix, n_pix).
    theta : numpy.ndarray
        Colatitude angles for active pixels in radians.
    phi : numpy.ndarray
        Longitude angles for active pixels in radians.
    lmax : int
        Maximum multipole for harmonic expansion.
    beam : numpy.ndarray or None, optional
        Beam window function B_ℓ for ℓ=2 to lmax.

    Examples
    --------
    >>> import numpy as np
    >>> from cosmocore.compression import HarmonicCompression
    >>> N_inv = np.diag(1.0 / noise_variance)
    >>> hc = HarmonicCompression(N_inv, theta, phi, lmax=100)
    >>> hc.setup()
    >>> fisher_element = hc.compute_fisher_element(C_ell, ell_i=10, ell_j=10)

    References
    ----------
    .. [1] Tegmark, M. "How to measure CMB power spectra without losing information"
       Phys. Rev. D 55, 5895 (1997)
    """

    @property
    def projector(self) -> np.ndarray:
        """
        Get the projection matrix V (n_modes × n_pix).

        Maps pixel space to harmonic mode space.
        """
        return self._V

    @property
    def n_compressed(self) -> int:
        """
        Size of compressed space (n_modes_total for multi-field, n_modes for single).
        """
        return self.n_modes_total if self.n_components > 1 else self.n_modes

    def setup(self) -> None:
        """
        Build V and precompute SMW components.

        Computes:
        - V: harmonic operator (n_modes × n_pix)
        - V N^{-1}: precomputed for SMW
        - V N^{-1} V^T: SMW kernel
        - V N V^T: compressed noise covariance
        - log|N|: for determinant calculations
        - Derivative diagonals: E_ℓ for all ℓ

        When switch optimization is enabled (lswitch_high < lmax):
        - V is built only for ℓ in [lswitch_low, lswitch_high]
        - S_fixed is computed for ℓ > lswitch_high using fiducial C_ell
        - N_eff = N + S_fixed is used instead of N
        This dramatically reduces the SMW dimension.
        """
        # Build harmonic operator for reduced ell range if switch optimization
        self._build_harmonic_operator()
        self._build_ell_mode_mapping()
        self._precompute_derivative_diagonals()

        # Compute effective noise matrix when switch optimization is enabled
        if self._use_switch_optimization:
            self._compute_effective_noise()

        self._compute_smw_components()

    def _compute_effective_noise(self) -> None:
        """
        Compute effective noise N_eff = N + S_fixed for switch optimization.

        S_fixed is the signal matrix contribution from multipoles outside
        the switch range (ℓ > lswitch_high). This contribution is constant
        across all parameter points and can be absorbed into an effective
        noise matrix.

        The SMW formula then applies only to the varying multipoles:
            C = N_eff + S_varied
            (N_eff + S_varied)^{-1} via SMW with reduced V

        This significantly reduces the SMW dimension from
        (lmax+1)² - 4 to (lswitch_high+1)² - 4.

        S_fixed can be provided directly (preferred) or computed from
        fiducial_C_ell (legacy). The direct approach ensures consistency
        with the parent class's signal matrix computation.
        """
        # Prefer S_fixed if provided (computed by Core using existing infrastructure)
        if self._S_fixed is not None:
            S_fixed = np.asarray(self._S_fixed, dtype=np.float64)
        elif self._fiducial_C_ell is not None:
            # Legacy path: compute S_fixed from fiducial_C_ell
            S_fixed = self._compute_s_fixed_from_fiducial()
        else:
            raise ValueError(
                "Either S_fixed or fiducial_C_ell must be provided when using "
                "switch optimization (lswitch_high < lmax)"
            )

        # Compute effective noise and its inverse
        self._N_eff = self._N + S_fixed
        self._N_eff_inv = matrix_inverse_symm(self._N_eff)

        # Replace N and N_inv with N_eff for SMW computations
        self._N_original = self._N
        self._N_inv_original = self.N_inv
        self._N = np.asfortranarray(self._N_eff)
        self.N_inv = np.asfortranarray(self._N_eff_inv)

        # Store log|N_eff| for determinant calculations
        _, self._log_det_N_eff = matrix_slogdet_symm(self._N_eff)

    def _compute_s_fixed_from_fiducial(self) -> np.ndarray:
        """
        Legacy method: compute S_fixed from fiducial_C_ell using compute_00_contribution.

        This is kept for backward compatibility but the preferred approach is to
        compute S_fixed externally using the existing signal matrix infrastructure.
        """
        from ..pixel import compute_00_contribution

        # Create C_ell array with zeros for varied multipoles, fiducial for fixed
        cl_fixed = np.zeros(self.lmax - 1, dtype=np.float64)
        for ell in range(self.lswitch_high + 1, self.lmax + 1):
            if ell - 2 < len(self._fiducial_C_ell):
                cl_fixed[ell - 2] = self._fiducial_C_ell[ell - 2]

        # Build point vectors from theta, phi (shape: n_pix x 3)
        point_vectors = np.column_stack(
            [
                np.sin(self.theta) * np.cos(self.phi),
                np.sin(self.theta) * np.sin(self.phi),
                np.cos(self.theta),
            ]
        )

        # Compute S_fixed using existing signal matrix infrastructure
        S_fixed = np.zeros((self.n_pix, self.n_pix), dtype=np.float64)
        legendre_buffer = np.empty(self.lmax, dtype=np.float64)

        compute_00_contribution(
            cl_fixed,
            S_fixed,
            point_vectors,
            point_vectors,
            legendre_buffer,
            mode=0,  # Symmetric matrix
            remove_dipole=False,
        )

        # Make S_fixed symmetric (compute_00_contribution fills lower triangle)
        S_fixed = S_fixed + S_fixed.T - np.diag(np.diag(S_fixed))

        return S_fixed

    def _compute_smw_components(self) -> None:
        """
        Precompute matrices needed for SMW formula.

        Computes:
            - V @ N^{-1} (used in SMW and for projecting data)
            - V @ N^{-1} @ V^T (SMW kernel matrix)
            - log|N| (for determinant computation)

        Note: V @ N @ V^T is computed lazily when needed (get_compressed_covariance)
        to avoid expensive O(n_pix³) inversion when only Fisher is needed.
        """
        # V @ N^{-1}
        self._V_N_inv = matrix_mult(self._V, self.N_inv)

        # V @ N^{-1} @ V^T (SMW kernel)
        self._V_Ninv_VT = matrix_mult(self._V_N_inv, self._V.T)
        self._V_Ninv_VT = 0.5 * (self._V_Ninv_VT + self._V_Ninv_VT.T)

        # V @ N @ V^T
        self._V_N_VT = matrix_mult(matrix_mult(self._V, self._N), self._V.T)
        self._V_N_VT = 0.5 * (self._V_N_VT + self._V_N_VT.T)

        # log|N| = -log|N^{-1}|
        _, logdet_N_inv = matrix_slogdet_symm(self.N_inv)
        self._log_det_N = -logdet_N_inv

        # Pre-allocate buffers for frequently called methods
        self._allocate_buffers()

    def _allocate_buffers(self) -> None:
        """
        Pre-allocate reusable buffers for intermediate computations.

        This reduces memory allocation overhead in frequently called methods
        like get_projected_inverse and compute_fisher_matrix.
        """
        # For multi-field, use n_modes_total; for single-field, use n_modes
        buffer_size = self.n_modes_total if self.n_components > 1 else self.n_modes

        # Buffer for K matrix in SMW: (buffer_size, buffer_size)
        self._K_buffer = np.empty((buffer_size, buffer_size), dtype=np.float64, order="F")
        # Buffer for M @ K_inv @ M product: (buffer_size, buffer_size)
        self._MKM_buffer = np.empty(
            (buffer_size, buffer_size), dtype=np.float64, order="F"
        )

    def get_projected_inverse(self, C_ell: np.ndarray) -> np.ndarray:
        """
        Compute V C^{-1} V^T efficiently using SMW formula.

        From the SMW formula, V C^{-1} V^T can be computed as:
            V C^{-1} V^T = M - M K^{-1} M
        where:
            M = V N^{-1} V^T (precomputed)
            K = Λ^{-1} + M

        This is O(n_modes³) instead of O(n_pix³).

        Parameters
        ----------
        C_ell : numpy.ndarray
            Power spectrum values for ell = 2 to lmax.

        Returns
        -------
        numpy.ndarray
            Projected inverse covariance V C^{-1} V^T of shape (n_modes, n_modes).
        """
        # Get Λ^{-1} diagonal
        Lambda_diag = self._build_lambda_diagonal(C_ell)
        Lambda_inv_diag = np.where(Lambda_diag > 1e-30, 1.0 / Lambda_diag, 1e30)

        # K = Λ^{-1} + M using pre-allocated buffer
        add_diagonal(self._V_Ninv_VT, Lambda_inv_diag, out=self._K_buffer)

        # K^{-1} using optimized symmetric inverse (K is positive definite)
        # Buffer is already Fortran-ordered
        K_inv = matrix_inverse_symm(self._K_buffer, overwrite=True)

        # V C^{-1} V^T = M - M K^{-1} M
        return self._V_Ninv_VT - matrix_mult(
            matrix_mult(self._V_Ninv_VT, K_inv), self._V_Ninv_VT
        )

    def get_derivative_matrix(self, ell: int) -> np.ndarray:
        """
        Get the derivative matrix ∂S/∂C_ℓ in compressed form.

        The traditional pixel-space derivative uses:
            ∂S/∂C_ℓ = (2ℓ+1)/(4π) × P_ℓ(cos γ)

        Since V^T E_ℓ V = P_ℓ(cos γ) (from the addition theorem), we need:
            E_ℓ = (2ℓ+1)/(4π) × diag(1s for modes at ℓ)

        This matches the traditional derivative from do_derivative_step.

        Uses precomputed diagonals from _precompute_derivative_diagonals().

        Parameters
        ----------
        ell : int
            Multipole for which to compute the derivative.

        Returns
        -------
        numpy.ndarray
            Derivative matrix of shape (n_modes, n_modes).
        """
        # Use precomputed diagonal to build the matrix
        return np.diag(self._derivative_diagonals[ell])

    def compute_fisher_matrix(
        self,
        C_ell: np.ndarray,
        ell_min: int = 2,
        ell_max: int | None = None,
    ) -> np.ndarray:
        """
        Compute full Fisher matrix with optimized diagonal derivative structure.

        This override exploits the fact that for HarmonicCompression, the
        derivative matrices E_ℓ are diagonal. This allows using fast column
        scaling instead of full matrix multiplication.

        The Fisher matrix element is:
            F_ij = (1/2) Tr[(V C^{-1} V^T) E_i (V C^{-1} V^T) E_j]

        Since E_ℓ is diagonal, (V C^{-1} V^T) @ E_ℓ = V_Cinv_VT * E_diag
        (column scaling by broadcasting).

        Parameters
        ----------
        C_ell : numpy.ndarray
            Power spectrum values for ell = 2 to lmax.
        ell_min : int, default 2
            Minimum multipole to include in Fisher matrix.
        ell_max : int or None, optional
            Maximum multipole. If None, uses self.lmax.

        Returns
        -------
        numpy.ndarray
            Fisher matrix of shape (n_ell, n_ell).
        """
        if ell_max is None:
            ell_max = self.lmax

        n_ell = ell_max - ell_min + 1
        fisher = np.zeros((n_ell, n_ell))

        # Precompute V C^{-1} V^T ONCE
        V_Cinv_VT = self.get_projected_inverse(C_ell)

        # Exploit diagonal structure: (V C^{-1} V^T) @ diag(E) = V_Cinv_VT * E
        # This is O(n²) instead of O(n³) for each ell
        VCinvVT_E = {}
        for ell in range(ell_min, ell_max + 1):
            E_diag = self._derivative_diagonals[ell]
            # Column scaling by broadcasting: (n_modes, n_modes) * (n_modes,)
            VCinvVT_E[ell] = V_Cinv_VT * E_diag

        # Compute Fisher elements using precomputed products
        # Use optimized matrix_trace which is O(n²) instead of
        # np.trace(A @ B) which is O(n³)
        for ell_i in range(ell_min, ell_max + 1):
            for ell_j in range(ell_i, ell_max + 1):
                # F_ij = 0.5 * Tr[A_i @ A_j] where A_k = V_Cinv_VT * E_k

                idx_i = ell_i - ell_min
                idx_j = ell_j - ell_min

                fisher_val = 0.5 * matrix_trace(VCinvVT_E[ell_i], VCinvVT_E[ell_j])
                fisher[idx_i, idx_j] = fisher_val
                if idx_i != idx_j:
                    fisher[idx_j, idx_i] = fisher_val  # Symmetry

        return fisher

    def get_compressed_covariance(self, C_ell: np.ndarray) -> np.ndarray:
        """
        Compute compressed covariance C̄ = V N V^T + Λ.

        Parameters
        ----------
        C_ell : numpy.ndarray
            Power spectrum values C_ell[ell] for ell = 2 to lmax.

        Returns
        -------
        numpy.ndarray
            Compressed covariance matrix of shape (n_modes, n_modes).
        """
        Lambda_diag = self._build_lambda_diagonal(C_ell)
        # C̄ = V N V^T + Λ (avoid creating full diagonal matrix)
        return add_diagonal(self._V_N_VT, Lambda_diag)

    def get_compressed_covariance_multi(
        self, C_ell_dict: dict[tuple[int, int], np.ndarray]
    ) -> np.ndarray:
        """
        Compute compressed covariance C̄ = V N V^T + Λ for multi-field.

        Parameters
        ----------
        C_ell_dict : dict
            Dictionary mapping (comp_i, comp_j) to C_ell array for each
            auto-spectrum and cross-spectrum. Keys are (i, j) tuples with
            i <= j for unique spectra.

        Returns
        -------
        numpy.ndarray
            Compressed covariance matrix of shape (n_modes_total, n_modes_total).
        """
        if self.n_components == 1:
            # Fall back to single-field for backward compatibility
            C_ell = C_ell_dict.get((0, 0), next(iter(C_ell_dict.values())))
            return self.get_compressed_covariance(C_ell)

        # Build full Lambda matrix with cross-spectra blocks
        Lambda_full = self._build_lambda_full(C_ell_dict)

        # C̄ = V N V^T + Λ
        return self._V_N_VT + Lambda_full

    # === Full pixel-space operations (if needed) ===

    def get_inverse(self, C_ell: np.ndarray) -> np.ndarray:
        """
        Compute full (N + S)^{-1} using SMW formula.

        Parameters
        ----------
        C_ell : numpy.ndarray
            Power spectrum values for ell = 2 to lmax.

        Returns
        -------
        numpy.ndarray
            Full inverse covariance matrix of shape (n_pix, n_pix).
        """
        Lambda_diag = self._build_lambda_diagonal(C_ell)
        return smw_inverse(self.N_inv, self._V_N_inv, self._V_Ninv_VT, Lambda_diag)

    def get_logdet(self, C_ell: np.ndarray) -> float:
        """
        Compute log|N + S| using SMW formula.

        Parameters
        ----------
        C_ell : numpy.ndarray
            Power spectrum values for ell = 2 to lmax.

        Returns
        -------
        float
            Log determinant of the full covariance matrix.
        """
        Lambda_diag = self._build_lambda_diagonal(C_ell)
        return smw_logdet(self._log_det_N, self._V_Ninv_VT, Lambda_diag)

    def get_weighted_compressed_data(
        self, data: np.ndarray, C_ell: np.ndarray, C_c_inv: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Compute V @ C^{-1} @ d for QML estimation in compressed space.

        This computes the weighted compressed data vector needed for QML:
            w = V @ C^{-1} @ d

        Using the SMW formula for C^{-1}:
            C^{-1} = N^{-1} - N^{-1} V^T K^{-1} V N^{-1}
        where K = Λ^{-1} + V N^{-1} V^T.

        Therefore:
            V @ C^{-1} @ d = V N^{-1} d - (V N^{-1} V^T) K^{-1} (V N^{-1} d)

        Parameters
        ----------
        data : numpy.ndarray
            Pixel-space data vector of length n_pix.
        C_ell : numpy.ndarray
            Power spectrum values for ell = 2 to lmax.
        C_c_inv : numpy.ndarray, optional
            Unused for harmonic compression. Provided for interface
            compatibility with pixel_projected compression.

        Returns
        -------
        numpy.ndarray
            Weighted compressed data vector w = V C^{-1} d of length n_modes.
        """
        # Note: C_c_inv is unused for harmonic compression - we use SMW formula
        del C_c_inv  # Silence unused parameter warning
        # y = V N^{-1} d
        y = self._V_N_inv @ data

        # Build K = Λ^{-1} + V N^{-1} V^T
        Lambda_diag = self._build_lambda_diagonal(C_ell)
        K = smw_kernel(self._V_Ninv_VT, Lambda_diag)

        # (V N^{-1} V^T) @ K^{-1} @ y = M @ K^{-1} @ y
        L = cholesky_decomposition(K)
        K_inv_y = cho_solve((L, True), y)

        M_K_inv_y = matrix_mult(self._V_Ninv_VT, K_inv_y)

        # w = y - M @ K^{-1} @ y = V C^{-1} d
        return y - M_K_inv_y

    def compute_quadratic_form(self, data: np.ndarray, C_ell: np.ndarray) -> float:
        """
        Compute d^T C^{-1} d efficiently using SMW formula.

        Parameters
        ----------
        data : numpy.ndarray
            Pixel-space data vector of length n_pix.
        C_ell : numpy.ndarray
            Power spectrum values for ell = 2 to lmax.

        Returns
        -------
        float
            Quadratic form value d^T C^{-1} d.
        """
        Lambda_diag = self._build_lambda_diagonal(C_ell)
        return smw_quadratic_form(
            data, self.N_inv, self._V_N_inv, self._V_Ninv_VT, Lambda_diag
        )

    # === Multi-field operations ===

    def get_projected_inverse_multi(
        self, C_ell_dict: dict[tuple[int, int], np.ndarray]
    ) -> np.ndarray:
        """
        Compute V C^{-1} V^T for multi-field using block SMW formula.

        Parameters
        ----------
        C_ell_dict : dict
            Dictionary mapping (comp_i, comp_j) to C_ell array for cross-spectra.

        Returns
        -------
        numpy.ndarray
            Projected inverse covariance of shape (n_modes_total, n_modes_total).
        """
        if self.n_components == 1:
            # Fall back to single-field for backward compatibility
            return self.get_projected_inverse(C_ell_dict.get((0, 0), C_ell_dict))

        # Build full Lambda from cross-spectra
        # Note: _build_lambda_full already creates a symmetric matrix
        # (both (i,j) and (j,i) blocks are filled by _build_lambda_blocks)
        Lambda_full = self._build_lambda_full(C_ell_dict)

        # Build Lambda^{-1} - use pseudo-inverse for robustness
        # Small regularization to handle near-singular cases
        Lambda_reg = Lambda_full + np.eye(Lambda_full.shape[0]) * 1e-20
        try:
            Lambda_inv = np.linalg.inv(Lambda_reg)
        except np.linalg.LinAlgError:
            # Fall back to pseudo-inverse
            Lambda_inv = np.linalg.pinv(Lambda_full)

        # K = Λ^{-1} + M where M = V N^{-1} V^T
        K = Lambda_inv + self._V_Ninv_VT

        # K^{-1} - use standard inverse for robustness
        K_inv = np.linalg.inv(K)

        # V C^{-1} V^T = M - M K^{-1} M
        return self._V_Ninv_VT - matrix_mult(
            matrix_mult(self._V_Ninv_VT, K_inv), self._V_Ninv_VT
        )

    def get_weighted_compressed_data_multi(
        self, data: np.ndarray, C_ell_dict: dict[tuple[int, int], np.ndarray]
    ) -> np.ndarray:
        """
        Compute V @ C^{-1} @ d for multi-field QML estimation.

        This computes the weighted compressed data vector for multi-field:
            w = V @ C^{-1} @ d

        Using the SMW formula with full Lambda matrix for cross-spectra.

        Parameters
        ----------
        data : numpy.ndarray
            Pixel-space data vector of length n_pix_total.
        C_ell_dict : dict
            Dictionary mapping (comp_i, comp_j) to C_ell array for cross-spectra.

        Returns
        -------
        numpy.ndarray
            Weighted compressed data vector w = V C^{-1} d of length n_modes_total.
        """
        if self.n_components == 1:
            # Fall back to single-field
            C_ell = C_ell_dict.get((0, 0), list(C_ell_dict.values())[0])
            return self.get_weighted_compressed_data(data, C_ell)

        # y = V N^{-1} d
        y = self._V_N_inv @ data

        # Build full Lambda from cross-spectra
        Lambda_full = self._build_lambda_full(C_ell_dict)

        # Build Lambda^{-1}
        Lambda_reg = Lambda_full + np.eye(Lambda_full.shape[0]) * 1e-20
        try:
            Lambda_inv = np.linalg.inv(Lambda_reg)
        except np.linalg.LinAlgError:
            Lambda_inv = np.linalg.pinv(Lambda_full)

        # K = Λ^{-1} + M where M = V N^{-1} V^T
        K = Lambda_inv + self._V_Ninv_VT

        # K^{-1} @ y
        K_inv_y = np.linalg.solve(K, y)

        # w = y - M @ K^{-1} @ y = V C^{-1} d
        M_K_inv_y = matrix_mult(self._V_Ninv_VT, K_inv_y)
        return y - M_K_inv_y

    def get_derivative_matrix_multi(
        self, ell: int, comp_i: int, comp_j: int
    ) -> np.ndarray:
        """
        Get derivative matrix for spectrum (comp_i, comp_j) at multipole ell.

        For multi-field, the derivative E_ℓ^{ij} is a block matrix with
        non-zero entries only in blocks (i,j) and (j,i).

        Parameters
        ----------
        ell : int
            Multipole for which to compute the derivative.
        comp_i : int
            First component index.
        comp_j : int
            Second component index.

        Returns
        -------
        numpy.ndarray
            Derivative matrix of shape (n_modes_total, n_modes_total).
        """
        if self.n_components == 1:
            # Fall back to single-field
            return self.get_derivative_matrix(ell)

        E = np.zeros((self.n_modes_total, self.n_modes_total), dtype=np.float64)

        chngconv = (2 * ell + 1) / (4 * np.pi)
        local_mode_indices = self._ell_to_modes_local[ell]

        # Block (comp_i, comp_j)
        row_offset = self._mode_offsets[comp_i]
        col_offset = self._mode_offsets[comp_j]
        for local_idx in local_mode_indices:
            E[row_offset + local_idx, col_offset + local_idx] = chngconv

        # Block (comp_j, comp_i) if i != j (symmetric)
        if comp_i != comp_j:
            for local_idx in local_mode_indices:
                E[col_offset + local_idx, row_offset + local_idx] = chngconv

        return E

    def get_derivative_matrix_with_spins(
        self, ell: int, comp_i: int, comp_j: int, mode: int = 0
    ) -> np.ndarray:
        """
        Get derivative matrix for spectrum (comp_i, comp_j, mode) at multipole ell.

        Handles spin-2 E/B sub-block structure. The mode parameter selects
        which sub-spectrum:
        - spin-0 x spin-0: mode 0 (TT)
        - spin-2 x spin-2: mode 0 (EE), 1 (BB), 2 (EB)
        - spin-0 x spin-2: mode 0 (TE), 1 (TB)

        Parameters
        ----------
        ell : int
            Multipole for which to compute the derivative.
        comp_i : int
            First component index.
        comp_j : int
            Second component index.
        mode : int, default 0
            Sub-spectrum mode index.

        Returns
        -------
        numpy.ndarray
            Derivative matrix of shape (n_modes_total, n_modes_total).
        """
        spin_i = self._spins[comp_i]
        spin_j = self._spins[comp_j]

        # For pure spin-0, delegate to existing method
        if spin_i == 0 and spin_j == 0:
            return self.get_derivative_matrix_multi(ell, comp_i, comp_j)

        E = np.zeros((self.n_modes_total, self.n_modes_total), dtype=np.float64)
        chngconv = (2 * ell + 1) / (4 * np.pi)
        local_mode_indices = self._ell_to_modes_local[ell]
        n_base = self._n_modes_base

        # Spin-dependent normalization factors matching pixel.py convention.
        # The pipeline's apply_normalization() applies these factors to C_ell,
        # so derivatives must include them too for consistency with the
        # traditional pixel-space derivative (do_derivative_step).
        factor2 = 1.0 / ((ell + 2) * (ell + 1) * ell * (ell - 1))
        factor = np.sqrt(factor2)

        if spin_i == 2 and spin_j == 2:
            # Spin-2 x Spin-2: EE (mode 0), BB (mode 1), EB (mode 2)
            deriv_val = chngconv * factor2
            row_start = self._mode_offsets[comp_i]
            col_start = self._mode_offsets[comp_j]

            if mode == 0:  # EE
                for idx in local_mode_indices:
                    E[row_start + idx, col_start + idx] = deriv_val
            elif mode == 1:  # BB
                for idx in local_mode_indices:
                    E[row_start + n_base + idx, col_start + n_base + idx] = deriv_val
            elif mode == 2:  # EB
                for idx in local_mode_indices:
                    E[row_start + idx, col_start + n_base + idx] = deriv_val
                    E[col_start + n_base + idx, row_start + idx] = deriv_val

            # Symmetric blocks for comp_i != comp_j
            if comp_i != comp_j:
                if mode == 0:  # EE
                    for idx in local_mode_indices:
                        E[col_start + idx, row_start + idx] = deriv_val
                elif mode == 1:  # BB
                    for idx in local_mode_indices:
                        E[col_start + n_base + idx, row_start + n_base + idx] = deriv_val
                elif mode == 2:  # EB
                    for idx in local_mode_indices:
                        E[col_start + idx, row_start + n_base + idx] = deriv_val
                        E[row_start + n_base + idx, col_start + idx] = deriv_val

        elif spin_i == 0 and spin_j == 2:
            # Scalar x Spin-2: TE (mode 0), TB (mode 1)
            # Negative sign matches the spin-2 convention: E = -(_2Y + _{-2}Y)/2.
            # See compute_02_contribution / derivative_step_02 in pixel.py.
            deriv_val = -chngconv * factor
            row_start = self._mode_offsets[comp_i]
            col_start = self._mode_offsets[comp_j]
            # mode 0 → E sub-block, mode 1 → B sub-block
            col_sub = col_start + mode * n_base

            for idx in local_mode_indices:
                E[row_start + idx, col_sub + idx] = deriv_val
                E[col_sub + idx, row_start + idx] = deriv_val

        elif spin_i == 2 and spin_j == 0:
            # Spin-2 x Scalar: transpose of above
            deriv_val = -chngconv * factor
            row_start = self._mode_offsets[comp_i]
            col_start = self._mode_offsets[comp_j]
            row_sub = row_start + mode * n_base

            for idx in local_mode_indices:
                E[row_sub + idx, col_start + idx] = deriv_val
                E[col_start + idx, row_sub + idx] = deriv_val

        return E

    def get_projected_inverse_with_spins(
        self, C_ell_dict: dict[tuple, np.ndarray]
    ) -> np.ndarray:
        """
        Compute V C^{-1} V^T for multi-field with spin-2 using SMW formula.

        Uses 3-tuple keys (comp_i, comp_j, mode) in C_ell_dict.

        Parameters
        ----------
        C_ell_dict : dict
            Dictionary with 3-tuple keys (comp_i, comp_j, mode).

        Returns
        -------
        numpy.ndarray
            Projected inverse covariance of shape (n_modes_total, n_modes_total).
        """
        Lambda_full = self._build_lambda_full_with_spins(C_ell_dict)

        # Build Lambda^{-1} with regularization
        Lambda_reg = Lambda_full + np.eye(Lambda_full.shape[0]) * 1e-20
        try:
            Lambda_inv = np.linalg.inv(Lambda_reg)
        except np.linalg.LinAlgError:
            Lambda_inv = np.linalg.pinv(Lambda_full)

        # K = Λ^{-1} + M where M = V N^{-1} V^T
        K = Lambda_inv + self._V_Ninv_VT

        K_inv = np.linalg.inv(K)

        # V C^{-1} V^T = M - M K^{-1} M
        return self._V_Ninv_VT - matrix_mult(
            matrix_mult(self._V_Ninv_VT, K_inv), self._V_Ninv_VT
        )

    def compute_fisher_matrix_with_spins(
        self,
        C_ell_dict: dict[tuple, np.ndarray],
        spectra_list: list[tuple[int, int, int]],
        ell_min: int = 2,
        ell_max: int | None = None,
    ) -> np.ndarray:
        """
        Compute Fisher matrix for multiple spectra with spin-2 support.

        Uses 3-tuple spectra_list entries (comp_i, comp_j, mode) and
        3-tuple C_ell_dict keys.

        Parameters
        ----------
        C_ell_dict : dict
            Dictionary with 3-tuple keys (comp_i, comp_j, mode).
        spectra_list : list of 3-tuple
            List of (comp_i, comp_j, mode) specifying which spectra to include.
        ell_min : int, default 2
            Minimum multipole.
        ell_max : int or None, optional
            Maximum multipole. If None, uses self.lmax.

        Returns
        -------
        numpy.ndarray
            Fisher matrix of shape (n_spectra * n_ell, n_spectra * n_ell).
        """
        if ell_max is None:
            ell_max = self.lmax

        n_ell = ell_max - ell_min + 1
        n_spec = len(spectra_list)
        fisher = np.zeros((n_spec * n_ell, n_spec * n_ell))

        # Precompute V C^{-1} V^T ONCE
        V_Cinv_VT = self.get_projected_inverse_with_spins(C_ell_dict)

        # Precompute (V C^{-1} V^T) @ E for all (spectrum, ell) pairs
        VCinvVT_E = {}
        for spec_idx, (comp_i, comp_j, mode) in enumerate(spectra_list):
            for ell in range(ell_min, ell_max + 1):
                E_matrix = self.get_derivative_matrix_with_spins(
                    ell, comp_i, comp_j, mode
                )
                VCinvVT_E[(spec_idx, ell)] = matrix_mult(V_Cinv_VT, E_matrix)

        # Compute Fisher elements
        for spec_a in range(n_spec):
            for ell_a in range(ell_min, ell_max + 1):
                idx_a = spec_a * n_ell + (ell_a - ell_min)

                for spec_b in range(spec_a, n_spec):
                    ell_b_start = ell_a if spec_a == spec_b else ell_min
                    for ell_b in range(ell_b_start, ell_max + 1):
                        idx_b = spec_b * n_ell + (ell_b - ell_min)

                        fisher_val = 0.5 * matrix_trace(
                            VCinvVT_E[(spec_a, ell_a)],
                            VCinvVT_E[(spec_b, ell_b)],
                        )

                        fisher[idx_a, idx_b] = fisher_val
                        if idx_a != idx_b:
                            fisher[idx_b, idx_a] = fisher_val

        return fisher

    def get_compressed_covariance_with_spins(
        self, C_ell_dict: dict[tuple, np.ndarray]
    ) -> np.ndarray:
        """
        Compute compressed covariance C̄ = V N V^T + Λ for spin-2.

        Parameters
        ----------
        C_ell_dict : dict
            Dictionary with 3-tuple keys (comp_i, comp_j, mode).

        Returns
        -------
        numpy.ndarray
            Compressed covariance of shape (n_modes_total, n_modes_total).
        """
        Lambda_full = self._build_lambda_full_with_spins(C_ell_dict)
        return self._V_N_VT + Lambda_full

    def get_compressed_inverse_with_spins(
        self, C_ell_dict: dict[tuple, np.ndarray]
    ) -> np.ndarray:
        """
        Compute inverse of compressed covariance with spin-2 support.

        Parameters
        ----------
        C_ell_dict : dict
            Dictionary with 3-tuple keys (comp_i, comp_j, mode).

        Returns
        -------
        numpy.ndarray
            Inverse compressed covariance of shape (n_modes_total, n_modes_total).
        """
        C_compressed = self.get_compressed_covariance_with_spins(C_ell_dict)
        return matrix_inverse_symm(C_compressed)

    def get_weighted_compressed_data_with_spins(
        self, data: np.ndarray, C_ell_dict: dict[tuple, np.ndarray]
    ) -> np.ndarray:
        """
        Compute V @ C^{-1} @ d for spin-2 multi-field QML.

        Parameters
        ----------
        data : numpy.ndarray
            Pixel-space data vector of length n_pix_total.
        C_ell_dict : dict
            Dictionary with 3-tuple keys (comp_i, comp_j, mode).

        Returns
        -------
        numpy.ndarray
            Weighted compressed data vector of length n_modes_total.
        """
        y = self._V_N_inv @ data

        Lambda_full = self._build_lambda_full_with_spins(C_ell_dict)

        Lambda_reg = Lambda_full + np.eye(Lambda_full.shape[0]) * 1e-20
        try:
            Lambda_inv = np.linalg.inv(Lambda_reg)
        except np.linalg.LinAlgError:
            Lambda_inv = np.linalg.pinv(Lambda_full)

        K = Lambda_inv + self._V_Ninv_VT
        K_inv_y = np.linalg.solve(K, y)
        M_K_inv_y = matrix_mult(self._V_Ninv_VT, K_inv_y)

        return y - M_K_inv_y

    def prepare_smw_with_spins(
        self, C_ell_dict: dict[tuple, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Precompute K Cholesky factor and log determinant terms for a C_ell_dict.

        This is the expensive step (Lambda build + inverse + Cholesky).
        Call once per parameter point, then reuse for all sims via
        quadratic_form_from_prepared and logdet_from_prepared.

        Parameters
        ----------
        C_ell_dict : dict
            Dictionary with 3-tuple keys (comp_i, comp_j, mode).

        Returns
        -------
        K_chol : numpy.ndarray
            Lower Cholesky factor of K = Lambda^{-1} + V N^{-1} V^T.
        d_Ninv_d_coeff : None
            Reserved (unused, kept for API symmetry).
        logdet : float
            log|N + V^T Lambda V| = log|N| + log|Lambda| + log|K|.
        """
        Lambda_full = self._build_lambda_full_with_spins(C_ell_dict)

        # log|Lambda|
        _, log_det_Lambda = matrix_slogdet_symm(Lambda_full)

        # Lambda^{-1}
        Lambda_reg = Lambda_full + np.eye(Lambda_full.shape[0]) * 1e-20
        Lambda_inv = np.linalg.inv(Lambda_reg)

        # K = Lambda^{-1} + V N^{-1} V^T
        K = Lambda_inv + self._V_Ninv_VT

        # Cholesky factorization of K (reusable for solve)
        K_chol = cholesky_decomposition(K)

        # log|K| from Cholesky: log|K| = 2 * sum(log(diag(L)))
        log_det_K = 2.0 * np.sum(np.log(np.diag(K_chol)))

        logdet = self._log_det_N + log_det_Lambda + log_det_K

        return K_chol, None, logdet

    def quadratic_form_from_prepared(self, data: np.ndarray, K_chol: np.ndarray) -> float:
        """
        Compute d^T C^{-1} d using precomputed K Cholesky factor.

        Parameters
        ----------
        data : numpy.ndarray
            Pixel-space data vector of length n_pix_total.
        K_chol : numpy.ndarray
            Lower Cholesky factor of K from prepare_smw_with_spins.

        Returns
        -------
        float
            Quadratic form value d^T C^{-1} d.
        """
        term1 = float(data.T @ self.N_inv @ data)
        y = self._V_N_inv @ data
        K_inv_y = cho_solve((K_chol, True), y)
        term2 = float(y.T @ K_inv_y)
        return term1 - term2

    def compute_quadratic_form_with_spins(
        self, data: np.ndarray, C_ell_dict: dict[tuple, np.ndarray]
    ) -> float:
        """
        Compute d^T C^{-1} d using SMW formula with full Lambda matrix.

        For multiple sims at the same C_ell_dict, prefer using
        prepare_smw_with_spins + quadratic_form_from_prepared instead.

        Parameters
        ----------
        data : numpy.ndarray
            Pixel-space data vector of length n_pix_total.
        C_ell_dict : dict
            Dictionary with 3-tuple keys (comp_i, comp_j, mode).

        Returns
        -------
        float
            Quadratic form value d^T C^{-1} d.
        """
        K_chol, _, _ = self.prepare_smw_with_spins(C_ell_dict)
        return self.quadratic_form_from_prepared(data, K_chol)

    def get_logdet_with_spins(self, C_ell_dict: dict[tuple, np.ndarray]) -> float:
        """
        Compute log|N + V^T Lambda V| using SMW formula with full Lambda matrix.

        For combined logdet + quadratic form, prefer using
        prepare_smw_with_spins which returns logdet directly.

        Parameters
        ----------
        C_ell_dict : dict
            Dictionary with 3-tuple keys (comp_i, comp_j, mode).

        Returns
        -------
        float
            Log determinant of the full covariance matrix.
        """
        _, _, logdet = self.prepare_smw_with_spins(C_ell_dict)
        return logdet

    def _get_derivative_diagonal_multi(
        self, ell: int, comp_i: int, comp_j: int
    ) -> np.ndarray:
        """
        Get derivative diagonal for spectrum (comp_i, comp_j) at multipole ell.

        Returns the diagonal of E_ℓ^{ij} for efficient matrix operations.

        Parameters
        ----------
        ell : int
            Multipole.
        comp_i : int
            First component index.
        comp_j : int
            Second component index.

        Returns
        -------
        numpy.ndarray
            Diagonal of derivative matrix, shape (n_modes_total,).
        """
        if self.n_components == 1:
            return self._derivative_diagonals[ell]

        E_diag = np.zeros(self.n_modes_total, dtype=np.float64)

        chngconv = (2 * ell + 1) / (4 * np.pi)
        local_mode_indices = self._ell_to_modes_local[ell]

        # For auto-spectrum (i == j): diagonal block
        if comp_i == comp_j:
            row_offset = self._mode_offsets[comp_i]
            for local_idx in local_mode_indices:
                E_diag[row_offset + local_idx] = chngconv
        # For cross-spectrum: off-diagonal blocks (not supported in diagonal form)
        # Cross-spectrum derivatives are not purely diagonal in the full matrix

        return E_diag

    def compute_fisher_matrix_multi(
        self,
        C_ell_dict: dict[tuple[int, int], np.ndarray],
        spectra_list: list[tuple[int, int]],
        ell_min: int = 2,
        ell_max: int | None = None,
    ) -> np.ndarray:
        """
        Compute Fisher matrix for multiple spectra (auto and cross).

        Parameters
        ----------
        C_ell_dict : dict
            Dictionary mapping (comp_i, comp_j) to C_ell array.
        spectra_list : list of tuple
            List of (comp_i, comp_j) tuples specifying which spectra to include.
        ell_min : int, default 2
            Minimum multipole.
        ell_max : int or None, optional
            Maximum multipole. If None, uses self.lmax.

        Returns
        -------
        numpy.ndarray
            Fisher matrix of shape (n_spectra * n_ell, n_spectra * n_ell).
            Ordering: spectrum-major (all ells for spectrum 0, then spectrum 1, ...).
        """
        if self.n_components == 1 and len(spectra_list) == 1:
            # Fall back to single-field
            C_ell = C_ell_dict.get((0, 0), list(C_ell_dict.values())[0])
            return self.compute_fisher_matrix(C_ell, ell_min, ell_max)

        if ell_max is None:
            ell_max = self.lmax

        n_ell = ell_max - ell_min + 1
        n_spec = len(spectra_list)
        fisher = np.zeros((n_spec * n_ell, n_spec * n_ell))

        # Precompute V C^{-1} V^T ONCE
        V_Cinv_VT = self.get_projected_inverse_multi(C_ell_dict)

        # Precompute (V C^{-1} V^T) @ E for all (spectrum, ell) pairs
        VCinvVT_E = {}
        for spec_idx, (comp_i, comp_j) in enumerate(spectra_list):
            for ell in range(ell_min, ell_max + 1):
                E_matrix = self.get_derivative_matrix_multi(ell, comp_i, comp_j)
                VCinvVT_E[(spec_idx, ell)] = matrix_mult(V_Cinv_VT, E_matrix)

        # Compute Fisher elements
        for spec_a in range(n_spec):
            for ell_a in range(ell_min, ell_max + 1):
                idx_a = spec_a * n_ell + (ell_a - ell_min)

                for spec_b in range(spec_a, n_spec):
                    ell_b_start = ell_a if spec_a == spec_b else ell_min
                    for ell_b in range(ell_b_start, ell_max + 1):
                        idx_b = spec_b * n_ell + (ell_b - ell_min)

                        fisher_val = 0.5 * matrix_trace(
                            VCinvVT_E[(spec_a, ell_a)], VCinvVT_E[(spec_b, ell_b)]
                        )

                        fisher[idx_a, idx_b] = fisher_val
                        if idx_a != idx_b:
                            fisher[idx_b, idx_a] = fisher_val

        return fisher
