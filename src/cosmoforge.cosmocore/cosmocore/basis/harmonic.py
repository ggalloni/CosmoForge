"""
Harmonic computation basis (Tegmark-like) for CMB Fisher matrix computation.

This module implements direct harmonic space projection where data is
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
from .base import (
    _LAMBDA_INV_CAP,
    _LAMBDA_ZERO_THRESHOLD,
    _MATRIX_REGULARIZATION,
    ComputationBasis,
    SMWPrepared,
)


class HarmonicBasis(ComputationBasis):
    """
    Direct harmonic space computation basis (Tegmark-like).

    Transforms directly to n_modes dimensions via V. Fast and efficient
    when n_modes << n_pix. No additional eigenvalue truncation.

    The key operations are:
    - Data projection: d̄ = P @ d where P = V
    - Projected covariance: C̄ = V N V^T + Λ
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
    compress : bool, optional
        If True, use m-block compression for K inversion.
        Approximates K as block-diagonal in azimuthal quantum number |m|,
        giving ~lmax^2 speedup. Default is False.
    delta_m : int, optional
        Bandwidth for m-block coupling. delta_m=0 means block-diagonal
        (no coupling between different |m|). delta_m=lmax recovers exact
        result. Default is 0.

    Examples
    --------
    >>> import numpy as np
    >>> from cosmocore.basis import HarmonicBasis
    >>> N_inv = np.diag(1.0 / noise_variance)
    >>> hc = HarmonicBasis(N_inv, theta, phi, lmax=100)
    >>> hc.setup()
    >>> fisher_element = hc.compute_fisher_element(C_ell, ell_i=10, ell_j=10)

    References
    ----------
    .. [1] Tegmark, M. "How to measure CMB power spectra without losing information"
       Phys. Rev. D 55, 5895 (1997)
    """

    def __init__(
        self,
        N: np.ndarray,
        N_inv: np.ndarray,
        theta: np.ndarray | tuple[np.ndarray, ...],
        phi: np.ndarray | tuple[np.ndarray, ...],
        lmax: int,
        beam: np.ndarray | None = None,
        spins: list[int] | None = None,
        lswitch_low: int | None = None,
        lswitch_high: int | None = None,
        fiducial_C_ell: np.ndarray | None = None,
        S_fixed: np.ndarray | None = None,
        compress: bool = False,
        delta_m: int = 0,
    ):
        super().__init__(
            N=N,
            N_inv=N_inv,
            theta=theta,
            phi=phi,
            lmax=lmax,
            beam=beam,
            spins=spins,
            lswitch_low=lswitch_low,
            lswitch_high=lswitch_high,
            fiducial_C_ell=fiducial_C_ell,
            S_fixed=S_fixed,
        )
        self._compress = compress
        self._delta_m = delta_m

        if self._compress:
            # Multi-field + compress not yet supported
            if self.n_components > 1:
                raise NotImplementedError(
                    "m-block compression is only supported for single-field "
                    "spin-0 configurations. Multi-field support will be added later."
                )
            if any(s != 0 for s in self._spins):
                raise NotImplementedError(
                    "m-block compression is only supported for spin-0 fields."
                )

    @property
    def method(self) -> str:
        """Computation basis name."""
        return "harmonic"

    @property
    def projector(self) -> np.ndarray:
        """Projection matrix V (n_modes × n_pix)."""
        return self._V

    @property
    def n_compressed(self) -> int:
        """
        Size of compressed space (n_modes_total, including spin-2 E+B doubling).
        """
        return self.n_modes_total

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
        import time as _time

        # Build harmonic operator, ell-mode mapping, and derivative diagonals
        _t0 = _time.time()
        self._build_basis()
        _t_basis = _time.time() - _t0

        # Compute effective noise matrix when switch optimization is enabled
        _t_eff = 0.0
        if self._use_switch_optimization:
            _t0 = _time.time()
            self._compute_effective_noise()
            _t_eff = _time.time() - _t0

        _t0 = _time.time()
        self._compute_smw_components()
        _t_smw = _time.time() - _t0

        from ..logger import get_logger

        logger = get_logger("basis")
        logger.debug(
            f"Basis setup: V={_t_basis:.2f}s, eff_noise={_t_eff:.2f}s, SMW={_t_smw:.2f}s"
        )

        if self._compress:
            self._compute_mblock_smw_components()

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
        # Buffer for M @ kernel_inv @ M product: (buffer_size, buffer_size)
        self._MKM_buffer = np.empty(
            (buffer_size, buffer_size), dtype=np.float64, order="F"
        )

    # =================================================================
    # SMW helpers
    # =================================================================

    def _smw_projected_inverse(
        self,
        M: np.ndarray,
        *,
        lambda_inv_diag: np.ndarray | None = None,
        lambda_inv_matrix: np.ndarray | None = None,
        K: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute M - M @ K^{-1} @ M where K = Lambda^{-1} + M.

        Parameters
        ----------
        M : numpy.ndarray
            V @ N^{-1} @ V^T (or a sub-block of it).
        lambda_inv_diag : numpy.ndarray or None
            Diagonal of Lambda^{-1}. Builds K = diag(lambda_inv) + M.
        lambda_inv_matrix : numpy.ndarray or None
            Full Lambda^{-1} matrix. Builds K = lambda_inv_matrix + M.
        K : numpy.ndarray or None
            Prebuilt K matrix (skips K construction).
        """
        if K is None:
            if lambda_inv_diag is not None:
                K = M.copy()
                K[np.diag_indices_from(K)] += lambda_inv_diag
            else:
                K = lambda_inv_matrix + M
        kernel_inv = matrix_inverse_symm(np.asfortranarray(K), overwrite=True)
        return M - matrix_mult(matrix_mult(M, kernel_inv), M)

    def _normalize_c_ell(self, C_ell):
        """Normalize C_ell input to canonical form.

        Returns (c_ell_array, c_ell_dict, is_single_field).
        Single-component spin-0 dicts with a single 1D entry are unwrapped.
        """
        if not isinstance(C_ell, dict):
            return C_ell, None, True
        if self.n_components == 1 and self._spins[0] == 0:
            first_val = next(iter(C_ell.values()))
            if isinstance(first_val, np.ndarray) and first_val.ndim == 1:
                return first_val, None, True
        return None, C_ell, False

    def _lambda_inv_diag_from_array(self, C_ell: np.ndarray) -> np.ndarray:
        """Build Lambda^{-1} diagonal from single-field C_ell array."""
        Lambda_diag = self._build_lambda_diagonal(C_ell)
        return np.where(
            Lambda_diag > _LAMBDA_ZERO_THRESHOLD, 1.0 / Lambda_diag, _LAMBDA_INV_CAP
        )

    # =================================================================
    # M-block compression
    # =================================================================

    def _compute_mblock_smw_components(self) -> None:
        """Compute block-wise V N^{-1} V^T for m-block compression.

        Instead of storing the full V N^{-1} V^T matrix, stores blocks
        indexed by (mi, mj) pairs within the delta_m bandwidth.
        The full V_N_inv is still computed for data projection.
        """
        m_to_modes = self._m_to_modes
        max_m = max(m_to_modes.keys())

        # Block-wise V_m N^{-1} V_{m'}^T
        self._vninvvt_blocks = {}
        for mi in sorted(m_to_modes.keys()):
            V_mi_Ninv = self._V_N_inv[m_to_modes[mi], :]
            for mj in range(
                max(0, mi - self._delta_m), min(max_m, mi + self._delta_m) + 1
            ):
                if mj not in m_to_modes:
                    continue
                V_mj = self._V[m_to_modes[mj], :]
                block = V_mi_Ninv @ V_mj.T
                self._vninvvt_blocks[(mi, mj)] = block

        # Build reverse mapping: global mode index -> (m, local_index)
        self._mode_to_m_local = {}
        for m, modes in m_to_modes.items():
            for local_idx, global_idx in enumerate(modes):
                self._mode_to_m_local[global_idx] = (m, local_idx)

        # Build mapping: for each m-block, which local indices belong to each ell
        self._mblock_ell_local_indices = {}
        for m in sorted(m_to_modes.keys()):
            modes = m_to_modes[m]
            mode_set = set(modes)
            mode_to_local = {g: i for i, g in enumerate(modes)}
            ell_map = {}
            for ell in range(self._lmin_smw, self._lmax_smw + 1):
                ell_modes = self._ell_to_modes[ell]
                local_indices = [mode_to_local[em] for em in ell_modes if em in mode_set]
                if local_indices:
                    ell_map[ell] = local_indices
            self._mblock_ell_local_indices[m] = ell_map

    def _get_projected_inverse_mblock(self, C_ell):
        """Compute V C^{-1} V^T block by block using SMW formula.

        For delta_m=0 (block-diagonal): each m-block is independent.
        Returns dict mapping m -> block matrix.

        Parameters
        ----------
        C_ell : numpy.ndarray
            Power spectrum values for ell = 2 to lmax.

        Returns
        -------
        dict
            Mapping from m -> block matrix (V C^{-1} V^T restricted to m).
        """
        lambda_inv_diag = self._lambda_inv_diag_from_array(C_ell)

        result_blocks = {}
        for m in sorted(self._m_to_modes.keys()):
            modes = self._m_to_modes[m]
            M_m = self._vninvvt_blocks[(m, m)]
            result_blocks[m] = self._smw_projected_inverse(
                M_m, lambda_inv_diag=lambda_inv_diag[modes]
            )

        return result_blocks

    def _get_projected_inverse_mblock_banded(self, C_ell):
        """Compute full V C^{-1} V^T using banded m-block K.

        Groups m-values into coupled sets within delta_m bandwidth,
        builds combined K for each group, inverts, and places entries
        into the full n_modes x n_modes result matrix.

        Parameters
        ----------
        C_ell : numpy.ndarray
            Power spectrum values for ell = 2 to lmax.

        Returns
        -------
        numpy.ndarray
            Full projected inverse matrix (n_modes, n_modes).
        """
        lambda_inv_diag = self._lambda_inv_diag_from_array(C_ell)

        m_values = sorted(self._m_to_modes.keys())
        max_m = max(m_values)

        # Build groups of coupled m-values via BFS
        assigned = set()
        groups = []
        for m in m_values:
            if m in assigned:
                continue
            group = []
            queue = [m]
            while queue:
                current = queue.pop(0)
                if current in assigned or current not in self._m_to_modes:
                    continue
                assigned.add(current)
                group.append(current)
                for neighbor in range(
                    max(0, current - self._delta_m),
                    min(max_m, current + self._delta_m) + 1,
                ):
                    if neighbor not in assigned and neighbor in self._m_to_modes:
                        queue.append(neighbor)
            groups.append(sorted(group))

        n = self.n_modes
        result = np.zeros((n, n), dtype=np.float64)

        for group in groups:
            # Build combined mode list
            combined_modes = []
            group_offsets = {}
            for mg in group:
                group_offsets[mg] = len(combined_modes)
                combined_modes.extend(self._m_to_modes[mg])

            n_combined = len(combined_modes)

            # Assemble combined M matrix
            M_combined = np.zeros((n_combined, n_combined), dtype=np.float64)
            for mi in group:
                oi = group_offsets[mi]
                ni = len(self._m_to_modes[mi])
                for mj in group:
                    if (mi, mj) not in self._vninvvt_blocks:
                        continue
                    oj = group_offsets[mj]
                    nj = len(self._m_to_modes[mj])
                    M_combined[oi : oi + ni, oj : oj + nj] = self._vninvvt_blocks[
                        (mi, mj)
                    ]

            result_combined = self._smw_projected_inverse(
                M_combined, lambda_inv_diag=lambda_inv_diag[combined_modes]
            )

            # Place combined result into output matrix
            ix = np.ix_(combined_modes, combined_modes)
            result[ix] = result_combined

        return result

    def _assemble_full_from_blocks(self, blocks):
        """Assemble full matrix from m-block dict (zeros in off-block positions).

        Parameters
        ----------
        blocks : dict
            Mapping from m -> block matrix.

        Returns
        -------
        numpy.ndarray
            Full (n_modes, n_modes) matrix with blocks on diagonal.
        """
        n = self.n_modes
        result = np.zeros((n, n), dtype=np.float64)
        for m, block in blocks.items():
            modes = self._m_to_modes[m]
            ix = np.ix_(modes, modes)
            result[ix] = block
        return result

    def _compute_fisher_mblock(self, C_ell, ell_min, ell_max):
        """Compute Fisher matrix using m-block projected inverse.

        For delta_m=0 (block-diagonal), computes Fisher directly from
        independent m-blocks. For delta_m>0, assembles the full projected
        inverse from banded blocks and uses the standard Fisher path.

        Parameters
        ----------
        C_ell : numpy.ndarray
            Power spectrum values for ell = 2 to lmax.
        ell_min : int
            Minimum multipole.
        ell_max : int
            Maximum multipole.

        Returns
        -------
        numpy.ndarray
            Fisher matrix of shape (n_ell, n_ell).
        """
        if self._delta_m > 0:
            # For banded case, cross-block contributions matter for Fisher.
            # Use full matrix from banded K inversion.
            V_Cinv_VT = self._get_projected_inverse_mblock_banded(C_ell)
            return self._compute_fisher_from_full(V_Cinv_VT, ell_min, ell_max)

        # delta_m=0: block-diagonal, each m contributes independently
        proj_inv_blocks = self._get_projected_inverse_mblock(C_ell)

        n_ell = ell_max - ell_min + 1
        fisher = np.zeros((n_ell, n_ell))

        for m, block in proj_inv_blocks.items():
            ell_local = self._mblock_ell_local_indices[m]
            block_size = block.shape[0]

            # Precompute block * E_diag for each ell in this m-block
            block_E = {}
            for ell, local_indices in ell_local.items():
                if ell < ell_min or ell > ell_max:
                    continue
                E_diag_local = np.zeros(block_size)
                for li in local_indices:
                    E_diag_local[li] = 1.0
                block_E[ell] = block * E_diag_local

            # F_{ij} += 0.5 * Tr[(block * E_i)(block * E_j)^T]
            ells_in_block = sorted(block_E.keys())
            for ii, ell_i in enumerate(ells_in_block):
                idx_i = ell_i - ell_min
                block_deriv_i = block_E[ell_i]
                for ell_j in ells_in_block[ii:]:
                    idx_j = ell_j - ell_min
                    block_deriv_j = block_E[ell_j]
                    val = np.sum(block_deriv_i * block_deriv_j.T)
                    fisher[idx_i, idx_j] += 0.5 * val
                    if idx_i != idx_j:
                        fisher[idx_j, idx_i] += 0.5 * val

        return fisher

    def _compute_fisher_from_full(self, V_Cinv_VT, ell_min, ell_max):
        """Compute Fisher matrix from full projected inverse (standard path)."""
        n_ell = ell_max - ell_min + 1
        fisher = np.zeros((n_ell, n_ell))

        VCinvVT_E = {}
        for ell in range(ell_min, ell_max + 1):
            E_diag = self._derivative_diagonals[ell]
            VCinvVT_E[ell] = V_Cinv_VT * E_diag

        for ell_i in range(ell_min, ell_max + 1):
            for ell_j in range(ell_i, ell_max + 1):
                idx_i = ell_i - ell_min
                idx_j = ell_j - ell_min
                fisher_val = 0.5 * matrix_trace(VCinvVT_E[ell_i], VCinvVT_E[ell_j])
                fisher[idx_i, idx_j] = fisher_val
                if idx_i != idx_j:
                    fisher[idx_j, idx_i] = fisher_val

        return fisher

    def _get_projected_inverse_field_blocks(
        self, C_ell_dict: dict, field_groups: list[list[int]]
    ) -> np.ndarray:
        """Compute V C^{-1} V^T exploiting field block-diagonal K.

        When K is block-diagonal across field groups, each group's block
        can be inverted independently.  The result is assembled into the
        full ``n_modes_total x n_modes_total`` matrix (with zeros in
        cross-group blocks).

        Parameters
        ----------
        C_ell_dict : dict
            Power spectrum dictionary (multi-field).
        field_groups : list of list of int
            Independent field groups from ``_detect_field_blocks``.

        Returns
        -------
        numpy.ndarray
            Projected inverse of shape ``(n_modes_total, n_modes_total)``.
        """
        lambda_matrix = self._build_lambda_matrix(C_ell_dict)

        n_total = self.n_modes_total
        result = np.zeros((n_total, n_total), dtype=np.float64)

        for group in field_groups:
            # Collect mode indices for this group
            mode_indices = []
            for comp in group:
                start = self._mode_offsets[comp]
                end = self._mode_offsets[comp + 1]
                mode_indices.extend(range(start, end))
            mode_indices = np.array(mode_indices, dtype=int)
            ix = np.ix_(mode_indices, mode_indices)

            M_block = self._V_Ninv_VT[ix]
            Lambda_block = lambda_matrix[ix]

            Lambda_reg = Lambda_block + np.eye(len(mode_indices)) * _MATRIX_REGULARIZATION
            Lambda_inv_block = matrix_inverse_symm(np.asfortranarray(Lambda_reg))

            result[ix] = self._smw_projected_inverse(
                M_block, lambda_inv_matrix=Lambda_inv_block
            )

        return result

    def get_projected_inverse(self, C_ell, field_groups=None):
        """
        Compute V C^{-1} V^T efficiently using SMW formula.

        Parameters
        ----------
        C_ell : numpy.ndarray or dict
            Power spectrum. Can be array (single-field) or dict (multi-field).
        field_groups : list of list of int or None, optional
            Independent field groups from ``_detect_field_blocks``.
            If provided and contains more than one group, exploits
            field block-diagonal structure for faster K inversion.

        Returns
        -------
        numpy.ndarray
            Projected inverse covariance V C^{-1} V^T.
        """
        c_ell_arr, c_ell_dict, is_single = self._normalize_c_ell(C_ell)

        # M-block compressed path (single-field only)
        if self._compress and is_single:
            if self._delta_m > 0:
                return self._get_projected_inverse_mblock_banded(c_ell_arr)
            blocks = self._get_projected_inverse_mblock(c_ell_arr)
            return self._assemble_full_from_blocks(blocks)

        if is_single:
            lambda_inv_diag = self._lambda_inv_diag_from_array(c_ell_arr)
            add_diagonal(self._V_Ninv_VT, lambda_inv_diag, out=self._K_buffer)
            return self._smw_projected_inverse(
                self._V_Ninv_VT, lambda_inv_diag=lambda_inv_diag
            )

        # Multi-field dict path
        if field_groups is not None and len(field_groups) > 1:
            return self._get_projected_inverse_field_blocks(c_ell_dict, field_groups)

        K, _ = self._build_smw_kernel(c_ell_dict)
        return self._smw_projected_inverse(self._V_Ninv_VT, K=K)

    def get_derivative_matrix(
        self,
        ell: int,
        comp_i: int | None = None,
        comp_j: int | None = None,
        mode: int = 0,
    ) -> np.ndarray:
        """
        Get the derivative matrix ∂S/∂C_ℓ in compressed form.

        Parameters
        ----------
        ell : int
            Multipole for which to compute the derivative.
        comp_i, comp_j : int or None
            Component indices for multi-field. None for single-field.
        mode : int
            Spin mode (0=EE/TE, 1=BB/TB, 2=EB). Only used with comp_i/comp_j.

        Returns
        -------
        numpy.ndarray
            Derivative matrix of shape (n_modes, n_modes).
        """
        if comp_i is None:
            return np.diag(self._derivative_diagonals[ell])
        if self.n_components == 1 and self._spins[0] == 0:
            return np.diag(self._derivative_diagonals[ell])
        return self._build_derivative_matrix_with_spins(ell, comp_i, comp_j, mode)

    def get_compressed_covariance(self, C_ell):
        """
        Compute compressed covariance C̄ = V N V^T + Λ.

        Parameters
        ----------
        C_ell : numpy.ndarray or dict
            Power spectrum. Can be:
            - numpy.ndarray: C_ell values for ell = 2 to lmax (single-field)
            - dict: Multi-field with 2-tuple or 3-tuple keys

        Returns
        -------
        numpy.ndarray
            Compressed covariance matrix.
        """
        c_ell_arr, c_ell_dict, is_single = self._normalize_c_ell(C_ell)
        if is_single:
            Lambda_diag = self._build_lambda_diagonal(c_ell_arr)
            return add_diagonal(self._V_N_VT, Lambda_diag)
        lambda_matrix = self._build_lambda_matrix(c_ell_dict)
        return self._V_N_VT + lambda_matrix

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

    def get_logdet(self, C_ell) -> float:
        """
        Compute log|N + S| using SMW formula.

        Parameters
        ----------
        C_ell : numpy.ndarray or dict
            Power spectrum (array for single-field, dict for multi-field).

        Returns
        -------
        float
            Log determinant of the full covariance matrix.
        """
        c_ell_arr, c_ell_dict, is_single = self._normalize_c_ell(C_ell)
        if not is_single:
            _, logdet = self.prepare_smw(c_ell_dict)
            return logdet
        Lambda_diag = self._build_lambda_diagonal(c_ell_arr)
        return smw_logdet(self._log_det_N, self._V_Ninv_VT, Lambda_diag)

    def get_full_logdet(self, C_ell) -> float:
        """Get exact log|N + S| via SMW formula."""
        return self.get_logdet(C_ell)

    def get_weighted_compressed_data(
        self, data: np.ndarray, C_ell, C_c_inv: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Compute V @ C^{-1} @ d for QML estimation in compressed space.

        Parameters
        ----------
        data : numpy.ndarray
            Pixel-space data vector of length n_pix.
        C_ell : numpy.ndarray or dict
            Power spectrum (array for single-field, dict for multi-field).
        C_c_inv : numpy.ndarray, optional
            Unused for harmonic compression.

        Returns
        -------
        numpy.ndarray
            Weighted compressed data vector w = V C^{-1} d.
        """
        c_ell_arr, c_ell_dict, is_single = self._normalize_c_ell(C_ell)

        y = self._V_N_inv @ data
        if is_single:
            Lambda_diag = self._build_lambda_diagonal(c_ell_arr)
            K = smw_kernel(self._V_Ninv_VT, Lambda_diag)
        else:
            K, _ = self._build_smw_kernel(c_ell_dict)

        L = cholesky_decomposition(K)
        kernel_inv_y = cho_solve((L, True), y)
        return y - matrix_mult(self._V_Ninv_VT, kernel_inv_y)

    def compute_quadratic_form(self, data: np.ndarray, C_ell) -> float:
        """
        Compute d^T C^{-1} d efficiently using SMW formula.

        Parameters
        ----------
        data : numpy.ndarray
            Pixel-space data vector of length n_pix.
        C_ell : numpy.ndarray or dict
            Power spectrum (array for single-field, dict for multi-field).

        Returns
        -------
        float
            Quadratic form value d^T C^{-1} d.
        """
        c_ell_arr, c_ell_dict, is_single = self._normalize_c_ell(C_ell)
        if not is_single:
            K_chol, _ = self.prepare_smw(c_ell_dict)
            return self.quadratic_form_from_prepared(data, K_chol)
        Lambda_diag = self._build_lambda_diagonal(c_ell_arr)
        return smw_quadratic_form(
            data, self.N_inv, self._V_N_inv, self._V_Ninv_VT, Lambda_diag
        )

    def _build_smw_kernel(self, C_ell_dict: dict) -> tuple[np.ndarray, np.ndarray]:
        """Build K = lambda_inv + V N^{-1} V^T and return (K, lambda_matrix)."""
        lambda_matrix = self._build_lambda_matrix(C_ell_dict)
        lambda_regularized = (
            lambda_matrix + np.eye(lambda_matrix.shape[0]) * _MATRIX_REGULARIZATION
        )
        lambda_inv = matrix_inverse_symm(np.asfortranarray(lambda_regularized))
        K = lambda_inv + self._V_Ninv_VT
        return K, lambda_matrix

    def prepare_smw(self, C_ell_dict: dict) -> SMWPrepared:
        """Precompute K Cholesky factor and log determinant for reuse across sims."""
        K, lambda_matrix = self._build_smw_kernel(C_ell_dict)

        _, log_det_Lambda = matrix_slogdet_symm(lambda_matrix)

        K_chol = cholesky_decomposition(K)
        log_det_K = 2.0 * np.sum(np.log(np.diag(K_chol)))

        logdet = self._log_det_N + log_det_Lambda + log_det_K

        return SMWPrepared(K_chol, logdet)

    def quadratic_form_from_prepared(self, data: np.ndarray, K_chol: np.ndarray) -> float:
        """Compute d^T C^{-1} d using precomputed K Cholesky factor."""
        term1 = float(data.T @ self.N_inv @ data)
        y = self._V_N_inv @ data
        kernel_inv_y = cho_solve((K_chol, True), y)
        term2 = float(y.T @ kernel_inv_y)
        return term1 - term2

    def _get_derivative_diagonal(
        self, ell: int, comp_i: int = 0, comp_j: int = 0, mode: int = 0
    ) -> np.ndarray:
        """
        Get derivative diagonal for auto-spectrum (comp_i, comp_j, mode).

        Only valid for auto-spectra (comp_i == comp_j) where the derivative
        matrix is diagonal. For spin-2 auto-spectra, ``mode`` selects EE (0)
        or BB (1); EB (mode=2) is NOT diagonal and must not use this method.

        Parameters
        ----------
        ell : int
            Multipole.
        comp_i : int
            First component index.
        comp_j : int
            Second component index (must equal comp_i).
        mode : int
            Spin mode: 0=TT/EE, 1=BB. Only used for spin-2 components.

        Returns
        -------
        numpy.ndarray
            Diagonal of derivative matrix, shape (n_kept,).
        """
        spin = self._spins[comp_i]
        local_mode_indices = self._ell_to_modes_local[ell]

        # Single-component spin-0: use precomputed diagonal directly
        if self.n_components == 1 and spin == 0:
            return self._derivative_diagonals[ell]

        E_diag = np.zeros(self.n_modes_total, dtype=np.float64)

        offset = self._mode_offsets[comp_i] if self.n_components > 1 else 0

        if spin == 0:
            for idx in local_mode_indices:
                E_diag[offset + idx] = 1.0
        elif spin == 2:
            n_base = self._n_modes_base
            if mode == 0:  # EE
                for idx in local_mode_indices:
                    E_diag[offset + idx] = 1.0
            elif mode == 1:  # BB
                for idx in local_mode_indices:
                    E_diag[offset + n_base + idx] = 1.0

        return E_diag

    def compute_fisher_matrix(
        self,
        C_ell,
        spectra_list: list[tuple] | None = None,
        ell_min: int = 2,
        ell_max: int | None = None,
    ) -> np.ndarray:
        """Compute Fisher matrix.

        Parameters
        ----------
        C_ell : numpy.ndarray or dict
            Power spectrum. Can be:
            - numpy.ndarray: for single-field (spectra_list should be None)
            - dict: for multi-field (spectra_list required)
        spectra_list : list of tuple or None
            For multi-field: list of 2-tuple or 3-tuple specifying spectra.
            For single-field: should be None (auto-detected from C_ell type).
        ell_min : int
            Minimum multipole.
        ell_max : int or None
            Maximum multipole.

        Returns
        -------
        numpy.ndarray
            Fisher matrix.
        """
        c_ell_arr, c_ell_dict, is_single = self._normalize_c_ell(C_ell)

        # Single-field path
        if is_single:
            if spectra_list is not None and not (len(spectra_list) == 1 and is_single):
                raise ValueError(
                    "spectra_list should be None for single-field (array) input"
                )
            if ell_max is None:
                ell_max = self.lmax

            if self._compress:
                return self._compute_fisher_mblock(c_ell_arr, ell_min, ell_max)

            n_ell = ell_max - ell_min + 1
            fisher = np.zeros((n_ell, n_ell))

            V_Cinv_VT = self.get_projected_inverse(c_ell_arr)

            # (V C⁻¹ V^T) @ diag(E) = V_Cinv_VT * E  [O(n²) column scaling]
            VCinvVT_E = {}
            for ell in range(ell_min, ell_max + 1):
                VCinvVT_E[ell] = V_Cinv_VT * self._derivative_diagonals[ell]

            # F_ij = 0.5 * Tr[A_i @ A_j]
            for ell_i in range(ell_min, ell_max + 1):
                for ell_j in range(ell_i, ell_max + 1):
                    idx_i = ell_i - ell_min
                    idx_j = ell_j - ell_min
                    fisher_val = 0.5 * matrix_trace(VCinvVT_E[ell_i], VCinvVT_E[ell_j])
                    fisher[idx_i, idx_j] = fisher_val
                    if idx_i != idx_j:
                        fisher[idx_j, idx_i] = fisher_val

            return fisher

        # Multi-field dict path
        if spectra_list is None:
            raise ValueError("spectra_list is required for multi-field (dict) input")

        if ell_max is None:
            ell_max = self.lmax

        n_ell = ell_max - ell_min + 1
        n_spec = len(spectra_list)
        fisher = np.zeros((n_spec * n_ell, n_spec * n_ell))

        field_groups = self._detect_field_blocks(c_ell_dict)
        V_Cinv_VT = self.get_projected_inverse(c_ell_dict, field_groups=field_groups)

        VCinvVT_E = {}
        for spec_idx, spec_entry in enumerate(spectra_list):
            comp_i, comp_j = spec_entry[0], spec_entry[1]
            mode = spec_entry[2] if len(spec_entry) == 3 else 0
            for ell in range(ell_min, ell_max + 1):
                E_matrix = self.get_derivative_matrix(ell, comp_i, comp_j, mode)
                VCinvVT_E[(spec_idx, ell)] = matrix_mult(V_Cinv_VT, E_matrix)

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
