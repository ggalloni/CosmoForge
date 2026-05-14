"""
Harmonic computation basis (Tegmark-like) for CMB Fisher matrix computation.

This module implements direct harmonic space projection where data is
transformed directly to n_modes dimensions via the harmonic operator V.
"""

from __future__ import annotations

from functools import cached_property

import numpy as np

from ..basics import (
    add_diagonal,
    cholesky_decomposition,
    cholesky_solve,
    inv,
    matrix_inverse_symm,
    matrix_mult,
    matrix_slogdet_symm,
    matrix_trace,
    smw_inverse,
    smw_logdet,
    smw_quadratic_form,
    solve_linear,
    symmetrize_inplace,
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
        Noise covariance matrix of shape (n_pix, n_pix). The basis takes
        ownership of this buffer; ``_factorise_noise()`` overwrites it
        in place during setup.
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
    >>> hc = HarmonicBasis(N, theta, phi, lmax=100)
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
        theta: np.ndarray | tuple[np.ndarray, ...],
        phi: np.ndarray | tuple[np.ndarray, ...],
        lmax_signal: int,
        beam: np.ndarray | None = None,
        spins: list[int] | None = None,
        lmin_signal: list[int] | None = None,
        lmin: int | None = None,
        lmax: int | None = None,
        fiducial_C_ell: np.ndarray | None = None,
        S_fixed: np.ndarray | None = None,
        compress: bool = False,
        delta_m: int = 0,
    ):
        super().__init__(
            N=N,
            theta=theta,
            phi=phi,
            lmax_signal=lmax_signal,
            beam=beam,
            spins=spins,
            lmin_signal=lmin_signal,
            lmin=lmin,
            lmax=lmax,
            fiducial_C_ell=fiducial_C_ell,
            S_fixed=S_fixed,
        )
        self._init_harmonic_internals()
        self.dim = self.n_modes_total
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

    def release_pixel_projector(self) -> None:
        """Drop V after SMW components are built.

        QML hot paths (Fisher trace, Spectra weighted-data) read only
        ``_V_N_inv`` and ``_V_Ninv_VT``. The remaining V consumers —
        ``get_compressed_covariance``/``get_inverse``,
        ``to_basis``, m-block compression, and PICSLike ``do_cross``
        with the harmonic basis — must not be invoked after this call.
        Raises if m-block compression was requested at construction.
        """
        if self._compress:
            raise RuntimeError(
                "release_pixel_projector incompatible with m-block compression"
            )
        self._harmonic_basis._V = None

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
        # _V_blocks is the C-order source PixelBasis re-reads during
        # apply_compression. HarmonicBasis never does; release here.
        self._harmonic_basis._V_blocks = None
        _t_basis = _time.time() - _t0

        # Compute effective noise matrix when switch optimization is enabled
        _t_eff = 0.0
        if self._use_switch_optimization:
            _t0 = _time.time()
            self._compute_effective_noise()
            _t_eff = _time.time() - _t0

        # Factor the (effective) noise in place; releases the symmetric N
        # buffer and any caller-provided dense N_inv. After this call,
        # cholesky_solve via self._N_chol is the only path to N^{-1}.
        self._factorise_noise()

        _t0 = _time.time()
        self._compute_smw_components()
        _t_smw = _time.time() - _t0

        from ..logger import get_logger

        logger = get_logger("basis", feedback_level=4)
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
        if self._S_fixed is not None:
            S_fixed = np.asarray(self._S_fixed, dtype=np.float64)
        elif self._fiducial_C_ell is not None:
            S_fixed = self._compute_s_fixed_from_fiducial()
        else:
            raise ValueError(
                "Either S_fixed or fiducial_C_ell must be provided when using "
                "switch optimization (lswitch_high < lmax)"
            )

        # In-place — _compute_smw_components rebuilds T from S_fixed via the
        # identity T = V_Ninv_VT − V_N_inv S_fixed V_N_inv^T, so we don't need
        # an _N_original copy.
        np.add(self._N, S_fixed, out=self._N)
        self._S_fixed = S_fixed

    def _compute_s_fixed_from_fiducial(self) -> np.ndarray:
        """
        Legacy method: compute S_fixed from fiducial_C_ell using compute_00_contribution.

        This is kept for backward compatibility but the preferred approach is to
        compute S_fixed externally using the existing signal matrix infrastructure.
        """
        from ..signal_kernels import compute_00_contribution

        cl_fixed = np.zeros(self.lmax_signal + 1, dtype=np.float64)
        for ell in range(self.lmax + 1, self.lmax_signal + 1):
            if ell < len(self._fiducial_C_ell):
                cl_fixed[ell] = self._fiducial_C_ell[ell]

        point_vectors = np.column_stack(
            [
                np.sin(self.theta) * np.cos(self.phi),
                np.sin(self.theta) * np.sin(self.phi),
                np.cos(self.theta),
            ]
        )

        S_fixed = np.zeros((self.n_pix, self.n_pix), dtype=np.float64)
        legendre_buffer = np.empty(self.lmax_signal + 1, dtype=np.float64)

        compute_00_contribution(
            cl_fixed,
            S_fixed,
            point_vectors,
            point_vectors,
            legendre_buffer,
            mode=0,
        )

        # compute_00_contribution fills lower triangle only; symmetrise.
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
        self._V_N_inv = cholesky_solve(self._N_chol, self._V.T).T

        # Symmetrise in place: M+M.T and 0.5*(...) each allocate a fresh
        # n_modes² buffer (~88 GB transient at QU/nside=64/lmax=192).
        self._V_Ninv_VT = matrix_mult(self._V_N_inv, self._V.T)
        symmetrize_inplace(self._V_Ninv_VT)

        # T = V N_eff^{-1} N_orig N_eff^{-1} V^T, consumed by
        # Spectra._compute_noise_cov_compressed. With N_orig = N_eff − S_fixed
        # the algebra reduces to T = V_Ninv_VT − V_N_inv S_fixed V_N_inv^T,
        # avoiding an n_pix² N_orig copy. Without switch optimisation, T = V_Ninv_VT.
        if self._S_fixed is not None:
            corr = matrix_mult(matrix_mult(self._V_N_inv, self._S_fixed), self._V_N_inv.T)
            T = self._V_Ninv_VT - corr
            self._noise_cov_T = symmetrize_inplace(T)
            self._S_fixed = None
        else:
            self._noise_cov_T = self._V_Ninv_VT

    @cached_property
    def _V_N_VT(self) -> np.ndarray:
        """Lazily compute V @ N @ V^T via the Cholesky factor.

        Uses N = L L^T so V N V^T = (V L)(V L)^T — cheaper than building
        N from L and tolerant of in-place factor storage in commit 3.
        """
        VL = matrix_mult(self._V, self._L)
        return matrix_mult(VL, VL.T)

    # =================================================================
    # SMW helpers
    # =================================================================

    def prepare_stable_inner_inv(
        self, C_ell, lambda_matrix: np.ndarray | None = None
    ) -> np.ndarray:
        """Build (I + Lambda M)^{-1} for the stable QML algebra.

        This matrix appears identically in three places of the QML
        algebra and replaces the unstable subtractive forms:

        - Projected inverse:    V C^{-1} V^T = M @ (I + Lambda M)^{-1}
        - Data weighting:       V C^{-1} d = (I + Lambda M)^{-1} @ V N^{-1} d
        - Noise bias matrix:    A = I - M K^{-1} = (I + Lambda M)^{-1}

        All three follow from the identity
        ``M - M K^{-1} M = M K^{-1} Lambda^{-1} = M (I + Lambda M)^{-1}``
        and never subtract large nearly-equal matrices, so they remain
        accurate in the cosmic-variance-limited regime where the legacy
        SMW form loses precision (e.g. T at low ell with sub-uK
        polarization noise).

        Parameters
        ----------
        C_ell : np.ndarray or dict
            Power spectrum (single-field array or multi-field dict).
        lambda_matrix : np.ndarray, optional
            Pre-built full Lambda matrix.  If None, built from C_ell.

        Returns
        -------
        np.ndarray
            (I + Lambda M)^{-1}, n_modes_total x n_modes_total.
        """
        if lambda_matrix is None:
            c_ell_arr, c_ell_dict, is_single = self._normalize_c_ell(C_ell)
            if is_single:
                lambda_diag = self._build_lambda_diagonal(c_ell_arr)
                lambda_matrix = np.diag(lambda_diag)
            else:
                lambda_matrix = self._build_lambda_matrix(c_ell_dict)
        inner = lambda_matrix @ self._V_Ninv_VT
        inner[np.diag_indices_from(inner)] += 1.0
        return inv(inner)

    def _smw_projected_inverse(
        self,
        M: np.ndarray,
        *,
        lambda_inv_diag: np.ndarray | None = None,
        lambda_inv_matrix: np.ndarray | None = None,
        K: np.ndarray | None = None,
        lambda_matrix: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute the projected inverse V (S+N)^{-1} V^T.

        Two algebraically equivalent forms are supported:

        - **Subtractive (legacy)**: ``M - M K^{-1} M`` with
          ``K = Lambda^{-1} + M``. Numerically unstable in the
          signal-dominated regime (``Lambda >> M^{-1}``) because the two
          terms become large and nearly equal — float64 cancellation can
          drive entries to spuriously negative values, breaking
          downstream Fisher PD-ness. Used when ``K`` or
          ``lambda_inv_*`` is supplied (preserves backward compatibility
          for callers that already built K).

        - **Stable**: ``(M^{-1} + Lambda)^{-1}``. Algebraically
          identical, but never subtracts large nearly-equal matrices —
          accurate in both noise- and signal-dominated regimes.
          Triggered automatically when only ``lambda_matrix`` is
          provided.

        Parameters
        ----------
        M : numpy.ndarray
            V @ N^{-1} @ V^T (or a sub-block of it).
        lambda_inv_diag, lambda_inv_matrix, K : optional
            Trigger the legacy subtractive form (see above).
        lambda_matrix : numpy.ndarray, optional
            Full Lambda matrix (signal-space block). Triggers the stable
            ``(M^{-1} + Lambda)^{-1}`` form. Mutually exclusive with the
            legacy keywords.
        """
        if lambda_matrix is not None:
            if (
                K is not None
                or lambda_inv_diag is not None
                or lambda_inv_matrix is not None
            ):
                raise ValueError(
                    "lambda_matrix is mutually exclusive with K / "
                    "lambda_inv_diag / lambda_inv_matrix"
                )
            # Stable form: V C^{-1} V^T = M (I + Lambda M)^{-1}.
            # Algebraically equivalent to M - M K^{-1} M but never
            # subtracts large nearly-equal matrices and never inverts
            # M (which is rank-deficient on a partial sky) or Lambda
            # (which can be ill-conditioned in the cosmic-variance
            # limit). The matrix (I + Lambda M) has eigenvalues 1 +
            # eig(Lambda^{1/2} M Lambda^{1/2}) >= 1, so it is always
            # well-conditioned and invertible. Result must be symmetric;
            # we enforce it to clean float-eps drift.
            # We want VCVT = M @ inv(I + Lambda @ M).  Compute via
            # solve((I + Lambda @ M)^T, M^T)^T which is equivalent.  By
            # symmetry of Lambda and M, (I + Lambda M)^T = I + M Lambda;
            # we use the form that lets numpy's solve do the work.
            inner = lambda_matrix @ M
            inner[np.diag_indices_from(inner)] += 1.0
            VCVT = solve_linear(inner.T, M.T).T
            symmetrize_inplace(VCVT)
            return np.asfortranarray(VCVT)

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
            return self._smw_projected_inverse(
                self._V_Ninv_VT, lambda_inv_diag=lambda_inv_diag
            )

        # Multi-field dict path
        if field_groups is not None and len(field_groups) > 1:
            return self._get_projected_inverse_field_blocks(c_ell_dict, field_groups)

        # Stable form: (M^{-1} + Lambda)^{-1}. Avoids the catastrophic
        # cancellation of M - M K^{-1} M in the signal-dominated regime,
        # which is otherwise the dominant source of numerical noise for
        # multi-spectrum QML at high SNR (e.g. T at low ell with sub-uK
        # polarization noise).
        lambda_matrix = self._build_lambda_matrix(c_ell_dict)
        return self._smw_projected_inverse(self._V_Ninv_VT, lambda_matrix=lambda_matrix)

    def get_derivative_matrix(
        self,
        ell: int,
        key,
        *,
        symmetry_mode=None,
    ) -> np.ndarray:
        """
        Get the derivative matrix ∂S/∂C_ℓ in compressed form.

        Parameters
        ----------
        ell : int
            Multipole.
        key : SpectrumKey
            Identifies the spectrum (component pair + spin kind).
        symmetry_mode : SymmetryMode or None
            ``None`` keeps the SYMMETRIC default.

        Returns
        -------
        numpy.ndarray
            Derivative matrix of shape (n_modes, n_modes).
        """
        from ..spectrum_key import SpectrumKind, SymmetryMode, kind_to_legacy_mode

        if self.n_components == 1 and self._spins[0] == 0:
            if key.kind is not SpectrumKind.SS or key.comp_i != 0 or key.comp_j != 0:
                raise ValueError(
                    f"single spin-0 basis only supports SpectrumKey(0, 0, SS); got {key}"
                )
            return np.diag(self._derivative_diagonals[ell])
        if symmetry_mode is None:
            symmetry_mode = SymmetryMode.SYMMETRIC
        is_cross = key.comp_i != key.comp_j
        mode = kind_to_legacy_mode(key.kind, is_cross=is_cross)
        return self._build_derivative_matrix_with_spins(
            ell, key.comp_i, key.comp_j, mode, symmetry_mode=symmetry_mode
        )

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

    def get_full_inverse(self, C_ell: np.ndarray) -> np.ndarray:
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

    def get_noise_for_bias(self) -> np.ndarray:
        """Return the SMW noise intermediate ``T = V N_eff^{-1} N N_eff^{-1} V^T``.

        See :meth:`ComputationBasis.get_noise_for_bias` for the cross-basis
        contract. ``T`` is precomputed during basis setup.
        """
        return self._noise_cov_T

    def get_weighted_compressed_data(
        self,
        data: np.ndarray,
        C_ell,
        C_c_inv: np.ndarray | None = None,
        stable_inner_inv: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute ``w = V C^{-1} d`` via the stable SMW algebra.

        ``w = (I + Lambda M)^{-T} (V N^{-1} d)`` — algebraically equivalent to
        ``M K^{-1} d`` but never subtracts large nearly-equal matrices, so it
        retains precision in the cosmic-variance-limited regime where the
        legacy subtractive form ``y - M K^{-1} y`` cancelled.

        Parameters
        ----------
        data : numpy.ndarray
            Pixel-space data, shape ``(n_pix,)`` or ``(n_pix, n_sims)``.
        C_ell : numpy.ndarray or dict
            Power spectrum (single-field array or multi-field dict).
        C_c_inv : numpy.ndarray, optional
            Unused for harmonic basis; accepted for ABC signature parity.
        stable_inner_inv : numpy.ndarray, optional
            Precomputed ``(I + Lambda M)^{-1}``. If supplied, avoids
            rebuilding it internally. Callers that weight several data
            vectors with the same ``C_ell`` should build it once via
            :meth:`prepare_stable_inner_inv` and pass it in.

        Returns
        -------
        numpy.ndarray
            Weighted compressed data ``w = V C^{-1} d``.
        """
        if stable_inner_inv is None:
            stable_inner_inv = self.prepare_stable_inner_inv(C_ell)
        return stable_inner_inv.T @ (self._V_N_inv @ data)

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
            data, self._N_chol, self._V_N_inv, self._V_Ninv_VT, Lambda_diag
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
        term1 = float(data.T @ cholesky_solve(self._N_chol, data))
        y = self._V_N_inv @ data
        kernel_inv_y = cholesky_solve((K_chol, True), y)
        term2 = float(y.T @ kernel_inv_y)
        return term1 - term2

    def get_derivative_diagonal(self, ell: int, key) -> np.ndarray:
        """Get derivative diagonal for an auto-pair diagonal SpectrumKey.

        Returns a 1D vector — the diagonal of ∂S/∂C_ℓ — for use in the
        sparse-trace fast path. Valid only for ``key.comp_i == key.comp_j``
        and diagonal kinds (``SS``, ``GG``, ``CC``) — i.e. TT, EE, or BB.
        Cross-component keys and ``GC/CG/SG/SC/GS/CS`` raise ``ValueError``.
        """
        from ..spectrum_key import SpectrumKind, kind_to_legacy_mode

        if key.comp_i != key.comp_j:
            raise ValueError(
                "get_derivative_diagonal requires an auto-pair key; "
                f"got comp_i={key.comp_i}, comp_j={key.comp_j}"
            )
        if key.kind not in (SpectrumKind.SS, SpectrumKind.GG, SpectrumKind.CC):
            raise ValueError(
                "get_derivative_diagonal requires a diagonal kind "
                f"(SS/GG/CC); got {key.kind.name}"
            )
        return self._get_derivative_diagonal(
            ell, key.comp_i, key.comp_j, kind_to_legacy_mode(key.kind)
        )

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
            Diagonal of derivative matrix, shape (dim,).
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
        spectra_list=None,
        ell_min: int = 2,
        ell_max: int | None = None,
        *,
        symmetry_mode=None,
    ) -> np.ndarray:
        """Compute Fisher matrix.

        Parameters
        ----------
        C_ell : numpy.ndarray or dict
            Power spectrum. Single-field: numpy.ndarray (spectra_list=None).
            Multi-field: dict keyed by SpectrumKey (spectra_list required).
        spectra_list : list[SpectrumKey] or None
            For multi-field, the list of SpectrumKey instances enumerating
            the spectra to include. None for single-field.
        ell_min : int
            Minimum multipole.
        ell_max : int or None
            Maximum multipole.
        symmetry_mode : SymmetryMode or None
            Forwarded to the per-ℓ derivative builder. ``None`` keeps the
            SYMMETRIC default. Required as ``DIRECTIONAL`` when
            ``spectra_list`` contains a ``CG`` cross-pair entry — the
            builder otherwise rejects the ``mode=2`` slot used by CG.

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
            for ell in range(ell_min, ell_max + 1):
                E_matrix = self.get_derivative_matrix(
                    ell, spec_entry, symmetry_mode=symmetry_mode
                )
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
