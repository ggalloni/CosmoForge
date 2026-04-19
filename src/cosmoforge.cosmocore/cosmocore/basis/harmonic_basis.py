"""
Harmonic basis construction for compression methods.

This module contains the HarmonicBasis class which encapsulates all spherical
harmonic operator (V), Lambda matrix, and derivative matrix construction logic.
It is an internal helper used by ComputationBasis and its subclasses.
"""

from __future__ import annotations

import numpy as np

from ..basics import legendre_plm, wigner_d_small


class HarmonicBasisBuilder:
    """Builds and caches harmonic operator V, Lambda matrices, and derivative matrices.

    This is an internal helper class owned by ComputationBasis. It groups all
    the spherical harmonic basis construction code that was previously spread
    across ComputationBasis methods.

    Parameters
    ----------
    parent : ComputationBasis
        Parent compression instance providing configuration. The following
        attributes are read (all set during ComputationBasis.__init__):
        _theta_tuple, _phi_tuple, _spins, n_components, lmax,
        _lmin_smw, _lmax_smw, _n_modes_base, _n_modes_per_component,
        _n_modes_per_component_list, n_modes, n_modes_total,
        _mode_offsets, _pix_offsets, n_pix, _beam.
    """

    def __init__(self, parent) -> None:
        # Copy configuration attributes (read-only, set once in parent.__init__).
        # Using the same attribute names means extracted method bodies are unchanged.
        self._theta_tuple = parent._theta_tuple
        self._phi_tuple = parent._phi_tuple
        self._spins = parent._spins
        self.n_components = parent.n_components
        self.lmax = parent.lmax
        self._lmin_smw = parent._lmin_smw
        self._lmax_smw = parent._lmax_smw
        self._n_modes_base = parent._n_modes_base
        self._n_modes_per_component = parent._n_modes_per_component
        self._n_modes_per_component_list = parent._n_modes_per_component_list
        self.n_modes = parent.n_modes
        self.n_modes_total = parent.n_modes_total
        self._mode_offsets = parent._mode_offsets
        self._pix_offsets = parent._pix_offsets
        self.n_pix = parent.n_pix
        self._beam = parent._beam

        # Output attributes (set by build())
        self._V = None
        self._V_blocks = None
        self._ell_to_modes = None
        self._ell_to_modes_local = None
        self._derivative_diagonals = None
        self._derivative_diagonals_local = None

    def build(self) -> None:
        """Build harmonic operator V, ell-mode mapping, and derivative diagonals."""
        self._build_harmonic_operator()
        self._build_ell_mode_mapping()
        self._precompute_derivative_diagonals()

    # =========================================================================
    # V operator construction
    # =========================================================================

    def _build_harmonic_operator(self) -> None:
        """
        Build the harmonic projection operator V using real spherical harmonics.

        For single-field: V is (n_modes x n_pix)
        For multi-field: V is block-diagonal (n_modes_total x n_pix_total)
        """
        self._V_blocks = []
        for comp_idx in range(self.n_components):
            spin = self._spins[comp_idx]
            if spin == 0:
                V_comp = self._build_harmonic_operator_single(
                    self._theta_tuple[comp_idx],
                    self._phi_tuple[comp_idx],
                )
            else:  # spin == 2
                V_comp = self._build_harmonic_operator_spin2(
                    self._theta_tuple[comp_idx],
                    self._phi_tuple[comp_idx],
                )
            self._V_blocks.append(V_comp)

        if self.n_components == 1:
            self._V = np.asfortranarray(self._V_blocks[0])
        else:
            V_full = np.zeros((self.n_modes_total, self.n_pix), dtype=np.float64)
            for comp_idx in range(self.n_components):
                row_start = self._mode_offsets[comp_idx]
                row_end = self._mode_offsets[comp_idx + 1]
                col_start = self._pix_offsets[comp_idx]
                col_end = self._pix_offsets[comp_idx + 1]
                V_full[row_start:row_end, col_start:col_end] = self._V_blocks[comp_idx]
            self._V = np.asfortranarray(V_full)

    def _build_harmonic_operator_single(
        self, theta: np.ndarray, phi: np.ndarray
    ) -> np.ndarray:
        """Build harmonic operator V for a single spin-0 component."""
        n_pix_comp = len(theta)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        V = np.zeros((self._n_modes_per_component, n_pix_comp), dtype=np.float64)

        lmin_v = self._lmin_smw
        lmax_v = self._lmax_smw

        cos_mphi = np.zeros((lmax_v + 1, n_pix_comp), dtype=np.float64)
        sin_mphi = np.zeros((lmax_v + 1, n_pix_comp), dtype=np.float64)
        for m in range(lmax_v + 1):
            for ipix in range(n_pix_comp):
                cos_mphi[m, ipix] = np.cos(m * phi[ipix])
                sin_mphi[m, ipix] = np.sin(m * phi[ipix])

        plm = np.zeros((self.lmax + 1, self.lmax + 1), dtype=np.float64)

        for ipix in range(n_pix_comp):
            legendre_plm(cos_theta[ipix], sin_theta[ipix], plm)

            mode_idx = 0
            for ell in range(lmin_v, lmax_v + 1):
                base_idx = mode_idx

                # m = 0
                V[base_idx + ell, ipix] = plm[ell, 0]

                # m > 0
                for m in range(1, ell + 1):
                    base = np.sqrt(2.0) * plm[ell, m]
                    V[base_idx + ell + m, ipix] = base * cos_mphi[m, ipix]
                    V[base_idx + ell - m, ipix] = base * sin_mphi[m, ipix]

                mode_idx += 2 * ell + 1

        return V

    def _build_harmonic_operator_spin2(
        self, theta: np.ndarray, phi: np.ndarray
    ) -> np.ndarray:
        """Build harmonic operator V for a spin-2 (polarization) component.

        V maps (Q, U) pixel data to (E, B) mode coefficients using
        spin-weighted spherical harmonics.

        Returns V of shape (2 * n_modes, 2 * n_pix) where:
        - Rows 0:n_modes are E modes, rows n_modes:2*n_modes are B modes
        - Cols 0:n_pix are Q pixels, cols n_pix:2*n_pix are U pixels
        """
        n_pix = len(theta)
        n_modes = self._n_modes_base

        V = np.zeros((2 * n_modes, 2 * n_pix), dtype=np.float64)

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        lmin_v = self._lmin_smw
        lmax_v = self._lmax_smw

        cos_mphi = np.zeros((lmax_v + 1, n_pix), dtype=np.float64)
        sin_mphi = np.zeros((lmax_v + 1, n_pix), dtype=np.float64)
        for m in range(lmax_v + 1):
            for ipix in range(n_pix):
                cos_mphi[m, ipix] = np.cos(m * phi[ipix])
                sin_mphi[m, ipix] = np.sin(m * phi[ipix])

        for ipix in range(n_pix):
            cos_th = cos_theta[ipix]
            sin_th = sin_theta[ipix]

            mode_idx = 0
            for ell in range(lmin_v, lmax_v + 1):
                scale_ell = np.sqrt((2 * ell + 1) / (4 * np.pi))

                for m in range(-ell, ell + 1):
                    abs_m = abs(m)

                    d_plus2 = wigner_d_small(ell, abs_m, 2, cos_th, sin_th)
                    d_minus2 = wigner_d_small(ell, abs_m, -2, cos_th, sin_th)

                    D_plus = d_minus2 + d_plus2
                    D_minus = d_minus2 - d_plus2

                    if m == 0:
                        scale = scale_ell * 0.5

                        V[mode_idx, ipix] = scale * D_plus
                        V[mode_idx, n_pix + ipix] = 0.0

                        V[n_modes + mode_idx, ipix] = 0.0
                        V[n_modes + mode_idx, n_pix + ipix] = scale * D_plus
                    elif m > 0:
                        scale = scale_ell * np.sqrt(2.0) * 0.5
                        cm = cos_mphi[m, ipix]
                        sm = sin_mphi[m, ipix]

                        V[mode_idx, ipix] = scale * D_plus * cm
                        V[mode_idx, n_pix + ipix] = scale * D_minus * sm

                        V[n_modes + mode_idx, ipix] = -scale * D_minus * sm
                        V[n_modes + mode_idx, n_pix + ipix] = scale * D_plus * cm
                    else:  # m < 0
                        scale = scale_ell * np.sqrt(2.0) * 0.5
                        cm = cos_mphi[abs_m, ipix]
                        sm = sin_mphi[abs_m, ipix]

                        V[mode_idx, ipix] = scale * D_plus * sm
                        V[mode_idx, n_pix + ipix] = -scale * D_minus * cm

                        V[n_modes + mode_idx, ipix] = scale * D_minus * cm
                        V[n_modes + mode_idx, n_pix + ipix] = scale * D_plus * sm

                    mode_idx += 1

        return V

    # =========================================================================
    # Ell-mode mapping and derivative diagonals
    # =========================================================================

    def _build_ell_mode_mapping(self) -> None:
        """Build mapping from multipole ell to mode indices."""
        self._ell_to_modes_local = {}
        mode_idx = 0
        for ell in range(self._lmin_smw, self._lmax_smw + 1):
            n_m = 2 * ell + 1
            self._ell_to_modes_local[ell] = list(range(mode_idx, mode_idx + n_m))
            mode_idx += n_m

        if self.n_components == 1:
            self._ell_to_modes = self._ell_to_modes_local
        else:
            self._ell_to_modes = self._ell_to_modes_local

    def _precompute_derivative_diagonals(self) -> None:
        """Precompute derivative matrix diagonals E_ell for multipoles in SMW range."""
        self._derivative_diagonals_local = {}
        for ell in range(self._lmin_smw, self._lmax_smw + 1):
            E_diag = np.zeros(self._n_modes_per_component, dtype=np.float64)
            if ell in self._ell_to_modes_local:
                for mode_idx in self._ell_to_modes_local[ell]:
                    E_diag[mode_idx] = 1.0
            self._derivative_diagonals_local[ell] = E_diag

        if self.n_components == 1:
            self._derivative_diagonals = self._derivative_diagonals_local
        else:
            self._derivative_diagonals = self._derivative_diagonals_local

    # =========================================================================
    # Derivative matrix construction
    # =========================================================================

    def _build_derivative_matrix_with_spins(
        self, ell: int, comp_i: int, comp_j: int, mode: int = 0
    ) -> np.ndarray:
        """Build full harmonic-space derivative matrix for (comp_i, comp_j, mode).

        Handles all spin combinations: 0x0, 2x2, 0x2, 2x0.
        Returns the E matrix in n_modes_total x n_modes_total space.
        """
        E = np.zeros((self.n_modes_total, self.n_modes_total), dtype=np.float64)
        local_mode_indices = self._ell_to_modes_local[ell]
        n_base = self._n_modes_base

        spin_i = self._spins[comp_i]
        spin_j = self._spins[comp_j]

        if spin_i == 0 and spin_j == 0:
            row_offset = self._mode_offsets[comp_i]
            col_offset = self._mode_offsets[comp_j]
            for idx in local_mode_indices:
                E[row_offset + idx, col_offset + idx] = 1.0
            if comp_i != comp_j:
                for idx in local_mode_indices:
                    E[col_offset + idx, row_offset + idx] = 1.0

        elif spin_i == 2 and spin_j == 2:
            deriv_val = 1.0
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
            deriv_val = -1.0
            row_start = self._mode_offsets[comp_i]
            col_start = self._mode_offsets[comp_j]
            col_sub = col_start + mode * n_base
            for idx in local_mode_indices:
                E[row_start + idx, col_sub + idx] = deriv_val
                E[col_sub + idx, row_start + idx] = deriv_val

        elif spin_i == 2 and spin_j == 0:
            deriv_val = -1.0
            row_start = self._mode_offsets[comp_i]
            col_start = self._mode_offsets[comp_j]
            row_sub = row_start + mode * n_base
            for idx in local_mode_indices:
                E[row_sub + idx, col_start + idx] = deriv_val
                E[col_start + idx, row_sub + idx] = deriv_val

        return E

    # =========================================================================
    # Lambda matrix construction
    # =========================================================================

    def _build_lambda_diagonal(self, C_ell: np.ndarray) -> np.ndarray:
        """Build Lambda diagonal from C_ell values in the (ell,m) basis.

        The input C_ell values are assumed to already include all normalization
        factors and beam smoothing from SpectraManager.
        """
        Lambda_diag = np.zeros(self.n_modes)
        idx = 0
        for ell in range(self._lmin_smw, self._lmax_smw + 1):
            n_m = 2 * ell + 1
            c_ell_value = C_ell[ell - 2] if ell - 2 < len(C_ell) else 0.0
            Lambda_diag[idx : idx + n_m] = c_ell_value
            idx += n_m
        return Lambda_diag

    def _build_lambda_blocks(
        self, C_ell_dict: dict[tuple[int, int], np.ndarray]
    ) -> dict[tuple[int, int], np.ndarray]:
        """Build Lambda blocks from cross-power spectra dictionary (2-tuple keys)."""
        Lambda_blocks = {}
        for (comp_i, comp_j), C_ell in C_ell_dict.items():
            Lambda_diag = self._build_lambda_diagonal(C_ell)
            Lambda_blocks[(comp_i, comp_j)] = Lambda_diag
            if comp_i != comp_j:
                Lambda_blocks[(comp_j, comp_i)] = Lambda_diag
        return Lambda_blocks

    def _build_lambda_matrix(self, C_ell_dict: dict) -> np.ndarray:
        """Build full Lambda matrix, auto-detecting 2-tuple or 3-tuple keys."""
        first_key = next(iter(C_ell_dict))
        if len(first_key) == 3:
            return self._build_lambda_matrix_3tuple(C_ell_dict)
        return self._build_lambda_matrix_2tuple(C_ell_dict)

    def _build_lambda_matrix_2tuple(
        self, C_ell_dict: dict[tuple[int, int], np.ndarray]
    ) -> np.ndarray:
        """Build full Lambda matrix from 2-tuple (comp_i, comp_j) keys."""
        Lambda_blocks = self._build_lambda_blocks(C_ell_dict)

        lambda_matrix = np.zeros(
            (self.n_modes_total, self.n_modes_total), dtype=np.float64
        )
        for (comp_i, comp_j), Lambda_diag in Lambda_blocks.items():
            row_start = self._mode_offsets[comp_i]
            col_start = self._mode_offsets[comp_j]
            for k, val in enumerate(Lambda_diag):
                lambda_matrix[row_start + k, col_start + k] = val

        return lambda_matrix

    def _build_lambda_block_spin2(
        self,
        C_ell_EE: np.ndarray,
        C_ell_BB: np.ndarray,
        C_ell_EB: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build Lambda block for a spin-2 auto-correlation (EE, BB, EB).

        For polarization, Lambda has 2x2 block structure at each (ell,m):
            Lambda_{ell,m} = | C_ell^EE  C_ell^EB |
                             | C_ell^EB  C_ell^BB |
        """
        n = self._n_modes_base
        Lambda = np.zeros((2 * n, 2 * n), dtype=np.float64)

        idx = 0
        for ell in range(self._lmin_smw, self._lmax_smw + 1):
            n_m = 2 * ell + 1
            c_ee = C_ell_EE[ell - 2] if ell - 2 < len(C_ell_EE) else 0.0
            c_bb = C_ell_BB[ell - 2] if ell - 2 < len(C_ell_BB) else 0.0
            c_eb = 0.0
            if C_ell_EB is not None and ell - 2 < len(C_ell_EB):
                c_eb = C_ell_EB[ell - 2]

            for _ in range(n_m):
                Lambda[idx, idx] = c_ee  # E-E block
                Lambda[n + idx, n + idx] = c_bb  # B-B block
                Lambda[idx, n + idx] = c_eb  # E-B block
                Lambda[n + idx, idx] = c_eb  # B-E block
                idx += 1

        return Lambda

    def _build_lambda_matrix_3tuple(
        self, C_ell_dict: dict[tuple, np.ndarray]
    ) -> np.ndarray:
        """Build full Lambda matrix handling mixed spin-0/spin-2 components.

        Accepts 3-tuple keys (comp_i, comp_j, mode) matching the
        get_cls(field_i, field_j, mode) API:
        - spin-0 x spin-0: mode 0 only
        - spin-2 x spin-2: mode 0=EE, 1=BB, 2=EB
        - spin-0 x spin-2: mode 0=TE, 1=TB
        """
        lambda_matrix = np.zeros(
            (self.n_modes_total, self.n_modes_total), dtype=np.float64
        )

        # Group entries by component pair
        pair_entries: dict[tuple[int, int], dict[int, np.ndarray]] = {}
        for (ci, cj, mode), C_ell in C_ell_dict.items():
            pair_entries.setdefault((ci, cj), {})[mode] = C_ell

        for (ci, cj), mode_dict in pair_entries.items():
            spin_i = self._spins[ci]
            spin_j = self._spins[cj]
            row_start = self._mode_offsets[ci]
            col_start = self._mode_offsets[cj]

            if spin_i == 0 and spin_j == 0:
                diag = self._build_lambda_diagonal(mode_dict[0])
                for k, val in enumerate(diag):
                    lambda_matrix[row_start + k, col_start + k] = val
                    if ci != cj:
                        lambda_matrix[col_start + k, row_start + k] = val

            elif spin_i == 2 and spin_j == 2:
                C_EE = mode_dict.get(0, np.zeros(self.lmax - 1))
                C_BB = mode_dict.get(1, np.zeros(self.lmax - 1))
                C_EB = mode_dict.get(2, None)
                block = self._build_lambda_block_spin2(C_EE, C_BB, C_EB)
                n_block = 2 * self._n_modes_base
                lambda_matrix[
                    row_start : row_start + n_block,
                    col_start : col_start + n_block,
                ] = block

            elif spin_i == 0 and spin_j == 2:
                n_base = self._n_modes_base
                for mode, C_ell in mode_dict.items():
                    diag = self._build_lambda_diagonal(C_ell)
                    col_sub = col_start + mode * n_base
                    for k, val in enumerate(diag):
                        lambda_matrix[row_start + k, col_sub + k] = -val
                        lambda_matrix[col_sub + k, row_start + k] = -val

            elif spin_i == 2 and spin_j == 0:
                n_base = self._n_modes_base
                for mode, C_ell in mode_dict.items():
                    diag = self._build_lambda_diagonal(C_ell)
                    row_sub = row_start + mode * n_base
                    for k, val in enumerate(diag):
                        lambda_matrix[row_sub + k, col_start + k] = -val
                        lambda_matrix[col_start + k, row_sub + k] = -val

        return lambda_matrix
