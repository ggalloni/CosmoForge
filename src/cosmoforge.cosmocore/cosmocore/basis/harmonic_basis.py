"""
Harmonic basis construction for computation basis methods.

This module contains the HarmonicBasisBuilder class which encapsulates all
spherical harmonic operator (V), Lambda matrix, and derivative matrix
construction logic. It is an internal helper used by ComputationBasis and
its subclasses.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from ..basics import legendre_plm, wigner_d_small
from ..spectrum_key import SpectrumKey, kind_to_legacy_mode


@njit(parallel=True, cache=True)
def _fill_V_spin0(V, cos_theta, sin_theta, cos_mphi, sin_mphi, lmin_v, lmax_v):
    """Fill V matrix for spin-0 in parallel over pixels."""
    lmax = lmax_v
    n_pix = len(cos_theta)
    sqrt2 = np.sqrt(2.0)

    for ipix in prange(n_pix):
        plm_local = np.zeros((lmax + 1, lmax + 1), dtype=np.float64)
        legendre_plm(cos_theta[ipix], sin_theta[ipix], plm_local)

        mode_idx = 0

        # m=0 block
        for ell in range(lmin_v, lmax_v + 1):
            V[mode_idx, ipix] = plm_local[ell, 0]
            mode_idx += 1

        # |m|>0 blocks
        for abs_m in range(1, lmax_v + 1):
            ell_start = abs_m if abs_m > lmin_v else lmin_v
            # cos rows
            for ell in range(ell_start, lmax_v + 1):
                base = sqrt2 * plm_local[ell, abs_m]
                V[mode_idx, ipix] = base * cos_mphi[abs_m, ipix]
                mode_idx += 1
            # sin rows
            for ell in range(ell_start, lmax_v + 1):
                base = sqrt2 * plm_local[ell, abs_m]
                V[mode_idx, ipix] = base * sin_mphi[abs_m, ipix]
                mode_idx += 1


@njit(parallel=True, cache=True)
def _fill_V_spin2(
    V, cos_theta, sin_theta, cos_mphi, sin_mphi, lmin_v, lmax_v, n_modes, n_pix
):
    """Fill V matrix for spin-2 in parallel over pixels."""
    sqrt2 = np.sqrt(2.0)
    pi4 = 4.0 * np.pi

    for ipix in prange(n_pix):
        cos_th = cos_theta[ipix]
        sin_th = sin_theta[ipix]

        mode_idx = 0

        # m=0 block
        for ell in range(lmin_v, lmax_v + 1):
            scale_ell = np.sqrt((2 * ell + 1) / pi4)
            d_plus2 = wigner_d_small(ell, 0, 2, cos_th, sin_th)
            d_minus2 = wigner_d_small(ell, 0, -2, cos_th, sin_th)
            D_plus = d_minus2 + d_plus2
            scale = scale_ell * 0.5

            V[mode_idx, ipix] = scale * D_plus
            V[mode_idx, n_pix + ipix] = 0.0
            V[n_modes + mode_idx, ipix] = 0.0
            V[n_modes + mode_idx, n_pix + ipix] = scale * D_plus
            mode_idx += 1

        # |m|>0 blocks
        for abs_m in range(1, lmax_v + 1):
            ell_start = abs_m if abs_m > lmin_v else lmin_v

            # cos rows
            for ell in range(ell_start, lmax_v + 1):
                scale_ell = np.sqrt((2 * ell + 1) / pi4)
                d_plus2 = wigner_d_small(ell, abs_m, 2, cos_th, sin_th)
                d_minus2 = wigner_d_small(ell, abs_m, -2, cos_th, sin_th)
                D_plus = d_minus2 + d_plus2
                D_minus = d_minus2 - d_plus2
                scale = scale_ell * sqrt2 * 0.5
                cm = cos_mphi[abs_m, ipix]
                sm = sin_mphi[abs_m, ipix]

                V[mode_idx, ipix] = scale * D_plus * cm
                V[mode_idx, n_pix + ipix] = scale * D_minus * sm
                V[n_modes + mode_idx, ipix] = -scale * D_minus * sm
                V[n_modes + mode_idx, n_pix + ipix] = scale * D_plus * cm
                mode_idx += 1

            # sin rows
            for ell in range(ell_start, lmax_v + 1):
                scale_ell = np.sqrt((2 * ell + 1) / pi4)
                d_plus2 = wigner_d_small(ell, abs_m, 2, cos_th, sin_th)
                d_minus2 = wigner_d_small(ell, abs_m, -2, cos_th, sin_th)
                D_plus = d_minus2 + d_plus2
                D_minus = d_minus2 - d_plus2
                scale = scale_ell * sqrt2 * 0.5
                cm = cos_mphi[abs_m, ipix]
                sm = sin_mphi[abs_m, ipix]

                V[mode_idx, ipix] = scale * D_plus * sm
                V[mode_idx, n_pix + ipix] = -scale * D_minus * cm
                V[n_modes + mode_idx, ipix] = scale * D_minus * cm
                V[n_modes + mode_idx, n_pix + ipix] = scale * D_plus * sm
                mode_idx += 1


class HarmonicBasisBuilder:
    """Builds and caches harmonic operator V, Lambda matrices, and derivative matrices.

    This is an internal helper class owned by ComputationBasis. It groups all
    the spherical harmonic basis construction code that was previously spread
    across ComputationBasis methods.

    Parameters
    ----------
    parent : ComputationBasis
        Parent computation basis instance providing configuration. The following
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
        self.lmax_signal = parent.lmax_signal
        self._lmin_smw = parent._lmin_smw
        self._lmax_smw = parent._lmax_smw
        self._lmin_signal = parent._lmin_signal
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
        self._m_to_modes = None
        self._m_to_modes_local = None
        self._derivative_diagonals = None
        self._derivative_diagonals_local = None

    def build(self) -> None:
        """Build harmonic operator V, mode mappings, and derivative diagonals."""
        self._build_harmonic_operator()
        self._build_ell_mode_mapping()
        self._build_m_mode_mapping()
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
            # Per-component V floor (ADR 0009): when lmin_signal[i] exceeds the
            # global SMW floor, this component skips its first few ell modes.
            # _n_modes_per_component_list already accounts for this; the V-fill
            # loops follow.
            lmin_v_comp = max(self._lmin_smw, self._lmin_signal[comp_idx])
            if spin == 0:
                V_comp = self._build_harmonic_operator_single(
                    self._theta_tuple[comp_idx],
                    self._phi_tuple[comp_idx],
                    lmin_v=lmin_v_comp,
                )
            else:  # spin == 2
                V_comp = self._build_harmonic_operator_spin2(
                    self._theta_tuple[comp_idx],
                    self._phi_tuple[comp_idx],
                    lmin_v=lmin_v_comp,
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
        self,
        theta: np.ndarray,
        phi: np.ndarray,
        lmin_v: int | None = None,
    ) -> np.ndarray:
        """Build harmonic operator V for a single spin-0 component.

        Modes are grouped by azimuthal quantum number |m|:
        - m=0 block: one row per ell (ell=lmin_v..lmax)
        - |m|>0 blocks: cos rows for all ell, then sin rows for all ell

        Parameters
        ----------
        theta, phi : np.ndarray
            Pointing angles for this component's active pixels.
        lmin_v : int or None
            Per-component V floor. Defaults to ``self._lmin_smw`` for the
            backward-compatible scalar path. Pass a higher value (typically
            ``max(lmin_smw, lmin_signal[i])``) to start the V from a
            component-specific floor (ADR 0009).

        Pixel loop is parallelized via Numba prange.
        """
        n_pix_comp = len(theta)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        n_modes = (
            (self._lmax_smw + 1) ** 2 - max(self._lmin_smw, lmin_v) ** 2
            if lmin_v is not None
            else self._n_modes_per_component
        )
        V = np.zeros((n_modes, n_pix_comp), dtype=np.float64)

        lmin_v_local = self._lmin_smw if lmin_v is None else lmin_v
        lmax_v = self._lmax_smw

        cos_mphi = np.zeros((lmax_v + 1, n_pix_comp), dtype=np.float64)
        sin_mphi = np.zeros((lmax_v + 1, n_pix_comp), dtype=np.float64)
        for m in range(lmax_v + 1):
            cos_mphi[m, :] = np.cos(m * phi)
            sin_mphi[m, :] = np.sin(m * phi)

        _fill_V_spin0(V, cos_theta, sin_theta, cos_mphi, sin_mphi, lmin_v_local, lmax_v)

        return V

    def _build_harmonic_operator_spin2(
        self,
        theta: np.ndarray,
        phi: np.ndarray,
        lmin_v: int | None = None,
    ) -> np.ndarray:
        """Build harmonic operator V for a spin-2 (polarization) component.

        V maps (Q, U) pixel data to (E, B) mode coefficients using
        spin-weighted spherical harmonics.

        Modes are grouped by azimuthal quantum number |m|:
        - m=0 block: one row per ell
        - |m|>0 blocks: cos rows for all ell, then sin rows for all ell

        Returns V of shape (2 * n_modes, 2 * n_pix) where:
        - Rows 0:n_modes are E modes, rows n_modes:2*n_modes are B modes
        - Cols 0:n_pix are Q pixels, cols n_pix:2*n_pix are U pixels

        ``lmin_v`` (per-component V floor; ADR 0009) defaults to
        ``self._lmin_smw``. Spin-2 representation theory pins ell ≥ 2 so
        callers should not pass anything below 2.

        Pixel loop is parallelized via Numba prange.
        """
        n_pix = len(theta)
        lmin_v_local = self._lmin_smw if lmin_v is None else lmin_v
        n_modes = (self._lmax_smw + 1) ** 2 - max(self._lmin_smw, lmin_v_local) ** 2

        V = np.zeros((2 * n_modes, 2 * n_pix), dtype=np.float64)

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        lmax_v = self._lmax_smw

        cos_mphi = np.zeros((lmax_v + 1, n_pix), dtype=np.float64)
        sin_mphi = np.zeros((lmax_v + 1, n_pix), dtype=np.float64)
        for m in range(lmax_v + 1):
            cos_mphi[m, :] = np.cos(m * phi)
            sin_mphi[m, :] = np.sin(m * phi)

        _fill_V_spin2(
            V,
            cos_theta,
            sin_theta,
            cos_mphi,
            sin_mphi,
            lmin_v_local,
            lmax_v,
            n_modes,
            n_pix,
        )

        return V

    # =========================================================================
    # Ell-mode mapping and derivative diagonals
    # =========================================================================

    def _build_ell_mode_mapping(self) -> None:
        """Build mapping from multipole ell to mode indices.

        With m-ordered modes, each ell's modes are scattered across m-blocks:
        - One mode in the m=0 block (at position ell-lmin within the block)
        - For each |m| from 1 to ell: two modes (cos and sin rows)
        """
        self._ell_to_modes_local = {}
        lmin_v = self._lmin_smw
        lmax_v = self._lmax_smw

        for ell in range(lmin_v, lmax_v + 1):
            modes = []
            # m=0: position within m=0 block
            modes.append(ell - lmin_v)

            # |m|>0: find position within each |m| block
            block_offset = lmax_v - lmin_v + 1  # size of m=0 block
            for abs_m in range(1, ell + 1):
                ell_start = max(abs_m, lmin_v)
                n_ell_m = lmax_v - ell_start + 1
                pos_in_block = ell - ell_start
                # cos row
                modes.append(block_offset + pos_in_block)
                # sin row
                modes.append(block_offset + n_ell_m + pos_in_block)
                block_offset += 2 * n_ell_m

            self._ell_to_modes_local[ell] = modes

        if self.n_components == 1:
            self._ell_to_modes = self._ell_to_modes_local
        else:
            self._ell_to_modes = self._ell_to_modes_local

    def _build_m_mode_mapping(self) -> None:
        """Build mapping from azimuthal quantum number |m| to mode indices.

        With m-ordered modes, each |m| block is contiguous:
        - m=0 block: one row per ell (lmin..lmax)
        - |m|>0 blocks: cos rows for all ell, then sin rows
        """
        self._m_to_modes_local = {}
        mode_idx = 0
        lmin_v = self._lmin_smw
        lmax_v = self._lmax_smw

        # m=0 block
        n_ell_m0 = lmax_v - lmin_v + 1
        self._m_to_modes_local[0] = list(range(mode_idx, mode_idx + n_ell_m0))
        mode_idx += n_ell_m0

        # |m|>0 blocks
        for abs_m in range(1, lmax_v + 1):
            ell_start = max(abs_m, lmin_v)
            n_ell_m = lmax_v - ell_start + 1
            if n_ell_m <= 0:
                continue
            block_size = 2 * n_ell_m  # cos + sin
            self._m_to_modes_local[abs_m] = list(range(mode_idx, mode_idx + block_size))
            mode_idx += block_size

        if self.n_components == 1:
            self._m_to_modes = self._m_to_modes_local
        else:
            self._m_to_modes = self._m_to_modes_local

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
        factors and beam smoothing from SpectraManager. Uses _ell_to_modes_local
        to place values at the correct mode indices for the current ordering.
        """
        if self._ell_to_modes_local is None:
            self._build_ell_mode_mapping()
        Lambda_diag = np.zeros(self.n_modes)
        for ell in range(self._lmin_smw, self._lmax_smw + 1):
            c_ell_value = C_ell[ell] if ell < len(C_ell) else 0.0
            for idx in self._ell_to_modes_local[ell]:
                Lambda_diag[idx] = c_ell_value
        return Lambda_diag

    def _build_lambda_blocks(self, C_ell_dict: dict) -> dict[tuple[int, int], np.ndarray]:
        """Build Lambda blocks from cross-power spectra dictionary.

        Accepts either 2-tuple ``(comp_i, comp_j)`` keys (legacy single-mode)
        or :class:`SpectrumKey` instances (kind is read off the key without
        materialising an intermediate tuple-keyed dict).
        """
        Lambda_blocks = {}
        for key, C_ell in C_ell_dict.items():
            if isinstance(key, SpectrumKey):
                comp_i, comp_j = key.comp_i, key.comp_j
            else:
                comp_i, comp_j = key
            Lambda_diag = self._build_lambda_diagonal(C_ell)
            Lambda_blocks[(comp_i, comp_j)] = Lambda_diag
            if comp_i != comp_j:
                Lambda_blocks[(comp_j, comp_i)] = Lambda_diag
        return Lambda_blocks

    def _build_lambda_matrix(self, C_ell_dict: dict) -> np.ndarray:
        """Build full Lambda matrix, auto-detecting the key shape.

        Accepts three key shapes (mid-migration):

        - :class:`SpectrumKey` instances (post-Slice-4 canonical)
        - 3-tuple ``(comp_i, comp_j, mode)`` — legacy
        - 2-tuple ``(comp_i, comp_j)`` — legacy single-mode
        """
        first_key = next(iter(C_ell_dict))
        if isinstance(first_key, SpectrumKey):
            return self._build_lambda_matrix_keyed(C_ell_dict)
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

        Uses _ell_to_modes_local for correct mode index placement.
        """
        if self._ell_to_modes_local is None:
            self._build_ell_mode_mapping()
        n = self._n_modes_base
        Lambda = np.zeros((2 * n, 2 * n), dtype=np.float64)

        for ell in range(self._lmin_smw, self._lmax_smw + 1):
            c_ee = C_ell_EE[ell] if ell < len(C_ell_EE) else 0.0
            c_bb = C_ell_BB[ell] if ell < len(C_ell_BB) else 0.0
            c_eb = 0.0
            if C_ell_EB is not None and ell < len(C_ell_EB):
                c_eb = C_ell_EB[ell]

            for idx in self._ell_to_modes_local[ell]:
                Lambda[idx, idx] = c_ee  # E-E block
                Lambda[n + idx, n + idx] = c_bb  # B-B block
                Lambda[idx, n + idx] = c_eb  # E-B block
                Lambda[n + idx, idx] = c_eb  # B-E block

        return Lambda

    def _build_lambda_matrix_keyed(
        self, C_ell_dict: dict[SpectrumKey, np.ndarray]
    ) -> np.ndarray:
        """Build full Lambda matrix from a :class:`SpectrumKey`-keyed dict.

        Mirrors :meth:`_build_lambda_matrix_3tuple` but reads
        ``key.comp_i`` / ``key.comp_j`` and translates ``key.kind`` to the
        int mode per-iteration via :func:`kind_to_legacy_mode`. No dict-level
        rewrap is performed.
        """
        lambda_matrix = np.zeros(
            (self.n_modes_total, self.n_modes_total), dtype=np.float64
        )

        pair_entries: dict[tuple[int, int], dict[int, np.ndarray]] = {}
        for key, C_ell in C_ell_dict.items():
            mode = kind_to_legacy_mode(key.kind)
            pair_entries.setdefault((key.comp_i, key.comp_j), {})[mode] = C_ell

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
                C_EE = mode_dict.get(0, np.zeros(self.lmax_signal + 1))
                C_BB = mode_dict.get(1, np.zeros(self.lmax_signal + 1))
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
                C_EE = mode_dict.get(0, np.zeros(self.lmax_signal + 1))
                C_BB = mode_dict.get(1, np.zeros(self.lmax_signal + 1))
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
