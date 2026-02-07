"""
Base class for compression methods.

This module provides the abstract base class that defines the interface
for all compression methods used in CMB Fisher matrix computation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np


class SMWPrepared(NamedTuple):
    """Pre-computed quantities for SMW-based likelihood evaluation.

    Parameters
    ----------
    factor : numpy.ndarray
        K Cholesky factor (harmonic) or C_c_inv (pixel_projected).
    reserved : None
        Unused, kept for API symmetry.
    logdet : float
        log|C| (full covariance log-determinant).
    """

    factor: np.ndarray
    reserved: None
    logdet: float


from ..basics import (
    legendre_plm,
    matrix_mult,
    matrix_slogdet_symm,
    matrix_trace,
    wigner_d_small,
)


class BaseCompression(ABC):
    """
    Abstract base class for compression methods.

    This class defines the interface for compression methods and provides
    shared implementations for spherical harmonic evaluation and ell-to-mode
    mapping.

    Supports both single-field and multi-field configurations. For multi-field,
    theta and phi are passed as tuples of arrays (one per component), and the
    harmonic operator V is built as a block-diagonal matrix.

    Parameters
    ----------
    N_inv : numpy.ndarray
        Precomputed noise inverse matrix of shape (n_pix_total, n_pix_total).
    theta : numpy.ndarray or tuple of numpy.ndarray
        Colatitude angles for active pixels in radians. Single array for
        single-field, tuple of arrays for multi-field.
    phi : numpy.ndarray or tuple of numpy.ndarray
        Longitude angles for active pixels in radians. Single array for
        single-field, tuple of arrays for multi-field.
    lmax : int
        Maximum multipole for harmonic expansion.
    beam : numpy.ndarray or None, optional
        Beam window function B_ℓ for ℓ=2 to lmax. Shape should be (lmax-1,).
        If provided, the harmonic operator V is multiplied by beam factors
        so that V_ℓm = B_ℓ × Y_ℓm. If None, no beam correction is applied.
    spins : list of int or None, optional
        Spin weight for each component. Default is [0, 0, ...] (all spin-0).
        Use spin=2 for polarization (Q/U) fields, which doubles pixel count
        and uses spin-weighted spherical harmonics for E/B decomposition.

    Attributes
    ----------
    n_pix : int
        Total number of active pixels across all components.
    n_modes : int
        Number of harmonic modes per component (for ℓ=2 to lmax).
    n_modes_total : int
        Total harmonic modes across all components (n_components × n_modes).
    n_kept : int
        Number of modes kept after compression (if applicable).
    n_components : int
        Number of field components (1 for single-field, N for multi-field).
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
    ):
        """
        Initialize compression base class.

        Parameters
        ----------
        N : numpy.ndarray
            Noise covariance matrix (n_pix_total, n_pix_total).
        N_inv : numpy.ndarray
            Precomputed noise inverse matrix (n_pix_total, n_pix_total).
        theta : numpy.ndarray or tuple of numpy.ndarray
            Colatitude angles for active pixels in radians. Single array for
            single-field, tuple of arrays for multi-field (one per component).
        phi : numpy.ndarray or tuple of numpy.ndarray
            Longitude angles for active pixels in radians. Single array for
            single-field, tuple of arrays for multi-field (one per component).
        lmax : int
            Maximum multipole for harmonic expansion.
        beam : numpy.ndarray or None, optional
            Beam window function B_ℓ for ℓ=2 to lmax.
        spins : list of int or None, optional
            Spin weight for each component (0 for scalar, 2 for polarization).
            Default is [0, ...] for all components. For spin-2 components,
            theta/phi represent physical pixel locations but V is built for
            (Q, U) → (E, B) transformation with doubled dimensions.
        lswitch_low : int or None, optional
            Minimum multipole where signal varies with parameters.
            If provided with lswitch_high, enables switch optimization.
        lswitch_high : int or None, optional
            Maximum multipole where signal varies with parameters.
            Multipoles above this use fixed fiducial spectrum.
        fiducial_C_ell : numpy.ndarray or None, optional
            Deprecated: Use S_fixed instead. Fiducial power spectrum for
            fixed multipoles (ℓ > lswitch_high).
        S_fixed : numpy.ndarray or None, optional
            Precomputed signal matrix for fixed multipoles (ℓ > lswitch_high).
            This is the recommended way to pass the fixed signal contribution.
            Shape should be (n_pix_total, n_pix_total).
        """
        self._N = np.asfortranarray(N, dtype=np.float64)
        self.N_inv = np.asfortranarray(N_inv, dtype=np.float64)
        self.lmax = lmax

        # Normalize theta/phi to tuple format for consistent handling
        # Single-field: 1D array → wrap as single-element tuple
        # Multi-field: already tuple
        if isinstance(theta, np.ndarray) and theta.ndim == 1:
            self._theta_tuple = (np.asarray(theta, dtype=np.float64),)
            self._phi_tuple = (np.asarray(phi, dtype=np.float64),)
        else:
            self._theta_tuple = tuple(np.asarray(t, dtype=np.float64) for t in theta)
            self._phi_tuple = tuple(np.asarray(p, dtype=np.float64) for p in phi)

        # Multi-field tracking
        self.n_components = len(self._theta_tuple)

        # Store spins for each component (0 for scalar, 2 for polarization)
        if spins is None:
            self._spins = [0] * self.n_components
        else:
            if len(spins) != self.n_components:
                raise ValueError(
                    f"spins list length ({len(spins)}) must match "
                    f"number of components ({self.n_components})"
                )
            for spin in spins:
                if spin not in (0, 2):
                    raise ValueError(
                        f"Spin must be 0 (scalar) or 2 (polarization), got {spin}"
                    )
            self._spins = list(spins)

        # Pixel count per component: spin-2 has 2x pixels (Q, U)
        self._n_physical_pix = [len(t) for t in self._theta_tuple]
        self._n_pix_per_component = [
            2 * n if self._spins[i] == 2 else n
            for i, n in enumerate(self._n_physical_pix)
        ]

        # Compute pixel offsets for block structure
        self._pix_offsets = [0]
        for n in self._n_pix_per_component:
            self._pix_offsets.append(self._pix_offsets[-1] + n)

        # For backward compatibility: theta/phi as concatenated arrays
        if self.n_components == 1:
            self.theta = self._theta_tuple[0]
            self.phi = self._phi_tuple[0]
        else:
            self.theta = np.concatenate(self._theta_tuple)
            self.phi = np.concatenate(self._phi_tuple)

        # Store lswitch parameters for reduced-dimension SMW
        self.lswitch_low = lswitch_low if lswitch_low is not None else 2
        self.lswitch_high = lswitch_high if lswitch_high is not None else lmax
        self._fiducial_C_ell = fiducial_C_ell
        self._S_fixed = S_fixed
        self._use_switch_optimization = lswitch_high is not None and lswitch_high < lmax

        # Store beam window function
        if beam is not None:
            beam = np.asarray(beam, dtype=np.float64)
            expected_len = lmax - 1  # ell = 2 to lmax
            if beam.shape[0] != expected_len:
                raise ValueError(
                    f"Beam must have length {expected_len} (ell=2 to lmax={lmax}), "
                    f"got {beam.shape[0]}"
                )
            self._beam = beam
        else:
            self._beam = None

        # Derived quantities
        self.n_pix = sum(self._n_pix_per_component)

        # n_modes per component depends on whether switch optimization is used
        if self._use_switch_optimization:
            # Only modes for ℓ in [lswitch_low, lswitch_high]
            self._n_modes_base = (self.lswitch_high + 1) ** 2 - (self.lswitch_low) ** 2
            self._lmin_smw = self.lswitch_low
            self._lmax_smw = self.lswitch_high
        else:
            # All modes from ℓ=2 to lmax
            self._n_modes_base = (lmax + 1) ** 2 - 4
            self._lmin_smw = 2
            self._lmax_smw = lmax

        # Mode count per component: spin-2 has 2x modes (E, B)
        self._n_modes_per_component_list = [
            2 * self._n_modes_base if self._spins[i] == 2 else self._n_modes_base
            for i in range(self.n_components)
        ]

        # For backward compatibility (single-field case)
        self._n_modes_per_component = self._n_modes_base

        # Total modes across all components
        self.n_modes_total = sum(self._n_modes_per_component_list)

        # For backward compatibility: n_modes is per-component for single-field,
        # total for multi-field Fisher computation
        self.n_modes = self._n_modes_base

        # Compute mode offsets for block structure
        self._mode_offsets = [0]
        for n in self._n_modes_per_component_list:
            self._mode_offsets.append(self._mode_offsets[-1] + n)

        # To be set by subclasses
        self._V = None
        self._V_blocks = None  # List of V_i for each component
        self._ell_to_modes = None
        self.n_kept = self.n_modes_total if self.n_components > 1 else self.n_modes

    @abstractmethod
    def setup(self) -> None:
        """
        Initialize compression-specific components.

        Must be called after initialization before using any other methods.
        """
        pass

    @property
    @abstractmethod
    def method(self) -> str:
        """Compression method name: "harmonic" or "pixel_projected"."""
        pass

    @property
    @abstractmethod
    def projector(self) -> np.ndarray:
        """
        Get the projection matrix that maps pixel space to compressed space.

        This is the fundamental operator that defines the compression:
        - HarmonicCompression: V (n_modes × n_pix)
        - PixelProjectedCompression: U^T (n_kept × n_pix)

        Returns
        -------
        numpy.ndarray
            Projection matrix of shape (n_compressed, n_pix).
        """
        pass

    @property
    @abstractmethod
    def n_compressed(self) -> int:
        """
        Size of the compressed space (number of rows in projector).

        - HarmonicCompression: n_modes
        - PixelProjectedCompression: n_kept

        Returns
        -------
        int
            Dimension of compressed space.
        """
        pass

    @abstractmethod
    def get_projected_inverse(self, C_ell: np.ndarray) -> np.ndarray:
        """
        Get the projected inverse for Fisher computation.

        Parameters
        ----------
        C_ell : numpy.ndarray
            Power spectrum values for ell = 2 to lmax.

        Returns
        -------
        numpy.ndarray
            Projected inverse covariance matrix.
        """
        pass

    @abstractmethod
    def get_derivative_matrix(self, ell: int) -> np.ndarray:
        """
        Get the derivative matrix ∂C/∂C_ℓ in the compressed basis.

        Parameters
        ----------
        ell : int
            Multipole for which to compute the derivative.

        Returns
        -------
        numpy.ndarray
            Derivative matrix of shape (n_compressed, n_compressed).
        """
        pass

    @abstractmethod
    def get_compressed_covariance(self, C_ell: np.ndarray) -> np.ndarray:
        """
        Compute covariance matrix in the compressed space.

        - HarmonicCompression: C̄ = V @ N @ V^T + Λ
        - PixelProjectedCompression: C_c = U^T @ C @ U

        Parameters
        ----------
        C_ell : numpy.ndarray
            Power spectrum values for ell = 2 to lmax.

        Returns
        -------
        numpy.ndarray
            Compressed covariance of shape (n_compressed, n_compressed).
        """
        pass

    @abstractmethod
    def get_weighted_compressed_data(
        self, data: np.ndarray, C_ell: np.ndarray
    ) -> np.ndarray:
        """
        Compute weighted compressed data for QML estimation.

        Returns the vector w such that QML estimator is:
            q_ℓ = (1/2) * w^T @ ∂C_c/∂C_ℓ @ w

        - HarmonicCompression: w = V @ C^{-1} @ d (via SMW)
        - PixelProjectedCompression: w = C_c^{-1} @ (U^T @ d)

        Parameters
        ----------
        data : numpy.ndarray
            Pixel-space data vector of shape (n_pix,) or (n_pix, n_sims).
        C_ell : numpy.ndarray
            Power spectrum values for ell = 2 to lmax.

        Returns
        -------
        numpy.ndarray
            Weighted compressed data of shape (n_compressed,) or (n_compressed, n_sims).
        """
        pass

    # === Shared implementations ===

    def get_compressed_inverse(self, C_ell: np.ndarray) -> np.ndarray:
        """
        Compute inverse of compressed covariance matrix.

        This default implementation inverts get_compressed_covariance().
        Subclasses may override for more efficient implementations.

        Parameters
        ----------
        C_ell : numpy.ndarray
            Power spectrum values for ell = 2 to lmax.

        Returns
        -------
        numpy.ndarray
            Inverse compressed covariance of shape (n_compressed, n_compressed).
        """
        from ..basics import matrix_inverse_symm

        C_compressed = self.get_compressed_covariance(C_ell)
        return matrix_inverse_symm(C_compressed, overwrite=True)

    def get_compressed_logdet(self, C_ell: np.ndarray) -> float:
        """
        Compute log determinant of compressed covariance matrix.

        Parameters
        ----------
        C_ell : numpy.ndarray
            Power spectrum values for ell = 2 to lmax.

        Returns
        -------
        float
            Log determinant of compressed covariance.
        """
        C_compressed = self.get_compressed_covariance(C_ell)
        _, logdet = matrix_slogdet_symm(np.asfortranarray(C_compressed))
        return logdet

    def get_full_logdet(self, C_ell: np.ndarray) -> float:
        """
        Get best available log determinant of full covariance.

        For harmonic compression, returns exact log|C| via SMW formula.
        For pixel_projected, returns log|C_compressed| (approximation).

        Subclasses may override to provide exact computation.
        """
        return self.get_compressed_logdet(C_ell)

    def get_full_logdet_multi(self, C_ell_dict: dict) -> float:
        """
        Get best available log determinant for multi-field covariance.

        For harmonic compression, returns exact log|C| via SMW formula.
        For pixel_projected, returns log|C_compressed| (approximation).

        Subclasses may override to provide exact computation.
        """
        return self.get_logdet_multi(C_ell_dict)

    def _build_harmonic_operator(self) -> None:
        """
        Build the harmonic projection operator V using real spherical harmonics.

        For single-field: V is (n_modes × n_pix)
        For multi-field: V is block-diagonal (n_components × n_modes, n_pix_total)

        V projects from pixel space to harmonic space. Each row of V
        corresponds to a (ell, m) mode, and each column to a pixel.

        For spin-0 (scalar) components:
            V[mode, pix] = Y_ℓm(theta[pix], phi[pix])

        For spin-2 (polarization) components:
            V maps (Q, U) pixel data to (E, B) harmonic modes using
            spin-weighted spherical harmonics _±2Y_ℓm.

        Beam effects are expected to be incorporated in the C_ell power spectrum
        passed to methods like get_compressed_covariance(). This is consistent
        with how spectra_manager stores beam-convolved power spectra.

        When switch optimization is enabled (lswitch_high < lmax), V is built
        only for multipoles in [lswitch_low, lswitch_high], significantly
        reducing the dimension of the SMW operations.

        This implementation uses JIT-compiled recurrence relations for associated
        Legendre polynomials (spin-0) and Wigner d-matrices (spin-2).
        """
        # Build V block for each component, dispatching based on spin
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

        # Assemble into full block-diagonal V
        if self.n_components == 1:
            # Single-field: V is just the single block
            self._V = np.asfortranarray(self._V_blocks[0])
        else:
            # Multi-field: block-diagonal V
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
        """
        Build harmonic operator V for a single component.

        Parameters
        ----------
        theta : numpy.ndarray
            Colatitude angles for this component's pixels.
        phi : numpy.ndarray
            Longitude angles for this component's pixels.

        Returns
        -------
        numpy.ndarray
            Harmonic operator V of shape (n_modes_per_component, n_pix_component).
        """
        n_pix_comp = len(theta)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        V = np.zeros((self._n_modes_per_component, n_pix_comp), dtype=np.float64)

        # Determine ell range for V (may be reduced if switch optimization is used)
        lmin_v = self._lmin_smw
        lmax_v = self._lmax_smw

        # Precompute cos(m*phi) and sin(m*phi) for all m and pixels
        cos_mphi = np.zeros((lmax_v + 1, n_pix_comp), dtype=np.float64)
        sin_mphi = np.zeros((lmax_v + 1, n_pix_comp), dtype=np.float64)
        for m in range(lmax_v + 1):
            for ipix in range(n_pix_comp):
                cos_mphi[m, ipix] = np.cos(m * phi[ipix])
                sin_mphi[m, ipix] = np.sin(m * phi[ipix])

        # Buffer for P_ℓ^m values for one pixel (need full lmax for Legendre)
        plm = np.zeros((self.lmax + 1, self.lmax + 1), dtype=np.float64)

        # Process each pixel
        for ipix in range(n_pix_comp):
            # Compute all normalized P_ℓ^m for this pixel
            legendre_plm(cos_theta[ipix], sin_theta[ipix], plm)

            # Fill V for this pixel (only for ℓ in [lmin_v, lmax_v])
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
        """
        Build harmonic operator V for a spin-2 (polarization) component.

        For spin-2, V maps (Q, U) pixel data to (E, B) mode coefficients
        using spin-weighted spherical harmonics _±2Y_ℓm.

        The transformation uses:
            E_ℓm = -1/2 (_{+2}a_ℓm + _{-2}a_ℓm)
            B_ℓm = i/2 (_{+2}a_ℓm - _{-2}a_ℓm)

        where _±2a_ℓm are the spin-2 harmonic coefficients of (Q ± iU).

        Parameters
        ----------
        theta : numpy.ndarray
            Colatitude angles for physical pixel locations.
        phi : numpy.ndarray
            Longitude angles for physical pixel locations.

        Returns
        -------
        numpy.ndarray
            Harmonic operator V of shape (2 * n_modes, 2 * n_pix), where:
            - Rows 0:n_modes are E modes
            - Rows n_modes:2*n_modes are B modes
            - Cols 0:n_pix are Q pixels
            - Cols n_pix:2*n_pix are U pixels

        References
        ----------
        .. [1] Zaldarriaga, M. & Seljak, U. "All-sky analysis of polarization
           in the microwave background" Phys. Rev. D 55, 1830 (1997)
        """
        n_pix = len(theta)
        n_modes = self._n_modes_base  # E modes or B modes count

        # V has shape (2*n_modes, 2*n_pix): [E,B] modes × [Q,U] pixels
        V = np.zeros((2 * n_modes, 2 * n_pix), dtype=np.float64)

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # Determine ell range
        lmin_v = self._lmin_smw
        lmax_v = self._lmax_smw

        # Precompute cos(mφ) and sin(mφ) for all m and pixels
        cos_mphi = np.zeros((lmax_v + 1, n_pix), dtype=np.float64)
        sin_mphi = np.zeros((lmax_v + 1, n_pix), dtype=np.float64)
        for m in range(lmax_v + 1):
            for ipix in range(n_pix):
                cos_mphi[m, ipix] = np.cos(m * phi[ipix])
                sin_mphi[m, ipix] = np.sin(m * phi[ipix])

        # Process each pixel
        for ipix in range(n_pix):
            cos_th = cos_theta[ipix]
            sin_th = sin_theta[ipix]

            mode_idx = 0
            for ell in range(lmin_v, lmax_v + 1):
                # Per-ℓ normalization: sqrt((ℓ+2)(ℓ+1)ℓ(ℓ-1)) ensures that
                # V^T Λ V reproduces the traditional pixel-space signal matrix
                # when Λ contains C_ℓ × (2ℓ+1)/(4π) / ((ℓ+2)(ℓ+1)ℓ(ℓ-1)).
                norm_ell = np.sqrt((ell + 2) * (ell + 1) * ell * (ell - 1))

                for m in range(-ell, ell + 1):
                    abs_m = abs(m)

                    # Compute Wigner d-matrix elements for spin ±2 at |m|
                    # Using |m| ensures correct real spin-2 harmonic basis:
                    # the m<0 real modes use the same d-functions as m>0.
                    d_plus2 = wigner_d_small(ell, abs_m, 2, cos_th, sin_th)
                    d_minus2 = wigner_d_small(ell, abs_m, -2, cos_th, sin_th)

                    # Combination for E and B modes
                    # D^+ = d_{|m|,-2} + d_{|m|,2} (E-mode pattern on Q)
                    # D^- = d_{|m|,-2} - d_{|m|,2} (B-mode pattern / E on U)
                    D_plus = d_minus2 + d_plus2
                    D_minus = d_minus2 - d_plus2

                    # V entries from real spin-2 harmonics derivation:
                    #
                    # From Q + iU = -Σ_m (E_lm - iB_lm) × _2Y_lm, using real
                    # mode expansion E_lm = (e_m - ie_{-m})/√2 for m>0:
                    #
                    # m>0: V_{E,Q} = D⁺ cos(mφ),    V_{E,U} = D⁻ sin(mφ)
                    # m=0: V_{E,Q} = D⁺,             V_{E,U} = 0
                    # m<0: V_{E,Q} = D⁺ sin(|m|φ),  V_{E,U} = -D⁻ cos(|m|φ)
                    #
                    # B modes follow by swapping D⁺ ↔ -D⁻ in the E pattern.

                    if m == 0:
                        scale = norm_ell * 0.5

                        # E mode: Q = D⁺, U = 0
                        V[mode_idx, ipix] = scale * D_plus
                        V[mode_idx, n_pix + ipix] = 0.0

                        # B mode: Q = 0, U = D⁺
                        V[n_modes + mode_idx, ipix] = 0.0
                        V[n_modes + mode_idx, n_pix + ipix] = scale * D_plus
                    elif m > 0:
                        scale = norm_ell * np.sqrt(2.0) * 0.5
                        cm = cos_mphi[m, ipix]
                        sm = sin_mphi[m, ipix]

                        # E mode
                        V[mode_idx, ipix] = scale * D_plus * cm
                        V[mode_idx, n_pix + ipix] = scale * D_minus * sm

                        # B mode
                        V[n_modes + mode_idx, ipix] = -scale * D_minus * sm
                        V[n_modes + mode_idx, n_pix + ipix] = scale * D_plus * cm
                    else:  # m < 0
                        scale = norm_ell * np.sqrt(2.0) * 0.5
                        cm = cos_mphi[abs_m, ipix]
                        sm = sin_mphi[abs_m, ipix]

                        # E mode (sin/cos swapped vs m>0, sign flip on U)
                        V[mode_idx, ipix] = scale * D_plus * sm
                        V[mode_idx, n_pix + ipix] = -scale * D_minus * cm

                        # B mode
                        V[n_modes + mode_idx, ipix] = scale * D_minus * cm
                        V[n_modes + mode_idx, n_pix + ipix] = scale * D_plus * sm

                    mode_idx += 1

        return V

    def _build_ell_mode_mapping(self) -> None:
        """
        Build mapping from multipole ell to mode indices.

        This is used for computing derivatives with respect to C_ell,
        where E_ell is a diagonal matrix with 1s for modes at multipole ell.

        When switch optimization is enabled, mapping is built only for
        multipoles in [lswitch_low, lswitch_high].

        For multi-field: also builds per-component local mode indices.
        """
        # Local mode indices within a single component
        self._ell_to_modes_local = {}
        mode_idx = 0
        for ell in range(self._lmin_smw, self._lmax_smw + 1):
            n_m = 2 * ell + 1
            self._ell_to_modes_local[ell] = list(range(mode_idx, mode_idx + n_m))
            mode_idx += n_m

        # For backward compatibility: single-field uses local = global
        if self.n_components == 1:
            self._ell_to_modes = self._ell_to_modes_local
        else:
            # For multi-field, we keep local mapping and compute global on demand
            self._ell_to_modes = self._ell_to_modes_local

    def _precompute_derivative_diagonals(self) -> None:
        """
        Precompute derivative matrix diagonals E_ℓ for multipoles in SMW range.

        The derivative matrix for multipole ℓ is diagonal with factor
        (2ℓ+1)/(4π) at positions corresponding to modes at that ℓ.
        Storing these diagonals avoids repeated array allocations.

        When switch optimization is enabled, only computes for ℓ in
        [lswitch_low, lswitch_high].

        For multi-field: precomputes diagonals for both single-component (local)
        and full block-diagonal structure.

        This is called during setup and the diagonals are reused in
        compute_fisher_matrix() and get_derivative_matrix().

        References
        ----------
        .. [1] Tegmark, M. "How to measure CMB power spectra without losing
           information" Phys. Rev. D 55, 5895 (1997) - Equation 16
        """
        # Precompute local (per-component) derivative diagonals
        self._derivative_diagonals_local = {}
        for ell in range(self._lmin_smw, self._lmax_smw + 1):
            E_diag = np.zeros(self._n_modes_per_component, dtype=np.float64)
            if ell in self._ell_to_modes_local:
                chngconv = (2 * ell + 1) / (4 * np.pi)
                for mode_idx in self._ell_to_modes_local[ell]:
                    E_diag[mode_idx] = chngconv
            self._derivative_diagonals_local[ell] = E_diag

        # For backward compatibility: single-field uses local = global
        if self.n_components == 1:
            self._derivative_diagonals = self._derivative_diagonals_local
        else:
            # For multi-field, we need block-diagonal derivatives
            # These are computed on-demand in get_derivative_matrix_multi()
            self._derivative_diagonals = self._derivative_diagonals_local

    def _build_derivative_matrix_with_spins(
        self, ell: int, comp_i: int, comp_j: int, mode: int = 0
    ) -> np.ndarray:
        """Build full harmonic-space derivative matrix for (comp_i, comp_j, mode).

        Handles all spin combinations: 0x0, 2x2, 0x2, 2x0.
        Returns the E matrix in n_modes_total x n_modes_total space.
        """
        E = np.zeros((self.n_modes_total, self.n_modes_total), dtype=np.float64)
        chngconv = (2 * ell + 1) / (4 * np.pi)
        local_mode_indices = self._ell_to_modes_local[ell]
        n_base = self._n_modes_base

        spin_i = self._spins[comp_i]
        spin_j = self._spins[comp_j]

        if spin_i == 0 and spin_j == 0:
            row_offset = self._mode_offsets[comp_i]
            col_offset = self._mode_offsets[comp_j]
            for idx in local_mode_indices:
                E[row_offset + idx, col_offset + idx] = chngconv
            if comp_i != comp_j:
                for idx in local_mode_indices:
                    E[col_offset + idx, row_offset + idx] = chngconv

        elif spin_i == 2 and spin_j == 2:
            factor2 = 1.0 / ((ell + 2) * (ell + 1) * ell * (ell - 1))
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
            factor = np.sqrt(1.0 / ((ell + 2) * (ell + 1) * ell * (ell - 1)))
            deriv_val = -chngconv * factor
            row_start = self._mode_offsets[comp_i]
            col_start = self._mode_offsets[comp_j]
            col_sub = col_start + mode * n_base
            for idx in local_mode_indices:
                E[row_start + idx, col_sub + idx] = deriv_val
                E[col_sub + idx, row_start + idx] = deriv_val

        elif spin_i == 2 and spin_j == 0:
            factor = np.sqrt(1.0 / ((ell + 2) * (ell + 1) * ell * (ell - 1)))
            deriv_val = -chngconv * factor
            row_start = self._mode_offsets[comp_i]
            col_start = self._mode_offsets[comp_j]
            row_sub = row_start + mode * n_base
            for idx in local_mode_indices:
                E[row_sub + idx, col_start + idx] = deriv_val
                E[col_start + idx, row_sub + idx] = deriv_val

        return E

    def _build_lambda_blocks(
        self, C_ell_dict: dict[tuple[int, int], np.ndarray]
    ) -> dict[tuple[int, int], np.ndarray]:
        """
        Build Lambda blocks from cross-power spectra dictionary.

        For multi-field compression, Lambda is a block matrix where
        Lambda^{ij} is diagonal with C_ℓ^{ij} entries.

        Parameters
        ----------
        C_ell_dict : dict
            Dictionary mapping (comp_i, comp_j) to C_ell array.
            Keys are component index pairs, values are power spectra
            with C_ell[0] = C_2, C_ell[1] = C_3, etc.

        Returns
        -------
        dict
            Dictionary mapping (comp_i, comp_j) to Lambda diagonal arrays.
        """
        Lambda_blocks = {}
        for (comp_i, comp_j), C_ell in C_ell_dict.items():
            Lambda_diag = self._build_lambda_diagonal(C_ell)
            Lambda_blocks[(comp_i, comp_j)] = Lambda_diag
            # Symmetric: Lambda^{ji} = Lambda^{ij}
            if comp_i != comp_j:
                Lambda_blocks[(comp_j, comp_i)] = Lambda_diag
        return Lambda_blocks

    def _build_lambda_full(self, C_ell_dict: dict) -> np.ndarray:
        """Build full Lambda matrix, auto-detecting 2-tuple or 3-tuple keys."""
        first_key = next(iter(C_ell_dict))
        if len(first_key) == 3:
            return self._build_lambda_full_3tuple(C_ell_dict)
        return self._build_lambda_full_2tuple(C_ell_dict)

    def _build_lambda_full_2tuple(
        self, C_ell_dict: dict[tuple[int, int], np.ndarray]
    ) -> np.ndarray:
        """Build full Lambda matrix from 2-tuple (comp_i, comp_j) keys."""
        Lambda_blocks = self._build_lambda_blocks(C_ell_dict)

        Lambda_full = np.zeros((self.n_modes_total, self.n_modes_total), dtype=np.float64)
        for (comp_i, comp_j), Lambda_diag in Lambda_blocks.items():
            row_start = self._mode_offsets[comp_i]
            col_start = self._mode_offsets[comp_j]
            for k, val in enumerate(Lambda_diag):
                Lambda_full[row_start + k, col_start + k] = val

        return Lambda_full

    def _build_lambda_diagonal(self, C_ell: np.ndarray) -> np.ndarray:
        """
        Build Λ diagonal from C_ell values in the (ℓ,m) basis.

        The input C_ell values are assumed to already include all necessary
        normalization factors and beam smoothing from SpectraManager:
        - (2ℓ+1)/(4π) from apply_normalization_factors()
        - 2π/(ℓ(ℓ+1)) × B_ℓ² from apply_smoothing()

        This matches the traditional pixel-space signal matrix formula where
        the cls already contain all the physics factors. Since the harmonic
        operator V uses unnormalized spherical harmonics satisfying
        Σ_m Y'_ℓm(p) × Y'_ℓm(q) = P_ℓ(cos γ), we use:
            Λ_ℓ = C_ℓ (as-is, no additional factors)

        This ensures V^T Λ V = S (the traditional signal matrix).

        When switch optimization is enabled, only builds Λ for multipoles
        in [lswitch_low, lswitch_high], matching the reduced V dimension.

        Note: The beam parameter stored in this class is NOT used here
        because the cls already have beam smoothing applied. The beam is
        stored for reference and for methods that need the raw beam values.

        Parameters
        ----------
        C_ell : numpy.ndarray
            Power spectrum with C_ell[0] = C_2, C_ell[1] = C_3, etc.
            Expected to already include all normalization and beam factors.

        Returns
        -------
        numpy.ndarray
            Diagonal elements of Λ, shape (n_modes,).
        """
        Lambda_diag = np.zeros(self.n_modes)
        idx = 0
        for ell in range(self._lmin_smw, self._lmax_smw + 1):
            n_m = 2 * ell + 1
            c_ell_value = C_ell[ell - 2] if ell - 2 < len(C_ell) else 0.0

            # Note: C_ell already includes all normalization and beam factors
            # from SpectraManager, so no additional processing needed
            Lambda_diag[idx : idx + n_m] = c_ell_value
            idx += n_m
        return Lambda_diag

    def _build_lambda_block_spin2(
        self,
        C_ell_EE: np.ndarray,
        C_ell_BB: np.ndarray,
        C_ell_EB: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Build Lambda block for a spin-2 auto-correlation (EE, BB, EB).

        For polarization, Lambda has 2x2 block structure at each (ℓ,m):
            Λ_{ℓm} = | C_ℓ^{EE}   C_ℓ^{EB} |
                      | C_ℓ^{EB}   C_ℓ^{BB} |

        Parameters
        ----------
        C_ell_EE : numpy.ndarray
            EE power spectrum, C_ell[0] = C_2.
        C_ell_BB : numpy.ndarray
            BB power spectrum.
        C_ell_EB : numpy.ndarray or None
            EB cross-spectrum. If None, assumed zero.

        Returns
        -------
        numpy.ndarray
            Lambda block of shape (2 * n_modes_base, 2 * n_modes_base).
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

    def _build_lambda_full_3tuple(
        self, C_ell_dict: dict[tuple, np.ndarray]
    ) -> np.ndarray:
        """
        Build full Lambda matrix handling mixed spin-0/spin-2 components.

        Accepts 3-tuple keys (comp_i, comp_j, mode) matching the
        get_cls(field_i, field_j, mode) API:
        - spin-0 x spin-0: mode 0 only
        - spin-2 x spin-2: mode 0=EE, 1=BB, 2=EB
        - spin-0 x spin-2: mode 0=TE, 1=TB

        Parameters
        ----------
        C_ell_dict : dict
            Dictionary with 3-tuple keys (comp_i, comp_j, mode).

        Returns
        -------
        numpy.ndarray
            Full Lambda matrix of shape (n_modes_total, n_modes_total).
        """
        Lambda_full = np.zeros((self.n_modes_total, self.n_modes_total), dtype=np.float64)

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
                # Scalar x Scalar: single diagonal (+ symmetric for cross-spectra)
                diag = self._build_lambda_diagonal(mode_dict[0])
                for k, val in enumerate(diag):
                    Lambda_full[row_start + k, col_start + k] = val
                    if ci != cj:
                        Lambda_full[col_start + k, row_start + k] = val

            elif spin_i == 2 and spin_j == 2:
                # Spin-2 x Spin-2: EE, BB, EB sub-blocks
                C_EE = mode_dict.get(0, np.zeros(self.lmax - 1))
                C_BB = mode_dict.get(1, np.zeros(self.lmax - 1))
                C_EB = mode_dict.get(2, None)
                block = self._build_lambda_block_spin2(C_EE, C_BB, C_EB)
                n_block = 2 * self._n_modes_base
                Lambda_full[
                    row_start : row_start + n_block,
                    col_start : col_start + n_block,
                ] = block

            elif spin_i == 0 and spin_j == 2:
                # Scalar x Spin-2: TE and TB sub-blocks
                # Sign convention: E = -(_2Y + _{-2}Y)/2 introduces a minus
                # sign in the spin-0 x spin-2 cross signal (see compute_02_contribution
                # in pixel.py). We negate here so V^T Lambda V matches the
                # traditional pixel-space signal matrix.
                n_base = self._n_modes_base
                for mode, C_ell in mode_dict.items():
                    diag = self._build_lambda_diagonal(C_ell)
                    # mode 0 = TE → E sub-block, mode 1 = TB → B sub-block
                    col_sub = col_start + mode * n_base
                    for k, val in enumerate(diag):
                        Lambda_full[row_start + k, col_sub + k] = -val
                        # Symmetric
                        Lambda_full[col_sub + k, row_start + k] = -val

            elif spin_i == 2 and spin_j == 0:
                # Spin-2 x Scalar: transpose of above (already handled by symmetry)
                n_base = self._n_modes_base
                for mode, C_ell in mode_dict.items():
                    diag = self._build_lambda_diagonal(C_ell)
                    row_sub = row_start + mode * n_base
                    for k, val in enumerate(diag):
                        Lambda_full[row_sub + k, col_start + k] = -val
                        Lambda_full[col_start + k, row_sub + k] = -val

        return Lambda_full

    def compute_fisher_matrix(
        self,
        C_ell: np.ndarray,
        ell_min: int = 2,
        ell_max: int | None = None,
    ) -> np.ndarray:
        """
        Compute full Fisher matrix efficiently with precomputed projected inverse.

        This method precomputes V C^{-1} V^T once and reuses it for all Fisher
        elements, providing O(ℓ²) speedup over calling compute_fisher_element()
        in a loop.

        The Fisher matrix element is:
            F_ij = (1/2) Tr[(V C^{-1} V^T) E_i (V C^{-1} V^T) E_j]

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
            Fisher matrix of shape (n_ell, n_ell) where n_ell = ell_max - ell_min + 1.
            Index [i, j] corresponds to ell_i = ell_min + i, ell_j = ell_min + j.

        Notes
        -----
        This implements the efficient Fisher computation from Gjerløw et al. (2015),
        Section 6.3, where V^T C^{-1} V is precomputed once:

            F_{bb'} = ½ tr[(V^T C^{-1} V) I_b (V^T C^{-1} V) I_{b'}]

        References
        ----------
        .. [1] Gjerløw, E., et al. (2015). "Optimized Large-Scale CMB Likelihood
           and Quadratic Maximum Likelihood Power Spectrum Estimation."
        """
        if ell_max is None:
            ell_max = self.lmax

        n_ell = ell_max - ell_min + 1
        fisher = np.zeros((n_ell, n_ell))

        # Precompute V C^{-1} V^T ONCE (the key optimization)
        V_Cinv_VT = self.get_projected_inverse(C_ell)

        # Precompute (V C^{-1} V^T) @ E_ℓ for all ℓ
        # This avoids redundant matrix multiplications
        VCinvVT_E = {}
        for ell in range(ell_min, ell_max + 1):
            E_ell = self.get_derivative_matrix(ell)
            VCinvVT_E[ell] = matrix_mult(V_Cinv_VT, E_ell)

        # Compute Fisher elements using precomputed products
        # Use optimized matrix_trace which is O(n²) instead of np.trace(A @ B)
        # which is O(n³)
        for ell_i in range(ell_min, ell_max + 1):
            for ell_j in range(ell_i, ell_max + 1):
                # F_ij = 0.5 * Tr[(V C^{-1} V^T @ E_i) @ (V C^{-1} V^T @ E_j)]
                fisher_val = 0.5 * matrix_trace(VCinvVT_E[ell_i], VCinvVT_E[ell_j])

                idx_i = ell_i - ell_min
                idx_j = ell_j - ell_min

                fisher[idx_i, idx_j] = fisher_val
                if idx_i != idx_j:
                    fisher[idx_j, idx_i] = fisher_val  # Symmetry

        return fisher

    def compress_data(self, data: np.ndarray) -> np.ndarray:
        """
        Project pixel data to compressed representation: d_c = P @ d.

        Uses the compression-specific projector P:
        - HarmonicCompression: P = V (n_modes × n_pix)
        - PixelProjectedCompression: P = U^T (n_kept × n_pix)

        Parameters
        ----------
        data : numpy.ndarray
            Pixel-space data vector of shape (n_pix,) or (n_pix, n_sims).

        Returns
        -------
        numpy.ndarray
            Compressed data of shape (n_compressed,) or (n_compressed, n_sims).
        """
        return matrix_mult(self.projector, data)

    def get_cls_vector(self) -> np.ndarray:
        """
        Get a placeholder C_ell vector for the configured lmax.

        Returns
        -------
        numpy.ndarray
            Array of zeros with length (lmax - 1).
        """
        return np.zeros(self.lmax - 1)

    @property
    def compression_ratio(self) -> float:
        """
        Ratio of kept modes to original modes.

        Returns
        -------
        float
            Compression ratio (1.0 means no compression).
        """
        return self.n_kept / self.n_modes
