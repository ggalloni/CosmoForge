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
    matrix_mult,
    matrix_slogdet_symm,
    matrix_trace,
)
from .harmonic_basis import HarmonicBasis


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

        # Harmonic basis helper (V, Lambda, derivative construction)
        self._harmonic_basis = HarmonicBasis(self)

        # To be set by _build_basis() during setup()
        self._V = None
        self._V_blocks = None
        self._ell_to_modes = None
        self._ell_to_modes_local = None
        self._derivative_diagonals = None
        self._derivative_diagonals_local = None
        self.n_kept = self.n_modes_total if self.n_components > 1 else self.n_modes

    @abstractmethod
    def setup(self) -> None:
        """
        Initialize compression-specific components.

        Must be called after initialization before using any other methods.
        """
        pass

    def _build_basis(self) -> None:
        """Build harmonic basis and copy results to self.

        Calls HarmonicBasis.build() then copies output attributes back so that
        subclasses and external code can access them via self._V, self._V_blocks,
        self._derivative_diagonals, etc.
        """
        self._harmonic_basis.build()
        self._V = self._harmonic_basis._V
        self._V_blocks = self._harmonic_basis._V_blocks
        self._ell_to_modes = self._harmonic_basis._ell_to_modes
        self._ell_to_modes_local = self._harmonic_basis._ell_to_modes_local
        self._derivative_diagonals = self._harmonic_basis._derivative_diagonals
        self._derivative_diagonals_local = (
            self._harmonic_basis._derivative_diagonals_local
        )

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

    # =========================================================================
    # Delegates to HarmonicBasis
    # =========================================================================
    # These methods forward to self._harmonic_basis so that subclasses and
    # external code (spectra.py) can keep using self._build_lambda_full() etc.

    def _build_lambda_diagonal(self, C_ell: np.ndarray) -> np.ndarray:
        """Build Lambda diagonal from C_ell. Delegates to HarmonicBasis."""
        return self._harmonic_basis._build_lambda_diagonal(C_ell)

    def _build_lambda_blocks(
        self, C_ell_dict: dict[tuple[int, int], np.ndarray]
    ) -> dict[tuple[int, int], np.ndarray]:
        """Build Lambda blocks from 2-tuple dict. Delegates to HarmonicBasis."""
        return self._harmonic_basis._build_lambda_blocks(C_ell_dict)

    def _build_lambda_full(self, C_ell_dict: dict) -> np.ndarray:
        """Build full Lambda matrix (auto-detects key format)."""
        return self._harmonic_basis._build_lambda_full(C_ell_dict)

    def _build_lambda_full_2tuple(
        self, C_ell_dict: dict[tuple[int, int], np.ndarray]
    ) -> np.ndarray:
        """Build full Lambda from 2-tuple keys. Delegates to HarmonicBasis."""
        return self._harmonic_basis._build_lambda_full_2tuple(C_ell_dict)

    def _build_lambda_full_3tuple(
        self, C_ell_dict: dict[tuple, np.ndarray]
    ) -> np.ndarray:
        """Build full Lambda from 3-tuple keys. Delegates to HarmonicBasis."""
        return self._harmonic_basis._build_lambda_full_3tuple(C_ell_dict)

    def _build_lambda_block_spin2(
        self,
        C_ell_EE: np.ndarray,
        C_ell_BB: np.ndarray,
        C_ell_EB: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build Lambda block for spin-2. Delegates to HarmonicBasis."""
        return self._harmonic_basis._build_lambda_block_spin2(
            C_ell_EE, C_ell_BB, C_ell_EB
        )

    def _build_derivative_matrix_with_spins(
        self, ell: int, comp_i: int, comp_j: int, mode: int = 0
    ) -> np.ndarray:
        """Build derivative matrix for (comp_i, comp_j, mode)."""
        return self._harmonic_basis._build_derivative_matrix_with_spins(
            ell, comp_i, comp_j, mode
        )

    def _precompute_derivative_diagonals(self) -> None:
        """Precompute derivative diagonals. Delegates to HarmonicBasis."""
        self._harmonic_basis._precompute_derivative_diagonals()
        self._derivative_diagonals = self._harmonic_basis._derivative_diagonals
        self._derivative_diagonals_local = (
            self._harmonic_basis._derivative_diagonals_local
        )

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
