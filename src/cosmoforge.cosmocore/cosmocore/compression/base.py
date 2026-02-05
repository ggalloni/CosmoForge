"""
Base class for compression methods.

This module provides the abstract base class that defines the interface
for all compression methods used in CMB Fisher matrix computation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..basics import legendre_plm, matrix_mult, matrix_slogdet_symm, matrix_trace


class BaseCompression(ABC):
    """
    Abstract base class for compression methods.

    This class defines the interface for compression methods and provides
    shared implementations for spherical harmonic evaluation and ell-to-mode
    mapping.

    Parameters
    ----------
    N_inv : numpy.ndarray
        Precomputed noise inverse matrix of shape (n_pix, n_pix).
    theta : numpy.ndarray
        Colatitude angles for active pixels in radians.
    phi : numpy.ndarray
        Longitude angles for active pixels in radians.
    lmax : int
        Maximum multipole for harmonic expansion.
    beam : numpy.ndarray or None, optional
        Beam window function B_ℓ for ℓ=2 to lmax. Shape should be (lmax-1,).
        If provided, the harmonic operator V is multiplied by beam factors
        so that V_ℓm = B_ℓ × Y_ℓm. If None, no beam correction is applied.

    Attributes
    ----------
    n_pix : int
        Number of active pixels.
    n_modes : int
        Number of harmonic modes (for ℓ=2 to lmax).
    n_kept : int
        Number of modes kept after compression (if applicable).
    """

    def __init__(
        self,
        N: np.ndarray,
        N_inv: np.ndarray,
        theta: np.ndarray,
        phi: np.ndarray,
        lmax: int,
        beam: np.ndarray | None = None,
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
            Noise covariance matrix (n_pix, n_pix).
        N_inv : numpy.ndarray
            Precomputed noise inverse matrix (n_pix, n_pix).
        theta : numpy.ndarray
            Colatitude angles for active pixels in radians.
        phi : numpy.ndarray
            Longitude angles for active pixels in radians.
        lmax : int
            Maximum multipole for harmonic expansion.
        beam : numpy.ndarray or None, optional
            Beam window function B_ℓ for ℓ=2 to lmax.
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
            Shape should be (n_pix, n_pix).
        """
        self._N = np.asfortranarray(N, dtype=np.float64)
        self.N_inv = np.asfortranarray(N_inv, dtype=np.float64)
        self.theta = np.asarray(theta, dtype=np.float64)
        self.phi = np.asarray(phi, dtype=np.float64)
        self.lmax = lmax

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
        self.n_pix = len(theta)

        # n_modes depends on whether switch optimization is used
        if self._use_switch_optimization:
            # Only modes for ℓ in [lswitch_low, lswitch_high]
            self.n_modes = (self.lswitch_high + 1) ** 2 - (self.lswitch_low) ** 2
            self._lmin_smw = self.lswitch_low
            self._lmax_smw = self.lswitch_high
        else:
            # All modes from ℓ=2 to lmax
            self.n_modes = (lmax + 1) ** 2 - 4
            self._lmin_smw = 2
            self._lmax_smw = lmax

        # To be set by subclasses
        self._V = None
        self._ell_to_modes = None
        self.n_kept = self.n_modes

    @abstractmethod
    def setup(self) -> None:
        """
        Initialize compression-specific components.

        Must be called after initialization before using any other methods.
        """
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

    def _build_harmonic_operator(self) -> None:
        """
        Build the harmonic projection operator V using real spherical harmonics.

        V projects from pixel space to harmonic space. Each row of V
        corresponds to a (ell, m) mode, and each column to a pixel.

        The operator is built WITHOUT beam effects:
            V[mode, pix] = Y_lm(theta[pix], phi[pix])

        Beam effects are expected to be incorporated in the C_ell power spectrum
        passed to methods like get_compressed_covariance(). This is consistent
        with how spectra_manager stores beam-convolved power spectra.

        Note: The _beam attribute is stored for reference but NOT applied to V.

        When switch optimization is enabled (lswitch_high < lmax), V is built
        only for multipoles in [lswitch_low, lswitch_high], significantly
        reducing the dimension of the SMW operations.

        This implementation uses JIT-compiled recurrence relations for associated
        Legendre polynomials, which is significantly faster than scipy.special.lpmv.
        """
        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)

        V = np.zeros((self.n_modes, self.n_pix), dtype=np.float64)

        # Determine ell range for V (may be reduced if switch optimization is used)
        lmin_v = self._lmin_smw
        lmax_v = self._lmax_smw

        # Precompute cos(m*phi) and sin(m*phi) for all m and pixels
        cos_mphi = np.zeros((lmax_v + 1, self.n_pix), dtype=np.float64)
        sin_mphi = np.zeros((lmax_v + 1, self.n_pix), dtype=np.float64)
        for m in range(lmax_v + 1):
            for ipix in range(self.n_pix):
                cos_mphi[m, ipix] = np.cos(m * self.phi[ipix])
                sin_mphi[m, ipix] = np.sin(m * self.phi[ipix])

        # Buffer for P_ℓ^m values for one pixel (need full lmax for Legendre)
        plm = np.zeros((self.lmax + 1, self.lmax + 1), dtype=np.float64)

        # Process each pixel
        for ipix in range(self.n_pix):
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

        self._V = np.asfortranarray(V)

    def _build_ell_mode_mapping(self) -> None:
        """
        Build mapping from multipole ell to mode indices.

        This is used for computing derivatives with respect to C_ell,
        where E_ell is a diagonal matrix with 1s for modes at multipole ell.

        When switch optimization is enabled, mapping is built only for
        multipoles in [lswitch_low, lswitch_high].
        """
        self._ell_to_modes = {}
        mode_idx = 0
        for ell in range(self._lmin_smw, self._lmax_smw + 1):
            n_m = 2 * ell + 1
            self._ell_to_modes[ell] = list(range(mode_idx, mode_idx + n_m))
            mode_idx += n_m

    def _precompute_derivative_diagonals(self) -> None:
        """
        Precompute derivative matrix diagonals E_ℓ for multipoles in SMW range.

        The derivative matrix for multipole ℓ is diagonal with factor
        (2ℓ+1)/(4π) at positions corresponding to modes at that ℓ.
        Storing these diagonals avoids repeated array allocations.

        When switch optimization is enabled, only computes for ℓ in
        [lswitch_low, lswitch_high].

        This is called during setup and the diagonals are reused in
        compute_fisher_matrix() and get_derivative_matrix().

        References
        ----------
        .. [1] Tegmark, M. "How to measure CMB power spectra without losing
           information" Phys. Rev. D 55, 5895 (1997) - Equation 16
        """
        self._derivative_diagonals = {}
        for ell in range(self._lmin_smw, self._lmax_smw + 1):
            E_diag = np.zeros(self.n_modes, dtype=np.float64)
            if ell in self._ell_to_modes:
                chngconv = (2 * ell + 1) / (4 * np.pi)
                for mode_idx in self._ell_to_modes[ell]:
                    E_diag[mode_idx] = chngconv
            self._derivative_diagonals[ell] = E_diag

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
