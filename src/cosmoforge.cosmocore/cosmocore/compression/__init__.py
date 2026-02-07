"""
Compression methods for CMB Fisher matrix computation.

This module provides two compression approaches:

1. **HarmonicCompression** (Tegmark-like): Direct transformation to harmonic space
   (n_pix → n_modes). Fast and efficient when n_modes << n_pix.

2. **PixelProjectedCompression** (Gjerløw-like): Pixel-space projector with
   eigenvalue compression (n_pix → n_kept). More flexible, handles systematics
   through custom projectors.

Use **CompressionManager** as the unified interface for both methods.

Available compression bases for PixelProjectedCompression:

- **harmonic**: P_h = V^T V (pure harmonic projector)
- **noise_weighted**: P_h N^{-1} P_h (inverse noise weighting)
- **total_covariance**: P_h C^{-1} P_h where C = N + S
- **snr**: S^{1/2} N^{-1} S^{1/2} (signal-to-noise ratio)

References
----------
.. [1] Tegmark, M. "How to measure CMB power spectra without losing information"
   Phys. Rev. D 55, 5895 (1997)
.. [2] Gjerløw, E. et al. "Component separation for the CMB with a
   low-resolution analysis" A&A 629, A51 (2019)
"""

from __future__ import annotations

import inspect

import numpy as np

from .base import BaseCompression, SMWPrepared
from .harmonic import HarmonicCompression
from .pixel_projected import COMPRESSION_BASES, PixelProjectedCompression

_COMPRESSION_CLASSES: dict[str, type[BaseCompression]] = {
    "harmonic": HarmonicCompression,
    "pixel_projected": PixelProjectedCompression,
}


def create_compression(
    method: str,
    N: np.ndarray,
    N_inv: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    lmax: int,
    **kwargs,
) -> BaseCompression:
    """
    Factory function to create the appropriate compression implementation.

    Parameters
    ----------
    method : str
        Compression method: "harmonic" or "pixel_projected".
    N : numpy.ndarray
        Noise covariance matrix.
    N_inv : numpy.ndarray
        Precomputed noise inverse matrix.
    theta : numpy.ndarray
        Colatitude angles for active pixels in radians.
    phi : numpy.ndarray
        Longitude angles for active pixels in radians.
    lmax : int
        Maximum multipole for harmonic expansion.
    **kwargs
        Additional keyword arguments passed to the compression constructor
        (beam, spins, basis, C_ell, epsilon, mode_fraction, etc.).
        Arguments not accepted by the chosen class are silently ignored.

    Returns
    -------
    BaseCompression
        Configured compression instance (not yet set up — call .setup()).
    """
    if method not in _COMPRESSION_CLASSES:
        raise ValueError(
            f"Unknown compression method '{method}'. "
            f"Available: {list(_COMPRESSION_CLASSES)}"
        )
    cls = _COMPRESSION_CLASSES[method]
    # Filter kwargs to only those accepted by the target class
    sig = inspect.signature(cls.__init__)
    accepted = {
        p.name
        for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    }
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    return cls(N, N_inv, theta, phi, lmax, **filtered)


class CompressionManager:
    """
    Unified interface for compression methods.

    This facade wraps both HarmonicCompression and PixelProjectedCompression,
    providing a single entry point for compression operations. Core and its
    subclasses (Fisher, Spectra, PICSLike) use this interface.

    Parameters
    ----------
    N     : numpy.ndarray
        Noise covariance matrix of shape (n_pix, n_pix).
    N_inv : numpy.ndarray
        Precomputed noise inverse matrix of shape (n_pix, n_pix).
    theta : numpy.ndarray
        Colatitude angles for active pixels in radians.
    phi : numpy.ndarray
        Longitude angles for active pixels in radians.
    lmax : int
        Maximum multipole for harmonic expansion.
    method : str, default "harmonic"
        Compression method: "harmonic" (Tegmark-like) or "pixel_projected"
        (Gjerløw-like).
    beam : numpy.ndarray or None, optional
        Beam window function B_ℓ for ℓ=2 to lmax.
    basis : str, default "noise_weighted"
        Compression basis for pixel_projected method. Options:
        "harmonic", "noise_weighted", "total_covariance", "snr".
    C_ell : numpy.ndarray or None, optional
        Power spectrum for bases that require it ("total_covariance", "snr").
    epsilon : float, optional
        Eigenvalue threshold relative to maximum.
    mode_fraction : float, optional
        Fraction of modes to keep.

    Examples
    --------
    >>> # Harmonic compression (default)
    >>> cm = CompressionManager(N_inv, theta, phi, lmax, method="harmonic")
    >>> cm.setup()
    >>>
    >>> # Pixel-projected with SNR basis
    >>> cm = CompressionManager(
    ...     N_inv, theta, phi, lmax,
    ...     method="pixel_projected",
    ...     basis="snr",
    ...     C_ell=C_ell,
    ... )
    >>> cm.setup()
    >>> cm.apply_compression(epsilon=1e-4)
    """

    def __init__(
        self,
        N: np.ndarray,
        N_inv: np.ndarray,
        theta: np.ndarray,
        phi: np.ndarray,
        lmax: int,
        method: str = "harmonic",
        beam: np.ndarray | None = None,
        spins: list[int] | None = None,
        basis: str = "noise_weighted",
        C_ell: np.ndarray | None = None,
        epsilon: float | list[float | tuple[float, float]] | None = None,
        mode_fraction: float | list[float | tuple[float, float]] | None = None,
        lswitch_low: int | None = None,
        lswitch_high: int | None = None,
        fiducial_C_ell: np.ndarray | None = None,
        S_fixed: np.ndarray | None = None,
    ):
        self.method = method
        self._basis = basis
        self._C_ell_for_basis = C_ell
        self._epsilon = epsilon
        self._mode_fraction = mode_fraction

        if method == "harmonic":
            self._impl = HarmonicCompression(
                N,
                N_inv,
                theta,
                phi,
                lmax,
                beam,
                spins=spins,
                lswitch_low=lswitch_low,
                lswitch_high=lswitch_high,
                fiducial_C_ell=fiducial_C_ell,
                S_fixed=S_fixed,
            )
        elif method == "pixel_projected":
            self._impl = PixelProjectedCompression(
                N,
                N_inv,
                theta,
                phi,
                lmax,
                beam,
                spins=spins,
                basis=basis,
                C_ell=C_ell,
                epsilon=epsilon,
                mode_fraction=mode_fraction,
            )
        else:
            raise ValueError(
                f"Unknown compression method '{method}'. "
                f"Available: 'harmonic', 'pixel_projected'"
            )

    def setup(self) -> None:
        """Build compression operator and prepare for operations."""
        self._impl.setup()

    # === Covariance operations ===

    def get_projected_inverse(self, C_ell: np.ndarray) -> np.ndarray:
        """
        Get the projected inverse for Fisher/QML computation.

        For harmonic method, returns V C^{-1} V^T via SMW formula.
        For pixel_projected method, returns C̄^{-1}.

        This is the matrix used in compute_fisher_element and should be
        used for QML E-operator computation to ensure consistency.

        Parameters
        ----------
        C_ell : numpy.ndarray
            Power spectrum values for ell = 2 to lmax.

        Returns
        -------
        numpy.ndarray
            Projected inverse covariance matrix.
        """
        return self._impl.get_projected_inverse(C_ell)

    def get_compressed_covariance(self, C_ell: np.ndarray) -> np.ndarray:
        """
        Get compressed covariance matrix.

        Parameters
        ----------
        C_ell : numpy.ndarray
            Power spectrum values for ell = 2 to lmax.

        Returns
        -------
        numpy.ndarray
            Compressed covariance matrix.
        """
        return self._impl.get_compressed_covariance(C_ell)

    def get_compressed_inverse(self, C_ell: np.ndarray) -> np.ndarray:
        """
        Get inverse of compressed covariance.

        Parameters
        ----------
        C_ell : numpy.ndarray
            Power spectrum values for ell = 2 to lmax.

        Returns
        -------
        numpy.ndarray
            Inverse of compressed covariance matrix.
        """
        return self._impl.get_compressed_inverse(C_ell)

    def get_compressed_logdet(self, C_ell: np.ndarray) -> float:
        """
        Get log determinant of compressed covariance.

        Parameters
        ----------
        C_ell : numpy.ndarray
            Power spectrum values for ell = 2 to lmax.

        Returns
        -------
        float
            Log determinant of compressed covariance.
        """
        return self._impl.get_compressed_logdet(C_ell)

    def get_full_logdet(self, C_ell: np.ndarray) -> float:
        """
        Get log determinant of full covariance using SMW formula.

        For harmonic method, returns exact log|C| via SMW formula.
        For pixel_projected, returns log|C_compressed| (approximation).

        Parameters
        ----------
        C_ell : numpy.ndarray
            Power spectrum values for ell = 2 to lmax.

        Returns
        -------
        float
            Log determinant of full (or compressed) covariance.
        """
        if self.method == "harmonic":
            return self._impl.get_logdet(C_ell)
        else:
            return self._impl.get_compressed_logdet(C_ell)

    def get_derivative_matrix(self, ell: int) -> np.ndarray:
        """
        Get derivative matrix dC/dC_ell in compressed space.

        Parameters
        ----------
        ell : int
            Multipole for derivative.

        Returns
        -------
        numpy.ndarray
            Derivative matrix.
        """
        return self._impl.get_derivative_matrix(ell)

    def compress_data(self, data: np.ndarray) -> np.ndarray:
        """
        Project pixel data to compressed representation.

        Parameters
        ----------
        data : numpy.ndarray
            Pixel-space data vector.

        Returns
        -------
        numpy.ndarray
            Compressed data vector.
        """
        return self._impl.compress_data(data)

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

        Parameters
        ----------
        C_ell : numpy.ndarray
            Power spectrum values for ell = 2 to lmax.
        ell_min : int, default 2
            Minimum multipole to include in Fisher matrix.
        ell_max : int or None, optional
            Maximum multipole. If None, uses compression lmax.

        Returns
        -------
        numpy.ndarray
            Fisher matrix of shape (n_ell, n_ell) where n_ell = ell_max - ell_min + 1.
            Index [i, j] corresponds to ell_i = ell_min + i, ell_j = ell_min + j.
        """
        return self._impl.compute_fisher_matrix(C_ell, ell_min, ell_max)

    def compute_fisher_matrix_multi(
        self,
        C_ell_dict: dict[tuple, np.ndarray],
        spectra_list: list[tuple],
        ell_min: int = 2,
        ell_max: int | None = None,
    ) -> np.ndarray:
        """
        Compute multi-field Fisher matrix for multiple auto and cross-spectra.

        Parameters
        ----------
        C_ell_dict : dict
            Dictionary mapping (comp_i, comp_j) or (comp_i, comp_j, mode)
            to C_ell array for each spectrum.
        spectra_list : list
            List of 2-tuple or 3-tuple specifying which spectra to include.
        ell_min : int, default 2
            Minimum multipole to include in Fisher matrix.
        ell_max : int or None, optional
            Maximum multipole. If None, uses compression lmax.

        Returns
        -------
        numpy.ndarray
            Fisher matrix of shape (n_spectra * n_ell, n_spectra * n_ell).
        """
        return self._impl.compute_fisher_matrix_multi(
            C_ell_dict, spectra_list, ell_min, ell_max
        )

    def get_weighted_compressed_data(
        self, data: np.ndarray, C_ell: np.ndarray, C_c_inv: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Compute weighted compressed data for QML estimation.

        For harmonic: w = V @ C^{-1} @ d (uses SMW formula)
        For pixel_projected: w = C_c^{-1} @ (U^T @ d)

        This is used for QML quadratic form computation in compressed space:
            q_l = 0.5 * w^T @ E_l @ w

        Parameters
        ----------
        data : numpy.ndarray
            Pixel-space data vector.
        C_ell : numpy.ndarray
            Power spectrum values for ell = 2 to lmax.
        C_c_inv : numpy.ndarray, optional
            Precomputed compressed covariance inverse. If provided, this is
            used directly instead of computing from C_ell. Useful when
            processing multiple simulations with the same C_ell.

        Returns
        -------
        numpy.ndarray
            Weighted compressed data vector.
        """
        return self._impl.get_weighted_compressed_data(data, C_ell, C_c_inv=C_c_inv)

    def get_weighted_compressed_data_multi(
        self, data: np.ndarray, C_ell_dict: dict[tuple, np.ndarray]
    ) -> np.ndarray:
        """
        Compute weighted compressed data for multi-field QML estimation.

        For harmonic: w = V @ C^{-1} @ d (uses SMW formula with full Lambda)

        Parameters
        ----------
        data : numpy.ndarray
            Pixel-space data vector.
        C_ell_dict : dict
            Dictionary mapping (comp_i, comp_j) or (comp_i, comp_j, mode)
            to C_ell array for cross-spectra.

        Returns
        -------
        numpy.ndarray
            Weighted compressed data vector.
        """
        return self._impl.get_weighted_compressed_data_multi(data, C_ell_dict)

    def compute_quadratic_form(self, data: np.ndarray, C_ell: np.ndarray) -> float:
        """
        Compute quadratic form d^T C^{-1} d efficiently.

        For harmonic compression, uses SMW formula for exact computation in O(n_modes²).
        For pixel_projected, uses compressed approximation.

        Parameters
        ----------
        data : numpy.ndarray
            Pixel-space data vector.
        C_ell : numpy.ndarray
            Power spectrum values for ell = 2 to lmax.

        Returns
        -------
        float
            Quadratic form value d^T C^{-1} d.
        """
        return self._impl.compute_quadratic_form(data, C_ell)

    # === Multi-field operations (unified, supports 2-tuple and 3-tuple keys) ===

    def get_compressed_covariance_multi(
        self, C_ell_dict: dict[tuple, np.ndarray]
    ) -> np.ndarray:
        """Get compressed covariance for multi-field."""
        return self._impl.get_compressed_covariance_multi(C_ell_dict)

    def get_compressed_inverse_multi(
        self, C_ell_dict: dict[tuple, np.ndarray]
    ) -> np.ndarray:
        """Get inverse compressed covariance for multi-field."""
        return self._impl.get_compressed_inverse_multi(C_ell_dict)

    def get_derivative_matrix_multi(
        self, ell: int, comp_i: int, comp_j: int, mode: int = 0
    ) -> np.ndarray:
        """Get derivative matrix dC/dC_ell for multi-field."""
        return self._impl.get_derivative_matrix_multi(ell, comp_i, comp_j, mode)

    def prepare_smw_multi(self, C_ell_dict: dict[tuple, np.ndarray]) -> SMWPrepared:
        """Precompute K Cholesky and logdet for reuse across sims."""
        return self._impl.prepare_smw_multi(C_ell_dict)

    def compute_quadratic_form_multi(
        self, data: np.ndarray, C_ell_dict: dict[tuple, np.ndarray]
    ) -> float:
        """Compute d^T C^{-1} d using SMW formula."""
        return self._impl.compute_quadratic_form_multi(data, C_ell_dict)

    def quadratic_form_from_prepared(self, data: np.ndarray, K_chol):
        """Compute d^T C^{-1} d using precomputed K Cholesky factor."""
        return self._impl.quadratic_form_from_prepared(data, K_chol)

    def get_logdet_multi(self, C_ell_dict: dict[tuple, np.ndarray]) -> float:
        """Get log|C| via SMW formula (harmonic only)."""
        return self._impl.get_logdet_multi(C_ell_dict)

    def get_full_logdet_multi(self, C_ell_dict: dict[tuple, np.ndarray]) -> float:
        """Get log|C| for multi-field.

        For harmonic method, uses exact SMW formula.
        For pixel_projected, uses compressed logdet approximation.
        """
        if self.method == "harmonic":
            return self._impl.get_logdet_multi(C_ell_dict)
        else:
            return self._impl.get_compressed_logdet(C_ell_dict)

    # === Deprecated _with_spins aliases ===

    def compute_fisher_matrix_with_spins(
        self, C_ell_dict, spectra_list, ell_min=2, ell_max=None
    ):
        """Deprecated: use compute_fisher_matrix_multi."""
        return self.compute_fisher_matrix_multi(
            C_ell_dict, spectra_list, ell_min, ell_max
        )

    def get_compressed_covariance_with_spins(self, C_ell_dict):
        """Deprecated: use get_compressed_covariance_multi."""
        return self.get_compressed_covariance_multi(C_ell_dict)

    def get_compressed_inverse_with_spins(self, C_ell_dict):
        """Deprecated: use get_compressed_inverse_multi."""
        return self.get_compressed_inverse_multi(C_ell_dict)

    def get_weighted_compressed_data_with_spins(self, data, C_ell_dict):
        """Deprecated: use get_weighted_compressed_data_multi."""
        return self.get_weighted_compressed_data_multi(data, C_ell_dict)

    def compute_quadratic_form_with_spins(self, data, C_ell_dict):
        """Deprecated: use compute_quadratic_form_multi."""
        return self.compute_quadratic_form_multi(data, C_ell_dict)

    def prepare_smw_with_spins(self, C_ell_dict):
        """Deprecated: use prepare_smw_multi."""
        return self.prepare_smw_multi(C_ell_dict)

    def get_full_logdet_with_spins(self, C_ell_dict):
        """Deprecated: use get_full_logdet_multi."""
        return self.get_full_logdet_multi(C_ell_dict)

    # === Properties ===

    @property
    def n_kept(self) -> int:
        """Number of modes after compression."""
        return self._impl.n_kept

    @property
    def n_modes(self) -> int:
        """Total number of harmonic modes."""
        return self._impl.n_modes

    @property
    def compression_ratio(self) -> float:
        """Ratio of kept modes to original size."""
        return self._impl.compression_ratio

    @property
    def n_components(self) -> int:
        """Number of field components (multi-field support)."""
        return self._impl.n_components


__all__ = [
    "BaseCompression",
    "COMPRESSION_BASES",
    "CompressionManager",
    "HarmonicCompression",
    "PixelProjectedCompression",
    "SMWPrepared",
    "create_compression",
]
