"""
Pixel-projected compression (Gjerløw-like) for CMB Fisher matrix computation.

This module implements pixel-space compression with a harmonic projector P_h,
followed by eigenvalue decomposition to find the optimal subspace for compression.

Available compression bases (from Gjerløw et al. 2019):
- "harmonic": Pure harmonic projector P_h = V^T V
- "noise_weighted": P_h N^{-1} P_h (default) - inverse noise weighting
- "total_covariance": P_h C^{-1} P_h where C = N + S - full covariance weighting
- "snr": S^{1/2} N^{-1} S^{1/2} - signal-to-noise ratio matrix
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import eigh

from ..basics import matrix_inverse_symm, matrix_mult
from .base import BaseCompression

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

# Available compression basis presets
COMPRESSION_BASES = {
    "harmonic": "P_h = V^T V (pure harmonic projector)",
    "noise_weighted": "P_h N^{-1} P_h (inverse noise weighting)",
    "total_covariance": "P_h C^{-1} P_h (full covariance weighting, requires C_ell)",
    "snr": "S^{1/2} N^{-1} S^{1/2} (signal-to-noise ratio, requires C_ell)",
}


class PixelProjectedCompression(BaseCompression):
    """
    Pixel-space compression with projector (Gjerløw-like).

    Stays in n_pix space with projector P_h, then uses eigenvalue
    decomposition to find optimal subspace. Compression: n_pix → n_kept.

    The key operations are:
    - Projector: P_h = V^T V (harmonic projector, n_pix × n_pix but rank n_modes)
    - Eigendecomposition of compression matrix to find optimal modes
    - Data compression: d_c = P @ d where P = U^T (n_kept × n_pix)

    This approach is more flexible than HarmonicCompression because it allows
    custom projectors to handle systematics (foreground deprojection, etc.).

    Available compression bases (from Gjerløw et al. 2019):

    - **harmonic**: Pure harmonic projector P_h = V^T V. Selects modes based
      purely on harmonic content, ignoring noise properties.
    - **noise_weighted**: P_h N^{-1} P_h (default). Weights modes by inverse
      noise, preferring low-noise harmonic modes.
    - **total_covariance**: P_h C^{-1} P_h where C = N + S. Uses full
      signal+noise covariance for optimal compression.
    - **snr**: S^{1/2} N^{-1} S^{1/2}. Signal-to-noise ratio matrix, prioritizing
      modes with highest SNR.

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
    basis : str, default "noise_weighted"
        Compression basis to use. Options:
        - "harmonic": P_h = V^T V (pure harmonic projector)
        - "noise_weighted": P_h N^{-1} P_h (inverse noise weighting)
        - "total_covariance": P_h C^{-1} P_h where C = N + S
        - "snr": S^{1/2} N^{-1} S^{1/2} (signal-to-noise ratio)
    C_ell : numpy.ndarray or None, optional
        Power spectrum values for ell = 2 to lmax. Required for
        "total_covariance" and "snr" bases.
    epsilon : float, optional
        Eigenvalue threshold relative to maximum.
    mode_fraction : float, optional
        Fraction of modes to keep.

    Attributes
    ----------
    n_kept : int
        Number of modes kept after compression (initially n_pix).

    Examples
    --------
    >>> import numpy as np
    >>> from cosmocore.compression import PixelProjectedCompression
    >>> N = np.diag(noise_variance)  # Noise covariance matrix
    >>> N_inv = np.diag(1.0 / noise_variance)
    >>> ppc = PixelProjectedCompression(N, N_inv, theta, phi, lmax=100)
    >>> ppc.setup()
    >>> # Inspect eigenvalue spectrum to choose threshold
    >>> fig, ax = ppc.plot_eigenvalue_spectrum(basis="noise_weighted")
    >>> # Apply compression with chosen threshold
    >>> ppc.apply_compression(epsilon=1e-4, basis="noise_weighted")
    >>> fisher_element = ppc.compute_fisher_element(C_ell, ell_i=10, ell_j=10)

    References
    ----------
    .. [1] Gjerløw, E. et al. "Component separation for the CMB with a
       low-resolution analysis" A&A 629, A51 (2019)
    """

    def __init__(
        self,
        N: np.ndarray,
        N_inv: np.ndarray,
        theta: np.ndarray,
        phi: np.ndarray,
        lmax: int,
        beam: np.ndarray | None = None,
        basis: str = "noise_weighted",
        C_ell: np.ndarray | None = None,
        epsilon: float | None = None,
        mode_fraction: float | None = None,
    ):
        super().__init__(N, N_inv, theta, phi, lmax, beam)
        # Before compression, n_kept = n_pix
        self.n_kept = self.n_pix
        # Compression quantities
        self._basis = basis
        self._C_ell_for_basis = C_ell
        self._epsilon = epsilon
        self._mode_fraction = mode_fraction
        self._eigenvectors = None

    @property
    def projector(self) -> np.ndarray:
        """
        Get the projection matrix U^T (n_kept × n_pix).

        Maps pixel space to eigenbasis compressed space.

        Raises
        ------
        RuntimeError
            If compression has not been applied yet.
        """
        if self._eigenvectors is None:
            raise RuntimeError("Compression not applied. Call apply_compression() first.")
        return self._eigenvectors.T

    @property
    def n_compressed(self) -> int:
        """Size of compressed space (n_kept)."""
        return self.n_kept

    def setup(self) -> None:
        """
        Build projector P_h and prepare for eigenvalue compression.

        Computes:
        - V: harmonic operator (n_modes × n_pix)
        - P_h = V^T V: harmonic projector (n_pix × n_pix, rank n_modes)
        - N from N^{-1}: needed for compressed covariance
        """
        self._build_harmonic_operator()
        self._build_ell_mode_mapping()

        # P_h = V^T V (harmonic projector, n_pix × n_pix but rank n_modes)
        self._P_h = matrix_mult(self._V.T, self._V)

        if self._epsilon or self._mode_fraction:
            self.apply_compression(
                epsilon=self._epsilon,
                mode_fraction=self._mode_fraction,
                basis=self._basis,
                C_ell=self._C_ell_for_basis,
            )

    def _build_compression_matrix(
        self,
        basis: str,
        C_ell: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Build the matrix for eigendecomposition based on compression basis.

        Parameters
        ----------
        basis : str
            Compression basis: "harmonic", "noise_weighted", "total_covariance", "snr".
        C_ell : numpy.ndarray, optional
            Power spectrum values for ell = 2 to lmax. Required for
            "total_covariance" and "snr" bases.

        Returns
        -------
        numpy.ndarray
            Compression matrix of shape (n_pix, n_pix).
        """
        if basis not in COMPRESSION_BASES:
            raise ValueError(
                f"Unknown compression basis '{basis}'. "
                f"Available: {list(COMPRESSION_BASES.keys())}"
            )

        if basis == "harmonic":
            # Pure harmonic projector P_h = V^T V
            return self._P_h

        elif basis == "noise_weighted":
            # P_h N^{-1} P_h
            return matrix_mult(matrix_mult(self._P_h, self.N_inv), self._P_h)

        elif basis == "total_covariance":
            # P_h C^{-1} P_h where C = N + S
            if C_ell is None:
                raise ValueError(
                    "C_ell is required for 'total_covariance' basis. "
                    "Provide power spectrum values for ell = 2 to lmax."
                )
            # Build signal covariance S = V^T Λ V
            # Use broadcasting to avoid creating full diagonal matrix
            Lambda_diag = self._build_lambda_diagonal(C_ell)
            # (n_modes, n_pix) * (n_modes, 1)
            V_scaled = self._V * Lambda_diag[:, np.newaxis]
            S = matrix_mult(self._V.T, V_scaled)
            # Total covariance C = N + S
            C = self._N + S
            # C^{-1}
            C_inv = matrix_inverse_symm(C, overwrite=True)
            # P_h C^{-1} P_h
            return matrix_mult(matrix_mult(self._P_h, C_inv), self._P_h)

        elif basis == "snr":
            # S^{1/2} N^{-1} S^{1/2} - signal-to-noise ratio matrix
            if C_ell is None:
                raise ValueError(
                    "C_ell is required for 'snr' basis. "
                    "Provide power spectrum values for ell = 2 to lmax."
                )
            # Build signal covariance S = V^T Λ V
            # Use broadcasting to avoid creating full diagonal matrix
            Lambda_diag = self._build_lambda_diagonal(C_ell)
            V_scaled = self._V * Lambda_diag[:, np.newaxis]
            S = matrix_mult(self._V.T, V_scaled)
            eigvals_S, eigvecs_S = eigh(S)
            sqrt_eigvals = np.sqrt(np.maximum(eigvals_S, 1e-30))
            Q_scaled = eigvecs_S * sqrt_eigvals  # (n_pix, n_pix) * (n_pix,)
            S_sqrt = matrix_mult(Q_scaled, eigvecs_S.T)
            # S^{1/2} N^{-1} S^{1/2}
            return matrix_mult(matrix_mult(S_sqrt, self.N_inv), S_sqrt)

        # Should never reach here
        raise ValueError(f"Unhandled basis: {basis}")

    def compute_eigenspectrum(
        self,
        basis: str = "noise_weighted",
        C_ell: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenvalue spectrum for a given compression basis.

        This method computes the eigendecomposition without applying compression,
        allowing inspection of the spectrum to choose an appropriate threshold.

        Parameters
        ----------
        basis : str, default "noise_weighted"
            Compression basis: "harmonic", "noise_weighted", "total_covariance", "snr".
        C_ell : numpy.ndarray, optional
            Power spectrum values for ell = 2 to lmax. Required for
            "total_covariance" and "snr" bases.

        Returns
        -------
        eigenvalues : numpy.ndarray
            Eigenvalues sorted in descending order.
        normalized_eigenvalues : numpy.ndarray
            Eigenvalues normalized by maximum value (for threshold selection).
        """
        compression_matrix = self._build_compression_matrix(basis, C_ell)
        eigenvalues, _ = eigh(compression_matrix)

        # Sort in descending order
        eigenvalues = np.sort(eigenvalues)[::-1]

        # Normalize by maximum
        max_eigenvalue = np.max(np.abs(eigenvalues))
        if max_eigenvalue > 0:
            normalized_eigenvalues = eigenvalues / max_eigenvalue
        else:
            normalized_eigenvalues = eigenvalues.copy()

        return eigenvalues, normalized_eigenvalues

    def plot_eigenvalue_spectrum(
        self,
        basis: str = "noise_weighted",
        C_ell: np.ndarray | None = None,
        ax: Axes | None = None,
        log_scale: bool = True,
        show_threshold_lines: bool = True,
        threshold_values: list[float] | None = None,
    ) -> tuple[Figure, Axes]:
        """
        Plot eigenvalue spectrum for compression threshold selection.

        The y-axis shows eigenvalues normalized by the maximum value, so values
        can be directly used as the epsilon threshold for apply_compression().

        Parameters
        ----------
        basis : str, default "noise_weighted"
            Compression basis: "harmonic", "noise_weighted", "total_covariance", "snr".
        C_ell : numpy.ndarray, optional
            Power spectrum values for ell = 2 to lmax. Required for
            "total_covariance" and "snr" bases.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure.
        log_scale : bool, default True
            Whether to use logarithmic y-axis.
        show_threshold_lines : bool, default True
            Whether to show reference threshold lines.
        threshold_values : list of float, optional
            Custom threshold values to show. Default: [1e-2, 1e-4, 1e-6, 1e-8].

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the plot.
        ax : matplotlib.axes.Axes
            The axes containing the plot.

        Examples
        --------
        >>> ppc = PixelProjectedCompression(N, N_inv, theta, phi, lmax=100)
        >>> ppc.setup()
        >>> fig, ax = ppc.plot_eigenvalue_spectrum(basis="noise_weighted")
        >>> # From the plot, decide threshold (e.g., 1e-4)
        >>> ppc.apply_compression(epsilon=1e-4, basis="noise_weighted")
        """
        import matplotlib.pyplot as plt

        _, normalized_eigenvalues = self.compute_eigenspectrum(basis, C_ell)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.get_figure()

        # Plot eigenvalue spectrum
        mode_indices = np.arange(1, len(normalized_eigenvalues) + 1)
        ax.plot(
            mode_indices, normalized_eigenvalues, "b-", linewidth=1.5, label="Eigenvalues"
        )

        # Add threshold reference lines
        if show_threshold_lines:
            if threshold_values is None:
                threshold_values = [1e-2, 1e-4, 1e-6, 1e-8]
            colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(threshold_values)))
            for thresh, color in zip(threshold_values, colors):
                n_kept = np.sum(normalized_eigenvalues > thresh)
                ax.axhline(
                    y=thresh,
                    color=color,
                    linestyle="--",
                    alpha=0.7,
                    label=f"ε={thresh:.0e} ({n_kept} modes)",
                )

        # Formatting
        if log_scale:
            ax.set_yscale("log")
        ax.set_xlabel("Mode index", fontsize=12)
        ax.set_ylabel("Normalized eigenvalue (ε threshold)", fontsize=12)
        ax.set_title(
            f"Eigenvalue Spectrum: {basis}\n{COMPRESSION_BASES[basis]}", fontsize=12
        )
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, len(normalized_eigenvalues))

        # Add text annotation for total modes
        n_total = len(normalized_eigenvalues)
        n_significant = np.sum(normalized_eigenvalues > 1e-10)
        ax.text(
            0.02,
            0.02,
            f"Total modes: {n_total}\nSignificant (>1e-10): {n_significant}",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        return fig, ax

    def plot_eigenvalue_comparison(
        self,
        bases: list[str] | None = None,
        C_ell: np.ndarray | None = None,
        ax: Axes | None = None,
        log_scale: bool = True,
    ) -> tuple[Figure, Axes]:
        """
        Compare eigenvalue spectra across different compression bases.

        Parameters
        ----------
        bases : list of str, optional
            Compression bases to compare. Default: all available bases.
            Note: "total_covariance" and "snr" require C_ell.
        C_ell : numpy.ndarray, optional
            Power spectrum values for ell = 2 to lmax. Required for
            "total_covariance" and "snr" bases.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, creates a new figure.
        log_scale : bool, default True
            Whether to use logarithmic y-axis.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the plot.
        ax : matplotlib.axes.Axes
            The axes containing the plot.
        """
        import matplotlib.pyplot as plt

        if bases is None:
            # Use all bases that don't require C_ell, or all if C_ell provided
            if C_ell is not None:
                bases = list(COMPRESSION_BASES.keys())
            else:
                bases = ["harmonic", "noise_weighted"]

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.get_figure()

        colors = plt.cm.tab10(np.linspace(0, 1, len(bases)))

        for basis, color in zip(bases, colors):
            try:
                _, normalized_eigenvalues = self.compute_eigenspectrum(basis, C_ell)
                mode_indices = np.arange(1, len(normalized_eigenvalues) + 1)
                ax.plot(
                    mode_indices,
                    normalized_eigenvalues,
                    color=color,
                    linewidth=1.5,
                    label=basis,
                )
            except ValueError as e:
                print(f"Skipping basis '{basis}': {e}")

        if log_scale:
            ax.set_yscale("log")
        ax.set_xlabel("Mode index", fontsize=12)
        ax.set_ylabel("Normalized eigenvalue", fontsize=12)
        ax.set_title("Eigenvalue Spectrum Comparison", fontsize=12)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, ax

    def apply_compression(
        self,
        epsilon: float | None = None,
        mode_fraction: float | None = None,
        basis: str = "noise_weighted",
        C_ell: np.ndarray | None = None,
    ) -> None:
        """
        Apply eigenvalue compression to find optimal subspace.

        Computes the eigendecomposition of the compression matrix (determined by
        the basis) and selects modes to keep based on either an eigenvalue
        threshold or a fraction of total modes.

        Parameters
        ----------
        epsilon : float, optional
            Eigenvalue threshold. Modes with eigenvalue > epsilon * max_eigenvalue
            are kept. Mutually exclusive with mode_fraction.
        mode_fraction : float, optional
            Fraction of modes to keep (between 0 and 1). Keeps the top modes
            ordered by eigenvalue. Mutually exclusive with epsilon.
        basis : str, default "noise_weighted"
            Compression basis determining which matrix to eigendecompose:

            - "harmonic": P_h = V^T V (pure harmonic projector)
            - "noise_weighted": P_h N^{-1} P_h (inverse noise weighting)
            - "total_covariance": P_h C^{-1} P_h where C = N + S
            - "snr": S^{1/2} N^{-1} S^{1/2} (signal-to-noise ratio)

        C_ell : numpy.ndarray, optional
            Power spectrum values for ell = 2 to lmax. Required for
            "total_covariance" and "snr" bases.

        Raises
        ------
        ValueError
            If both epsilon and mode_fraction are provided.
            If mode_fraction is not in (0, 1].
            If C_ell is required but not provided.

        Examples
        --------
        >>> ppc = PixelProjectedCompression(N, N_inv, theta, phi, lmax=100)
        >>> ppc.setup()
        >>> # Use eigenvalue spectrum plot to choose threshold
        >>> fig, ax = ppc.plot_eigenvalue_spectrum(basis="snr", C_ell=C_ell)
        >>> # Apply compression with chosen parameters
        >>> ppc.apply_compression(epsilon=1e-4, basis="snr", C_ell=C_ell)
        """
        # Validate mutual exclusivity
        if epsilon is not None and mode_fraction is not None:
            raise ValueError(
                "epsilon and mode_fraction are mutually exclusive. "
                "Provide only one compression criterion."
            )

        if mode_fraction is not None:
            if not (0 < mode_fraction <= 1):
                raise ValueError(f"mode_fraction must be in (0, 1], got {mode_fraction}")

        # Default epsilon if neither specified
        if epsilon is None and mode_fraction is None:
            epsilon = 1e-6

        # Build compression matrix based on chosen basis
        compression_matrix = self._build_compression_matrix(basis, C_ell)

        # Eigendecompose compression matrix
        eigenvalues, eigenvectors = eigh(compression_matrix)

        # Sort in descending order
        sort_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]

        # Determine which modes to keep
        if mode_fraction is not None:
            # Keep top mode_fraction of modes
            n_significant = np.sum(eigenvalues > 1e-10 * np.max(np.abs(eigenvalues)))
            n_to_keep = max(1, int(np.ceil(mode_fraction * n_significant)))
            mask = np.zeros(len(eigenvalues), dtype=bool)
            mask[:n_to_keep] = True
        else:
            # Use epsilon threshold (relative to max eigenvalue)
            max_eigenvalue = np.max(np.abs(eigenvalues))
            threshold = epsilon * max_eigenvalue
            mask = np.abs(eigenvalues) > threshold

        self._eigenvalues = eigenvalues[mask]
        self._eigenvectors = eigenvectors[:, mask]  # U: (n_pix, n_kept)
        self.n_kept = np.sum(mask)
        self._compression_basis = basis

        # Precompute compression-dependent quantities that don't depend on C_ell
        self._precompute_compression_products()

    def _precompute_compression_products(self) -> None:
        """
        Precompute matrices that depend only on compression eigenvectors.

        These quantities are independent of C_ell and can be computed once
        after compression is applied, providing O(ℓ²) speedup for Fisher
        matrix computation.

        Precomputes:
        - _U_N_U: U^T @ N @ U (compressed noise covariance)
        - _VU: V @ U (harmonic operator applied to eigenvectors)
        - _V_N_inv: V @ N^{-1} (for SMW formula in QML)
        - _V_Ninv_VT: V @ N^{-1} @ V^T (SMW kernel matrix)
        - Derivative diagonals: E_ℓ for all ℓ (inherited from base class)
        - Buffers for intermediate computations

        References
        ----------
        .. [1] Gjerløw, E., et al. (2015). Section 6.2 - "Precompute: PY"
        """
        U = self._eigenvectors  # (n_pix, n_kept)

        # U^T @ N @ U - compressed noise covariance (independent of C_ell)
        self._U_N_U = U.T @ self._N @ U

        # V @ U - used for signal covariance transformation (independent of C_ell)
        # VU has shape (n_modes, n_kept)
        self._VU = self._V @ U

        # Make U_N_U symmetric (numerical stability)
        self._U_N_U = 0.5 * (self._U_N_U + self._U_N_U.T)

        # SMW components for get_weighted_compressed_data
        # V @ N^{-1} (n_modes, n_pix)
        self._V_N_inv = matrix_mult(self._V, self.N_inv)
        # V @ N^{-1} @ V^T (n_modes, n_modes) - the M matrix in SMW
        self._V_Ninv_VT = matrix_mult(self._V_N_inv, self._V.T)
        self._V_Ninv_VT = 0.5 * (self._V_Ninv_VT + self._V_Ninv_VT.T)

        # Precompute derivative diagonals (from base class)
        self._precompute_derivative_diagonals()

        # Pre-allocate buffers for frequently called methods
        self._allocate_buffers()

    def _allocate_buffers(self) -> None:
        """
        Pre-allocate reusable buffers for intermediate computations.

        This reduces memory allocation overhead in frequently called methods
        like get_derivative_matrix and get_projected_inverse.
        """
        # Buffer for VU * diagonal scaling: (n_modes, n_kept)
        self._VU_scaled_buffer = np.empty(
            (self.n_modes, self.n_kept), dtype=np.float64, order="C"
        )
        # Buffer for U_S_U computation: (n_kept, n_kept)
        self._U_S_U_buffer = np.empty(
            (self.n_kept, self.n_kept), dtype=np.float64, order="F"
        )
        # Buffer for compressed covariance: (n_kept, n_kept)
        self._C_compressed_buffer = np.empty(
            (self.n_kept, self.n_kept), dtype=np.float64, order="F"
        )

    def get_projected_inverse(self, C_ell: np.ndarray) -> np.ndarray:
        """
        Compute inverse of compressed covariance (U^T @ C @ U)^{-1}.

        For pixel-projected compression, this returns the same as
        get_compressed_inverse(), which is the correct quantity for
        computing the compressed Fisher matrix:
            F_ij = (1/2) Tr[C_c^{-1} @ dC_c_i @ C_c^{-1} @ dC_c_j]

        where C_c = U^T @ C @ U is the compressed covariance.

        Note: This differs from the "projected inverse" U^T @ C^{-1} @ U,
        which is NOT equivalent for rectangular projection matrix U.
        The compressed Fisher formula requires the inverse of the compressed
        covariance, not the projected inverse.

        Parameters
        ----------
        C_ell : numpy.ndarray
            Power spectrum values for ell = 2 to lmax.

        Returns
        -------
        numpy.ndarray
            Inverse of compressed covariance (U^T @ C @ U)^{-1},
            shape (n_kept, n_kept).
        """
        if self._eigenvectors is None:
            raise RuntimeError("Compression not applied. Call apply_compression() first.")

        # Return inverse of compressed covariance
        return self.get_compressed_inverse(C_ell)

    def get_derivative_matrix(self, ell: int) -> np.ndarray:
        """
        Get ∂C_compressed/∂C_ℓ = (VU)^T E_ℓ (VU).

        The derivative matrix E_ℓ includes the (2ℓ+1)/(4π) factor to match
        the traditional pixel-space derivative, consistent with HarmonicCompression.

        Uses precomputed VU and derivative diagonals for efficiency.

        Parameters
        ----------
        ell : int
            Multipole for which to compute the derivative.

        Returns
        -------
        numpy.ndarray
            Derivative matrix of shape (n_kept, n_kept).
        """
        if self._eigenvectors is None:
            raise RuntimeError("Compression not applied. Call apply_compression() first.")

        # Use precomputed VU and E_ℓ diagonal
        E_diag = self._derivative_diagonals[ell]

        # (VU)^T @ diag(E_diag) @ VU = (VU * E_diag)^T @ VU
        # Use buffer for intermediate VU_scaled to reduce allocations
        np.multiply(self._VU, E_diag[:, np.newaxis], out=self._VU_scaled_buffer)
        return matrix_mult(self._VU_scaled_buffer.T, self._VU)

    def get_compressed_covariance(self, C_ell: np.ndarray) -> np.ndarray:
        """
        Compute compressed covariance C_compressed = U^T C U.

        Uses precomputed U^T N U and VU from _precompute_compression_products()
        for efficiency.

        Parameters
        ----------
        C_ell : numpy.ndarray
            Power spectrum values for ell = 2 to lmax.

        Returns
        -------
        numpy.ndarray
            Compressed covariance matrix of shape (n_kept, n_kept).
        """
        if self._eigenvectors is None:
            raise RuntimeError("Compression not applied. Call apply_compression() first.")

        # Use precomputed U^T N U
        U_N_U = self._U_N_U

        # U^T V^T Λ V U = (VU)^T Λ (VU) using precomputed VU
        # Use buffers to reduce allocations
        Lambda_diag = self._build_lambda_diagonal(C_ell)
        np.multiply(self._VU, Lambda_diag[:, np.newaxis], out=self._VU_scaled_buffer)
        np.matmul(self._VU.T, self._VU_scaled_buffer, out=self._U_S_U_buffer)

        # Return copy since caller may modify result
        return U_N_U + self._U_S_U_buffer

    def get_weighted_compressed_data(
        self, data: np.ndarray, C_ell: np.ndarray, C_c_inv: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Compute C_c^{-1} @ U^T @ d for QML estimation in compressed space.

        This implements the standard QML formula in compressed space, following
        Gjerløw et al. (2015) equation 18. The weighted compressed data is:

            w = C_c^{-1} @ d_c

        where:
            d_c = U^T @ d (compressed data)
            C_c = U^T @ C @ U (compressed covariance)

        The QML estimate is then: q_ℓ = (1/2) w^T @ E_ℓ @ w

        Parameters
        ----------
        data : numpy.ndarray
            Pixel-space data vector of shape (n_pix,) or (n_pix, n_sims).
        C_ell : numpy.ndarray
            Power spectrum values for ell = 2 to lmax.
        C_c_inv : numpy.ndarray, optional
            Precomputed compressed covariance inverse. If provided, C_ell is
            ignored and this matrix is used directly. This is useful when
            processing multiple simulations with the same C_ell.

        Returns
        -------
        numpy.ndarray
            Weighted compressed data of shape (n_kept,) or (n_kept, n_sims).

        References
        ----------
        .. [1] Gjerløw, E., et al. (2015). Section 5, Equation 18.
        """
        if self._eigenvectors is None:
            raise RuntimeError("Compression not applied. Call apply_compression() first.")

        # d_c = U^T @ d (compressed data)
        d_compressed = self._eigenvectors.T @ data

        # w = C_c^{-1} @ d_c
        if C_c_inv is None:
            C_c_inv = self.get_compressed_inverse(C_ell)
        return matrix_mult(C_c_inv, d_compressed)

    def compute_quadratic_form(self, data: np.ndarray, C_ell: np.ndarray) -> float:
        """
        Compute d^T C^{-1} d approximately in compressed space.

        For pixel_projected compression, we approximate:
            d^T C^{-1} d ≈ d_c^T @ C_c^{-1} @ d_c

        where d_c = U^T @ d and C_c = U^T @ C @ U.

        Note: This is an approximation that becomes exact as more modes are kept.

        Parameters
        ----------
        data : numpy.ndarray
            Pixel-space data vector of length n_pix.
        C_ell : numpy.ndarray
            Power spectrum values for ell = 2 to lmax.

        Returns
        -------
        float
            Approximate quadratic form value.
        """
        if self._eigenvectors is None:
            raise RuntimeError("Compression not applied. Call apply_compression() first.")

        # d_compressed = U^T @ d
        d_compressed = self._eigenvectors.T @ data

        # C_compressed^{-1}
        C_compressed_inv = self.get_projected_inverse(C_ell)

        return float(d_compressed.T @ C_compressed_inv @ d_compressed)

    @property
    def compression_ratio(self) -> float:
        """
        Ratio of kept modes to original pixels.

        Returns
        -------
        float
            Compression ratio (1.0 means no compression).
        """
        return self.n_kept / self.n_pix

    @property
    def eigenvalues(self) -> np.ndarray | None:
        """
        Eigenvalues of kept modes (sorted descending).

        Returns
        -------
        numpy.ndarray or None
            Eigenvalues if compression has been applied, None otherwise.
        """
        return self._eigenvalues

    @property
    def compression_basis(self) -> str | None:
        """
        The compression basis used in apply_compression().

        Returns
        -------
        str or None
            Basis name if compression has been applied, None otherwise.
        """
        return self._compression_basis

    @classmethod
    def available_bases(cls) -> dict[str, str]:
        """
        Get available compression bases and their descriptions.

        Returns
        -------
        dict
            Dictionary mapping basis names to their descriptions.

        Examples
        --------
        >>> PixelProjectedCompression.available_bases()
        {'harmonic': 'P_h = V^T V (pure harmonic projector)',
         'noise_weighted': 'P_h N^{-1} P_h (inverse noise weighting)',
         'total_covariance': 'P_h C^{-1} P_h (full covariance weighting, requires C_ell)',
         'snr': 'S^{1/2} N^{-1} S^{1/2} (signal-to-noise ratio, requires C_ell)'}
        """
        return COMPRESSION_BASES.copy()
