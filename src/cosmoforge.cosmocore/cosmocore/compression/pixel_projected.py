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

from ..basics import matrix_inverse_symm, matrix_mult, matrix_trace
from .base import BaseCompression, SMWPrepared

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
        spins: list[int] | None = None,
        basis: str = "noise_weighted",
        C_ell: np.ndarray | None = None,
        epsilon: float | list[float | tuple[float, float]] | None = None,
        mode_fraction: float | list[float | tuple[float, float]] | None = None,
    ):
        super().__init__(N, N_inv, theta, phi, lmax, beam, spins=spins)
        # Before compression, n_kept = n_pix
        self.n_kept = self.n_pix
        # Compression quantities
        self._basis = basis
        self._C_ell_for_basis = C_ell
        self._epsilon = epsilon
        self._mode_fraction = mode_fraction
        self._eigenvectors = None

        # Parse per-field thresholds
        self._epsilon_per_field = self._parse_per_field_thresholds(epsilon, "epsilon")
        self._mode_fraction_per_field = self._parse_per_field_thresholds(
            mode_fraction, "mode_fraction"
        )

    def _parse_per_field_thresholds(
        self,
        value: float | list[float | tuple[float, float]] | None,
        name: str,
    ) -> list[float | tuple[float, float]] | None:
        """
        Normalize threshold parameter to per-field list.

        Parameters
        ----------
        value : float, list, or None
            Threshold value(s). float broadcasts to all fields; list must match
            n_components; tuples in list are only valid for spin-2 fields.
        name : str
            Parameter name for error messages.

        Returns
        -------
        list or None
            Per-field threshold list, or None if value is None.
        """
        if value is None:
            return None

        if isinstance(value, (int, float)):
            return [float(value)] * self.n_components

        if isinstance(value, list):
            if len(value) != self.n_components:
                raise ValueError(
                    f"{name} list length ({len(value)}) must match "
                    f"number of components ({self.n_components})"
                )
            for i, v in enumerate(value):
                if isinstance(v, tuple):
                    if self._spins[i] != 2:
                        raise ValueError(
                            f"{name}[{i}] is a tuple (E/B split) but component "
                            f"{i} has spin {self._spins[i]}, not 2"
                        )
                    if len(v) != 2:
                        raise ValueError(
                            f"{name}[{i}] tuple must have exactly 2 elements "
                            f"(E_threshold, B_threshold), got {len(v)}"
                        )
            return value

        raise TypeError(
            f"{name} must be float, list, or None, got {type(value).__name__}"
        )

    def _eigendecompose_single(
        self,
        comp_matrix: np.ndarray,
        epsilon: float | None,
        mode_fraction: float | None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Eigendecompose a compression matrix and apply threshold.

        Parameters
        ----------
        comp_matrix : numpy.ndarray
            Symmetric compression matrix.
        epsilon : float or None
            Eigenvalue threshold relative to maximum.
        mode_fraction : float or None
            Fraction of modes to keep.

        Returns
        -------
        U : numpy.ndarray
            Kept eigenvectors, shape (n, n_kept).
        eigenvalues : numpy.ndarray or None
            Kept eigenvalues (descending), or None.
        """
        eigenvalues, eigenvectors = eigh(comp_matrix)

        # Sort descending
        sort_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]

        if mode_fraction is not None:
            n_significant = np.sum(eigenvalues > 1e-10 * np.max(np.abs(eigenvalues)))
            n_to_keep = max(1, int(np.ceil(mode_fraction * n_significant)))
            mask = np.zeros(len(eigenvalues), dtype=bool)
            mask[:n_to_keep] = True
        elif epsilon is not None:
            max_eigenvalue = np.max(np.abs(eigenvalues))
            threshold = epsilon * max_eigenvalue
            mask = np.abs(eigenvalues) > threshold
        else:
            # Default: keep everything above numerical noise
            max_eigenvalue = np.max(np.abs(eigenvalues))
            mask = np.abs(eigenvalues) > 1e-10 * max_eigenvalue

        return eigenvectors[:, mask], eigenvalues[mask]

    def _build_compression_matrix_for_subfield(
        self,
        V_sub: np.ndarray,
        N_field: np.ndarray,
        N_field_inv: np.ndarray,
        basis: str,
        C_ell_sub: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Build compression matrix for a sub-field (E or B rows of spin-2, or full spin-0).

        Parameters
        ----------
        V_sub : numpy.ndarray
            V sub-block, shape (n_sub_modes, n_field_pix).
        N_field : numpy.ndarray
            Noise covariance for this field, shape (n_field_pix, n_field_pix).
        N_field_inv : numpy.ndarray
            Noise inverse for this field.
        basis : str
            Compression basis.
        C_ell_sub : numpy.ndarray or None
            Auto-spectrum diagonal for this sub-field (EE for E, BB for B).

        Returns
        -------
        numpy.ndarray
            Compression matrix, shape (n_field_pix, n_field_pix).
        """
        P_sub = matrix_mult(V_sub.T, V_sub)

        if basis == "harmonic":
            return P_sub
        elif basis == "noise_weighted":
            return matrix_mult(matrix_mult(P_sub, N_field_inv), P_sub)
        elif basis == "total_covariance":
            if C_ell_sub is None:
                raise ValueError("C_ell required for 'total_covariance' basis")
            V_scaled = V_sub * C_ell_sub[:, np.newaxis]
            S_sub = matrix_mult(V_sub.T, V_scaled)
            C_sub = N_field + S_sub
            C_sub_inv = matrix_inverse_symm(C_sub, overwrite=True)
            return matrix_mult(matrix_mult(P_sub, C_sub_inv), P_sub)
        elif basis == "snr":
            if C_ell_sub is None:
                raise ValueError("C_ell required for 'snr' basis")
            V_scaled = V_sub * C_ell_sub[:, np.newaxis]
            S_sub = matrix_mult(V_sub.T, V_scaled)
            eigvals_S, eigvecs_S = eigh(S_sub)
            sqrt_eigvals = np.sqrt(np.maximum(eigvals_S, 1e-30))
            Q_scaled = eigvecs_S * sqrt_eigvals
            S_sqrt = matrix_mult(Q_scaled, eigvecs_S.T)
            return matrix_mult(matrix_mult(S_sqrt, N_field_inv), S_sqrt)
        else:
            raise ValueError(f"Unknown compression basis '{basis}'")

    def _eigendecompose_spin2_split(
        self,
        V_comp: np.ndarray,
        N_field: np.ndarray,
        N_field_inv: np.ndarray,
        basis: str,
        epsilon: tuple[float, float] | None,
        mode_fraction: tuple[float, float] | None,
        C_ell: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Eigendecompose a spin-2 field with separate E/B thresholds.

        Parameters
        ----------
        V_comp : numpy.ndarray
            Full V block for this spin-2 component, shape (2*n_base, 2*n_phys).
        N_field : numpy.ndarray
            Noise covariance for this field.
        N_field_inv : numpy.ndarray
            Noise inverse for this field.
        basis : str
            Compression basis.
        epsilon : tuple of float or None
            (E_epsilon, B_epsilon) thresholds.
        mode_fraction : tuple of float or None
            (E_fraction, B_fraction) mode fractions.
        C_ell : numpy.ndarray or None
            C_ell for basis that needs it. For spin-2, this should be a dict-like
            or we extract EE/BB diagonals.

        Returns
        -------
        U_combined : numpy.ndarray
            Orthogonalized eigenvectors, shape (n_field_pix, n_kept).
        """
        n_base = self._n_modes_base

        # Split V into E and B rows
        V_E = V_comp[:n_base, :]  # E modes, all QU pixels
        V_B = V_comp[n_base:, :]  # B modes, all QU pixels

        # Build per-sub-field C_ell diagonals if needed
        C_ell_E = None
        C_ell_B = None
        if C_ell is not None and isinstance(C_ell, dict):
            # Extract EE and BB diagonals
            C_ell_EE = C_ell.get("EE", None)
            C_ell_BB = C_ell.get("BB", None)
            if C_ell_EE is not None:
                C_ell_E = self._build_lambda_diagonal(C_ell_EE)
            if C_ell_BB is not None:
                C_ell_B = self._build_lambda_diagonal(C_ell_BB)

        # Build compression matrices for E and B separately
        comp_E = self._build_compression_matrix_for_subfield(
            V_E, N_field, N_field_inv, basis, C_ell_E
        )
        comp_B = self._build_compression_matrix_for_subfield(
            V_B, N_field, N_field_inv, basis, C_ell_B
        )

        # Eigendecompose each with separate thresholds
        eps_E = epsilon[0] if epsilon is not None else None
        eps_B = epsilon[1] if epsilon is not None else None
        mf_E = mode_fraction[0] if mode_fraction is not None else None
        mf_B = mode_fraction[1] if mode_fraction is not None else None

        U_E, _ = self._eigendecompose_single(comp_E, eps_E, mf_E)
        U_B, _ = self._eigendecompose_single(comp_B, eps_B, mf_B)

        # Combine and SVD-orthogonalize (E and B pixel patterns overlap on cut sky)
        U_combined = np.hstack([U_E, U_B])
        Q, S, _ = np.linalg.svd(U_combined, full_matrices=False)
        # Keep columns where singular values are significant
        keep = S > 1e-10 * S[0]
        return Q[:, keep]

    def _eigendecompose_field(
        self,
        comp_idx: int,
        basis: str,
        epsilon: float | tuple[float, float] | None,
        mode_fraction: float | tuple[float, float] | None,
        C_ell: np.ndarray | dict | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Eigendecompose a single field component.

        Routes to E/B split for spin-2 with tuple thresholds, or standard
        eigendecomposition otherwise.

        Parameters
        ----------
        comp_idx : int
            Component index.
        basis : str
            Compression basis.
        epsilon : float, tuple, or None
            Threshold(s).
        mode_fraction : float, tuple, or None
            Mode fraction(s).
        C_ell : array or dict or None
            Power spectrum for basis that needs it.

        Returns
        -------
        U : numpy.ndarray
            Eigenvectors for this field, shape (n_field_pix, n_kept).
        eigenvalues : numpy.ndarray or None
            Eigenvalues if available (None for E/B split).
        """
        spin = self._spins[comp_idx]
        pix_start = self._pix_offsets[comp_idx]
        pix_end = self._pix_offsets[comp_idx + 1]

        # Extract field noise blocks
        N_field = self._N[pix_start:pix_end, pix_start:pix_end]
        N_field_inv = self.N_inv[pix_start:pix_end, pix_start:pix_end]

        V_comp = self._V_blocks[comp_idx]

        use_eb_split = spin == 2 and (
            isinstance(epsilon, tuple) or isinstance(mode_fraction, tuple)
        )

        if use_eb_split:
            U = self._eigendecompose_spin2_split(
                V_comp,
                N_field,
                N_field_inv,
                basis,
                epsilon if isinstance(epsilon, tuple) else None,
                mode_fraction if isinstance(mode_fraction, tuple) else None,
                C_ell,
            )
            return U, None
        else:
            # Standard: build compression matrix for full field
            eps_scalar = epsilon if not isinstance(epsilon, tuple) else epsilon[0]
            mf_scalar = mode_fraction if not isinstance(mode_fraction, tuple) else None

            # Build P_h for this field
            P_sub = matrix_mult(V_comp.T, V_comp)

            if basis == "harmonic":
                comp_matrix = P_sub
            elif basis == "noise_weighted":
                comp_matrix = matrix_mult(matrix_mult(P_sub, N_field_inv), P_sub)
            elif basis == "total_covariance":
                if C_ell is None:
                    raise ValueError("C_ell required for 'total_covariance' basis")
                Lambda_diag = self._build_lambda_diagonal(
                    C_ell if not isinstance(C_ell, dict) else next(iter(C_ell.values()))
                )
                V_scaled = V_comp * Lambda_diag[: V_comp.shape[0], np.newaxis]
                S = matrix_mult(V_comp.T, V_scaled)
                C_total = N_field + S
                C_inv = matrix_inverse_symm(C_total, overwrite=True)
                comp_matrix = matrix_mult(matrix_mult(P_sub, C_inv), P_sub)
            elif basis == "snr":
                if C_ell is None:
                    raise ValueError("C_ell required for 'snr' basis")
                Lambda_diag = self._build_lambda_diagonal(
                    C_ell if not isinstance(C_ell, dict) else next(iter(C_ell.values()))
                )
                V_scaled = V_comp * Lambda_diag[: V_comp.shape[0], np.newaxis]
                S = matrix_mult(V_comp.T, V_scaled)
                eigvals_S, eigvecs_S = eigh(S)
                sqrt_eigvals = np.sqrt(np.maximum(eigvals_S, 1e-30))
                Q_scaled = eigvecs_S * sqrt_eigvals
                S_sqrt = matrix_mult(Q_scaled, eigvecs_S.T)
                comp_matrix = matrix_mult(matrix_mult(S_sqrt, N_field_inv), S_sqrt)
            else:
                raise ValueError(f"Unknown compression basis '{basis}'")

            U, eigvals = self._eigendecompose_single(comp_matrix, eps_scalar, mf_scalar)
            return U, eigvals

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

        if self._epsilon is not None or self._mode_fraction is not None:
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
        epsilon: float | list[float | tuple[float, float]] | None = None,
        mode_fraction: float | list[float | tuple[float, float]] | None = None,
        basis: str = "noise_weighted",
        C_ell: np.ndarray | None = None,
    ) -> None:
        """
        Apply eigenvalue compression to find optimal subspace.

        Computes the eigendecomposition of the compression matrix (determined by
        the basis) and selects modes to keep based on either an eigenvalue
        threshold or a fraction of total modes.

        Supports per-field thresholds and separate E/B thresholds for spin-2:
        - float: broadcast to all fields
        - list[float]: per-field threshold
        - list[float | tuple[float, float]]: tuples give (E, B) split for spin-2

        Parameters
        ----------
        epsilon : float, list, or None
            Eigenvalue threshold. Modes with eigenvalue > epsilon * max_eigenvalue
            are kept. Mutually exclusive with mode_fraction.
        mode_fraction : float, list, or None
            Fraction of modes to keep (between 0 and 1). Keeps the top modes
            ordered by eigenvalue. Mutually exclusive with epsilon.
        basis : str, default "noise_weighted"
            Compression basis determining which matrix to eigendecompose.
        C_ell : numpy.ndarray, optional
            Power spectrum values for ell = 2 to lmax. Required for
            "total_covariance" and "snr" bases.

        Raises
        ------
        ValueError
            If both epsilon and mode_fraction are provided.
            If mode_fraction is not in (0, 1].
            If C_ell is required but not provided.
        """
        # Parse per-field thresholds
        eps_list = self._parse_per_field_thresholds(epsilon, "epsilon")
        mf_list = self._parse_per_field_thresholds(mode_fraction, "mode_fraction")

        # Validate mutual exclusivity (check scalar level)
        if eps_list is not None and mf_list is not None:
            raise ValueError(
                "epsilon and mode_fraction are mutually exclusive. "
                "Provide only one compression criterion."
            )

        if mf_list is not None:
            # Use original error message format for scalar inputs
            scalar_input = isinstance(mode_fraction, (int, float))
            for i, mf in enumerate(mf_list):
                if isinstance(mf, tuple):
                    for v in mf:
                        if not (0 < v <= 1):
                            raise ValueError(
                                f"mode_fraction[{i}] values must be in (0, 1], got {mf}"
                            )
                else:
                    if not (0 < mf <= 1):
                        if scalar_input:
                            raise ValueError(f"mode_fraction must be in (0, 1], got {mf}")
                        raise ValueError(
                            f"mode_fraction[{i}] must be in (0, 1], got {mf}"
                        )

        # Default epsilon if neither specified
        if eps_list is None and mf_list is None:
            eps_list = [1e-6] * self.n_components

        # Determine if we need per-field decomposition
        has_any_tuple = (
            eps_list is not None and any(isinstance(e, tuple) for e in eps_list)
        ) or (mf_list is not None and any(isinstance(m, tuple) for m in mf_list))
        need_per_field = (
            has_any_tuple or self.n_components > 1 or any(s == 2 for s in self._spins)
        )

        if need_per_field:
            # Per-field decomposition path
            U_blocks = []
            for comp_idx in range(self.n_components):
                eps_i = eps_list[comp_idx] if eps_list is not None else None
                mf_i = mf_list[comp_idx] if mf_list is not None else None

                U_i, _ = self._eigendecompose_field(comp_idx, basis, eps_i, mf_i, C_ell)
                U_blocks.append(U_i)

            # Assemble block-diagonal U
            total_kept = sum(U_i.shape[1] for U_i in U_blocks)
            U_full = np.zeros((self.n_pix, total_kept), dtype=np.float64)
            col = 0
            for comp_idx, U_i in enumerate(U_blocks):
                pix_start = self._pix_offsets[comp_idx]
                pix_end = self._pix_offsets[comp_idx + 1]
                n_kept_i = U_i.shape[1]
                U_full[pix_start:pix_end, col : col + n_kept_i] = U_i
                col += n_kept_i

            self._eigenvectors = U_full
            self._eigenvalues = None  # Not meaningful for combined E/B
            self.n_kept = total_kept
        else:
            # Single-field spin-0 path (backward compatible)
            eps_scalar = eps_list[0] if eps_list is not None else None
            mf_scalar = mf_list[0] if mf_list is not None else None

            compression_matrix = self._build_compression_matrix(basis, C_ell)
            eigenvectors, eigenvalues = self._eigendecompose_single(
                compression_matrix, eps_scalar, mf_scalar
            )

            self._eigenvalues = eigenvalues
            self._eigenvectors = eigenvectors
            self.n_kept = eigenvectors.shape[1]

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

    # === Spin-2 aware operations ===

    def get_compressed_covariance_multi(self, C_ell_dict: dict) -> np.ndarray:
        """Compute compressed covariance U^T C U. Accepts 2-tuple or 3-tuple keys."""
        if self._eigenvectors is None:
            raise RuntimeError("Compression not applied. Call apply_compression() first.")

        Lambda_full = self._build_lambda_full(C_ell_dict)
        VU_Lambda = matrix_mult(Lambda_full, self._VU)
        U_S_U = matrix_mult(self._VU.T, VU_Lambda)
        return self._U_N_U + U_S_U

    def get_compressed_inverse_multi(self, C_ell_dict: dict) -> np.ndarray:
        """Compute inverse of compressed covariance. Accepts 2-tuple or 3-tuple keys."""
        return matrix_inverse_symm(self.get_compressed_covariance_multi(C_ell_dict))

    def get_derivative_matrix_multi(
        self, ell: int, comp_i: int, comp_j: int, mode: int = 0
    ) -> np.ndarray:
        """Get compressed derivative matrix for (comp_i, comp_j, mode) at ell."""
        if self._eigenvectors is None:
            raise RuntimeError("Compression not applied. Call apply_compression() first.")

        if (
            self._spins[comp_i] == 0
            and self._spins[comp_j] == 0
            and self.n_components == 1
        ):
            return self.get_derivative_matrix(ell)

        E = self._build_derivative_matrix_with_spins(ell, comp_i, comp_j, mode)
        E_VU = matrix_mult(E, self._VU)
        return matrix_mult(self._VU.T, E_VU)

    def compute_fisher_matrix_multi(
        self,
        C_ell_dict: dict,
        spectra_list: list[tuple],
        ell_min: int = 2,
        ell_max: int | None = None,
    ) -> np.ndarray:
        """Compute Fisher matrix. Accepts 2-tuple or 3-tuple spectra_list entries."""
        if ell_max is None:
            ell_max = self.lmax

        n_ell = ell_max - ell_min + 1
        n_spec = len(spectra_list)
        fisher = np.zeros((n_spec * n_ell, n_spec * n_ell))

        C_c_inv = self.get_compressed_inverse_multi(C_ell_dict)

        Cinv_dC = {}
        for spec_idx, spec_entry in enumerate(spectra_list):
            comp_i, comp_j = spec_entry[0], spec_entry[1]
            mode = spec_entry[2] if len(spec_entry) == 3 else 0
            for ell in range(ell_min, ell_max + 1):
                dC = self.get_derivative_matrix_multi(ell, comp_i, comp_j, mode)
                Cinv_dC[(spec_idx, ell)] = matrix_mult(C_c_inv, dC)

        for spec_a in range(n_spec):
            for ell_a in range(ell_min, ell_max + 1):
                idx_a = spec_a * n_ell + (ell_a - ell_min)

                for spec_b in range(spec_a, n_spec):
                    ell_b_start = ell_a if spec_a == spec_b else ell_min
                    for ell_b in range(ell_b_start, ell_max + 1):
                        idx_b = spec_b * n_ell + (ell_b - ell_min)

                        fisher_val = 0.5 * matrix_trace(
                            Cinv_dC[(spec_a, ell_a)],
                            Cinv_dC[(spec_b, ell_b)],
                        )

                        fisher[idx_a, idx_b] = fisher_val
                        if idx_a != idx_b:
                            fisher[idx_b, idx_a] = fisher_val

        return fisher

    def get_weighted_compressed_data_multi(
        self, data: np.ndarray, C_ell_dict: dict
    ) -> np.ndarray:
        """Compute C_c^{-1} @ U^T @ d. Accepts 2-tuple or 3-tuple keys."""
        if self._eigenvectors is None:
            raise RuntimeError("Compression not applied. Call apply_compression() first.")

        d_compressed = self._eigenvectors.T @ data
        C_c_inv = self.get_compressed_inverse_multi(C_ell_dict)
        return matrix_mult(C_c_inv, d_compressed)

    def prepare_smw_multi(self, C_ell_dict: dict) -> SMWPrepared:
        """Precompute compressed inverse and logdet for reuse across sims."""
        if self._eigenvectors is None:
            raise RuntimeError("Compression not applied. Call apply_compression() first.")

        from ..basics import matrix_slogdet_symm

        C_c = self.get_compressed_covariance_multi(C_ell_dict)
        C_c_inv = matrix_inverse_symm(C_c)
        _, logdet = matrix_slogdet_symm(C_c)
        return SMWPrepared(C_c_inv, None, logdet)

    def quadratic_form_from_prepared(
        self, data: np.ndarray, C_c_inv: np.ndarray
    ) -> float:
        """Compute d^T C^{-1} d using precomputed compressed inverse."""
        if self._eigenvectors is None:
            raise RuntimeError("Compression not applied. Call apply_compression() first.")

        d_c = self._eigenvectors.T @ data
        return float(d_c.T @ C_c_inv @ d_c)

    def compute_quadratic_form_multi(self, data: np.ndarray, C_ell_dict: dict) -> float:
        """Compute d^T C^{-1} d in compressed space."""
        C_c_inv, _, _ = self.prepare_smw_multi(C_ell_dict)
        return self.quadratic_form_from_prepared(data, C_c_inv)

    def get_logdet_multi(self, C_ell_dict: dict) -> float:
        """Compute log determinant of compressed covariance."""
        _, _, logdet = self.prepare_smw_multi(C_ell_dict)
        return logdet

    # === Deprecated _with_spins aliases (delegate to _multi) ===

    def get_compressed_covariance_with_spins(self, C_ell_dict: dict) -> np.ndarray:
        """Deprecated: use get_compressed_covariance_multi."""
        return self.get_compressed_covariance_multi(C_ell_dict)

    def get_compressed_inverse_with_spins(self, C_ell_dict: dict) -> np.ndarray:
        """Deprecated: use get_compressed_inverse_multi."""
        return self.get_compressed_inverse_multi(C_ell_dict)

    def get_derivative_matrix_with_spins(
        self, ell: int, comp_i: int, comp_j: int, mode: int = 0
    ) -> np.ndarray:
        """Deprecated: use get_derivative_matrix_multi."""
        return self.get_derivative_matrix_multi(ell, comp_i, comp_j, mode)

    def compute_fisher_matrix_with_spins(
        self,
        C_ell_dict: dict,
        spectra_list: list,
        ell_min: int = 2,
        ell_max: int | None = None,
    ) -> np.ndarray:
        """Deprecated: use compute_fisher_matrix_multi."""
        return self.compute_fisher_matrix_multi(
            C_ell_dict, spectra_list, ell_min, ell_max
        )

    def get_weighted_compressed_data_with_spins(
        self, data: np.ndarray, C_ell_dict: dict
    ) -> np.ndarray:
        """Deprecated: use get_weighted_compressed_data_multi."""
        return self.get_weighted_compressed_data_multi(data, C_ell_dict)

    def prepare_smw_with_spins(self, C_ell_dict: dict) -> SMWPrepared:
        """Deprecated: use prepare_smw_multi."""
        return self.prepare_smw_multi(C_ell_dict)

    def compute_quadratic_form_with_spins(
        self, data: np.ndarray, C_ell_dict: dict
    ) -> float:
        """Deprecated: use compute_quadratic_form_multi."""
        return self.compute_quadratic_form_multi(data, C_ell_dict)

    def get_logdet_with_spins(self, C_ell_dict: dict) -> float:
        """Deprecated: use get_logdet_multi."""
        return self.get_logdet_multi(C_ell_dict)

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
