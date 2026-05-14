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

from ..basics import (
    cholesky_solve,
    eigh,
    eigvalsh,
    matrix_inverse_symm,
    matrix_mult,
    matrix_trace,
    svd,
    symmetrize_inplace,
)
from .base import ComputationBasis, SMWPrepared

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# Available compression basis presets
COMPRESSION_BASES = {
    "harmonic": "P_h = V^T V (pure harmonic projector)",
    "noise_weighted": "P_h N^{-1} P_h (inverse noise weighting)",
    "total_covariance": "P_h C^{-1} P_h (full covariance weighting, requires C_ell)",
    "snr": "S^{1/2} N^{-1} S^{1/2} (signal-to-noise ratio, requires C_ell)",
}


class PixelBasis(ComputationBasis):
    """
    Pixel-space compression with projector (Gjerløw-like).

    Stays in n_pix space with projector P_h, then uses eigenvalue
    decomposition to find optimal subspace. Compression: n_pix → dim.

    The key operations are:
    - Projector: P_h = V^T V (harmonic projector, n_pix × n_pix but rank n_modes)
    - Eigendecomposition of compression matrix to find optimal modes
    - Data compression: d_c = P @ d where P = U^T (dim × n_pix)

    This approach is more flexible than HarmonicBasis because it allows
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
        Noise covariance matrix of shape (n_pix, n_pix). The basis takes
        ownership of this buffer; ``_factorise_noise()`` overwrites it
        in place during setup (V-based mode only; pixel-direct keeps it).
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
    dim : int
        Number of modes kept after compression (initially n_pix).

    Examples
    --------
    >>> import numpy as np
    >>> from cosmocore.basis import PixelBasis
    >>> N = np.diag(noise_variance)  # Noise covariance matrix
    >>> ppc = PixelBasis(N, theta, phi, lmax=100)
    >>> ppc.setup()
    >>> # Inspect eigenvalue spectrum to choose threshold
    >>> fig, axes = ppc.plot_eigenvalue_spectrum(basis="noise_weighted")
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
        theta: np.ndarray,
        phi: np.ndarray,
        lmax_signal: int,
        beam: np.ndarray | None = None,
        spins: list[int] | None = None,
        basis: str = "noise_weighted",
        C_ell: np.ndarray | None = None,
        epsilon: float | list[float | tuple[float, float]] | None = None,
        mode_fraction: float | list[float | tuple[float, float]] | None = None,
        lmin_signal: list[int] | None = None,
        lmin: int | None = None,
        lmax: int | None = None,
        S_fixed: np.ndarray | None = None,
        fields=None,
        use_direct: bool = False,
    ):
        super().__init__(
            N,
            theta,
            phi,
            lmax_signal,
            beam,
            spins=spins,
            lmin_signal=lmin_signal,
            lmin=lmin,
            lmax=lmax,
            S_fixed=S_fixed,
        )
        self._fields = fields
        self._use_direct = use_direct

        if not self._use_direct:
            self._init_harmonic_internals()

        # Before compression, dim = n_pix
        self.dim = self.n_pix
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
            Kept eigenvectors, shape (n, dim).
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
            Orthogonalized eigenvectors, shape (n_field_pix, dim).
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
        Q, S, _ = svd(U_combined)
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
            Eigenvectors for this field, shape (n_field_pix, dim).
        eigenvalues : numpy.ndarray or None
            Eigenvalues if available (None for E/B split).
        """
        spin = self._spins[comp_idx]
        pix_start = self._pix_offsets[comp_idx]
        pix_end = self._pix_offsets[comp_idx + 1]

        # Extract field noise blocks. Reads via _N_symmetric and N_inv so the
        # method works whether called during setup (symmetric N still alive)
        # or post-setup (lazy reconstruction via Cholesky factor).
        N_field = self._N_symmetric[pix_start:pix_end, pix_start:pix_end]
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
    def method(self) -> str:
        """Compression method name."""
        return "pixel"

    @property
    def projector(self) -> np.ndarray:
        """
        Get the projection matrix U^T (dim × n_pix).

        Maps pixel space to eigenbasis compressed space. In direct mode
        without compression, the projector is the identity.

        Raises
        ------
        RuntimeError
            If compression has not been applied yet (V-based mode only).
        """
        if self._use_direct:
            # Cache the identity projector — n_pix×n_pix dense, allocated once.
            cached = getattr(self, "_direct_projector", None)
            if cached is None:
                cached = np.eye(self.n_pix)
                self._direct_projector = cached
            return cached
        if self._eigenvectors is None:
            raise RuntimeError("Compression not applied. Call apply_compression() first.")
        return self._eigenvectors.T

    def compress_data(self, data: np.ndarray) -> np.ndarray:
        """
        Project pixel-space data into the basis.

        Pixel-direct mode short-circuits the multiplication against the
        identity projector (otherwise an ``n_pix × n_sims`` matmul against
        an ``n_pix × n_pix`` identity, allocated and run on every call).
        """
        if self._use_direct:
            return data
        return super().compress_data(data)

    def setup(self) -> None:
        """
        Initialize the basis.

        In V-based mode: build V and P_h, optionally apply eigenvalue
        compression. In direct mode (n_pix small enough that V is overkill):
        skip V construction and rely on existing pixel-space machinery
        (compute_signal_matrix, do_derivative_step).
        """
        if self._use_direct:
            self._setup_direct()
        else:
            self._setup_v_based()

    def _setup_v_based(self) -> None:
        """V-based setup: build V, P_h, optionally apply compression.

        When switch optimization is active (lswitch_high < lmax), absorbs
        high-ℓ signal into N_eff and uses N_eff for compression matrix
        construction. Raw N is preserved as ``self._N_original`` for
        noise-bias computations downstream.
        """
        self._build_basis()
        if self._use_switch_optimization:
            self._compute_effective_noise()
        self._P_h = matrix_mult(self._V.T, self._V)
        if self._epsilon is not None or self._mode_fraction is not None:
            self.apply_compression(
                epsilon=self._epsilon,
                mode_fraction=self._mode_fraction,
                basis=self._basis,
                C_ell=self._C_ell_for_basis,
            )
        # Precompute U^T N_raw U so _N_original can be released; the
        # noise-bias path reads only the compressed form thereafter.
        self._precompute_compressed_noise()
        # Factor self._N in place. After this point the symmetric N buffer
        # is gone; consumers go through cholesky_solve via _N_chol or the
        # lazy _N_symmetric reconstruction (deferred compression-basis paths).
        self._factorise_noise()

    def _precompute_compressed_noise(self) -> None:
        """Cache U^T N_orig U and release the n_pix x n_pix raw N reference.

        Only possible when compression is active (``_eigenvectors`` is set)
        and switch optimization captured a raw ``_N_original``. Otherwise
        ``get_noise_for_bias`` falls back to ``_N`` / ``_N_original``.
        """
        N_original = getattr(self, "_N_original", None)
        if N_original is None or self._eigenvectors is None:
            return
        U = self._eigenvectors
        UN_raw_U = U.T @ N_original @ U
        self._compressed_noise_cache = symmetrize_inplace(UN_raw_U)
        del self._N_original

    def _compute_effective_noise(self) -> None:
        """Absorb high-ℓ signal into N_eff (mirrors HarmonicBasis).

        Replaces ``self._N`` with ``N_eff = N + S_fixed``. The factorisation
        happens later, in ``_factorise_noise`` at the end of setup, after
        compression-basis selection has read symmetric N for U^T N U.
        """
        if self._S_fixed is None:
            raise ValueError(
                "S_fixed must be provided when using switch optimization "
                "(lswitch_high < lmax)"
            )
        S_fixed = np.asarray(self._S_fixed, dtype=np.float64)

        # Preserve raw N for noise-bias computations (still read lazily by
        # get_noise_for_bias, so we cannot release it here).
        self._N_original = self._N

        # Build N_eff = N + S_fixed straight into a preallocated F-order
        # buffer to avoid the C-order intermediate that `_N + S_fixed`
        # would otherwise produce (one extra n_pix^2 alloc at peak).
        N_eff = np.empty_like(self._N, order="F")
        np.add(self._N, S_fixed, out=N_eff)
        self._N = N_eff

        # S_fixed has been absorbed into N_eff and is not read again; release.
        self._S_fixed = None

    def get_noise_for_bias(self) -> np.ndarray:
        """U^T N_raw U — raw noise projected into the eigenmode basis.

        See :meth:`ComputationBasis.get_noise_for_bias` for the cross-basis
        contract. Use this for noise-bias computations:
        ``Tr[C^{-1} N C^{-1} dS]`` requires the actual noise N, not the
        effective N_eff that includes the absorbed high-ℓ signal.

        Without switch optimization, N_eff = N and this is identical to
        ``get_compressed_covariance(0)``.
        """
        cached = getattr(self, "_compressed_noise_cache", None)
        if cached is not None:
            return cached
        N_raw = getattr(self, "_N_original", self._N_symmetric)
        if self._eigenvectors is None:
            return N_raw
        U = self._eigenvectors
        UN_raw_U = U.T @ N_raw @ U
        return symmetrize_inplace(UN_raw_U)

    def _setup_direct(self) -> None:
        """Direct pixel-space setup. No V operator, no harmonic machinery.

        Direct mode reuses existing pixel-space machinery from
        ``cosmocore.signal_kernels`` — no Legendre or geometry recomputation.
        """
        if self._fields is None:
            raise ValueError(
                "Direct mode requires fields (FieldCollection) to be passed to PixelBasis"
            )
        # Build (label -> spectrum_idx) mapping for derivative dispatch
        self._spec_label_to_idx = {
            label: idx for idx, label in enumerate(self._fields.spectra_labels)
        }

    # =========================================================================
    # Direct pixel-space methods (no V operator, reuse cosmocore.signal_kernels)
    # =========================================================================

    def _spectrum_idx_from_components(self, comp_i: int, comp_j: int, mode: int) -> int:
        """Map (comp_i, comp_j, mode) to spectrum_idx for do_derivative_step.

        Mode encoding follows the QML/CosmoForge convention:
        - spin-0 × spin-0: mode 0 = auto (e.g. TT)
        - spin-2 × spin-2 same component: mode 0 = EE, 1 = BB, 2 = EB
        - spin-2 × spin-2 cross-component: linear over label pairs
        - spin-0 × spin-2: linear over label pairs (e.g. TE, TB)
        """
        fi = self._fields.fields[comp_i]
        fj = self._fields.fields[comp_j]
        labels_i = fi.maps_label if isinstance(fi.maps_label, list) else [fi.maps_label]
        labels_j = fj.maps_label if isinstance(fj.maps_label, list) else [fj.maps_label]

        if fi.spin == 2 and fj.spin == 2 and comp_i == comp_j:
            # Auto-spectrum on a single spin-2 field: mode 0=EE, 1=BB, 2=EB
            if mode == 0:
                label = labels_i[0] + labels_j[0]
            elif mode == 1:
                label = labels_i[1] + labels_j[1]
            else:  # mode == 2
                label = labels_i[0] + labels_j[1]
        else:
            combined = [a + b for a in labels_i for b in labels_j]
            label = combined[mode]
        return self._spec_label_to_idx[label]

    @staticmethod
    def _symmetrize_inplace(block: np.ndarray) -> None:
        """Set block[i,j] = block[j,i] for i < j, in-place. Block must be square."""
        # Equivalent to: block += block.T - diag(block)
        # but uses in-place ops to avoid temporary allocations of size n_pix^2.
        idx = np.triu_indices_from(block, k=1)
        block[idx] = block[(idx[1], idx[0])]

    def _direct_buffer(self) -> np.ndarray:
        """Return a reusable n_pix × n_pix Fortran-ordered buffer.

        Allocated lazily on first use. Repeatedly zeroed and refilled by the
        per-spectrum contribution functions to avoid 8 GB+ allocation churn
        during binned-derivative loops.
        """
        if not hasattr(self, "_direct_pix_buffer"):
            self._direct_pix_buffer = np.asfortranarray(
                np.zeros((self.n_pix, self.n_pix), dtype=np.float64)
            )
        return self._direct_pix_buffer

    def _get_derivative_direct(
        self,
        ell: int,
        comp_i: int | None = None,
        comp_j: int | None = None,
        mode: int = 0,
    ) -> np.ndarray:
        """Compute dS/dC_ell via existing pixel-space machinery.

        Reuses ``do_derivative_step`` from ``cosmocore.signal_kernels`` — no
        duplication of Legendre or dispatch logic.
        """
        from ..signal_kernels import do_derivative_step

        ci = 0 if comp_i is None else comp_i
        cj = 0 if comp_j is None else comp_j
        spectrum_idx = self._spectrum_idx_from_components(ci, cj, mode)

        dS = self._direct_buffer()
        dS.fill(0.0)
        do_derivative_step(
            dS,
            spectrum_idx,
            self._fields.n_active,
            self._fields.spin,
            ell,
            self._fields,
        )
        return dS

    def get_binned_derivative_direct(
        self,
        bin_idx: int,
        bins,
        beam_smoothing: np.ndarray | None,
        key,
    ) -> np.ndarray:
        """Build the binned derivative dC^b in a single Legendre/Wigner pass.

        Mathematically equivalent to summing per-ℓ ``_get_derivative_direct``
        results with weights ``beam²(ℓ)`` over ℓ ∈ [bins.lmins[bin_idx],
        bins.lmaxs[bin_idx]], but uses one ``compute_*_contribution`` call
        instead of N (where N = bin width). The contribution functions
        accumulate ``Σ_ℓ cl[ell] × kernel[ell]`` per pixel pair on
        ℓ-indexed arrays, so populating ``cl[ell] = beam²(ell)`` inside the
        bin (zero elsewhere) yields the exact binned derivative.

        Parameters
        ----------
        key : SpectrumKey
            Identifies the spectrum (component pair + spin kind). Decomposed
            into ``(comp_i, comp_j, mode)`` once at entry for the Numba
            kernels below.
        """
        from ..signal_kernels import (
            compute_00_contribution,
            compute_02_contribution,
            compute_22_contribution,
        )
        from ..spectrum_key import kind_to_legacy_mode

        is_cross = key.comp_i != key.comp_j
        comp_i = key.comp_i
        comp_j = key.comp_j
        mode = kind_to_legacy_mode(key.kind, is_cross=is_cross)

        lmin_b = bins.lmins[bin_idx]
        lmax_b = bins.lmaxs[bin_idx]

        # Build ℓ-indexed weight array w[ell] = beam²(ell) inside the bin and
        # 0 elsewhere. The contribution functions evaluate Σ_ℓ w[ell] ×
        # kernel[ell] for ell=lmin..lmax, giving Σ_{ell∈bin} beam²(ell) ×
        # dC/dC_ell. beam_smoothing (when provided) is the inference-range
        # slice (ell=bins.lmin..lmax, offset-from-bins.lmin; ADR 0009).
        weights = np.zeros(self.lmax + 1, dtype=np.float64)
        for ell in range(lmin_b, lmax_b + 1):
            if ell > self.lmax:
                continue
            w = 1.0
            if beam_smoothing is not None:
                w = beam_smoothing[ell - bins.lmin]
            weights[ell] = w

        fi = self._fields.fields[comp_i]
        fj = self._fields.fields[comp_j]
        spin_i = fi.spin
        spin_j = fj.spin

        # Pixel layout matches do_derivative_step's convention.
        ri = sum(
            2 * n if s == 2 else n
            for n, s in zip(
                self._fields.n_active[:comp_i],
                self._fields.spin[:comp_i],
            )
        )
        rj = sum(
            2 * n if s == 2 else n
            for n, s in zip(
                self._fields.n_active[:comp_j],
                self._fields.spin[:comp_j],
            )
        )
        nrow = self._n_pix_per_component[comp_i]
        ncol = self._n_pix_per_component[comp_j]

        # Allocate a fresh output array — Fisher stores all binned derivatives
        # in a dict and reuses them later, so we cannot share a buffer across
        # binned calls without aliasing.
        dS = np.asfortranarray(np.zeros((self.n_pix, self.n_pix), dtype=np.float64))
        block = dS[rj : rj + ncol, ri : ri + nrow]
        legendre = np.empty(self.lmax + 1, dtype=np.float64)
        zeros = np.zeros(self.lmax + 1, dtype=np.float64)

        if spin_i == 0 and spin_j == 0:
            sub_mode = 0 if comp_i == comp_j else 1
            compute_00_contribution(
                weights, block, fi.point_vectors, fj.point_vectors, legendre, sub_mode
            )
            if sub_mode == 0:
                # compute_00_contribution mode=0 fills only lower + diagonal;
                # derivative_step_00 mode=0 symmetrizes upper at the end.
                self._symmetrize_inplace(block)
            else:
                # Cross-field (mode=1): both halves filled inside, but the
                # transposed cross-block in dS must be set to match
                # do_derivative_step's outer-level mirror.
                dS[ri : ri + nrow, rj : rj + ncol] = block.T
        elif spin_i == 2 and spin_j == 2 and comp_i == comp_j:
            f1 = np.empty(self.lmax + 1, dtype=np.float64)
            f2 = np.empty(self.lmax + 1, dtype=np.float64)
            if mode == 0:  # EE
                cl11, cl22, cl12 = weights, zeros, zeros
            elif mode == 1:  # BB
                cl11, cl22, cl12 = zeros, weights, zeros
            elif mode == 2:  # EB
                cl11, cl22, cl12 = zeros, zeros, weights
            else:
                raise ValueError(f"Invalid spin-2 mode {mode}")
            compute_22_contribution(
                cl11,
                cl22,
                cl12,
                block,
                fi.point_vectors,
                fj.point_vectors,
                legendre,
                f1,
                f2,
            )
            # derivative_step_22 symmetrizes its full output; compute_22 does
            # not, so do it here. The block is the full square spin-2 derivative.
            self._symmetrize_inplace(block)
        elif (spin_i, spin_j) in ((0, 2), (2, 0)):
            cl12 = weights if mode == 0 else zeros
            cl13 = weights if mode == 1 else zeros
            if spin_i == 0:
                compute_02_contribution(
                    cl12,
                    cl13,
                    block,
                    fi.point_vectors,
                    fj.point_vectors,
                    legendre,
                )
                # Mirror to the (i,j) block so the full dS is symmetric, matching
                # do_derivative_step's outer-level mirror (cosmocore/pixel.py
                # line ~781) for the spin-0×spin-2 case.
                dS[ri : ri + nrow, rj : rj + ncol] = block.T
            else:
                block_swapped = dS[ri : ri + nrow, rj : rj + ncol]
                compute_02_contribution(
                    cl12,
                    cl13,
                    block_swapped,
                    fj.point_vectors,
                    fi.point_vectors,
                    legendre,
                )
                # Same mirror, opposite direction.
                dS[rj : rj + ncol, ri : ri + nrow] = block_swapped.T
        else:
            raise NotImplementedError(
                f"Direct binned derivative not implemented for spins ({spin_i}, {spin_j})"
            )

        return dS

    def _build_signal_matrix_direct(self) -> np.ndarray:
        """Build signal matrix S via existing pixel-space machinery.

        Uses the spectra currently set on the FieldCollection.
        Reuses ``compute_signal_matrix`` from ``cosmocore.signal_kernels``.
        """
        from ..signal_kernels import compute_signal_matrix

        S = np.zeros((self.n_pix, self.n_pix), dtype=np.float64)
        S = np.asfortranarray(S)
        compute_signal_matrix(S, self.lmax_signal, self._fields)
        return S

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
            # Total covariance C = N + S (symmetric N via lazy reconstruction
            # if the basis has been factorised in place).
            C = self._N_symmetric + S
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

    def compute_eigenspectrum_per_field(
        self,
        basis: str = "noise_weighted",
        C_ell: np.ndarray | dict | None = None,
    ) -> list[dict]:
        """
        Compute per-component eigenvalue spectra.

        Returns one entry per component with eigenvalues and normalized
        eigenvalues.  For spin-2 components the result additionally contains
        separate E- and B-mode eigenspectra.

        Parameters
        ----------
        basis : str, default "noise_weighted"
            Compression basis.
        C_ell : numpy.ndarray, dict, or None
            Power spectrum (required for "total_covariance" and "snr" bases).

        Returns
        -------
        list of dict
            One dict per component with keys: ``component``, ``spin``,
            ``label``, ``eigenvalues``, ``normalized_eigenvalues``.
            Spin-2 components additionally have ``E_eigenvalues``,
            ``E_normalized``, ``B_eigenvalues``, ``B_normalized``.
        """
        if basis not in COMPRESSION_BASES:
            raise ValueError(
                f"Unknown compression basis '{basis}'. "
                f"Available: {list(COMPRESSION_BASES.keys())}"
            )

        results: list[dict] = []

        for comp_idx in range(self.n_components):
            spin = self._spins[comp_idx]
            pix_start = self._pix_offsets[comp_idx]
            pix_end = self._pix_offsets[comp_idx + 1]

            N_field = self._N_symmetric[pix_start:pix_end, pix_start:pix_end]
            N_field_inv = self.N_inv[pix_start:pix_end, pix_start:pix_end]
            V_comp = self._V_blocks[comp_idx]

            # Resolve per-mode C_ell diagonals for this component
            cell_diag_0, cell_diag_1 = self._resolve_cell_diagonals(C_ell, comp_idx, spin)

            # Full-field C_ell diagonal
            if spin == 2 and cell_diag_0 is not None:
                cell_sub_full = np.concatenate(
                    [
                        cell_diag_0,
                        cell_diag_1
                        if cell_diag_1 is not None
                        else np.zeros_like(cell_diag_0),
                    ]
                )
            else:
                cell_sub_full = cell_diag_0

            comp_matrix = self._build_compression_matrix_for_subfield(
                V_comp, N_field, N_field_inv, basis, cell_sub_full
            )
            eigenvalues = np.sort(eigvalsh(comp_matrix))[::-1]
            max_ev = np.max(np.abs(eigenvalues))
            normalized = eigenvalues / max_ev if max_ev > 0 else eigenvalues.copy()

            entry: dict = {
                "component": comp_idx,
                "spin": spin,
                "label": f"Field {comp_idx} (spin-{spin})",
                "eigenvalues": eigenvalues,
                "normalized_eigenvalues": normalized,
            }

            if spin == 2:
                n_base = self._n_modes_base
                V_E = V_comp[:n_base, :]
                V_B = V_comp[n_base:, :]

                comp_E = self._build_compression_matrix_for_subfield(
                    V_E, N_field, N_field_inv, basis, cell_diag_0
                )
                ev_E = np.sort(eigvalsh(comp_E))[::-1]
                max_E = np.max(np.abs(ev_E))
                norm_E = ev_E / max_E if max_E > 0 else ev_E.copy()

                comp_B = self._build_compression_matrix_for_subfield(
                    V_B, N_field, N_field_inv, basis, cell_diag_1
                )
                ev_B = np.sort(eigvalsh(comp_B))[::-1]
                max_B = np.max(np.abs(ev_B))
                norm_B = ev_B / max_B if max_B > 0 else ev_B.copy()

                entry["E_eigenvalues"] = ev_E
                entry["E_normalized"] = norm_E
                entry["B_eigenvalues"] = ev_B
                entry["B_normalized"] = norm_B

            results.append(entry)

        return results

    def _resolve_cell_diagonals(
        self,
        C_ell: np.ndarray | dict | None,
        comp_idx: int,
        spin: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Extract per-mode C_ell diagonals for a component.

        Returns
        -------
        cell_diag_0 : numpy.ndarray or None
            Auto-spectrum diagonal (TT for spin-0, EE for spin-2).
        cell_diag_1 : numpy.ndarray or None
            BB diagonal (only for spin-2, None otherwise).
        """
        if C_ell is None:
            return None, None

        if not isinstance(C_ell, dict):
            return self._build_lambda_diagonal(C_ell), None

        # Dict: try 3-tuple key (comp, comp, mode), then 2-tuple
        arr_0 = C_ell.get((comp_idx, comp_idx, 0))
        if arr_0 is None:
            arr_0 = C_ell.get((comp_idx, comp_idx))
        diag_0 = self._build_lambda_diagonal(arr_0) if arr_0 is not None else None

        diag_1 = None
        if spin == 2:
            arr_1 = C_ell.get((comp_idx, comp_idx, 1))
            if arr_1 is not None:
                diag_1 = self._build_lambda_diagonal(arr_1)

        return diag_0, diag_1

    def plot_eigenvalue_spectrum(
        self,
        basis: str = "noise_weighted",
        C_ell: np.ndarray | dict | None = None,
        axes: np.ndarray | None = None,
        log_scale: bool = True,
        show_threshold_lines: bool = True,
        threshold_values: list[float] | None = None,
        show_eb_split: bool = True,
    ) -> tuple[Figure, np.ndarray]:
        """
        Plot eigenvalue spectrum for compression threshold selection.

        Creates one subplot per component.  The y-axis shows eigenvalues
        normalized by the maximum value, so values can be directly used as the
        ``epsilon`` threshold for :meth:`apply_compression`.  For spin-2
        components the E and B sub-spectra are shown as dashed curves when
        ``show_eb_split`` is True.

        Parameters
        ----------
        basis : str, default "noise_weighted"
            Compression basis.
        C_ell : numpy.ndarray, dict, or None
            Power spectrum (required for "total_covariance" and "snr" bases).
        axes : numpy.ndarray of Axes or None
            Pre-created axes array (length ``n_components``).  If None, a new
            figure is created.
        log_scale : bool, default True
            Whether to use logarithmic y-axis.
        show_threshold_lines : bool, default True
            Whether to show reference threshold lines.
        threshold_values : list of float or None
            Custom threshold values to show.  Default: [1e-2, 1e-4, 1e-6, 1e-8].
        show_eb_split : bool, default True
            For spin-2 components, overlay E and B sub-spectra.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the plot.
        axes : numpy.ndarray
            1-D array of Axes (length ``n_components``).

        Examples
        --------
        >>> ppc = PixelBasis(N, N_inv, theta, phi, lmax=100)
        >>> ppc.setup()
        >>> fig, axes = ppc.plot_eigenvalue_spectrum(basis="noise_weighted")
        >>> # From the plot, decide threshold (e.g., 1e-4)
        >>> ppc.apply_compression(epsilon=1e-4, basis="noise_weighted")
        """
        import matplotlib.pyplot as plt

        per_field = self.compute_eigenspectrum_per_field(basis, C_ell)
        n_comp = len(per_field)

        if axes is None:
            fig, axes_raw = plt.subplots(
                1, n_comp, figsize=(6 * n_comp, 6), squeeze=False
            )
            axes_arr: np.ndarray = axes_raw[0]
        else:
            axes_arr = np.atleast_1d(axes)
            fig = axes_arr[0].get_figure()

        if threshold_values is None:
            threshold_values = [1e-2, 1e-4, 1e-6, 1e-8]

        for idx, entry in enumerate(per_field):
            ax = axes_arr[idx]
            normalized = entry["normalized_eigenvalues"]
            mode_indices = np.arange(1, len(normalized) + 1)
            ax.plot(mode_indices, normalized, "b-", linewidth=1.5, label="Eigenvalues")

            # E/B overlay for spin-2
            if show_eb_split and entry["spin"] == 2:
                norm_E = entry["E_normalized"]
                norm_B = entry["B_normalized"]
                ax.plot(
                    np.arange(1, len(norm_E) + 1),
                    norm_E,
                    "r--",
                    linewidth=1.2,
                    label="E modes",
                )
                ax.plot(
                    np.arange(1, len(norm_B) + 1),
                    norm_B,
                    "g--",
                    linewidth=1.2,
                    label="B modes",
                )

            if show_threshold_lines:
                colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(threshold_values)))
                for thresh, color in zip(threshold_values, colors):
                    dim = int(np.sum(normalized > thresh))
                    ax.axhline(
                        y=thresh,
                        color=color,
                        linestyle="--",
                        alpha=0.7,
                        label=f"\u03b5={thresh:.0e} ({dim} modes)",
                    )

            if log_scale:
                ax.set_yscale("log")
            ax.set_xlabel("Mode index", fontsize=12)
            ax.set_ylabel("Normalized eigenvalue", fontsize=12)
            ax.set_title(f"{entry['label']}: {basis}", fontsize=12)
            ax.legend(loc="upper right", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(1, len(normalized))

            n_total = len(normalized)
            n_significant = int(np.sum(normalized > 1e-10))
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
        return fig, axes_arr

    def plot_eigenvalue_comparison(
        self,
        bases: list[str] | None = None,
        C_ell: np.ndarray | dict | None = None,
        axes: np.ndarray | None = None,
        log_scale: bool = True,
    ) -> tuple[Figure, np.ndarray]:
        """
        Compare eigenvalue spectra across different compression bases.

        Creates one subplot per component, overlaying the different bases.

        Parameters
        ----------
        bases : list of str or None
            Compression bases to compare.  Default: all available (or just
            "harmonic"/"noise_weighted" if C_ell is not provided).
        C_ell : numpy.ndarray, dict, or None
            Power spectrum (required for "total_covariance" and "snr" bases).
        axes : numpy.ndarray of Axes or None
            Pre-created axes array (length ``n_components``).  If None, a new
            figure is created.
        log_scale : bool, default True
            Whether to use logarithmic y-axis.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure containing the plot.
        axes : numpy.ndarray
            1-D array of Axes (length ``n_components``).
        """
        import matplotlib.pyplot as plt

        if bases is None:
            if C_ell is not None:
                bases = list(COMPRESSION_BASES.keys())
            else:
                bases = ["harmonic", "noise_weighted"]

        n_comp = self.n_components

        if axes is None:
            fig, axes_raw = plt.subplots(
                1, n_comp, figsize=(6 * n_comp, 6), squeeze=False
            )
            axes_arr: np.ndarray = axes_raw[0]
        else:
            axes_arr = np.atleast_1d(axes)
            fig = axes_arr[0].get_figure()

        basis_colors = plt.cm.tab10(np.linspace(0, 1, len(bases)))

        for basis, color in zip(bases, basis_colors):
            try:
                per_field = self.compute_eigenspectrum_per_field(basis, C_ell)
            except ValueError as e:
                print(f"Skipping basis '{basis}': {e}")
                continue

            for idx, entry in enumerate(per_field):
                ax = axes_arr[idx]
                normalized = entry["normalized_eigenvalues"]
                mode_indices = np.arange(1, len(normalized) + 1)
                ax.plot(
                    mode_indices,
                    normalized,
                    color=color,
                    linewidth=1.5,
                    label=basis,
                )

        for idx in range(n_comp):
            ax = axes_arr[idx]
            spin = self._spins[idx]
            if log_scale:
                ax.set_yscale("log")
            ax.set_xlabel("Mode index", fontsize=12)
            ax.set_ylabel("Normalized eigenvalue", fontsize=12)
            ax.set_title(f"Field {idx} (spin-{spin}): Comparison", fontsize=12)
            ax.legend(loc="upper right", fontsize=10)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, axes_arr

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
            self.dim = total_kept
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
            self.dim = eigenvectors.shape[1]

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
        U = self._eigenvectors  # (n_pix, dim)

        # U^T @ N @ U - compressed noise covariance (independent of C_ell).
        # Goes through _N_symmetric so a post-setup apply_compression call
        # still works after _factorise_noise has overwritten self._N.
        self._U_N_U = U.T @ self._N_symmetric @ U

        # V @ U - used for signal covariance transformation (independent of C_ell)
        # VU has shape (n_modes, dim)
        self._VU = self._V @ U

        # Make U_N_U symmetric (numerical stability) — in place to avoid
        # the dim² broadcast temporaries.
        symmetrize_inplace(self._U_N_U)

        # SMW components for get_weighted_compressed_data
        # V @ N^{-1} via Cholesky solve: (N^{-1} V^T)^T = cholesky_solve(N_chol, V^T)^T
        self._V_N_inv = cholesky_solve(self._N_chol, self._V.T).T
        # V @ N^{-1} @ V^T (n_modes, n_modes) - the M matrix in SMW
        self._V_Ninv_VT = matrix_mult(self._V_N_inv, self._V.T)
        symmetrize_inplace(self._V_Ninv_VT)

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
        # Buffer for VU * diagonal scaling: (n_modes, dim)
        self._VU_scaled_buffer = np.empty(
            (self.n_modes, self.dim), dtype=np.float64, order="C"
        )
        # Buffer for U_S_U computation: (dim, dim)
        self._U_S_U_buffer = np.empty((self.dim, self.dim), dtype=np.float64, order="F")

    def get_projected_inverse(self, C_ell) -> np.ndarray:
        """
        Compute inverse of compressed covariance (U^T @ C @ U)^{-1}.

        Parameters
        ----------
        C_ell : numpy.ndarray or dict
            Power spectrum (array for single-field, dict for multi-field).

        Returns
        -------
        numpy.ndarray
            Inverse of compressed covariance, shape (dim, dim).
        """
        if self._use_direct:
            C = self._build_signal_matrix_direct() + self._N
            return matrix_inverse_symm(np.asfortranarray(C))
        if self._eigenvectors is None:
            raise RuntimeError("Compression not applied. Call apply_compression() first.")
        return self.get_compressed_inverse(C_ell)

    def get_derivative_matrix(
        self,
        ell: int,
        key,
        *,
        symmetry_mode=None,
    ) -> np.ndarray:
        """
        Get derivative matrix ∂C/∂C_ℓ in the compressed basis.

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
            Derivative matrix of shape (dim, dim).
        """
        from ..spectrum_key import SpectrumKind, SymmetryMode, kind_to_legacy_mode

        if symmetry_mode is None:
            symmetry_mode = SymmetryMode.SYMMETRIC

        is_cross = key.comp_i != key.comp_j
        mode = kind_to_legacy_mode(key.kind, is_cross=is_cross)

        if self._use_direct:
            return self._get_derivative_direct(ell, key.comp_i, key.comp_j, mode)

        if self._eigenvectors is None:
            raise RuntimeError("Compression not applied. Call apply_compression() first.")

        # Single spin-0 field: use precomputed diagonal. The precomputed
        # diagonal is the SS derivative; reject other kinds so a bogus key
        # doesn't silently return wrong-kind data.
        if self.n_components == 1 and self._spins[0] == 0:
            if key.kind is not SpectrumKind.SS or key.comp_i != 0 or key.comp_j != 0:
                raise ValueError(
                    f"single spin-0 basis only supports SpectrumKey(0, 0, SS); got {key}"
                )
            E_diag = self._derivative_diagonals[ell]
            np.multiply(self._VU, E_diag[:, np.newaxis], out=self._VU_scaled_buffer)
            return matrix_mult(self._VU_scaled_buffer.T, self._VU)

        E = self._build_derivative_matrix_with_spins(
            ell, key.comp_i, key.comp_j, mode, symmetry_mode=symmetry_mode
        )
        E_VU = matrix_mult(E, self._VU)
        return matrix_mult(self._VU.T, E_VU)

    def get_compressed_covariance(self, C_ell) -> np.ndarray:
        """
        Compute covariance matrix in the compressed space.

        Parameters
        ----------
        C_ell : numpy.ndarray or dict
            Power spectrum (array for single-field, dict for multi-field).

        Returns
        -------
        numpy.ndarray
            Covariance matrix of shape (dim, dim).
        """
        if self._use_direct:
            return self._build_signal_matrix_direct() + self._N

        if self._eigenvectors is None:
            raise RuntimeError("Compression not applied. Call apply_compression() first.")

        if isinstance(C_ell, dict):
            lambda_matrix = self._build_lambda_matrix(C_ell)
            VU_Lambda = matrix_mult(lambda_matrix, self._VU)
            U_S_U = matrix_mult(self._VU.T, VU_Lambda)
            return self._U_N_U + U_S_U

        # Single-field: use buffers for efficiency
        Lambda_diag = self._build_lambda_diagonal(C_ell)
        np.multiply(self._VU, Lambda_diag[:, np.newaxis], out=self._VU_scaled_buffer)
        np.matmul(self._VU.T, self._VU_scaled_buffer, out=self._U_S_U_buffer)
        return self._U_N_U + self._U_S_U_buffer

    def get_weighted_compressed_data(
        self,
        data: np.ndarray,
        C_ell,
        C_c_inv: np.ndarray | None = None,
        stable_inner_inv: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute C^{-1} @ d for QML estimation.

        Parameters
        ----------
        data : numpy.ndarray
            Pixel-space data vector of shape (n_pix,) or (n_pix, n_sims).
        C_ell : numpy.ndarray or dict
            Power spectrum (array for single-field, dict for multi-field).
        C_c_inv : numpy.ndarray, optional
            Precomputed inverse covariance.
        stable_inner_inv : numpy.ndarray, optional
            Unused for pixel basis; accepted for ABC signature parity.

        Returns
        -------
        numpy.ndarray
            Weighted data of shape (dim,) or (dim, n_sims).
        """
        if self._use_direct:
            if C_c_inv is None:
                C_c_inv = self.get_projected_inverse(C_ell)
            return matrix_mult(C_c_inv, data)

        if self._eigenvectors is None:
            raise RuntimeError("Compression not applied. Call apply_compression() first.")

        d_compressed = self._eigenvectors.T @ data
        if C_c_inv is None:
            C_c_inv = self.get_compressed_inverse(C_ell)
        return matrix_mult(C_c_inv, d_compressed)

    def compute_quadratic_form(self, data: np.ndarray, C_ell) -> float:
        """
        Compute d^T C^{-1} d in the compressed space.

        Parameters
        ----------
        data : numpy.ndarray
            Pixel-space data vector of length n_pix.
        C_ell : numpy.ndarray or dict
            Power spectrum (array for single-field, dict for multi-field).

        Returns
        -------
        float
            Quadratic form value.
        """
        if self._use_direct:
            C_inv = self.get_projected_inverse(C_ell)
            return float(data.T @ C_inv @ data)

        if self._eigenvectors is None:
            raise RuntimeError("Compression not applied. Call apply_compression() first.")

        d_compressed = self._eigenvectors.T @ data
        C_compressed_inv = self.get_compressed_inverse(C_ell)
        return float(d_compressed.T @ C_compressed_inv @ d_compressed)

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
            Power spectrum. Single-field: numpy.ndarray. Multi-field: dict
            keyed by SpectrumKey.
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
        if ell_max is None:
            ell_max = self.lmax

        # Single-field path
        if not isinstance(C_ell, dict):
            if spectra_list is not None:
                raise ValueError(
                    "spectra_list should be None for single-field (array) input"
                )

            n_ell = ell_max - ell_min + 1
            fisher = np.zeros((n_ell, n_ell))

            C_c_inv = self.get_compressed_inverse(C_ell)

            from ..spectrum_key import SpectrumKey, SpectrumKind

            # Array C_ell input is the spin-0 single-field signature; spin-2
            # single fields carry multiple spectra (EE/BB/EB) and must enter
            # through the dict path below.
            if self._spins[0] != 0:
                raise ValueError(
                    f"Array C_ell requires spin-0; got spin={self._spins[0]}. "
                    "Pass a dict for spin-2 single-field cases."
                )
            single_field_key = SpectrumKey(0, 0, SpectrumKind.SS, spins=(0,))
            cinv_times_dc = {}
            for ell in range(ell_min, ell_max + 1):
                dC = self.get_derivative_matrix(ell, single_field_key)
                cinv_times_dc[ell] = matrix_mult(C_c_inv, dC)

            for ell_i in range(ell_min, ell_max + 1):
                for ell_j in range(ell_i, ell_max + 1):
                    idx_i = ell_i - ell_min
                    idx_j = ell_j - ell_min

                    fisher_val = 0.5 * matrix_trace(
                        cinv_times_dc[ell_i], cinv_times_dc[ell_j]
                    )
                    fisher[idx_i, idx_j] = fisher_val
                    if idx_i != idx_j:
                        fisher[idx_j, idx_i] = fisher_val

            return fisher

        # Multi-field path
        if spectra_list is None:
            raise ValueError("spectra_list is required for multi-field (dict) input")

        n_ell = ell_max - ell_min + 1
        n_spec = len(spectra_list)
        fisher = np.zeros((n_spec * n_ell, n_spec * n_ell))

        C_c_inv = self.get_compressed_inverse(C_ell)

        cinv_times_dc = {}
        for spec_idx, spec_entry in enumerate(spectra_list):
            for ell in range(ell_min, ell_max + 1):
                dC = self.get_derivative_matrix(
                    ell, spec_entry, symmetry_mode=symmetry_mode
                )
                cinv_times_dc[(spec_idx, ell)] = matrix_mult(C_c_inv, dC)

        for spec_a in range(n_spec):
            for ell_a in range(ell_min, ell_max + 1):
                idx_a = spec_a * n_ell + (ell_a - ell_min)

                for spec_b in range(spec_a, n_spec):
                    ell_b_start = ell_a if spec_a == spec_b else ell_min
                    for ell_b in range(ell_b_start, ell_max + 1):
                        idx_b = spec_b * n_ell + (ell_b - ell_min)

                        fisher_val = 0.5 * matrix_trace(
                            cinv_times_dc[(spec_a, ell_a)],
                            cinv_times_dc[(spec_b, ell_b)],
                        )

                        fisher[idx_a, idx_b] = fisher_val
                        if idx_a != idx_b:
                            fisher[idx_b, idx_a] = fisher_val

        return fisher

    def prepare_smw(self, C_ell_dict: dict) -> SMWPrepared:
        """Precompute compressed inverse and logdet for reuse across sims."""
        if not self._use_direct and self._eigenvectors is None:
            raise RuntimeError("Compression not applied. Call apply_compression() first.")

        from ..basics import matrix_slogdet_symm

        C_c = self.get_compressed_covariance(C_ell_dict)
        C_c_inv = matrix_inverse_symm(C_c)
        _, logdet = matrix_slogdet_symm(C_c)
        return SMWPrepared(C_c_inv, logdet)

    def quadratic_form_from_prepared(
        self, data: np.ndarray, C_c_inv: np.ndarray
    ) -> float:
        """Compute d^T C^{-1} d using precomputed compressed inverse."""
        if self._use_direct:
            return float(data.T @ C_c_inv @ data)
        if self._eigenvectors is None:
            raise RuntimeError("Compression not applied. Call apply_compression() first.")

        d_c = self._eigenvectors.T @ data
        return float(d_c.T @ C_c_inv @ d_c)

    def get_logdet(self, C_ell) -> float:
        """
        Compute log determinant of compressed covariance.

        Parameters
        ----------
        C_ell : numpy.ndarray or dict
            Power spectrum (array for single-field, dict for multi-field).
        """
        if isinstance(C_ell, dict):
            _, logdet = self.prepare_smw(C_ell)
            return logdet
        from ..basics import matrix_slogdet_symm

        C_c = self.get_compressed_covariance(C_ell)
        _, logdet = matrix_slogdet_symm(np.asfortranarray(C_c))
        return logdet

    def get_full_inverse(self, C_ell) -> np.ndarray:
        """Reconstruct full ``n_pix x n_pix`` inverse from the basis-space form.

        Exact in pixel-direct mode (``U`` is the identity). Lossy on a
        truncated compressed pixel basis — the result lives in the kept
        subspace and is zero on the truncated complement.
        """
        C_basis_inv = self.get_compressed_inverse(C_ell)
        if self._use_direct or self._eigenvectors is None:
            return C_basis_inv
        U = self._eigenvectors
        return U @ C_basis_inv @ U.T

    @property
    def compression_ratio(self) -> float:
        """
        Ratio of kept modes to original pixels.

        Returns
        -------
        float
            Compression ratio (1.0 means no compression).
        """
        return self.dim / self.n_pix

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
        >>> PixelBasis.available_bases()
        {'harmonic': 'P_h = V^T V (pure harmonic projector)',
         'noise_weighted': 'P_h N^{-1} P_h (inverse noise weighting)',
         'total_covariance': 'P_h C^{-1} P_h (full covariance weighting, requires C_ell)',
         'snr': 'S^{1/2} N^{-1} S^{1/2} (signal-to-noise ratio, requires C_ell)'}
        """
        return COMPRESSION_BASES.copy()
