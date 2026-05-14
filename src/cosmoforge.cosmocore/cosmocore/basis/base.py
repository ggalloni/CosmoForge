"""
Base class for computation basis methods.

This module provides the abstract base class that defines the interface
for all computation basis methods used in CMB Fisher matrix computation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from typing import NamedTuple

import numpy as np

# Regularization thresholds for numerical stability
_LAMBDA_ZERO_THRESHOLD = 1e-30  # Below this, Lambda eigenvalue treated as zero
_LAMBDA_INV_CAP = 1e30  # Cap for 1/Lambda when Lambda ≈ 0
_MATRIX_REGULARIZATION = 1e-20  # Tikhonov regularization for matrix inversion


class BasisPrepared(NamedTuple):
    """Per-``C_ell`` quantities pre-computed for fast likelihood evaluation.

    Returned by :meth:`ComputationBasis.prepare_for_basis`. The stored
    ``factor`` is basis-specific in form but always usable as the
    second argument of
    :meth:`ComputationBasis.quadratic_form_from_prepared`.

    Parameters
    ----------
    factor : numpy.ndarray
        K Cholesky factor (harmonic) or basis-space inverse (pixel).
    logdet : float
        Log determinant. Semantics are basis-specific and intentionally
        match what ``quadratic_form_from_prepared`` consumes:

        - HarmonicBasis: exact full pixel-space ``log|N + S|`` via SMW.
        - PixelBasis in pixel-direct mode: exact full ``log|N + S|``
          (``U`` is the identity, so basis-space equals full-space).
        - PixelBasis on a truncated compressed basis: basis-space
          ``log|U^T (N + S) U|`` — the logdet of the restricted
          operator, NOT the full ``log|N + S|``. Combine with
          :meth:`quadratic_form_from_prepared` (also basis-space) for
          an internally-consistent basis-space likelihood.
    """

    factor: np.ndarray
    logdet: float


from ..basics import (
    cholesky_decomposition,
    cholesky_factor,
    cholesky_solve,
    matrix_inverse_symm,
    matrix_mult,
    matrix_slogdet_symm,
)
from .harmonic_basis import HarmonicBasisBuilder


class ComputationBasis(ABC):
    """
    Abstract base class for computation basis methods.

    This class defines the interface for computation basis methods and provides
    shared implementations for spherical harmonic evaluation and ell-to-mode
    mapping.

    Supports both single-field and multi-field configurations. For multi-field,
    theta and phi are passed as tuples of arrays (one per component), and the
    harmonic operator V is built as a block-diagonal matrix.

    Parameters
    ----------
    N : numpy.ndarray
        Noise covariance matrix of shape (n_pix_total, n_pix_total). The
        basis takes ownership of this buffer; the in-place Cholesky factor
        overwrites it during setup.
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
    dim : int
        Number of modes kept after compression (if applicable).
    n_components : int
        Number of field components (1 for single-field, N for multi-field).
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
    ):
        """
        Initialize computation basis.

        Parameters
        ----------
        N : numpy.ndarray
            Noise covariance matrix (n_pix_total, n_pix_total). The basis
            takes ownership; ``_factorise_noise()`` overwrites this buffer
            in place during setup.
        theta : numpy.ndarray or tuple of numpy.ndarray
            Colatitude angles for active pixels in radians. Single array for
            single-field, tuple of arrays for multi-field (one per component).
        phi : numpy.ndarray or tuple of numpy.ndarray
            Longitude angles for active pixels in radians. Single array for
            single-field, tuple of arrays for multi-field (one per component).
        lmax_signal : int
            Signal-cov ceiling. The basis represents multipoles up to and
            including this value.
        beam : numpy.ndarray or None, optional
            Beam window function B_ℓ for ℓ=0..lmax_signal.
        spins : list of int or None, optional
            Spin weight for each component (0 for scalar, 2 for polarization).
            Default is [0, ...] for all components. For spin-2 components,
            theta/phi represent physical pixel locations but V is built for
            (Q, U) → (E, B) transformation with doubled dimensions.
        lmin_signal : list of int or None, optional
            Per-component signal-cov floor. Defaults to ``[2]*n_components``.
            Each entry must satisfy ``lmin_signal[i] >= |spins[i]|``.
        lmin : int or None, optional
            Inference window lower bound. Defaults to ``min(lmin_signal)``.
            Together with ``lmax`` enables the switch optimisation:
            ``S_fixed`` absorbs the signal contribution outside
            ``[lmin, lmax]``.
        lmax : int or None, optional
            Inference window upper bound. Defaults to ``lmax_signal``.
            Multipoles in ``(lmax, lmax_signal]`` are absorbed into ``S_fixed``.
        fiducial_C_ell : numpy.ndarray or None, optional
            Deprecated: Use S_fixed instead. Fiducial power spectrum for
            fixed multipoles (ℓ outside ``[lmin, lmax]``).
        S_fixed : numpy.ndarray or None, optional
            Precomputed signal matrix for fixed multipoles outside
            ``[lmin, lmax]``. Recommended way to pass the fixed signal
            contribution. Shape ``(n_pix_total, n_pix_total)``.
        """
        self._N = np.asfortranarray(N, dtype=np.float64)
        # Pre-factor cache slot for the lazy N_inv property (populated on
        # first read of self.N_inv before _factorise_noise runs, then
        # released by _factorise_noise). The basis owns the noise buffer
        # end-to-end; the constructor no longer accepts a precomputed inverse.
        self._N_inv_dense = None
        self.lmax_signal = lmax_signal

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

        # Multipole-range bookkeeping (see ADR 0009). lmin_signal is
        # per-component; lmin/lmax are scalar inference window edges.
        # Spin-floor validation lives at the user-facing layer (Fisher /
        # PICSLike __init__); a defensive check still runs here.
        if lmin_signal is None:
            self._lmin_signal = [2] * self.n_components
        else:
            if len(lmin_signal) != self.n_components:
                raise ValueError(
                    f"lmin_signal length ({len(lmin_signal)}) must match "
                    f"number of components ({self.n_components})"
                )
            for i, (lm, s) in enumerate(zip(lmin_signal, self._spins)):
                if lm < abs(s):
                    raise ValueError(
                        f"lmin_signal[{i}]={lm} < |spin|={abs(s)} "
                        f"(component {i} is spin-{s})"
                    )
            self._lmin_signal = list(lmin_signal)
        self.lmin = lmin if lmin is not None else min(self._lmin_signal)
        self.lmax = lmax if lmax is not None else lmax_signal
        self._fiducial_C_ell = fiducial_C_ell
        self._S_fixed = S_fixed
        # Switch optimization triggers when the inference window is strictly
        # narrower than the signal-cov band on either side.
        self._use_switch_optimization = (
            self.lmin > min(self._lmin_signal) or self.lmax < lmax_signal
        )

        # Store beam window function (ℓ-indexed: beam[ell] for ell=0..lmax_signal).
        if beam is not None:
            beam = np.asarray(beam, dtype=np.float64)
            expected_len = lmax_signal + 1
            if beam.shape[0] != expected_len:
                raise ValueError(
                    f"Beam must have length {expected_len} "
                    f"(ell=0..lmax_signal={lmax_signal}), got {beam.shape[0]}"
                )
            self._beam = beam
        else:
            self._beam = None

        # Derived quantities
        self.n_pix = sum(self._n_pix_per_component)

    def _init_harmonic_internals(self) -> None:
        """Initialize harmonic mode counts, offsets, and the HarmonicBasisBuilder.

        Subclasses that need harmonic machinery (V operator, Lambda, derivatives)
        must call this after ``super().__init__()``.
        """
        # Mode count per component over the inference window, applying the
        # per-component lmin_signal floor. The V-fill loops sweep
        # [_lmin_smw.._lmax_smw] and zero rows below lmin_signal[i] for any
        # component i with lmin_signal[i] > _lmin_smw.
        self._lmin_smw = self.lmin
        self._lmax_smw = self.lmax
        self._n_modes_per_component_list = [
            (
                2 * ((self._lmax_smw + 1) ** 2 - max(self._lmin_smw, lm) ** 2)
                if self._spins[i] == 2
                else (self._lmax_smw + 1) ** 2 - max(self._lmin_smw, lm) ** 2
            )
            for i, lm in enumerate(self._lmin_signal)
        ]
        # Single-field convenience: a scalar base mode count drives the legacy
        # spin-0 paths (Lambda diagonals, single-component derivative slabs).
        self._n_modes_base = (self._lmax_smw + 1) ** 2 - max(
            self._lmin_smw, self._lmin_signal[0]
        ) ** 2

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
        self._harmonic_basis = HarmonicBasisBuilder(self)

    @abstractmethod
    def setup(self) -> None:
        """
        Initialize basis-specific components.

        Must be called after initialization before using any other methods.
        """
        pass

    @cached_property
    def _L(self) -> np.ndarray:
        # Pre-_factorise_noise fallback. Load-bearing: PixelBasis
        # _setup_v_based reads cholesky_solve(self._N_chol, V.T)
        # to build _V_N_inv during apply_compression, which runs *before*
        # _factorise_noise in _setup_v_based. Without this fallback, that
        # access would AttributeError. _factorise_noise then writes the
        # in-place factor into __dict__["_L"], which shadows this descriptor
        # and orphans the separate buffer allocated here (~1× cov, GC'd at
        # the next allocation). Net cost: one transient cov-sized buffer
        # during apply_compression. Removing it requires reordering setup
        # so the in-place factor exists before any consumer reads _N_chol.
        return cholesky_decomposition(self._N)

    @property
    def _N_chol(self) -> tuple:
        return (self._L, True)

    @property
    def _N_symmetric(self) -> np.ndarray:
        # If self._N still exists, return it directly. Once _factorise_noise
        # has overwritten and deleted it, reconstruct L @ L.T on demand
        # (transient n_pix² allocation; only deferred consumers hit this).
        N = getattr(self, "_N", None)
        if N is not None:
            return N
        return self._L @ self._L.T

    @property
    def N_inv(self) -> np.ndarray | None:
        # Caller-provided / pre-factor cached dense N_inv if available.
        stored = getattr(self, "_N_inv_dense", None)
        if stored is not None:
            return stored
        # Pre-factor: materialise from symmetric N once and cache, so
        # apply_compression's repeated reads share one allocation. The
        # cache is freed in _factorise_noise.
        N = getattr(self, "_N", None)
        if N is not None:
            self._N_inv_dense = matrix_inverse_symm(N)
            return self._N_inv_dense
        # Post-factor: lazy materialise from L, do not cache (would re-add
        # the 1× cov footprint we just eliminated). Reachable from the
        # deferred compression-basis paths in pixel.py (compute_compression_matrix
        # for "noise_weighted"/"snr"/per-field), and from the test-only
        # HarmonicBasis.get_inverse via smw_inverse. Never on the QML hot
        # path after the refactor.
        L = self.__dict__.get("_L")
        if L is None:
            return None
        return cholesky_solve((L, True), np.eye(L.shape[0]))

    def _factorise_noise(self) -> None:
        """In-place Cholesky factor of self._N.

        Replaces self._N's buffer contents with the lower-triangular factor
        L (upper triangle becomes garbage; no consumer reads it because all
        access goes through cholesky_solve or the _N_symmetric reconstruction).
        After this method, ``self._L`` is the in-place factor, ``self._N_chol``
        is the (L, True) tuple, ``self._log_det_N`` is set, and the original
        symmetric-N attribute ``self._N`` is removed so any stale read
        raises AttributeError. Calling twice is a programming error.
        """
        if not hasattr(self, "_N"):
            raise RuntimeError(
                "_factorise_noise called twice (or after self._N was removed). "
                "The basis is meant to factorise once during setup."
            )

        # Scan off-diagonal noise blocks for cross-field coupling while N is
        # still symmetric. The result is C_ell-independent and consumed by
        # _detect_field_blocks; computing it here avoids a later
        # _N_symmetric reconstruction (n_pix² transient buffer).
        self._noise_block_adjacency = self._compute_noise_block_adjacency()

        L_buf = self._N
        L, lower = cholesky_factor(L_buf, overwrite_a=True)
        # Bind into __dict__ to shadow the cached_property fallback. If the
        # fallback ran earlier (e.g., during apply_compression), its separate
        # buffer is orphaned here and GC'd at the next allocation.
        self.__dict__["_L"] = L
        self._log_det_N = 2.0 * float(np.sum(np.log(np.diag(L))))
        # Free the caller-provided dense inverse if any: the factor replaces it.
        self._N_inv_dense = None
        del self._N

    def _compute_noise_block_adjacency(self) -> set[tuple[int, int]]:
        """Return ``{(ci, cj)}`` pairs (ci<cj) with non-zero off-diagonal noise.

        Scans ``self._N`` once while symmetric. Empty for single-field bases.
        Uses a relative threshold scaled by the diagonal magnitude so that
        float32-loaded noise covariances do not flip the decision on
        spuriously tiny pivot fill.
        """
        pairs: set[tuple[int, int]] = set()
        if self.n_components <= 1:
            return pairs
        N_sym = self._N_symmetric
        diag_max = float(np.max(np.abs(np.diag(N_sym)))) if N_sym.size else 0.0
        # Floor the absolute threshold for the (unphysical) all-zero diagonal
        # case; otherwise scale by the dominant noise magnitude.
        rel_tol = 1e-12
        threshold = max(diag_max * rel_tol, _LAMBDA_ZERO_THRESHOLD)
        for ci in range(self.n_components):
            for cj in range(ci + 1, self.n_components):
                ri = self._pix_offsets[ci]
                re = self._pix_offsets[ci + 1]
                cj_start = self._pix_offsets[cj]
                cj_end = self._pix_offsets[cj + 1]
                N_block = N_sym[ri:re, cj_start:cj_end]
                if np.any(np.abs(N_block) > threshold):
                    pairs.add((ci, cj))
        return pairs

    def _build_basis(self) -> None:
        """Build harmonic operator V, mode mappings, and derivative diagonals."""
        self._harmonic_basis.build()

    # Properties delegating to _harmonic_basis (avoids copy-back)

    @property
    def _V(self):
        return self._harmonic_basis._V

    @property
    def _V_blocks(self):
        return self._harmonic_basis._V_blocks

    @property
    def _ell_to_modes(self):
        return self._harmonic_basis._ell_to_modes

    @property
    def _ell_to_modes_local(self):
        return self._harmonic_basis._ell_to_modes_local

    @property
    def _m_to_modes(self):
        return self._harmonic_basis._m_to_modes

    @property
    def _m_to_modes_local(self):
        return self._harmonic_basis._m_to_modes_local

    @property
    def _derivative_diagonals(self):
        return self._harmonic_basis._derivative_diagonals

    @property
    def _derivative_diagonals_local(self):
        return self._harmonic_basis._derivative_diagonals_local

    @property
    @abstractmethod
    def method(self) -> str:
        """Computation basis name: "harmonic" or "pixel"."""
        pass

    @property
    @abstractmethod
    def projector(self) -> np.ndarray:
        """
        Get the projection matrix that maps pixel space to the basis.

        The fundamental change-of-basis operator:
        - HarmonicBasis: V (n_modes × n_pix)
        - PixelBasis: U^T (dim × n_pix) when compression is active;
          identity-equivalent in pixel-direct mode.

        Returns
        -------
        numpy.ndarray
            Projection matrix of shape (dim, n_pix).
        """
        pass

    @abstractmethod
    def get_projected_inverse(self, C_ell) -> np.ndarray:
        """
        Get the projected inverse for Fisher computation.

        Parameters
        ----------
        C_ell : numpy.ndarray or dict
            Power spectrum (array for single-field, dict for multi-field).

        Returns
        -------
        numpy.ndarray
            Projected inverse covariance matrix.
        """
        pass

    @abstractmethod
    def get_derivative_matrix(
        self,
        ell: int,
        key,
        *,
        symmetry_mode=None,
    ) -> np.ndarray:
        """
        Get the derivative matrix ∂C/∂C_ℓ in the basis space.

        Parameters
        ----------
        ell : int
            Multipole for which to compute the derivative.
        key : SpectrumKey
            Identifies the spectrum (component pair + spin kind).
        symmetry_mode : SymmetryMode or None
            ``None`` keeps the SYMMETRIC default.

        Returns
        -------
        numpy.ndarray
            Derivative matrix of shape (dim, dim).
        """
        pass

    @abstractmethod
    def get_covariance(self, C_ell) -> np.ndarray:
        """
        Compute the covariance matrix in the basis space.

        - HarmonicBasis: C̄ = V @ N @ V^T + Λ
        - PixelBasis: C_c = U^T @ C @ U

        Parameters
        ----------
        C_ell : numpy.ndarray or dict
            Power spectrum (array for single-field, dict for multi-field).

        Returns
        -------
        numpy.ndarray
            Basis-space covariance of shape (dim, dim).
        """
        pass

    @abstractmethod
    def get_weighted_data(
        self,
        data: np.ndarray,
        C_ell,
        C_c_inv: np.ndarray | None = None,
        stable_inner_inv: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute weighted basis-space data for QML estimation.

        Parameters
        ----------
        data : numpy.ndarray
            Pixel-space data vector of shape (n_pix,) or (n_pix, n_sims).
        C_ell : numpy.ndarray or dict
            Power spectrum (array for single-field, dict for multi-field).
        C_c_inv : numpy.ndarray, optional
            Precomputed basis-space inverse. Used by PixelBasis only;
            ignored by HarmonicBasis.
        stable_inner_inv : numpy.ndarray, optional
            Precomputed ``(I + Lambda M)^{-1}``, as returned by
            :meth:`HarmonicBasis.prepare_stable_inner_inv`. Used by
            HarmonicBasis only; ignored by PixelBasis.

        Returns
        -------
        numpy.ndarray
            Weighted basis-space data of shape (dim,) or (dim, n_sims).
        """
        pass

    @abstractmethod
    def get_noise_for_bias(self) -> np.ndarray:
        """Basis-projected noise consumed by the QML noise-bias sandwich.

        Spectra computes ``Cov(w | noise) = X get_noise_for_bias() X^T``
        where ``X`` is the basis's natural inverse-equivalent
        (``A = (I + Lambda M)^{-T}`` for the harmonic basis;
        ``C_c^{-1}`` for the pixel basis). The *form* of the returned
        matrix is therefore basis-specific:

        - HarmonicBasis: ``V N_eff^{-1} N N_eff^{-1} V^T`` (the SMW
          intermediate ``T`` from ``N_eff = N + S_fixed`` when the
          switch optimisation is active; otherwise
          ``V N^{-1} N N^{-1} V^T``).
        - PixelBasis: ``U^T N U`` (raw noise projected once into the
          eigenmode basis).

        The two are not interchangeable. Callers must compose with the
        basis's own inverse-equivalent.

        Returns
        -------
        numpy.ndarray
            Symmetric (dim, dim) matrix.
        """
        pass

    # === Field block-diagonal detection ===

    def _detect_field_blocks(
        self,
        C_ell_dict: dict,
    ) -> list[list[int]]:
        """Detect independent field groups from signal spectra and noise structure.

        When cross-spectra are absent between field groups and noise is
        independent per field group, K is exactly block-diagonal across
        field groups.  Inverting each block independently is exact and faster.

        The coupling is determined by the *signal* (Lambda from C_ell_dict)
        and noise structure, not by which derivatives are requested.

        Parameters
        ----------
        C_ell_dict : dict
            Power spectrum dictionary with 2-tuple ``(comp_i, comp_j)`` keys,
            3-tuple ``(comp_i, comp_j, mode)`` keys, or :class:`SpectrumKey`
            instances.  Any cross-component entry couples those components.

        Returns
        -------
        list of list of int
            Groups of component indices that are coupled.  E.g.
            ``[[0], [1]]`` for two independent fields,
            ``[[0, 1]]`` for two coupled fields.
        """
        from ..spectrum_key import SpectrumKey

        if self.n_components <= 1:
            return [[0]] if self.n_components == 1 else []

        adj: dict[int, set[int]] = {i: set() for i in range(self.n_components)}
        for key in C_ell_dict:
            if isinstance(key, SpectrumKey):
                ci, cj = key.comp_i, key.comp_j
            else:
                ci, cj = key[0], key[1]
            if ci != cj:
                adj[ci].add(cj)
                adj[cj].add(ci)

        # Merge noise off-diagonal coupling (precomputed by _factorise_noise
        # while N was symmetric; falls back to an on-demand scan if absent).
        noise_pairs = getattr(self, "_noise_block_adjacency", None)
        if noise_pairs is None:
            noise_pairs = self._compute_noise_block_adjacency()
        for ci, cj in noise_pairs:
            adj[ci].add(cj)
            adj[cj].add(ci)

        visited: set[int] = set()
        groups: list[list[int]] = []
        for start in range(self.n_components):
            if start in visited:
                continue
            group: list[int] = []
            queue = [start]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                group.append(node)
                for neighbor in sorted(adj[node]):
                    if neighbor not in visited:
                        queue.append(neighbor)
            groups.append(sorted(group))

        return groups

    # === Shared implementations ===

    def get_inverse(self, C_ell) -> np.ndarray:
        """
        Compute the inverse of the basis-space covariance.

        Parameters
        ----------
        C_ell : numpy.ndarray or dict
            Power spectrum (array for single-field, dict for multi-field).

        Returns
        -------
        numpy.ndarray
            Basis-space inverse covariance of shape (dim, dim).
        """
        from ..basics import matrix_inverse_symm

        C_basis = self.get_covariance(C_ell)
        return matrix_inverse_symm(C_basis, overwrite=True)

    def get_logdet(self, C_ell) -> float:
        """
        Compute the log determinant of the basis-space covariance.

        Parameters
        ----------
        C_ell : numpy.ndarray
            Power spectrum values for ell = 2 to lmax.

        Returns
        -------
        float
            Log determinant of basis-space covariance.
        """
        C_basis = self.get_covariance(C_ell)
        _, logdet = matrix_slogdet_symm(np.asfortranarray(C_basis))
        return logdet

    def get_full_logdet(self, C_ell) -> float:
        """
        Return ``log|N + S|``, the full pixel-space log determinant.

        HarmonicBasis overrides this to compute it exactly via the SMW
        formula. The default below applies to PixelBasis and is exact
        only in pixel-direct mode (``U`` is the identity, so the
        basis-space logdet equals the full logdet).

        On a *truncated compressed* pixel basis the full logdet cannot
        be recovered from the kept quantities: the discarded complement
        of ``N + S`` is needed and is not stored. The default falls
        through to :meth:`get_logdet`, which returns the logdet of the
        restricted operator ``U^T (N + S) U`` — a different matrix, not
        an approximation to ``log|N + S|``. Callers that need the exact
        full logdet on a truncated basis must arrange for it elsewhere.

        Parameters
        ----------
        C_ell : numpy.ndarray or dict
            Power spectrum (array for single-field, dict for multi-field).
        """
        return self.get_logdet(C_ell)

    @abstractmethod
    def get_full_inverse(self, C_ell) -> np.ndarray:
        """Return the full ``n_pix x n_pix`` inverse covariance.

        For HarmonicBasis: exact ``(N + S)^{-1}`` via the SMW formula.
        For PixelBasis in pixel-direct mode: also exact (``U`` is the
        identity, so ``U @ get_inverse(C_ell) @ U.T`` equals the full
        inverse).

        For PixelBasis on a *truncated compressed* basis: the returned
        matrix is ``U @ get_inverse(C_ell) @ U.T``, which lives in the
        kept subspace and is zero on the truncated complement. It is
        *not* the inverse of the full pixel-space covariance; it is the
        inverse of the restricted operator, lifted back to ``n_pix``
        dimensions. Use with care.

        Callers who only need the basis-space inverse should use
        :meth:`get_inverse` instead — cheaper and equivalent in
        basis-space algebra.
        """
        pass

    @abstractmethod
    def quadratic_form(self, data: np.ndarray, C_ell) -> float:
        """Compute ``d^T C^{-1} d`` for full pixel-space data.

        The basis chooses its fastest path. Returns a single ``float``
        for a 1-D ``data`` vector, or a length-``n_sims`` array if
        ``data`` is shape ``(n_pix, n_sims)``.

        Parameters
        ----------
        data : numpy.ndarray
            Pixel-space data, shape ``(n_pix,)`` or ``(n_pix, n_sims)``.
        C_ell : numpy.ndarray or dict
            Power spectrum (array for single-field, dict for multi-field).
        """
        pass

    @abstractmethod
    def quadratic_form_from_prepared(self, data: np.ndarray, factor: np.ndarray) -> float:
        """Compute ``d^T C^{-1} d`` using a pre-baked ``factor`` from
        :meth:`prepare_for_basis`.

        Fast path for evaluating the quadratic form many times against
        the same ``C_ell``: caller calls :meth:`prepare_for_basis` once
        and reuses the resulting ``factor`` across simulations.

        Parameters
        ----------
        data : numpy.ndarray
            Pixel-space data, shape ``(n_pix,)`` or ``(n_pix, n_sims)``.
        factor : numpy.ndarray
            Basis-specific pre-baked quantity:
            ``BasisPrepared.factor`` from :meth:`prepare_for_basis`.
        """
        pass

    @abstractmethod
    def prepare_for_basis(self, C_ell_dict: dict) -> BasisPrepared:
        """Pre-bake per-``C_ell`` quantities for fast likelihood reuse.

        Returns a :class:`BasisPrepared` carrying the basis-specific
        ``factor`` and the full log determinant. The ``factor`` is
        consumable by :meth:`quadratic_form_from_prepared` on the same
        basis instance.

        Parameters
        ----------
        C_ell_dict : dict
            Multi-field power spectrum dict keyed by :class:`SpectrumKey`.
        """
        pass

    # =========================================================================
    # Delegates to HarmonicBasis
    # =========================================================================
    # These methods forward to self._harmonic_basis so that subclasses and
    # external code (spectra.py) can keep using self._build_lambda_matrix() etc.

    def _build_lambda_diagonal(self, C_ell: np.ndarray) -> np.ndarray:
        """Build Lambda diagonal from C_ell. Delegates to HarmonicBasis."""
        return self._harmonic_basis._build_lambda_diagonal(C_ell)

    def _build_lambda_blocks(self, C_ell_dict: dict) -> dict[tuple[int, int], np.ndarray]:
        """Build Lambda blocks. Accepts 2-tuple or :class:`SpectrumKey` keys."""
        return self._harmonic_basis._build_lambda_blocks(C_ell_dict)

    def _build_lambda_matrix(self, C_ell_dict: dict) -> np.ndarray:
        """Build full Lambda matrix.

        Auto-detects key format: 2-tuple, 3-tuple, or :class:`SpectrumKey`.
        """
        return self._harmonic_basis._build_lambda_matrix(C_ell_dict)

    def _build_lambda_matrix_2tuple(
        self, C_ell_dict: dict[tuple[int, int], np.ndarray]
    ) -> np.ndarray:
        """Build full Lambda from 2-tuple keys. Delegates to HarmonicBasis."""
        return self._harmonic_basis._build_lambda_matrix_2tuple(C_ell_dict)

    def _build_lambda_matrix_3tuple(
        self, C_ell_dict: dict[tuple, np.ndarray]
    ) -> np.ndarray:
        """Build full Lambda from 3-tuple keys. Delegates to HarmonicBasis."""
        return self._harmonic_basis._build_lambda_matrix_3tuple(C_ell_dict)

    def _build_lambda_matrix_keyed(self, C_ell_dict: dict) -> np.ndarray:
        """Build full Lambda from SpectrumKey keys. Delegates to HarmonicBasis."""
        return self._harmonic_basis._build_lambda_matrix_keyed(C_ell_dict)

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
        self,
        ell: int,
        comp_i: int,
        comp_j: int,
        mode: int = 0,
        *,
        symmetry_mode=None,
    ) -> np.ndarray:
        """Build derivative matrix for (comp_i, comp_j, mode)."""
        return self._harmonic_basis._build_derivative_matrix_with_spins(
            ell, comp_i, comp_j, mode, symmetry_mode=symmetry_mode
        )

    def _precompute_derivative_diagonals(self) -> None:
        """Precompute derivative diagonals. Delegates to HarmonicBasis."""
        self._harmonic_basis._precompute_derivative_diagonals()

    def release_pixel_projector(self) -> None:
        """No-op default. HarmonicBasis overrides to drop V post-SMW build."""
        return None

    def to_basis(self, data: np.ndarray) -> np.ndarray:
        """
        Project pixel data into the basis: ``d_basis = P @ d``.

        Uses the basis-specific projector P:
        - HarmonicBasis: P = V (n_modes × n_pix)
        - PixelBasis: P = U^T (dim × n_pix) when compression is active;
          identity short-circuit in pixel-direct mode.

        Parameters
        ----------
        data : numpy.ndarray
            Pixel-space data vector of shape (n_pix,) or (n_pix, n_sims).

        Returns
        -------
        numpy.ndarray
            Basis-space data of shape (dim,) or (dim, n_sims).
        """
        return matrix_mult(self.projector, data)

    def get_cls_vector(self) -> np.ndarray:
        """
        Get a placeholder C_ell vector for the configured lmax_signal.

        Returns
        -------
        numpy.ndarray
            ℓ-indexed array of zeros with length ``lmax_signal + 1``.
        """
        return np.zeros(self.lmax_signal + 1)

    @property
    def compression_ratio(self) -> float:
        """
        Ratio of kept modes to original modes.

        Returns
        -------
        float
            Compression ratio (1.0 means no compression).
        """
        return self.dim / self.n_modes_total
