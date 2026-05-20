"""Spectra IO and layout: SpectraManager + flat-vector conversions.

Moved from ``cosmocore.harmonic`` (2026-05). See ADR-0002 follow-up
(architecture backlog #5) for the rename rationale.

The main class is:
    - SpectraManager: Manages power spectra for collections of cosmological fields

Key functions:
    - cl_to_vec, vec_to_cl: Convert between C_ℓ matrices and vectorized forms

References
----------
Power Spectrum Conventions:
.. [1] Tegmark, M. "How to measure CMB power spectra without losing information"
   Phys. Rev. D 55, 5895 (1997)
.. [2] Hivon, E. et al. "MASTER of the Cosmic Microwave Background Anisotropy
   Power Spectrum: A Fast Method for Statistical Analysis of Large and Complex
   Cosmic Microwave Background Data Sets" Astrophys. J. 567, 2-17 (2002)

Spherical Harmonic Transforms:
.. [3] Reinecke, M. & Seljak, U. "Libsharp - spherical harmonic transforms revisited"
   Astron. Astrophys. 554, A112 (2013)
.. [4] Gorski, K.M. et al. "HEALPix: A Framework for High-Resolution Discretization
   and Fast Analysis of Data Distributed on the Sphere"
   Astrophys. J. 622, 759-771 (2005)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit

if TYPE_CHECKING:
    from cosmocore import BaseField
    from cosmocore.beam import BeamManager


@njit(cache=True)
def cl_to_vec(cl, vec):
    """
    Reshape a (n_bins, n_spec) matrix into a flat spectrum-major vector.

    Parameters
    ----------
    cl : numpy.ndarray, shape (n_bins, n_spec)
        Input matrix. Caller owns the bin-axis semantics; this function
        does not interpret the rows as multipoles or impose a low-ℓ floor.
    vec : numpy.ndarray, shape (n_bins * n_spec,)
        Output vector. Must be pre-allocated. Layout is spectrum-major:
        ``vec[ispec * n_bins + k] = cl[k, ispec]``.

    Examples
    --------
    >>> import numpy as np
    >>> cl = np.random.random((10, 3))
    >>> vec = np.zeros(30)
    >>> cl_to_vec(cl, vec)
    """
    n_bins = cl.shape[0]
    n_spec = cl.shape[1]
    counter = 0
    for ispec in range(n_spec):
        for k in range(n_bins):
            vec[counter] = cl[k, ispec]
            counter += 1


@njit(cache=True)
def vec_to_cl(vec, cl):
    """
    Inverse of :func:`cl_to_vec`: scatter a flat spectrum-major vector into
    a (n_bins, n_spec) matrix.

    Parameters
    ----------
    vec : numpy.ndarray, shape (n_bins * n_spec,)
        Input vector with spectrum-major layout.
    cl : numpy.ndarray, shape (n_bins, n_spec)
        Output matrix. Must be pre-allocated.

    Examples
    --------
    >>> import numpy as np
    >>> vec = np.random.random(30)
    >>> cl = np.zeros((10, 3))
    >>> vec_to_cl(vec, cl)
    """
    n_bins = cl.shape[0]
    n_spec = cl.shape[1]
    counter = 0
    for ispec in range(n_spec):
        for k in range(n_bins):
            cl[k, ispec] = vec[counter]
            counter += 1


class SpectraManager:
    """
    Manages power spectra for a collection of cosmological fields.

    This class provides a comprehensive interface for handling power spectra
    calculations for collections of cosmological fields (e.g., temperature,
    E-mode, B-mode polarization). It automatically constructs the appropriate
    auto- and cross-spectra based on the field properties and manages the
    mapping between field pairs and spectrum labels.

    The manager handles both dictionary and matrix representations of power
    spectra, applies normalization factors, and integrates with BeamManager
    for instrumental effects.

    Parameters
    ----------
    fields : list of BaseField
        Collection of cosmological fields for which to manage power spectra.
        Each field should have defined labels, spin, and lmax properties.

    Attributes
    ----------
    fields : list of BaseField
        The input collection of fields.
    labels : list of str
        List of all spectrum labels (e.g., ['TT', 'EE', 'BB', 'TE', 'TB', 'EB']).
    n_spectra : int
        Total number of power spectra (auto + cross).

    Examples
    --------
    >>> from cosmocore import ScalarField, PolarizationField
    >>> temp_field = ScalarField(['T'], spin=0, lmax=100)
    >>> pol_field = PolarizationField(['E', 'B'], spin=2, lmax=100)
    >>> spectra_mgr = SpectraManager([temp_field, pol_field])
    >>> print(spectra_mgr.labels)  # ['TT', 'EE', 'BB', 'TE']

    Notes
    -----
    The class automatically determines which cross-spectra are physically
    meaningful based on the field types and spin properties. For example,
    temperature-B mode cross-spectra (TB) are typically zero in standard
    cosmological models and may be excluded.
    """

    def __init__(self, fields: list[BaseField]):
        self.fields = fields
        self._spectra_labels = []
        self._spectra_map = {}
        self._cls_dict = {}
        self._cls_matrix = None

        self._build_spectra_structure()

    def _build_spectra_structure(self) -> None:
        """
        Build the mapping between field pairs and spectrum labels.

        This private method constructs the internal data structures that map
        field pair indices to power spectrum labels. It processes both
        auto-spectra (field with itself) and cross-spectra (between different
        fields) based on the field properties.

        Notes
        -----
        This method is called automatically during initialization and should
        not be called directly by users.
        """
        # Auto-spectra
        for i, field in enumerate(self.fields):
            labels = field.get_spectrum_labels()
            for mode, label in enumerate(labels):
                self._spectra_labels.append(label)
                self._spectra_map[(i, i, mode)] = label

        # Cross-spectra
        for i in range(len(self.fields)):
            for j in range(i + 1, len(self.fields)):
                labels = self.fields[i].get_cross_spectrum_labels(self.fields[j])
                for mode, label in enumerate(labels):
                    self._spectra_labels.append(label)
                    self._spectra_map[(i, j, mode)] = label

    @property
    def labels(self) -> list[str]:
        """
        Get list of all power spectrum labels.

        Returns
        -------
        list of str
            Copy of the spectrum labels list (e.g., ['TT', 'EE', 'BB', 'TE']).
            The order matches the column ordering in the power spectra matrix.
        """
        return self._spectra_labels.copy()

    @property
    def n_spectra(self) -> int:
        """
        Get total number of power spectra.

        Returns
        -------
        int
            Total number of auto- and cross-power spectra managed by this instance.
        """
        return len(self._spectra_labels)

    def get_spectrum_label(self, field_i: int, field_j: int, mode: int = 0) -> str:
        """
        Get spectrum label for given field pair and mode.

        Parameters
        ----------
        field_i : int
            Index of first field in the field collection.
        field_j : int
            Index of second field in the field collection.
        mode : int, optional
            Mode index for fields with multiple modes (default: 0).

        Returns
        -------
        str or None
            Spectrum label (e.g., 'TT', 'TE', 'EE') if the combination exists,
            None otherwise.

        Examples
        --------
        >>> # Get auto-spectrum label for temperature field (index 0)
        >>> label = spectra_mgr.get_spectrum_label(0, 0)  # 'TT'
        >>> # Get cross-spectrum label between temperature (0) and E-mode (1)
        >>> label = spectra_mgr.get_spectrum_label(0, 1)  # 'TE'
        """
        return self._spectra_map.get((field_i, field_j, mode))

    def set_cls(
        self, cls_data: dict[str, np.ndarray] | np.ndarray, lmax: int | None = None
    ) -> None:
        """
        Set power spectra from dictionary or matrix.

        This method accepts power spectra in two formats and automatically
        converts between dictionary and matrix representations for internal use.

        Parameters
        ----------
        cls_data : dict[str, np.ndarray] or np.ndarray
            Power spectra data. ℓ-indexed: ``cls_data[label][ell]`` is C_ℓ.
            If dict, keys should be spectrum labels (e.g., 'TT', 'EE', 'TE')
            and values should be 1D arrays of length ``lmax + 1``. If array,
            should have shape ``(lmax + 1, n_spectra)`` with columns
            corresponding to the spectrum labels in order.
        lmax : int, optional
            Maximum multipole to use. If None, uses the field's lmax.
            This allows setting Cls up to a different lmax than the field's lmax.

        Raises
        ------
        ValueError
            If dictionary is missing required spectrum labels, or if array
            has wrong number of columns.

        Examples
        --------
        >>> # Set from dictionary
        >>> cls_dict = {'TT': tt_spectrum, 'EE': ee_spectrum, 'TE': te_spectrum}
        >>> spectra_mgr.set_cls(cls_dict)
        >>>
        >>> # Set from matrix
        >>> cls_matrix = np.column_stack([tt_spectrum, ee_spectrum, te_spectrum])
        >>> spectra_mgr.set_cls(cls_matrix)
        """
        effective_lmax = lmax if lmax is not None else self.fields[0].lmax
        n_ell = effective_lmax + 1
        if isinstance(cls_data, dict):
            self._cls_dict = cls_data.copy()
            # Build matrix from dictionary
            self._cls_matrix = np.zeros((n_ell, self.n_spectra))
            for idx, label in enumerate(self._spectra_labels):
                if label not in cls_data:
                    raise ValueError(f"Missing power spectrum for {label}")
                self._cls_matrix[:, idx] = cls_data[label][:n_ell]

        elif isinstance(cls_data, np.ndarray):
            if cls_data.shape[1] != self.n_spectra:
                raise ValueError(
                    f"Expected {self.n_spectra} spectra columns, got {cls_data.shape[1]}"
                )
            self._cls_matrix = cls_data[:n_ell].copy()
            # Build dictionary from matrix
            self._cls_dict = {
                label: self._cls_matrix[:, idx]
                for idx, label in enumerate(self._spectra_labels)
            }

    def get_cls(self, field_i: int, field_j: int, mode: int = 0) -> np.ndarray:
        """
        Get power spectrum for field pair and mode.

        Parameters
        ----------
        field_i : int
            Index of first field.
        field_j : int
            Index of second field.
        mode : int, optional
            Mode index (default: 0).

        Returns
        -------
        np.ndarray
            ℓ-indexed power-spectrum array of length ``lmax + 1``: index
            ``ell`` holds C_ℓ. Indices below the spectrum's physical floor
            (ℓ < 2 for spin-0, ℓ < |s| for spin-s) are zero.

        Raises
        ------
        ValueError
            If no power spectrum is found for the specified field combination.

        Examples
        --------
        >>> # Get temperature auto-spectrum
        >>> tt_spectrum = spectra_mgr.get_cls(0, 0)
        >>> # Get temperature-E mode cross-spectrum
        >>> te_spectrum = spectra_mgr.get_cls(0, 1)
        """
        label = self.get_spectrum_label(field_i, field_j, mode)
        if label not in self._cls_dict:
            raise ValueError(f"No power spectrum found for {label}")
        return self._cls_dict[label]

    def build_inputs(self, *, symmetry_mode=None):
        """Build C_ell_dict and spectra_list keyed by SpectrumKey.

        Parameters
        ----------
        symmetry_mode : SymmetryMode | str | None, optional
            How cross-component spin-2 × spin-2 EB-like spectra are emitted.
            SYMMETRIC (default) emits one combined GC per cross-pair; the
            companion CG entry from the underlying spectra map is filtered
            out. DIRECTIONAL emits both GC and CG (B_i × E_j) as separate
            spectra. See ADR-0011. Strings are coerced via
            ``SymmetryMode[name.upper()]``.

        Returns
        -------
        C_ell_dict : dict[SpectrumKey, ndarray]
            Dictionary mapping each spectrum identifier to its C_ell array.
        spectra_list : list[SpectrumKey]
            Ordered list of spectrum identifiers. DIRECTIONAL keeps the
            extra CG entries in their original positions; SYMMETRIC skips
            them so the spectrum count stays at the legacy 3-per-cross-pair.
        """
        from cosmocore.spectrum_key import (
            SpectrumKey,
            SpectrumKind,
            SymmetryMode,
            _spin_pair_mode_to_kind,
        )

        # Coerce so downstream `is SymmetryMode.SYMMETRIC` checks behave
        # correctly when a caller passes the string form (e.g. from YAML).
        if symmetry_mode is None:
            symmetry_mode = SymmetryMode.SYMMETRIC
        elif isinstance(symmetry_mode, str):
            try:
                symmetry_mode = SymmetryMode[symmetry_mode.upper()]
            except KeyError as exc:
                raise ValueError(
                    f"unknown symmetry_mode {symmetry_mode!r}; expected one of "
                    f"{[m.name for m in SymmetryMode]}"
                ) from exc
        elif not isinstance(symmetry_mode, SymmetryMode):
            raise TypeError(
                f"symmetry_mode must be SymmetryMode | str | None; "
                f"got {type(symmetry_mode).__name__}"
            )

        spins = tuple(f.spin for f in self.fields)
        C_ell_dict: dict = {}
        spectra_list: list = []
        for fi, fj, mode in self._spectra_map:
            kind = _spin_pair_mode_to_kind(
                spins[fi], spins[fj], mode, is_cross=(fi != fj)
            )
            if symmetry_mode is SymmetryMode.SYMMETRIC and kind is SpectrumKind.CG:
                continue
            key = SpectrumKey(fi, fj, kind, spins=spins)
            C_ell_dict[key] = self.get_cls(fi, fj, mode)
            spectra_list.append(key)

        return C_ell_dict, spectra_list

    def compute_smoothing_factors(
        self, beam_manager: BeamManager, lmax: int | None = None
    ) -> dict[str, np.ndarray]:
        """
        Compute smoothing factors for all spectrum labels.

        Parameters:
        -----------
        beam_manager : BeamManager
            BeamManager instance for smoothing factors
        lmax : int, optional
            Maximum multipole to use. If None, uses the field's lmax.

        Returns:
        --------
        dict[str, np.ndarray]
            Dictionary mapping spectrum labels to smoothing factor arrays
        """
        effective_lmax = lmax if lmax is not None else self.fields[0].lmax
        n_ell = effective_lmax + 1  # ℓ-indexed: indices 0..lmax

        # Get beam dictionary
        beam_dict = beam_manager.get_beam_dict()
        smoothing_factors = {}

        for label in self._spectra_labels:
            smooth_factor = np.ones(n_ell, dtype=np.float64)

            # Extract field labels from spectrum label
            # (e.g., "TE" -> "T", "E")
            label1, label2 = label[0], label[1]
            if label1 in beam_dict and label2 in beam_dict:
                # Truncate beams to match output lmax if they were computed
                # with a larger lmax_signal (e.g., 4*nside for signal matrix).
                # Validate beam lengths before truncation to catch config errors.
                beam1_full = beam_dict[label1]
                beam2_full = beam_dict[label2]

                if beam1_full.shape[0] < n_ell:
                    raise ValueError(
                        f"Beam for field '{label1}' in spectrum '{label}' is too short: "
                        f"expected at least {n_ell} multipoles "
                        f"(up to ell={effective_lmax}), "
                        f"got {beam1_full.shape[0]}. "
                        "Check lmax_signal vs beam computation."
                    )
                if beam2_full.shape[0] < n_ell:
                    raise ValueError(
                        f"Beam for field '{label2}' in spectrum '{label}' is too short: "
                        f"expected at least {n_ell} multipoles "
                        f"(up to ell={effective_lmax}), "
                        f"got {beam2_full.shape[0]}. "
                        "Check lmax_signal vs beam computation."
                    )

                beam1 = beam1_full[:n_ell]
                beam2 = beam2_full[:n_ell]
                smooth_factor = beam1 * beam2

            smoothing_factors[label] = smooth_factor

        return smoothing_factors
