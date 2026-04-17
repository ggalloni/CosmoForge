"""
Harmonic analysis utilities for cosmological power spectra.

This module provides tools for working with spherical harmonic coefficients and
power spectra in cosmological analyses. It includes functions for converting
between different representations of power spectra, beam handling, and spectral
management for field collections.

The main classes are:
    - SpectraManager: Manages power spectra for collections of cosmological fields
    - BeamManager: Handles beam functions and smoothing operations

Key functions:
    - cl_to_vec, vec_to_cl: Convert between C_ℓ matrices and vectorized forms
    - coswinbeam: Generate cosine window beam functions

References
----------
Power Spectrum Conventions:
.. [1] Tegmark, M. "How to measure CMB power spectra without losing information"
   Phys. Rev. D 55, 5895 (1997)
.. [2] Hivon, E. et al. "MASTER of the Cosmic Microwave Background Anisotropy
   Power Spectrum: A Fast Method for Statistical Analysis of Large and Complex
   Cosmic Microwave Background Data Sets" Astrophys. J. 567, 2-17 (2002)

Beam Window Functions:
.. [3] Page, L. et al. "First-Year Wilkinson Microwave Anisotropy Probe (WMAP)
   Observations: Beam Profiles and Window Functions"
   Astrophys. J. Suppl. 148, 39-50 (2003)
.. [4] Mitra, S., Rocha, G., Gorski, K.M. et al. "Fast and Efficient Template Fitting
   of Deterministic Anisotropic Cosmological Models Applied to WMAP Data"
   Astrophys. J. Suppl. 193, 5 (2011)

Spherical Harmonic Transforms:
.. [5] Reinecke, M. & Seljak, U. "Libsharp - spherical harmonic transforms revisited"
   Astron. Astrophys. 554, A112 (2013)
.. [6] Gorski, K.M. et al. "HEALPix: A Framework for High-Resolution Discretization
   and Fast Analysis of Data Distributed on the Sphere"
   Astrophys. J. 622, 759-771 (2005)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit

if TYPE_CHECKING:
    from cosmocore import BaseField, InputParams


@njit(cache=True)
def cl_to_vec(cl, vec):
    """
    Convert power spectra matrix to vectorized form.

    This function flattens a 2D power spectra matrix (multipoles × spectra)
    into a 1D vector. The vectorization follows the order: all multipoles for
    spectrum 0, then all multipoles for spectrum 1, etc. Only multipoles ℓ ≥ 2
    are included.

    Parameters
    ----------
    cl : numpy.ndarray, shape (lmax-1, n_spec)
        Input power spectra matrix. First dimension corresponds to multipoles
        ℓ=2 to ℓ=lmax, second dimension corresponds to different power spectra.
    vec : numpy.ndarray, shape ((lmax-1) * n_spec,)
        Output vector to be filled. Must be pre-allocated with correct size.

    Notes
    -----
    This function is JIT-compiled with Numba for performance. The input vector
    `vec` is modified in-place.

    Examples
    --------
    >>> import numpy as np
    >>> cl = np.random.random((10, 3))  # lmax=12, 3 spectra
    >>> vec = np.zeros(30)  # 10 * 3
    >>> cl_to_vec(cl, vec)
    """
    lmax = cl.shape[0] + 1
    n_spec = cl.shape[1]
    counter = 0
    for ispec in range(n_spec):
        for il in range(2, lmax):
            vec[counter] = cl[il - 2, ispec]
            counter += 1


@njit(cache=True)
def vec_to_cl(vec, cl):
    """
    Convert vectorized power spectra back to matrix form.

    This function is the inverse of cl_to_vec, converting a 1D vector of power
    spectra back to a 2D matrix format. The devectorization follows the same
    ordering convention: all multipoles for spectrum 0, then all multipoles
    for spectrum 1, etc.

    Parameters
    ----------
    vec : numpy.ndarray, shape ((lmax-1) * n_spec,)
        Input vector containing flattened power spectra.
    cl : numpy.ndarray, shape (lmax-1, n_spec)
        Output power spectra matrix to be filled. Must be pre-allocated with
        correct shape.

    Notes
    -----
    This function is JIT-compiled with Numba for performance. The input matrix
    `cl` is modified in-place.

    Examples
    --------
    >>> import numpy as np
    >>> vec = np.random.random(30)  # 10 * 3 elements
    >>> cl = np.zeros((10, 3))  # lmax=12, 3 spectra
    >>> vec_to_cl(vec, cl)
    """
    lmax = cl.shape[0] + 1
    n_spec = cl.shape[1]
    counter = 0
    for ispec in range(n_spec):
        for il in range(2, lmax):
            cl[il - 2, ispec] = vec[counter]
            counter += 1


def coswinbeam(nside):
    """
    Generate a cosine window beam function.

    This function creates a beam window function with a flat top up to nside
    multipoles, followed by a cosine roll-off between nside and 3*nside, and
    zero beyond 3*nside. This type of beam is commonly used to simulate
    realistic instrumental response in cosmological surveys.

    Parameters
    ----------
    nside : int
        HEALPix nside parameter that determines the beam characteristics.
        The flat-top region extends to ℓ = nside, and the cosine roll-off
        extends from ℓ = nside to ℓ = 3*nside.

    Returns
    -------
    numpy.ndarray, shape (4*nside + 1,)
        Beam window function B(ℓ) for multipoles ℓ = 0 to ℓ = 4*nside.

    Notes
    -----
    The beam function is defined as:
        - B(ℓ) = 1.0 for ℓ ≤ nside
        - B(ℓ) = 0.5 * (1 + cos((ℓ - nside) * π / (2*nside))) for nside < ℓ ≤ 3*nside
        - B(ℓ) = 0.0 for ℓ > 3*nside

    Examples
    --------
    >>> beam = coswinbeam(512)
    >>> print(beam.shape)  # (2049,)
    >>> print(beam[0:513].min(), beam[0:513].max())  # (1.0, 1.0)
    """
    L = 4 * nside + 1
    beam = np.zeros(L, dtype=np.float64)
    # flat top
    beam[: nside + 1] = 1.0
    # cosine roll-off
    ell = np.arange(nside + 1, 3 * nside + 1)
    beam[nside + 1 : 3 * nside + 1] = 0.5 * (
        1.0 + np.cos((ell - nside) * np.pi / (2.0 * nside))
    )
    return beam


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
    >>> from cosmoforge.cosmocore import ScalarField, PolarizationField
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
            Power spectra data. If dict, keys should be spectrum labels
            (e.g., 'TT', 'EE', 'TE') and values should be 1D arrays of length
            (lmax-1). If array, should have shape (lmax-1, n_spectra) with
            columns corresponding to the spectrum labels in order.
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
        if isinstance(cls_data, dict):
            self._cls_dict = cls_data.copy()
            # Build matrix from dictionary
            self._cls_matrix = np.zeros((effective_lmax - 1, self.n_spectra))
            for idx, label in enumerate(self._spectra_labels):
                if label not in cls_data:
                    raise ValueError(f"Missing power spectrum for {label}")
                self._cls_matrix[:, idx] = cls_data[label][: effective_lmax - 1]

        elif isinstance(cls_data, np.ndarray):
            if cls_data.shape[1] != self.n_spectra:
                raise ValueError(
                    f"Expected {self.n_spectra} spectra columns, got {cls_data.shape[1]}"
                )
            self._cls_matrix = cls_data[: effective_lmax - 1].copy()
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
            Power spectrum array of length (lmax-1) containing C_ℓ values
            for multipoles ℓ = 2 to ℓ = lmax.

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

    def build_inputs(
        self,
    ) -> tuple[dict[tuple[int, int, int], np.ndarray], list[tuple[int, int, int]]]:
        """
        Build C_ell_dict and spectra_list for compressed multi-field operations.

        Iterates over the spectra map to build a dictionary with 3-tuple keys
        (comp_i, comp_j, mode) and an ordered list of spectra.

        Returns
        -------
        C_ell_dict : dict
            Dictionary mapping (comp_i, comp_j, mode) to C_ell arrays.
        spectra_list : list
            Ordered list of (comp_i, comp_j, mode) tuples.
        """
        C_ell_dict = {}
        spectra_list = []
        for fi, fj, mode in self._spectra_map:
            C_ell_dict[(fi, fj, mode)] = self.get_cls(fi, fj, mode)
            spectra_list.append((fi, fj, mode))
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
        n_ell = effective_lmax - 1  # ell from 2 to lmax

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


class BeamManager:
    """
    Manages beam functions for a collection of cosmological fields.

    This class handles the computation and application of instrumental beam
    window functions that model the finite angular resolution of cosmological
    surveys. It supports multiple beam types (Gaussian, cosine window, or
    custom from file) and manages beam assignment to different field types.

    The manager computes beam functions for temperature and polarization
    measurements separately, as they may have different instrumental responses.
    It integrates with SpectraManager to apply beam smoothing to theoretical
    power spectra.

    Parameters
    ----------
    fields : list of BaseField
        Collection of cosmological fields for which to manage beam functions.
        Each field should have defined labels and spin properties.

    Attributes
    ----------
    fields : list of BaseField
        The input collection of fields.

    Examples
    --------
    >>> from cosmoforge.cosmocore import ScalarField, PolarizationField
    >>> temp_field = ScalarField(['T'], spin=0, lmax=100)
    >>> pol_field = PolarizationField(['E', 'B'], spin=2, lmax=100)
    >>> beam_mgr = BeamManager([temp_field, pol_field])
    >>> # Set Gaussian beam with 5 arcmin FWHM
    >>> params = InputParams(lmax=100, nside=512, smoothing_type=1, fwhmarcmin=5.0)
    >>> beam_mgr.set_beams_from_params(params)

    Notes
    -----
    Supported beam types:
        - smoothtype=0: No smoothing (beam = 1)
        - smoothtype=1: Gaussian beam with specified FWHM
        - smoothtype=2: Cosine window beam based on nside
        - smoothtype=3: Custom beam from file
    """

    def __init__(self, fields: list[BaseField]):
        self.fields = fields
        self._beam_dict = {}

    def compute_beams(
        self, lmax: int, nside: int, smoothtype: str, fwhmarcmin: float, beam_file: str
    ) -> dict[str, np.ndarray]:
        """
        Compute beam functions based on smoothing type.

        Parameters:
        -----------
        lmax : int
            Maximum multipole
        nside : int
            HEALPix nside parameter
        smoothtype : str
            Type of smoothing: ``"none"``, ``"gaussian"``, ``"cosine"``,
            or ``"file"``.
        fwhmarcmin : float
            FWHM in arcminutes for Gaussian beam
        beam_file : str
            Path to beam file for smoothtype="file"

        Returns:
        --------
        Dict[str, np.ndarray]
            Dictionary with beam functions for T, E, B
        """
        import healpy as hp

        if smoothtype == "none":
            beam = np.ones((3, lmax - 1), dtype=np.float64)
        elif smoothtype == "gaussian":
            # fwhmarcmin in arcminutes → fwhm_rad
            beam = np.array(
                hp.gauss_beam(np.deg2rad(fwhmarcmin / 60.0), lmax=lmax + 1, pol=True)[
                    2 : lmax + 1, :-1
                ],
                dtype=np.float64,
            ).T
        elif smoothtype == "cosine":
            b = coswinbeam(nside)[2 : lmax + 1]
            beam = np.column_stack([b] * 3).T
        elif smoothtype == "file":
            # Beam file must contain at least 3 columns: T, E, B window functions.
            # Additional columns (e.g., cross-terms like T-E, T-B) are ignored
            # as they are not needed for power spectrum smoothing.
            bls = hp.read_cl(beam_file.strip()).astype(np.float64)
            if bls.shape[0] < 3:
                raise ValueError(
                    f"Beam file must have at least 3 columns (T, E, B), "
                    f"got {bls.shape[0]}"
                )
            beam = np.column_stack([bls[i][2 : lmax + 1] for i in range(3)]).T
        else:
            raise ValueError(f"Unknown smoothtype='{smoothtype}'")

        if beam.shape[0] != 3 or beam.shape[1] != lmax - 1:
            raise ValueError(
                f"Beam shape mismatch: expected (3, {lmax - 1}), got {beam.shape}"
            )

        return {
            "T": beam[0, :],
            "E": beam[1, :],
            "B": beam[2, :],
        }

    def set_beams_from_params(self, params: InputParams, lmax: int | None = None) -> None:
        """
        Set beams for all fields using parameter configuration.

        This method computes the appropriate beam functions based on the
        parameter settings and assigns them to each field in the collection.
        Scalar fields receive a single beam function, while polarization
        fields receive separate E and B mode beam functions.

        Parameters
        ----------
        params : InputParams
            Configuration object containing beam parameters including:
            - lmax: Maximum multipole
            - nside: HEALPix resolution parameter
            - smoothing_type: Type of beam ("none", "gaussian", "cosine", "file")
            - fwhmarcmin: FWHM in arcminutes (for Gaussian beams)
            - beam_file: Path to beam file (for custom beams)
        lmax : int, optional
            Maximum multipole to use. If None, uses params.lmax.
            This allows computing beams up to a different lmax than params.lmax.

        Notes
        -----
        The method automatically handles field-beam matching. For scalar fields,
        it uses the 'T' beam. For polarization fields, it uses 'E' and 'B' beams.
        If specific labels aren't found, it falls back to available beams.

        Examples
        --------
        >>> params = InputParams(lmax=100, smoothing_type=1, fwhmarcmin=5.0)
        >>> beam_mgr.set_beams_from_params(params)
        """
        effective_lmax = lmax if lmax is not None else params.lmax
        beam_dict = self.compute_beams(
            lmax=effective_lmax,
            nside=params.nside,
            smoothtype=params.smoothing_type,
            fwhmarcmin=params.fwhmarcmin,
            beam_file=params.beam_file,
        )

        # Build internal beam dictionary with field labels
        self._beam_dict = {}
        for field in self.fields:
            if field.spin == 0:
                # Scalar field - single beam
                beam_label = field.labels[0]
                if beam_label in beam_dict:
                    self._beam_dict[beam_label] = beam_dict[beam_label]
                else:
                    # Fallback to generic beam if label not found
                    self._beam_dict[beam_label] = beam_dict.get(
                        "T", list(beam_dict.values())[0]
                    )
            elif field.spin == 2:
                # Polarization field - E and B beams
                e_beam = beam_dict.get(
                    "E", beam_dict.get("P", list(beam_dict.values())[0])
                )
                b_beam = beam_dict.get(
                    "B", beam_dict.get("P", list(beam_dict.values())[0])
                )
                self._beam_dict[field.labels[0]] = e_beam
                self._beam_dict[field.labels[1]] = b_beam

        # Only set beams on fields if lmax matches field.lmax
        # (otherwise the validation would fail)
        if lmax is None or lmax == self.fields[0].lmax:
            for field in self.fields:
                if field.spin == 0:
                    field.set_beam(self._beam_dict[field.labels[0]])
                elif field.spin == 2:
                    beam_array = np.column_stack(
                        [
                            self._beam_dict[field.labels[0]],
                            self._beam_dict[field.labels[1]],
                        ]
                    )
                    field.set_beam(beam_array)

    def get_beam_dict(self) -> dict[str, np.ndarray]:
        """
        Get dictionary mapping field labels to beam functions.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary with field labels as keys (e.g., 'T', 'E', 'B') and
            corresponding beam window functions as values. Each beam is a
            1D array of length (lmax-1) containing B(ℓ) for ℓ = 2 to ℓ = lmax.

        Raises
        ------
        ValueError
            If beams have not been set.

        Notes
        -----
        For scalar fields, the dictionary contains one entry per field label.
        For polarization fields, the beam array is split into separate E and B
        mode entries.

        Examples
        --------
        >>> beam_dict = beam_mgr.get_beam_dict()
        >>> print(beam_dict.keys())  # ['T', 'E', 'B']
        >>> t_beam = beam_dict['T']  # Temperature beam function
        """
        # Return internally stored beams (set by set_beams_from_params)
        if not self._beam_dict:
            raise ValueError("Beams have not been set. Call set_beams_from_params first.")
        return self._beam_dict.copy()

    def get_beam(self, label: str) -> np.ndarray:
        """
        Get beam window function for a specific field label.

        Parameters
        ----------
        label : str
            Field label (e.g., 'T', 'E', 'B').

        Returns
        -------
        np.ndarray
            Beam window function B(ℓ) for ℓ = 2 to ℓ = lmax.

        Raises
        ------
        ValueError
            If beams have not been set or if label is not found.
        """
        beam_dict = self.get_beam_dict()
        if label not in beam_dict:
            raise ValueError(f"No beam found for label '{label}'")
        return beam_dict[label]

    def apply_smoothing(
        self, spectra_manager: SpectraManager, lmax: int | None = None
    ) -> None:
        """
        Apply beam smoothing to power spectra.

        This method applies instrumental beam effects to all power spectra
        managed by the SpectraManager. The smoothing accounts for both the
        beam window functions and the geometric conversion factors.

        Parameters
        ----------
        spectra_manager : SpectraManager
            The SpectraManager instance containing the power spectra to smooth.
            The spectra are modified in-place.
        lmax : int, optional
            Maximum multipole to use. If None, uses the field's lmax.
            This allows applying smoothing up to a different lmax.

        Notes
        -----
        The smoothing operation multiplies each power spectrum by:
        C_ℓ^smoothed = C_ℓ^theory × B₁(ℓ) × B₂(ℓ)

        where B₁(ℓ) and B₂(ℓ) are the beam functions for the two fields
        involved in the spectrum.

        Examples
        --------
        >>> beam_mgr.set_beams_from_params(params)
        >>> spectra_mgr.set_cls_from_file('theory_cls.dat', params)
        >>> beam_mgr.apply_smoothing(spectra_mgr)  # Apply instrumental effects
        """
        # Use precomputed smoothing factors
        smoothing_factors = spectra_manager.compute_smoothing_factors(self, lmax=lmax)

        for label in spectra_manager.labels:
            if label not in spectra_manager._cls_dict:
                raise ValueError(f"No power spectrum found for {label}.")

            if label in smoothing_factors:
                # Apply smoothing factor
                spectra_manager._cls_dict[label] *= smoothing_factors[label]

        # Update matrix from dictionary
        for idx, label in enumerate(spectra_manager._spectra_labels):
            spectra_manager._cls_matrix[:, idx] = spectra_manager._cls_dict[label]
