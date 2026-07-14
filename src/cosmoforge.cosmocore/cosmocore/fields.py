"""
Cosmological field representations and management.

This module provides a comprehensive framework for handling different types of
cosmological fields (scalar and polarization) on the sphere. It includes abstract
base classes and concrete implementations for temperature and polarization fields,
along with configuration management and field collection utilities.

The main components are:
    - FieldConfig: Configuration dataclass for field parameters
    - BaseField: Abstract base class for all cosmological fields
    - ScalarField: Implementation for spin-0 fields (e.g., temperature)
    - PolarizationField: Implementation for spin-2 fields (E/B modes)
    - FieldCollection: Manager for collections of multiple fields

Key features:
    - HEALPix map handling and spherical harmonic transforms
    - Beam function management for instrumental effects
    - Power spectra computation and cross-correlation support
    - Mask handling for partial sky analyses
    - Type-safe field configuration with validation
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import healpy as hp
import numpy as np

from cosmocore.beam import BeamManager
from cosmocore.geometry import ACTIVE_THRESHOLD
from cosmocore.in_out import readcl
from cosmocore.settings import InputParams
from cosmocore.spectra_io import SpectraManager


@dataclass
class FieldConfig:
    """
    Configuration for a cosmological field.

    This dataclass encapsulates all the parameters needed to define a cosmological
    field on the sphere, including geometric properties, resolution parameters,
    and field-specific metadata.

    Parameters
    ----------
    spin : int
        Spin weight of the field. Must be 0 (scalar) or 2 (polarization).
    nside : int
        HEALPix nside parameter determining map resolution. Total number of
        pixels is 12*nside².
    lmax : int
        Maximum multipole moment for spherical harmonic analysis. Must satisfy
        lmax ≤ 4*nside for numerical stability.
    mask : numpy.ndarray
        1D boolean or float array of length 12*nside² indicating which pixels
        are observed (1) or masked (0).
    labels : str or list of str
        Field label(s). Single string for spin-0 (e.g., "T"), list of two
        strings for spin-2 (e.g., ["E", "B"]).

    Raises
    ------
    ValueError
        If spin is not 0 or 2, if mask is not 1D, if lmax < 2, if lmax > 4*nside,
        or if number of labels doesn't match spin requirements.

    Examples
    --------
    >>> import numpy as np
    >>> # Temperature field configuration
    >>> temp_config = FieldConfig(
    ...     spin=0, nside=512, lmax=1024,
    ...     mask=np.ones(12*512**2), labels="T"
    ... )
    >>>
    >>> # Polarization field configuration
    >>> pol_config = FieldConfig(
    ...     spin=2, nside=512, lmax=1024,
    ...     mask=np.ones(12*512**2), labels=["E", "B"]
    ... )

    Notes
    -----
    The configuration is validated automatically upon creation. The mask array
    is used for partial sky analyses and should have the same ordering as the
    HEALPix maps (typically RING ordering).
    """

    spin: int
    nside: int
    lmax: int
    mask: np.ndarray
    labels: str | list[str]  # "T" for spin-0, ["E", "B"] for spin-2

    def __post_init__(self):
        """
        Validate configuration parameters.

        This method is called automatically after dataclass initialization to
        ensure all parameters are valid and consistent. It also normalizes the
        labels format to a consistent list representation.

        Raises
        ------
        ValueError
            If any validation check fails, including invalid spin values,
            incorrect mask dimensions, or mismatched label counts.
        """
        if self.spin not in (0, 2):
            raise ValueError(f"Invalid spin {self.spin}. Must be 0 or 2.")
        if self.mask.ndim != 1:
            raise ValueError("Mask must be a 1D array.")
        # ADR 0009: spin-0 fields can carry monopole/dipole multipoles
        # (lmax >= 0). Spin-2 still requires lmax >= 2 by representation
        # theory.
        spin_floor = 2 if self.spin == 2 else 0
        if self.lmax < spin_floor:
            raise ValueError(
                f"lmax={self.lmax} below spin-{self.spin} floor {spin_floor}."
            )
        if self.lmax > self.nside * 4:
            raise ValueError("lmax is too large for the given nside.")

        # Normalize labels to list format
        if isinstance(self.labels, str):
            self.labels = [self.labels]

        # Validate label count matches spin
        expected_labels = 1 if self.spin == 0 else 2
        if len(self.labels) != expected_labels:
            raise ValueError(
                f"Spin-{self.spin} field requires {expected_labels} labels, "
                f"got {len(self.labels)}"
            )


class BaseField(ABC):
    """
    Abstract base class for cosmological fields.

    This class provides the common interface and functionality for all types of
    cosmological fields defined on the sphere. It handles pixel indexing, beam
    functions, spherical harmonic transforms, and power spectra computations.

    The class is designed to work with HEALPix pixelization and supports both
    full-sky and partial-sky (masked) analyses. Concrete subclasses implement
    the specific behavior for different field types (scalar vs. polarization).

    Parameters
    ----------
    config : FieldConfig
        Configuration object containing all field parameters including spin,
        resolution, mask, and labels.

    Attributes
    ----------
    config : FieldConfig
        The input configuration object.
    npix : int
        Total number of pixels in the HEALPix map (12*nside²).
    n_components : int
        Number of field components (1 for scalar, 2 for polarization).
    spin : int
        Spin weight of the field (0 or 2).
    nside : int
        HEALPix nside parameter.
    lmax : int
        Maximum multipole for spherical harmonic analysis.
    labels : list of str
        Field labels (e.g., ["T"] or ["E", "B"]).
    mask : numpy.ndarray
        Boolean mask indicating observed pixels.
    beam : numpy.ndarray or None
        Beam window function(s) for instrumental effects.

    Notes
    -----
    This is an abstract class and cannot be instantiated directly. Use the
    concrete subclasses ScalarField or PolarizationField instead.

    Examples
    --------
    >>> # Use concrete subclasses
    >>> temp_field = ScalarField(temp_config)
    >>> pol_field = PolarizationField(pol_config)
    """

    def __init__(self, config: FieldConfig):
        self.config = config
        self._active_pixels = None
        self._point_vectors = None
        self._beam = None

        # Derived properties
        self.npix = hp.nside2npix(config.nside)
        self.n_components = 1 if config.spin == 0 else 2

    @property
    def spin(self) -> int:
        """Spin weight of the field (0 for scalar, 2 for polarization)."""
        return self.config.spin

    @property
    def nside(self) -> int:
        """HEALPix nside parameter determining map resolution."""
        return self.config.nside

    @property
    def lmax(self) -> int:
        """Maximum multipole moment for spherical harmonic analysis."""
        return self.config.lmax

    @property
    def labels(self) -> list[str]:
        """Field labels (e.g., ['T'] for temperature, ['E', 'B'] for polarization)."""
        return self.config.labels

    @property
    def maps_label(self) -> str | list[str]:
        """
        Backward compatibility property for legacy code.

        Returns
        -------
        str or list of str
            Single string for scalar fields, list for polarization fields.
            This property is maintained for compatibility with older code.

        Deprecated
        -----------
        Use the `labels` property instead for new code.
        """
        if len(self.labels) == 1:
            return self.labels[0]
        return self.labels

    @property
    def mask(self) -> np.ndarray:
        """
        Boolean mask indicating observed pixels.

        Returns
        -------
        numpy.ndarray
            1D array of length npix with 1 for observed pixels and 0 for masked pixels.
        """
        return self.config.mask

    @property
    def active_pixels(self) -> np.ndarray:
        """
        Indices of unmasked (active) pixels.

        Returns
        -------
        numpy.ndarray
            1D array of pixel indices where the mask exceeds
            ``geometry.ACTIVE_THRESHOLD``. Computed lazily and cached.
        """
        if self._active_pixels is None:
            self._active_pixels = np.flatnonzero(self.mask > ACTIVE_THRESHOLD)
        return self._active_pixels

    @property
    def n_active(self) -> list[int]:
        """
        Number of active pixels.

        Returns
        -------
        list of int
            List containing the count of active pixels. Returned as list
            for consistency with multi-component fields.
        """
        return [len(self.active_pixels)]

    @property
    def point_vectors(self) -> np.ndarray | None:
        """
        Pointing vectors for active pixels.

        Returns
        -------
        numpy.ndarray or None
            Array of shape (n_active, 3) containing unit vectors pointing to
            active pixel centers, or None if not set.
        """
        return self._point_vectors

    @property
    def beam(self) -> np.ndarray | None:
        """
        Beam window function for instrumental effects.

        Returns
        -------
        numpy.ndarray or None
            Beam window function(s). For scalar fields: 1D array of length
            ``lmax + 1`` (ℓ-indexed: ``beam[ell]``). For polarization fields:
            2D array of shape ``(lmax + 1, 2)`` for E and B modes.
            Returns None if beam has not been set.
        """
        return self._beam

    def set_point_vectors(self, vectors: np.ndarray) -> None:
        """
        Set pointing vectors for active pixels.

        Parameters
        ----------
        vectors : numpy.ndarray
            Array of shape (n_active, 3) containing unit vectors pointing
            to the center of each active pixel in 3D space.

        Raises
        ------
        ValueError
            If vectors shape doesn't match (n_active, 3).

        Notes
        -----
        Pointing vectors are used for spherical harmonic analysis and
        coordinate transformations. The vectors should be normalized
        unit vectors in the chosen coordinate system.

        Examples
        --------
        >>> field = ScalarField(...)
        >>> vectors = get_pixel_pointing_vectors(field.active_pixels)
        >>> field.set_point_vectors(vectors)
        """
        if vectors.shape[0] != self.n_active[0]:
            raise ValueError(
                f"Expected {self.n_active} point vectors, got {vectors.shape[0]}"
            )
        if vectors.shape[1] != 3:
            raise ValueError("Point vectors must have 3 components (x,y,z)")
        self._point_vectors = vectors.copy()

    def set_beam(self, beam: np.ndarray) -> None:
        """
        Set the beam window function for this field.

        Parameters
        ----------
        beam : numpy.ndarray
            Beam window function. Expected shape depends on field type:
            - Scalar fields: 1D array of length ``lmax + 1`` (ℓ-indexed)
            - Polarization fields: 2D array of shape ``(lmax + 1, 2)``
              for E and B modes

        Raises
        ------
        ValueError
            If beam shape doesn't match expected dimensions for this field type.

        Notes
        -----
        The beam function is applied to correct for instrumental effects in
        harmonic space analysis. The function is expected to be normalized
        appropriately for the specific instrument and observation strategy.

        Examples
        --------
        >>> field = ScalarField(...)
        >>> beam_window = load_beam_function(lmax=field.lmax)
        >>> field.set_beam(beam_window)
        """
        expected_rows = self.lmax + 1
        if beam.shape[0] != expected_rows:
            raise ValueError(f"Beam must have {expected_rows} rows, got {beam.shape[0]}")
        self._beam = beam.copy()

    @abstractmethod
    def get_spectrum_labels(self) -> list[str]:
        """
        Get labels for all auto-spectra of this field.

        Returns
        -------
        list of str
            List of spectrum labels for this field's auto-correlations.
            For scalar fields: ["TT"]
            For polarization fields: ["EE", "BB"]

        Notes
        -----
        This abstract method must be implemented by concrete subclasses
        to define the auto-spectra that can be computed for the field type.
        """
        pass

    @abstractmethod
    def get_cross_spectrum_labels(self, other: BaseField) -> list[str]:
        """
        Get labels for cross-spectra with another field.

        Parameters
        ----------
        other : BaseField
            Another field to compute cross-spectra with.

        Returns
        -------
        list of str
            List of spectrum labels for cross-correlations between this
            field and the other field. Examples:
            - T x T: ["TT"]
            - T x QU: ["TQ", "TU"]
            - QU x QU: ["QQ", "QU", "UQ", "UU"]

        Notes
        -----
        This abstract method must be implemented by concrete subclasses
        to define valid cross-spectra between different field types.
        """
        pass


class ScalarField(BaseField):
    """
    Spin-0 cosmological field representation (e.g., temperature).

    Represents scalar fields on the sphere such as temperature anisotropies
    in the cosmic microwave background. Supports single-component analysis
    with auto-correlation power spectra.

    Parameters
    ----------
    config : FieldConfig
        Configuration containing maps, mask, and metadata for this field.

    Attributes
    ----------
    config : FieldConfig
        Field configuration and data.
    labels : list of str
        Component labels (single element for scalar fields).
    maps : numpy.ndarray
        Map data with shape (1, npix).
    npix : int
        Number of pixels in the map.
    nside : int
        HEALPix resolution parameter.
    lmax : int
        Maximum multipole for harmonic analysis.
    spin : int
        Spin weight (0 for scalar fields).

    Examples
    --------
    >>> import numpy as np
    >>> from cosmocore.fields import FieldConfig, ScalarField
    >>>
    >>> # Create temperature field
    >>> nside = 64
    >>> npix = 12 * nside**2
    >>> temp_map = np.random.randn(npix)
    >>> mask = np.ones(npix)  # No masking for this example
    >>>
    >>> config = FieldConfig(
    ...     maps=[temp_map],
    ...     labels=["T"],
    ...     mask=mask,
    ...     nside=nside,
    ...     lmax=2*nside
    ... )
    >>> temp_field = ScalarField(config)
    >>> print(f"Field type: {temp_field.spin=}")
    >>> print(f"Auto-spectra: {temp_field.get_spectrum_labels()}")
    """

    def __init__(self, config: FieldConfig):
        """
        Initialize a scalar field.

        Parameters
        ----------
        config : FieldConfig
            Field configuration. Must have spin=0 for scalar fields.

        Raises
        ------
        ValueError
            If config.spin != 0, as scalar fields must have zero spin.
        """
        if config.spin != 0:
            raise ValueError("ScalarField requires spin=0")
        super().__init__(config)

    def get_spectrum_labels(self) -> list[str]:
        """
        Get auto-spectrum labels for scalar field.

        Returns
        -------
        list of str
            Single-element list containing "TT" for temperature auto-correlation.
        """
        label = self.labels[0].upper()
        return [f"{label}{label}"]

    def get_cross_spectrum_labels(self, other: BaseField) -> list[str]:
        """
        Get cross-spectrum labels with another field.

        Parameters
        ----------
        other : BaseField
            Another field to compute cross-spectra with.

        Returns
        -------
        list of str
            Cross-spectrum labels. Examples:
            - With ScalarField: ["TT"] (if both are temperature)
            - With PolarizationField: ["TQ", "TU"] (temperature x Q/U)

        Raises
        ------
        TypeError
            If other field type is not recognized.

        Examples
        --------
        >>> temp_field = ScalarField(temp_config)
        >>> pol_field = PolarizationField(pol_config)
        >>> cross_labels = temp_field.get_cross_spectrum_labels(pol_field)
        >>> print(cross_labels)  # ["TQ", "TU"]
        """
        self_label = self.labels[0].upper()
        if isinstance(other, ScalarField):
            other_label = other.labels[0].upper()
            return [f"{self_label}{other_label}"]
        elif isinstance(other, PolarizationField):
            return [
                f"{self_label}{other.labels[0].upper()}",
                f"{self_label}{other.labels[1].upper()}",
            ]
        else:
            raise TypeError(f"Unknown field type: {type(other)}")


class PolarizationField(BaseField):
    """
    Spin-2 cosmological field representation (e.g., CMB polarization E/B modes).

    Represents spin-2 fields on the sphere such as polarization anisotropies
    in the cosmic microwave background. Supports two-component analysis with
    auto-correlation and cross-correlation power spectra between E and B modes.

    Parameters
    ----------
    config : FieldConfig
        Configuration containing maps, mask, and metadata for this field.
        Must have exactly 2 maps for Q and U components or E and B components.

    Attributes
    ----------
    config : FieldConfig
        Field configuration and data.
    labels : list of str
        Component labels (two elements for polarization fields, e.g., ["Q", "U"]).
    maps : numpy.ndarray
        Map data with shape (2, npix) for Q/U or E/B components.
    npix : int
        Number of pixels in the map.
    nside : int
        HEALPix resolution parameter.
    lmax : int
        Maximum multipole for harmonic analysis.
    spin : int
        Spin weight (2 for polarization fields).

    Examples
    --------
    >>> import numpy as np
    >>> from cosmocore.fields import FieldConfig, PolarizationField
    >>>
    >>> # Create polarization field
    >>> nside = 64
    >>> npix = 12 * nside**2
    >>> q_map = np.random.randn(npix)
    >>> u_map = np.random.randn(npix)
    >>> mask = np.ones(npix)  # No masking
    >>>
    >>> config = FieldConfig(
    ...     maps=[q_map, u_map],
    ...     labels=["Q", "U"],
    ...     mask=mask,
    ...     nside=nside,
    ...     lmax=2*nside,
    ...     spin=2
    ... )
    >>> pol_field = PolarizationField(config)
    >>> print(f"Field type: {pol_field.spin=}")
    >>> print(f"Auto-spectra: {pol_field.get_spectrum_labels()}")
    """

    def __init__(self, config: FieldConfig):
        """
        Initialize a polarization field.

        Parameters
        ----------
        config : FieldConfig
            Field configuration. Must have spin=2 for polarization fields.

        Raises
        ------
        ValueError
            If config.spin != 2, as polarization fields must have spin=2.
        """
        if config.spin != 2:
            raise ValueError("PolarizationField requires spin=2")
        super().__init__(config)

    def get_spectrum_labels(self) -> list[str]:
        """
        Get auto-spectrum labels for polarization field.

        Returns
        -------
        list of str
            Three-element list containing ["EE", "BB", "EB"] for E/B mode
            auto-correlations and cross-correlation.

        Notes
        -----
        The labels are generated using the field's component labels converted
        to uppercase. For Q/U fields, these become E/B spectrum labels.
        """
        e_label, b_label = (label.upper() for label in self.labels)
        return [f"{e_label}{e_label}", f"{b_label}{b_label}", f"{e_label}{b_label}"]

    def get_cross_spectrum_labels(self, other: BaseField) -> list[str]:
        """
        Get cross-spectrum labels with another field.

        Parameters
        ----------
        other : BaseField
            Another field to compute cross-spectra with.

        Returns
        -------
        list of str
            Cross-spectrum labels. Examples:
            - With ScalarField: ["TQ", "TU"] (temperature x Q/U)
            - With PolarizationField: ["QQ", "QU", "UQ", "UU"] (all combinations)

        Raises
        ------
        TypeError
            If other field type is not recognized.

        Examples
        --------
        >>> pol_field = PolarizationField(pol_config)
        >>> temp_field = ScalarField(temp_config)
        >>> cross_labels = pol_field.get_cross_spectrum_labels(temp_field)
        >>> print(cross_labels)  # ["TQ", "TU"]
        """
        if isinstance(other, ScalarField):
            other_label = other.labels[0].upper()
            return [
                f"{other_label}{self.labels[0].upper()}",
                f"{other_label}{self.labels[1].upper()}",
            ]
        elif isinstance(other, PolarizationField):
            cross_labels = []
            for self_label in self.labels:
                for other_label in other.labels:
                    cross_labels.append(f"{self_label.upper()}{other_label.upper()}")
            return cross_labels
        else:
            raise TypeError(f"Unknown field type: {type(other)}")


class FieldCollection:
    """
    Collection of cosmological fields for joint analysis.

    Manages multiple cosmological fields (scalar and/or polarization) for
    joint power spectrum analysis. Provides utilities for cross-correlations,
    spectrum indexing, and data vector construction.

    Parameters
    ----------
    core_params : InputParams
        Core analysis parameters including cosmological and numerical settings.
    fields : list of BaseField
        List of fields to include in the collection. Can mix scalar and
        polarization fields.

    Attributes
    ----------
    core_params : InputParams
        Analysis parameters.
    fields : list of BaseField
        Individual fields in the collection.
    n_fields : int
        Number of fields in the collection.
    n_cross : int
        Total number of cross-correlations (including auto-correlations).
    spectrum_labels : list of str
        Labels for all spectra in the collection.

    Examples
    --------
    >>> from cosmocore.fields import FieldCollection, ScalarField, PolarizationField
    >>> from cosmocore.core import InputParams
    >>>
    >>> # Create field collection
    >>> params = InputParams()
    >>> temp_field = ScalarField(temp_config)
    >>> pol_field = PolarizationField(pol_config)
    >>>
    >>> collection = FieldCollection(params, [temp_field, pol_field])
    >>> print(f"Fields: {collection.n_fields}")
    >>> print(f"Cross-correlations: {collection.n_cross}")
    >>> print(f"Spectra: {collection.spectrum_labels}")

    Notes
    -----
    The collection automatically validates field consistency (same nside, lmax)
    and computes all possible auto- and cross-correlations between fields.
    """

    def __init__(self, core_params: InputParams, fields: list[BaseField], logger=None):
        """
        Initialize field collection.

        Parameters
        ----------
        core_params : InputParams
            Core analysis parameters.
        fields : list of BaseField
            Fields to include in the collection.
        logger : CosmoLogger, optional
            CosmoLogger instance for output messages.

        Raises
        ------
        ValueError
            If fields have inconsistent parameters (nside, lmax, etc.).
        """
        self.core_params = core_params
        self.logger = logger
        self.fields = fields
        self.n_fields = len(fields)

        # Validate consistency
        self._validate_fields()

        # Initialize managers
        self.spectra_manager = SpectraManager(fields)
        self.beam_manager = BeamManager(fields)

        # Cache commonly used properties
        self._lmax = fields[0].lmax
        self._nside = fields[0].nside

    def _validate_fields(self) -> None:
        """
        Ensure all fields have consistent lmax and nside.

        Raises
        ------
        ValueError
            If fields have inconsistent lmax or nside values, or if no fields provided.

        Notes
        -----
        All fields in a collection must have the same resolution (nside) and
        maximum multipole (lmax) for joint analysis to be valid.
        """
        if not self.fields:
            raise ValueError("Must provide at least one field")

        reference = self.fields[0]
        for i, field in enumerate(self.fields[1:], 1):
            if field.lmax != reference.lmax:
                raise ValueError(
                    f"Field {i} has lmax {field.lmax}, expected {reference.lmax}"
                )
            if field.nside != reference.nside:
                raise ValueError(
                    f"Field {i} has nside {field.nside}, expected {reference.nside}"
                )

    @property
    def lmax(self) -> int:
        """Maximum multipole for all fields in the collection."""
        return self._lmax

    @property
    def nside(self) -> int:
        """HEALPix resolution parameter for all fields in the collection."""
        return self._nside

    @property
    def spin(self) -> list[int]:
        """
        Spin weights for all fields in the collection.

        Returns
        -------
        list of int
            Spin weight for each field (0 for scalar, 2 for polarization).
        """
        return [field.spin for field in self.fields]

    @property
    def n_active(self) -> list[int]:
        """
        Number of active pixels per field component.

        Returns
        -------
        list of int
            Number of active pixels for each component. Scalar fields contribute
            one entry, polarization fields contribute two entries (one per component).

        Notes
        -----
        For a collection with temperature (T) and polarization (QU) fields,
        this returns [n_active_T, n_active_Q, n_active_U].
        """
        n_active = []
        for field in self.fields:
            n_active += field.n_active
            if field.spin == 2:
                n_active += field.n_active
        return n_active

    @property
    def total_active_pixels(self) -> int:
        """
        Total number of active pixels across all field components.

        Returns
        -------
        int
            Sum of active pixels for all components in the collection.
        """
        return sum(self.n_active)

    def get_active_pixels(self) -> list[np.ndarray]:
        """
        Get the active pixels of each component (T, Q, U), in order.

        Returns
        -------
        list of numpy.ndarray
            One index array per *component*: scalar fields contribute one,
            polarization fields two (Q and U share a single pixel set).

        Notes
        -----
        A plain list, not an object array: ``np.array(..., dtype=object)``
        silently collapses to 2-D when every component has the same
        ``n_active`` and stays 1-D ragged otherwise, so its rank depended on
        the mask. Consumers index it as ``active[i][j]``.
        """
        active_list = []
        for field in self.fields:
            active_list.append(field.active_pixels)
            if field.spin == 2:
                active_list.append(field.active_pixels)  # Same for Q and U
        return active_list

    def set_pointing_vectors(self, point_vectors: list[np.ndarray]) -> None:
        """
        Set pointing vectors for all fields.

        Parameters
        ----------
        point_vectors : list of numpy.ndarray
            List of pointing vector arrays, one per field. Each array should
            have shape (n_active, 3) for the corresponding field.

        Notes
        -----
        The pointing vectors are distributed to individual fields in the
        collection. Each field validates that the vector count matches
        its number of active pixels.
        """

        for field, vectors in zip(self.fields, point_vectors):
            field.set_point_vectors(vectors)

    def set_cls(
        self,
        cls_data: dict[str, np.ndarray] | np.ndarray | None = None,
        lmax: int | None = None,
    ) -> None:
        """
        Set power spectra and apply normalization.

        Parameters
        ----------
        cls_data : dict or numpy.ndarray or None, optional
            Power spectrum data. Can be:
            - dict: Spectrum name -> array mapping
            - array: Pre-formatted spectrum array
            - None: Load from core_params.inputclfile
        lmax : int, optional
            Maximum multipole to use. If None, uses core_params.lmax.
            This allows loading Cls up to a different lmax than the analysis lmax.

        Notes
        -----
        This method sets the theoretical power spectra for the field collection
        and applies any necessary normalization based on the analysis parameters.
        """
        if cls_data is None:
            cls_data = readcl(
                self.core_params.inputclfile, self.core_params, self.logger, lmax=lmax
            )

        self.spectra_manager.set_cls(cls_data, lmax=lmax)

    def get_cls(self, field_i: int, field_j: int, mode: int = 0) -> np.ndarray:
        """
        Get power spectrum for field pair.

        Parameters
        ----------
        field_i : int
            Index of first field.
        field_j : int
            Index of second field.
        mode : int, optional
            Spectrum mode index for polarization fields. Default is 0.

        Returns
        -------
        numpy.ndarray
            Power spectrum C_l values for the specified field pair and mode.

        Examples
        --------
        >>> collection = FieldCollection(params, [temp_field, pol_field])
        >>> tt_spectrum = collection.get_cls(0, 0)  # T auto-spectrum
        >>> te_spectrum = collection.get_cls(0, 1, 0)  # T-E cross-spectrum
        """
        return self.spectra_manager.get_cls(field_i, field_j, mode)

    def set_beams(
        self, lmax: int | None = None, injected_beam: np.ndarray | None = None
    ) -> None:
        """
        Set beam functions for all fields.

        Parameters
        ----------
        lmax : int, optional
            Maximum multipole to use. If None, uses core_params.lmax.
            This allows computing beams up to a different lmax than the analysis lmax.
        injected_beam : numpy.ndarray, optional
            In-memory beam injected in place of ``core_params.beam_file`` (ADR-0017);
            see :meth:`BeamManager.compute_beams` for the contract.
        """
        self.beam_manager.set_beams_from_params(
            self.core_params, lmax=lmax, injected_beam=injected_beam
        )
        self.beam_manager.apply_smoothing(self.spectra_manager, lmax=lmax)

    @property
    def fields(self) -> list[BaseField]:
        """Provide access to fields for backward compatibility."""
        return self._fields

    @fields.setter
    def fields(self, value: list[BaseField]) -> None:
        """Setter to allow modification of fields."""
        self._fields = value
        self._validate_fields()

    @property
    def spectra_labels(self) -> list[str]:
        """Backward compatibility property for Fisher class."""
        return self.spectra_manager.labels

    @property
    def n_spectra(self) -> int:
        """Backward compatibility property for number of spectra."""
        return self.spectra_manager.n_spectra


def create_field(
    spin: int, nside: int, lmax: int, mask: np.ndarray, labels: str | list[str]
) -> BaseField:
    """Factory function to create appropriate field type."""
    config = FieldConfig(spin=spin, nside=nside, lmax=lmax, mask=mask, labels=labels)

    if spin == 0:
        return ScalarField(config)
    elif spin == 2:
        return PolarizationField(config)
    else:
        raise ValueError(f"Unsupported spin value: {spin}")
