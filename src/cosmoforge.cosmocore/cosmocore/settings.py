"""
Configuration management for cosmological analysis parameters.

This module provides the `InputParams` class for managing configuration
parameters used in cosmological analysis workflows. It handles parameter
validation, default settings, YAML file loading, and derived parameter
computation for CMB analysis pipelines.

Classes
-------
InputParams
    Parameter management class for cosmological analysis configuration.

Functions
---------
spec2idx
    Convert field indices to spectrum index for compressed storage.
idx2spec
    Convert spectrum index back to field indices.

Notes
-----
The module integrates with HEALPix for spherical harmonic analysis
and supports YAML-based configuration files for parameter management.
Key functionality includes:
- Default parameter initialization
- Configuration file loading and parsing
- Automatic computation of derived parameters
- Automatic field label expansion (e.g., "QU" -> ["Q", "U"])
- Parameter validation and updates
"""

import healpy as hp
import numpy as np
import yaml

from .basics import spec2idx


class InputParams:
    r"""
    Parameter management class for cosmological analysis configuration.

    This class handles all configuration parameters for cosmological analysis
    workflows, including HEALPix pixelization settings, field specifications,
    file paths, analysis options, and automatic computation of derived parameters.

    Parameters
    ----------
    None
        Initialized with default parameter values. Use `update()` method or
        `read_parameter_file()` to load custom configurations.

    Attributes
    ----------
    nside : int
        HEALPix resolution parameter (default: 16).
    spins : list of int
        Spin values for fields, e.g., [0, 2] for temperature and polarization.
    labels : list of str
        Field labels, e.g., ["T", "E", "B"] for temperature and polarization.
        Supports automatic expansion: "QU" -> ["Q", "U"], "T1_T2" -> ["T1", "T2"].
    physical_labels : list of str or None
        Physical field labels for map data (defaults to same as labels).
        Supports automatic expansion: "TQU" -> ["T", "Q", "U"],
        "LOW_HIGH" -> ["LOW", "HIGH"].
    lmax : int
        Maximum multipole moment for spherical harmonic analysis.
    feedback : int
        Verbosity level for output (0=silent, 1=normal, 2=verbose).
    inputclfile : str
        Path to input power spectrum file.
    maskfile : str
        Path to analysis mask file.
    covmatfile1, covmatfile2 : str
        Paths to noise covariance matrix files.
    outinvcovmatfile1, outinvcovmatfile2 : str
        Paths for output inverse covariance matrices.
    beam_file : str
        Path to beam window function file.
    fwhmarcmin : float
        Beam FWHM in arcminutes for Gaussian beam approximation.
    apply_pixwin : bool
        Whether to apply HEALPix pixel window functions.
    smooth_pol : bool
        Whether to apply smoothing to polarization fields.
    calibration : float
        Overall calibration factor for maps.
    load_reduced : bool
        If True, read noise covariance matrices that have already been reduced
        to active pixels (via ``read_covmat_reduced``) instead of extracting
        the active-pixel sub-block from a full-sky covariance (via ``read_covmat``).
        Default is False.
    ordering : str
        HEALPix map ordering: ``"RING"`` or ``"NESTED"``.
    input_convention : str
        Power spectrum convention of input files: ``"Cl"`` (default) or ``"Dl"``.
        When set to ``"Dl"``, input spectra are converted from
        :math:`D_\ell = \ell(\ell+1) C_\ell / (2\pi)` to :math:`C_\ell` on read.
    output_convention : str
        Power spectrum convention for QML output: ``"Cl"`` (default) or ``"Dl"``.
        When set to ``"Dl"``, output spectra are converted from internal
        :math:`C_\ell` to :math:`D_\ell` before being returned.

    Derived Attributes
    ------------------
    nfields : int
        Number of fields (computed from labels).
    nspectra : int
        Number of power spectra including auto and cross correlations.
    npix : int
        Number of pixels for the given HEALPix resolution.
    cross_idxs : numpy.ndarray
        Indices for cross-correlation spectra.
    auto_idxs : numpy.ndarray
        Indices for auto-correlation spectra.

    Examples
    --------
    Create default parameters:

    >>> params = InputParams()
    >>> print(params.nside, params.lmax)
    16 64

    Load from YAML file:

    >>> params = InputParams.read_parameter_file('config.yaml')

    Update specific parameters:

    >>> params.update({'nside': 32, 'lmax': 128})
    >>> print(params.npix)  # Automatically recomputed
    12288

    Field label expansion:

    >>> params.update({'physical_labels': ['QU']})
    >>> print(params.physical_labels)  # Expands to ['Q', 'U']
    ['Q', 'U']
    >>> params.update({'labels': ['T1_T2']})
    >>> print(params.labels)  # Expands to ['T1', 'T2']
    ['T1', 'T2']

    Notes
    -----
    The class automatically computes derived parameters when base parameters
    are updated. This ensures consistency between related parameters like
    nside and npix, or labels and nfields.

    **Field Label Expansion:**
    The class supports automatic expansion of concatenated field labels:

    - Single-character concatenation: "QU" -> ["Q", "U"], "TEB" -> ["T", "E", "B"]
    - Underscore separation: "T1_T2" -> ["T1", "T2"], "LOW_HIGH" -> ["LOW", "HIGH"]
    - Mixed formats: ["T", "QU", "E1_E2"] -> ["T", "Q", "U", "E1", "E2"]

    This expansion is applied automatically when updating 'labels' or
    'physical_labels' parameters, providing backward compatibility while
    supporting flexible field specification formats.
    """

    def __init__(self):
        """Initialize with default parameter values."""
        self.set_defaults()

    def set_defaults(self):
        """
        Set default values for all configuration parameters.

        Notes
        -----
        Establishes standard defaults suitable for CMB analysis:
        - HEALPix nside=16 for moderate resolution
        - Temperature and polarization fields [0, 2] spins
        - Standard CMB field labels ["T", "E", "B"]
        - Typical file paths for inputs and outputs
        - Reasonable beam and analysis parameters
        """
        self.nside = 16
        self.spins = [0, 2]  # TQU
        self.labels = ["T", "E", "B"]
        self.physical_labels = None

        self.feedback = 1
        # Enhanced logging parameters
        self.log_file = None
        self.log_format = "scientific"
        self.log_timing = True

        self.inputclfile = "inputs/cls.dat"
        self.maskfile = "inputs/mask.fits"
        self.do_cross = True
        self.covmatfile1 = "inputs/NCVM1.bin"
        # out* default to None: persistence is opt-in (ADR-0015). A write happens
        # only when the caller sets the path.
        self.outinvcovmatfile1 = None
        self.covmatfile2 = "inputs/NCVM2.bin"
        self.outinvcovmatfile2 = None
        self.outnoisecovmat1 = None
        self.outnoisecovmat2 = None
        self.calibration = 1.0
        self.load_inverted = False
        # When True, noise covariance files are already reduced to active
        # pixels and can be read directly with read_covmat_reduced(), skipping
        # the full-sky extraction step performed by read_covmat().
        self.load_reduced = False
        self.output_geometry_file = None
        self.smoothing_type = "cosine_legacy"
        self.apply_pixwin = True
        self.smooth_pol = True
        self.fwhmarcmin = 440.0
        self.beam_file = "inputs/beam.fits"
        self.lmax = 64
        self.delta_ell = 1
        self.bin_lmins = None
        self.bin_lmaxs = None
        self.outfilefisher = None
        self.ordering = "RING"

        self.nsims = None
        self.inputmapfile1 = ""
        self.inputmapfile2 = ""
        self.outcovmatfile = None
        self.outerrfile = None
        self.remove_nb = True

        self.input_convention = "Cl"  # "Cl" or "Dl"
        self.output_convention = "Cl"  # "Cl" or "Dl"

        self.parameters = {}
        self.root_dir = "inputs"
        self.root_filename = "theory_spectra"

        # Multipole-range API (see ADR 0009). Basis covers the signal-cov
        # band [min(lmin_signal), lmax_signal]; C_ell vary inside the
        # inference window [lmin, lmax]; outside this window but inside the
        # signal-cov band, the fiducial spectrum is used.
        self.lmin_signal = 2
        self.lmax_signal = None  # None resolves to 4*nside at basis-setup time
        self.lmin = 2
        self.fiducialfile = None

        self.compute_derived()

    def compute_derived(self):
        """
        Compute derived parameters from base configuration.

        Notes
        -----
        Automatically calculates:
        - nfields from the length of labels
        - nspectra for auto and cross correlations
        - npix from HEALPix nside parameter
        - cross_idxs and auto_idxs arrays for spectrum indexing

        This method is called automatically when parameters are updated
        to ensure consistency between related parameters.
        """
        self.nfields = len(self.labels)
        self.nspectra = self.nfields * (self.nfields + 1) // 2

        self.npix = hp.nside2npix(self.nside)

        self.cross_idxs = np.array(
            [
                spec2idx(spec1, spec2, self.nfields)
                for spec1 in range(self.nfields)
                for spec2 in range(spec1 + 1, self.nfields)
            ]
        )
        self.auto_idxs = np.array(
            [spec2idx(spec, spec, self.nfields) for spec in range(self.nfields)]
        )

    @staticmethod
    def _normalize_ordering(value):
        """Normalize ordering to ``"RING"`` or ``"NESTED"``.

        Accepts strings (case-insensitive) or legacy integer codes.
        Legacy integers are accepted for backward compatibility:
        the code convention is 0=RING, 1=NESTED (matching the
        ``nest`` flag in HEALPix), and existing configs use 2 for RING
        (where any value != 1 maps to RING).
        """
        if isinstance(value, str):
            canonical = {"ring": "RING", "nested": "NESTED"}
            key = value.strip().lower()
            if key not in canonical:
                raise ValueError(
                    f"Unknown ordering '{value}'. Must be 'RING' or 'NESTED'."
                )
            return canonical[key]
        if isinstance(value, (int, float)):
            import warnings

            warnings.warn(
                f"Integer ordering={value} is deprecated. "
                "Use 'RING' or 'NESTED' instead.",
                DeprecationWarning,
                stacklevel=4,
            )
            return "NESTED" if int(value) == 1 else "RING"
        raise TypeError(f"ordering must be str or int, got {type(value).__name__}")

    @staticmethod
    def _normalize_smoothing_type(value):
        """Normalize smoothing_type to a string.

        Accepts strings (case-insensitive) or legacy integer codes
        (0=none, 1=gaussian, 2=cosine_legacy, 3=file). The bare string
        ``"cosine"`` is a deprecated alias for ``"cosine_legacy"``.
        """
        if isinstance(value, str):
            canonical = {
                "none": "none",
                "gaussian": "gaussian",
                "cosine_legacy": "cosine_legacy",
                "cosine_npipe": "cosine_npipe",
                "file": "file",
            }
            key = value.strip().lower()
            if key == "cosine":
                import warnings

                warnings.warn(
                    "smoothing_type='cosine' is deprecated; "
                    "use 'cosine_legacy' (Benabed+ 2009 / Aghanim+ 2019) "
                    "or 'cosine_npipe' (Akrami+ 2020).",
                    DeprecationWarning,
                    stacklevel=4,
                )
                return "cosine_legacy"
            if key not in canonical:
                raise ValueError(
                    f"Unknown smoothing_type '{value}'. Must be one of "
                    "'none', 'gaussian', 'cosine_legacy', 'cosine_npipe', 'file'."
                )
            return canonical[key]
        if isinstance(value, (int, float)):
            import warnings

            int_to_str = {0: "none", 1: "gaussian", 2: "cosine_legacy", 3: "file"}
            iv = int(value)
            if iv not in int_to_str:
                raise ValueError(
                    f"Unknown smoothing_type={iv}. Integer codes: "
                    "0=none, 1=gaussian, 2=cosine_legacy, 3=file."
                )
            warnings.warn(
                f"Integer smoothing_type={iv} is deprecated. "
                f"Use '{int_to_str[iv]}' instead.",
                DeprecationWarning,
                stacklevel=4,
            )
            return int_to_str[iv]
        raise TypeError(f"smoothing_type must be str or int, got {type(value).__name__}")

    def _expand_concatenated_fields(self, field_labels):
        """
        Expand concatenated field labels to individual components.

        Parameters
        ----------
        field_labels : list of str
            Field labels that may contain concatenated fields like "QU" or "T1_T2".

        Returns
        -------
        list of str
            Expanded field labels with concatenated fields split into components.

        Notes
        -----
        This method detects and expands concatenated field labels:
        - Single character fields can be concatenated: "QU" -> ["Q", "U"]
        - Multi-character fields can use underscore: "T1_T2" -> ["T1", "T2"]
        - Mixed cases are handled appropriately

        The expansion follows these rules:
        1. If a field label contains underscores, split on underscores
        2. Else if a field label has multiple characters and all characters are
           valid single-character field names (T, Q, U, E, B), expand it
        3. Otherwise, keep the field label as-is

        This provides backward compatibility while supporting both concatenated
        and explicit field specifications.
        """
        expanded = []
        valid_single_chars = {"T", "Q", "U", "E", "B", "I"}  # Common CMB field names

        for label in field_labels:
            if "_" in label:
                # Underscore-separated multi-character fields: "T1_T2" -> ["T1", "T2"]
                expanded.extend(label.split("_"))
            elif len(label) > 1 and all(c in valid_single_chars for c in label):
                # Concatenated single-character fields: "QU" -> ["Q", "U"]
                expanded.extend(list(label))
            else:
                # Single field or multi-character field name without underscores
                expanded.append(label)

        return expanded

    def update(self, config_dict):
        """
        Update parameters from a configuration dictionary.

        Parameters
        ----------
        config_dict : dict
            Dictionary containing parameter names and values to update.

        Notes
        -----
        Updates any existing parameter attributes with values from the
        configuration dictionary. After updating, automatically recomputes
        derived parameters and sets physical_labels if not already defined.

        Field expansion is applied to both 'labels' and 'physical_labels' if
        they contain concatenated field specifications like "QU" -> ["Q", "U"].
        Non-existent attributes are silently ignored.
        """
        for key, value in config_dict.items():
            if hasattr(self, key):
                # Apply field expansion for field label parameters
                if key in ("labels", "physical_labels") and isinstance(value, list):
                    expanded_value = self._expand_concatenated_fields(value)
                    setattr(self, key, expanded_value)
                elif key == "ordering":
                    setattr(self, key, self._normalize_ordering(value))
                elif key == "smoothing_type":
                    setattr(self, key, self._normalize_smoothing_type(value))
                else:
                    setattr(self, key, value)

        self.compute_derived()
        if self.physical_labels is None:
            self.physical_labels = self.labels.copy()

    @staticmethod
    def read_parameter_file(yaml_file):
        """
        Load parameters from a YAML configuration file.

        Parameters
        ----------
        yaml_file : str
            Path to YAML file containing parameter configurations.

        Returns
        -------
        InputParams
            New InputParams instance with parameters loaded from file.

        Notes
        -----
        Creates a new InputParams instance, loads the YAML file, and updates
        the parameters with the file contents. Useful for loading standardized
        analysis configurations from file.
        """
        with open(yaml_file) as file:
            config = yaml.safe_load(file)

        params = InputParams()
        params.update(config)
        return params

    def __str__(self):
        """
        Return string representation of all parameters.

        Returns
        -------
        str
            Formatted string with all parameter names and values.
        """
        return "\n".join(
            f"{key}: {value}" for key, value in sorted(self.__dict__.items())
        )

    def __repr__(self):
        """Return string representation (same as __str__)."""
        return self.__str__()

    def __eq__(self, other):
        """
        Check equality with another InputParams instance.

        Parameters
        ----------
        other : object
            Object to compare with.

        Returns
        -------
        bool
            True if all parameter values are equal, False otherwise.

        Notes
        -----
        Compares all attributes for equality. Useful for testing and
        validation of parameter configurations.
        """
        if not isinstance(other, InputParams):
            return False
        return all(
            getattr(self, key) == getattr(other, key) for key in self.__dict__.keys()
        )
