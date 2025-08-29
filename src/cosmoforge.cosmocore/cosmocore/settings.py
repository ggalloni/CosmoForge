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
- Parameter validation and updates
"""

import healpy as hp
import numpy as np
import yaml

from .basics import spec2idx


class InputParams:
    """
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
    physical_labels : list of str or None
        Physical field labels (defaults to same as labels).
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
    ordering : int
        HEALPix map ordering (1=RING, 2=NESTED).

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

    Notes
    -----
    The class automatically computes derived parameters when base parameters
    are updated. This ensures consistency between related parameters like
    nside and npix, or labels and nfields.
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
        self.inputclfile = "inputs/cls.dat"
        self.maskfile = "inputs/mask.fits"
        self.do_cross = True
        self.covmatfile1 = "inputs/NCVM1.bin"
        self.outinvcovmatfile1 = "outputs/invCOV1.bin"
        self.covmatfile2 = "inputs/NCVM2.bin"
        self.outinvcovmatfile2 = "outputs/invCOV2.bin"
        self.outnoisecovmat1 = "outputs/reducedNCVM1.bin"
        self.calibration = 1.0
        self.load_inverted = False
        self.output_geometry_file = "outputs/geometry.dat"
        self.smoothing_type = 2
        self.apply_pixwin = True
        self.smooth_pol = True
        self.fwhmarcmin = 440.0
        self.beam_file = "inputs/beam.fits"
        self.lmax = 64
        self.outfilefisher = "outputs/fisher.dat"
        self.ordering = 1

        self.nsims = None
        self.inputmapfile1 = ""
        self.inputmapfile2 = ""
        self.outcovmatfile = ""
        self.outerrfile = ""
        self.remove_nb = True

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
        Non-existent attributes are silently ignored.
        """
        for key, value in config_dict.items():
            if hasattr(self, key):
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
