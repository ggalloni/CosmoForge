"""
Core analysis classes for cosmological computations.

This module provides the foundational classes for cosmological power spectrum
analysis, Fisher matrix calculations, and related computations. The Core class
serves as an abstract base class for specific analysis tools like QML and
Fisher matrix estimators.

Classes
-------
Core
    Abstract base class for cosmological analysis tools.

Notes
-----
The core module follows a clean architecture pattern with separation of
concerns between data management (fields), computation (pixel/harmonic),
and I/O operations. The Core class provides common functionality that can
be extended by concrete analysis implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import healpy as hp
import numpy as np

from .fields import (
    BaseField,
    FieldCollection,
    create_field,
)
from .in_out import output_geometry, read_covmat, read_mask, write_covmat_reduced
from .pixel import compute_pointings
from .settings import InputParams


class Core(ABC):
    """
    Abstract base class for cosmological analysis tools.

    This class provides common functionality for Fisher matrix calculations,
    power spectrum analysis, and other cosmological computations. It handles
    parameter management, field setup, and coordinate system operations.

    Parameters
    ----------
    params : InputParams or str or dict or None, optional
        Analysis parameters. Can be:
        - InputParams instance: Use directly
        - str: Path to parameter file to load
        - dict: Parameter dictionary to initialize InputParams
        - None: Use default parameters

    Attributes
    ----------
    params : InputParams
        Analysis parameters and configuration.

    Notes
    -----
    This is an abstract class that cannot be instantiated directly. Use
    concrete subclasses that implement the required abstract methods for
    specific analysis tasks.
    """

    def __init__(
        self,
        params: InputParams | str | dict | None = None,
    ):
        """
        Initialize the core analysis framework.

        Parameters
        ----------
        params : InputParams or str or dict or None, optional
            Analysis parameters in various formats.
        """
        self.read_params(params)

    def read_params(self, params: InputParams | str | dict):
        """
        Read and validate analysis parameters.

        Parameters
        ----------
        params : InputParams or str or dict
            Parameters in various formats:
            - InputParams: Use directly
            - str: Path to parameter file
            - dict: Dictionary to initialize InputParams

        Raises
        ------
        TypeError
            If params format is not recognized.

        Notes
        -----
        This method provides flexible parameter input handling for different
        use cases (script files, interactive sessions, programmatic access).
        """
        if isinstance(params, InputParams):
            self.params = params
        elif isinstance(params, str):
            self.params = InputParams.read_parameter_file(params)
        elif isinstance(params, dict):
            self.params = InputParams()
            self.params.update(params)
        else:
            msg = (
                "params must be an instance of InputParams, "
                "a string with the path to a parameter file, "
                "or a dictionary with parameters."
            )
            raise TypeError(msg)

    def setup_fields(self) -> FieldCollection:
        """
        Set up cosmological fields using the new clean architecture.

        Returns
        -------
        FieldCollection
            Collection of fields configured according to analysis parameters.

        Notes
        -----
        This method:
        1. Loads masks from file using the parameters
        2. Creates appropriate field types (scalar/polarization) based on spin
        3. Assembles fields into a collection for joint analysis

        The field creation uses type-safe factory functions to ensure
        proper initialization based on the spin parameter.
        """
        npix = hp.nside2npix(self.params.nside)
        mask = np.empty((self.params.nfields, npix), dtype=np.float64)
        mask = read_mask(self.params.maskfile, mask)

        if len(mask.shape) == 1:
            mask = mask[:, np.newaxis]

        # Create fields using the new factory function
        fields: list[BaseField] = []
        counter = 0
        for spin in self.params.spins:
            if spin == 0:
                labels = self.params.labels[counter]
                counter += 1
            else:
                labels = [self.params.labels[counter], self.params.labels[counter + 1]]
                counter += 2

            # Use new factory function for type-safe field creation
            field = create_field(
                spin=spin,
                nside=self.params.nside,
                lmax=self.params.lmax,
                mask=mask[:, counter - 1],
                labels=labels,
            )
            fields.append(field)

        # Create collection using new design
        self.collection = FieldCollection(self.params, fields)

        return self.collection

    def setup_geometry(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Set up geometry including active pixels and pointing vectors.

        Returns
        -------
        tuple of numpy.ndarray
            Tuple containing (active_pixels, pointing_vectors) where:
            - active_pixels: Object array of active pixel indices per field component
            - pointing_vectors: Tuple of unit vector arrays for each component

        Raises
        ------
        ValueError
            If fields have not been set up before calling this method.

        Notes
        -----
        This method:
        1. Extracts active pixel counts from each field
        2. Computes 3D pointing vectors for each active pixel
        3. Sets pointing vectors in the field collection
        4. Optionally outputs geometry data to file

        The pointing vectors are unit vectors in 3D space pointing to pixel
        centers, used for spherical harmonic transforms and coordinate operations.
        """
        if self.collection is None:
            raise ValueError("Fields must be set up before geometry")

        # Get active pixels
        self.npixs = []
        for lf in self.collection.fields:
            self.npixs += lf.n_active

        self.point_vectors = tuple(
            np.empty((self.npixs[i], 3), dtype=np.float64) for i in range(len(self.npixs))
        )

        self.pixact = self.collection.get_active_pixels()

        # Compute pointing vectors
        self.point_vectors = compute_pointings(
            self.params.nside,
            self.npixs,
            self.point_vectors,
            self.pixact,
            self.params.ordering,
        )

        # Set pointing vectors in collection
        self.collection.set_pointing_vectors(self.point_vectors)

        # Output geometry if requested
        if hasattr(self.params, "output_geometry_file"):
            output_geometry(
                self.params.output_geometry_file,
                self.npixs,
                self.point_vectors,
                self.pixact,
            )

        return self.pixact, self.point_vectors

    def setup_covariance_matrices(self) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Set up noise covariance matrices for the analysis.

        Returns
        -------
        tuple of numpy.ndarray
            Tuple containing (primary_covariance, secondary_covariance) where:
            - primary_covariance: Main noise covariance matrix
            - secondary_covariance: Optional second covariance matrix for cross-analysis

        Raises
        ------
        ValueError
            If geometry has not been set up before calling this method.

        Notes
        -----
        This method:
        1. Reads noise covariance matrices from file
        2. Applies calibration corrections
        3. Optionally outputs reduced covariance matrices
        4. Sets up secondary covariance for cross-correlation analysis if enabled

        The covariance matrices are essential for proper weighting in maximum
        likelihood and Fisher matrix calculations.
        """
        if self.pixact is None:
            raise ValueError("Geometry must be set up before covariance matrices")

        npix = hp.nside2npix(self.params.nside)
        concatenate_pixact = np.concatenate(
            [self.pixact[i] + i * npix for i in range(len(self.pixact))]
        )

        shape = (concatenate_pixact.shape[0], concatenate_pixact.shape[0])
        self.NCov1 = np.empty(shape, dtype=np.float64)

        self.NCov1 = (
            read_covmat(
                self.params.covmatfile1,
                npix,
                self.params.nfields,
                concatenate_pixact,
                self.NCov1,
            )
            * self.params.calibration**2
        )

        if hasattr(self.params, "outnoisecovmat1"):
            write_covmat_reduced(
                self.params.outnoisecovmat1,
                self.NCov1,
            )

        self.NCov2 = None
        if self.params.do_cross:
            self.NCov2 = np.empty(shape, dtype=np.float64)
            self.NCov2 = (
                read_covmat(
                    self.params.covmatfile2,
                    npix,
                    self.params.nfields,
                    concatenate_pixact,
                    self.NCov2,
                )
                * self.params.calibration**2
            )

            if hasattr(self.params, "outnoisecovmat2"):
                write_covmat_reduced(
                    self.params.outnoisecovmat2,
                    self.NCov2,
                )

        return self.NCov1, self.NCov2

    def setup_cls(self):
        """
        Set up power spectra using the new field design.

        Notes
        -----
        This method configures theoretical power spectra for the field
        collection, loading from file and applying necessary normalizations.

        Raises
        ------
        ValueError
            If fields have not been set up before calling this method.
        """
        if self.collection is None:
            raise ValueError("Fields must be set up before Cls and beams")
        self.collection.set_cls()

    def setup_beams(self):
        """
        Set up beam functions for each field using the new design.

        Notes
        -----
        This method configures beam window functions for instrumental
        effects correction in harmonic space analysis.

        Raises
        ------
        ValueError
            If fields have not been set up before calling this method.
        """
        if self.collection is None:
            raise ValueError("Fields must be set up before Cls and beams")
        self.collection.set_beams()

    def log(self, message: str, level: int = 1):
        """
        Log a message based on feedback level.

        Parameters
        ----------
        message : str
            Message to log to console.
        level : int, optional
            Minimum feedback level required to display message. Default is 1.

        Notes
        -----
        Messages are only displayed if the analysis parameters include a
        feedback level greater than or equal to the specified level.
        This allows for controllable verbosity in analysis runs.
        """
        if hasattr(self.params, "feedback") and self.params.feedback >= level:
            print(message)

    @abstractmethod
    def compute(self):
        """
        Abstract method for performing the main computation.

        Notes
        -----
        This method must be implemented by concrete subclasses to define
        the specific analysis computation (e.g., Fisher matrix calculation,
        QML power spectrum estimation).

        Raises
        ------
        NotImplementedError
            Always raised in the base class to enforce implementation in subclasses.
        """
        raise NotImplementedError("Subclasses must implement 'compute' method")

    @abstractmethod
    def run(self):
        """
        Abstract method for running the full analysis pipeline.

        Notes
        -----
        This method must be implemented by concrete subclasses to define
        the complete analysis workflow from setup to final output.

        Raises
        ------
        NotImplementedError
            Always raised in the base class to enforce implementation in subclasses.
        """
        raise NotImplementedError("Subclasses must implement 'run' method")
