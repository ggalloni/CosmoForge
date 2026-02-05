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

from .basics import matrix_inverse_symm, matrix_slogdet_symm
from .compression import CompressionManager
from .fields import (
    BaseField,
    FieldCollection,
    create_field,
)
from .in_out import output_geometry, read_covmat, read_mask, readcl, write_covmat_reduced
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

        # Initialize enhanced logger
        from .logger import get_logger_from_params

        self.logger = get_logger_from_params(
            self.params, name=self.__class__.__name__.lower()
        )

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
            raise TypeError(
                "params must be an instance of InputParams, "
                "a string with the path to a parameter file, "
                "or a dictionary with parameters."
            )

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
        self.collection = FieldCollection(self.params, fields, self.logger)

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
        3. Computes theta (colatitude) and phi (longitude) for compression support
        4. Sets pointing vectors in the field collection
        5. Optionally outputs geometry data to file

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
        self.theta_vectors = tuple(
            np.empty((self.npixs[i]), dtype=np.float64) for i in range(len(self.npixs))
        )
        self.phi_vectors = tuple(
            np.empty((self.npixs[i]), dtype=np.float64) for i in range(len(self.npixs))
        )

        self.pixact = self.collection.get_active_pixels()

        # Compute pointing vectors
        self.point_vectors, self.theta, self.phi = compute_pointings(
            self.params.nside,
            self.npixs,
            self.point_vectors,
            self.theta_vectors,
            self.phi_vectors,
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

    def setup_cls(self, lmax: int | None = None):
        """
        Set up power spectra using the new field design.

        Parameters
        ----------
        lmax : int, optional
            Maximum multipole to use. If None, uses core_params.lmax.
            This allows loading Cls up to a different lmax than the analysis lmax.

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
        self.collection.set_cls(lmax=lmax)

    def setup_beams(self, lmax: int | None = None):
        """
        Set up beam functions for each field using the new design.

        Parameters
        ----------
        lmax : int, optional
            Maximum multipole to use. If None, uses core_params.lmax.
            This allows computing beams up to a different lmax than the analysis lmax.

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
        self.collection.set_beams(lmax=lmax)

    def setup_compression(
        self,
        method: str = "harmonic",
        epsilon: float | None = 1e-6,
        mode_fraction: float | None = None,
        beam: np.ndarray | None = None,
        basis: str = "noise_weighted",
        C_ell: np.ndarray | None = None,
        lmax: int | None = None,
        use_smw_optimization: bool = True,
    ):
        """
        Create and configure a CompressionManager for SMW-based operations.

        This method sets up the Sherman-Morrison-Woodbury compression framework
        for efficient covariance matrix operations. It requires that geometry
        and covariance matrices have already been set up.

        For harmonic compression, this method automatically enables SMW
        optimization: signal from multipoles ℓ > params.lmax is absorbed into
        an effective noise term, reducing the harmonic subspace dimension from
        (4*nside+1)² - 4 modes to (params.lmax+1)² - 4 modes while preserving
        numerical accuracy for the estimated multipoles.

        Parameters
        ----------
        method : str, default "harmonic"
            Compression method to use:
            - "harmonic": Tegmark-style direct harmonic transformation
            - "pixel_projected": Gjerløw-style pixel-space projector
        epsilon : float or None, optional
            Eigenvalue threshold for mode compression. Modes with eigenvalue
            < epsilon * max_eigenvalue are discarded. Default is 1e-6.
        lmax : int or None, optional
            Maximum multipole for harmonic expansion. If None, defaults to
            4 * nside to match the traditional signal matrix computation.
            Mutually exclusive with mode_fraction.
        mode_fraction : float or None, optional
            Fraction of modes to keep (between 0 and 1). Keeps the top modes
            ordered by eigenvalue. Mutually exclusive with epsilon.
        beam : numpy.ndarray or None, optional
            Beam window function B_ℓ for ℓ=2 to lmax. Shape should be (lmax-1,).
            If None and beams have been set up via setup_beams(), the first
            field's beam is automatically extracted from the beam manager.
        basis : str, default "noise_weighted"
            Compression basis for pixel_projected method. Options:
            "harmonic", "noise_weighted", "total_covariance", "snr".
        C_ell : numpy.ndarray or None, optional
            Power spectrum for bases that require it ("total_covariance", "snr").
        use_smw_optimization : bool, default True
            For harmonic compression, whether to use the SMW optimization that
            absorbs high-ℓ signal (ℓ > params.lmax) into effective noise. This
            reduces computation while preserving accuracy for estimated multipoles.

        Returns
        -------
        CompressionManager
            Configured compression manager ready for use.

        Raises
        ------
        ValueError
            If geometry or covariance matrices have not been set up.

        Examples
        --------
        >>> core = SomeConcreteCore("config.yaml")
        >>> core.setup_fields()
        >>> core.setup_geometry()
        >>> core.setup_covariance_matrices()
        >>> core.setup_beams()
        >>> # Harmonic compression (default)
        >>> cm = core.setup_compression(method="harmonic")
        >>> # Pixel-projected with SNR basis
        >>> cm = core.setup_compression(
        ...     method="pixel_projected",
        ...     basis="snr",
        ...     C_ell=C_ell,
        ...     epsilon=1e-4,
        ... )
        """
        if not hasattr(self, "theta") or self.theta is None:
            raise ValueError(
                "Geometry must be set up before compression. Call setup_geometry() first."
            )
        if not hasattr(self, "NCov1") or self.NCov1 is None:
            raise ValueError(
                "Covariance matrices must be set up before compression. "
                "Call setup_covariance_matrices() first."
            )

        compression_lmax = lmax if lmax is not None else 4 * self.params.nside

        # Extract beam from field collection if not provided
        if beam is None and hasattr(self, "collection") and self.collection is not None:
            beam_dict = self.collection.beam_manager.get_beam_dict()
            first_label = self.collection.fields[0].labels[0]
            beam = beam_dict[first_label]

        # Truncate beam to match compression_lmax (beam is for ell=2 to lmax)
        expected_beam_len = compression_lmax - 1
        if beam is not None and len(beam) > expected_beam_len:
            beam = beam[:expected_beam_len]

        # Extract theta/phi for first field
        theta_arr = self.theta[0] if isinstance(self.theta, tuple) else self.theta
        phi_arr = self.phi[0] if isinstance(self.phi, tuple) else self.phi

        # Compression only supports single-field
        n_fields = len(self.collection.fields) if hasattr(self, "collection") else 1
        if n_fields > 1:
            raise ValueError(
                f"Compression currently only supports single-field analysis, "
                f"but {n_fields} fields were found. "
                f"Use traditional method for multi-field cases."
            )

        # SMW optimization: absorb high-ℓ signal into effective noise
        lswitch_low = None
        lswitch_high = None
        S_fixed = None

        if use_smw_optimization and method == "harmonic":
            config_lswitch_low = getattr(self.params, "lswitch_low", None)
            config_lswitch_high = getattr(self.params, "lswitch_high", None)

            if config_lswitch_low is not None and config_lswitch_high is not None:
                # Explicit config (PICSLIKE): use fiducialfile
                lswitch_low = config_lswitch_low
                lswitch_high = config_lswitch_high
                fiducial_file = getattr(self.params, "fiducialfile", None)
            else:
                # Automatic (QML): use params.lmax as subspace limit
                params_lmax = getattr(self.params, "lmax", None)
                if params_lmax is not None and params_lmax < compression_lmax:
                    lswitch_low = 2
                    lswitch_high = params_lmax
                    fiducial_file = getattr(self.params, "inputclfile", None)
                else:
                    fiducial_file = None

            # Compute S_fixed for ℓ > lswitch_high
            if lswitch_high is not None and lswitch_high < compression_lmax:
                has_coll = hasattr(self, "collection") and self.collection is not None
                if fiducial_file is not None and has_coll:
                    fiducial_spectrum = readcl(
                        inputclfile=fiducial_file.strip(),
                        Params=self.params,
                        lmax=compression_lmax,
                    )

                    # Zero for ℓ ≤ lswitch_high, fiducial for ℓ > lswitch_high
                    fixed_spectra = {}
                    for key, cl_array in fiducial_spectrum.items():
                        cl_fixed = np.zeros_like(cl_array)
                        for ell in range(lswitch_high + 1, compression_lmax + 1):
                            if ell - 2 < len(cl_array):
                                cl_fixed[ell - 2] = cl_array[ell - 2]
                        fixed_spectra[key] = cl_fixed

                    # Save original spectra (already beam-smoothed)
                    original_spectra_smoothed = {
                        k: v.copy()
                        for k, v in self.collection.spectra_manager._cls_dict.items()
                    }

                    self.collection.set_cls(fixed_spectra, lmax=compression_lmax)
                    self.collection.beam_manager.apply_smoothing(
                        self.collection.spectra_manager, lmax=compression_lmax
                    )

                    from .pixel import compute_signal_matrix as _compute_signal_matrix

                    S_fixed = np.zeros_like(self.NCov1, dtype=np.float64)
                    _compute_signal_matrix(
                        S=S_fixed,
                        lmax=compression_lmax,
                        fields=self.collection,
                    )

                    # Restore original spectra (already smoothed - don't re-apply beam)
                    for key, value in original_spectra_smoothed.items():
                        self.collection.spectra_manager._cls_dict[key] = value

        # Create compression manager
        self.compression_manager = CompressionManager(
            N=self.NCov1,
            N_inv=matrix_inverse_symm(self.NCov1),
            theta=theta_arr,
            phi=phi_arr,
            lmax=compression_lmax,
            method=method,
            beam=beam,
            basis=basis,
            C_ell=C_ell,
            epsilon=epsilon,
            mode_fraction=mode_fraction,
            lswitch_low=lswitch_low,
            lswitch_high=lswitch_high,
            S_fixed=S_fixed,
        )

        # Build harmonic operator and precompute SMW components
        self.compression_manager.setup()

        return self.compression_manager

    # =========================================================================
    # Unified Covariance API
    # =========================================================================
    # These methods provide a compression-agnostic interface. Subclasses use
    # these methods without knowing whether compression is enabled.
    # =========================================================================

    def get_total_covariance(self, C_ell: np.ndarray) -> np.ndarray:
        """
        Get total covariance matrix C = N + S.

        If compression is enabled, returns the compressed covariance.
        Otherwise, returns the full pixel-space covariance.

        Parameters
        ----------
        C_ell : numpy.ndarray
            Power spectrum values for ell = 2 to lmax.

        Returns
        -------
        numpy.ndarray
            Total covariance matrix (compressed or full).
        """
        if hasattr(self, "compression_manager") and self.compression_manager is not None:
            return self.compression_manager.get_compressed_covariance(C_ell)
        else:
            return self.NCov1 + self._build_signal_matrix(C_ell)

    def get_covariance_inverse(self, C_ell: np.ndarray) -> np.ndarray:
        """
        Get inverse covariance matrix C^{-1}.

        If compression is enabled, returns the compressed inverse.
        Otherwise, computes the full inverse.

        Parameters
        ----------
        C_ell : numpy.ndarray
            Power spectrum values for ell = 2 to lmax.

        Returns
        -------
        numpy.ndarray
            Inverse covariance matrix (compressed or full).
        """
        if hasattr(self, "compression_manager") and self.compression_manager is not None:
            return self.compression_manager.get_compressed_inverse(C_ell)
        else:
            return matrix_inverse_symm(self.get_total_covariance(C_ell), overwrite=True)

    def get_covariance_logdet(self, C_ell: np.ndarray) -> float:
        """
        Get log determinant of covariance matrix log|C|.

        If compression is enabled, uses SMW formula for correct result.
        Otherwise, computes directly.

        Parameters
        ----------
        C_ell : numpy.ndarray
            Power spectrum values for ell = 2 to lmax.

        Returns
        -------
        float
            Log determinant of covariance matrix.
        """
        if hasattr(self, "compression_manager") and self.compression_manager is not None:
            return self.compression_manager.get_full_logdet(C_ell)
        else:
            _, logdet = matrix_slogdet_symm(self.get_total_covariance(C_ell))
            return logdet

    def get_derivative_matrix(self, ell: int) -> np.ndarray:
        """
        Get derivative matrix dC/dC_ell.

        If compression is enabled, returns compressed derivative.
        Otherwise, returns full pixel-space derivative.

        Parameters
        ----------
        ell : int
            Multipole for derivative.

        Returns
        -------
        numpy.ndarray
            Derivative matrix dC/dC_ell.
        """
        if hasattr(self, "compression_manager") and self.compression_manager is not None:
            return self.compression_manager.get_derivative_matrix(ell)
        else:
            return self._build_derivative_matrix(ell)

    def compute_quadratic_form(self, data: np.ndarray, C_ell: np.ndarray) -> float:
        """
        Compute quadratic form d^T C^{-1} d.

        If compression is enabled, uses efficient SMW-based computation.
        Otherwise, computes directly with full matrices.

        Parameters
        ----------
        data : numpy.ndarray
            Data vector in pixel space.
        C_ell : numpy.ndarray
            Power spectrum values for ell = 2 to lmax.

        Returns
        -------
        float
            Quadratic form value d^T C^{-1} d.
        """
        if hasattr(self, "compression_manager") and self.compression_manager is not None:
            return self.compression_manager.compute_quadratic_form(data, C_ell)
        else:
            C_inv = self.get_covariance_inverse(C_ell)
            return float(data.T @ C_inv @ data)

    def _build_signal_matrix(self, C_ell: np.ndarray) -> np.ndarray:
        """Build signal covariance matrix. Subclasses must override."""
        raise NotImplementedError("Subclasses must implement _build_signal_matrix")

    def _build_derivative_matrix(self, ell: int) -> np.ndarray:
        """Build derivative matrix dC/dC_ell. Subclasses must override."""
        raise NotImplementedError("Subclasses must implement _build_derivative_matrix")

    def log(self, message: str, level: int = 1):
        """
        Log a message based on feedback level (backward compatibility).

        Parameters
        ----------
        message : str
            Message to log to console.
        level : int, optional
            Minimum feedback level required to display message. Default is 1.

        Notes
        -----
        This method maintains backward compatibility with the existing feedback
        system while using the enhanced logger infrastructure. Messages are
        logged with appropriate formatting and can be output to both console
        and file if configured.
        """
        self.logger.log_with_feedback(message, level)

    @abstractmethod
    def compute(self):
        """Perform the main computation. Subclasses must implement."""
        raise NotImplementedError("Subclasses must implement 'compute' method")

    @abstractmethod
    def run(self):
        """Run the full analysis pipeline. Subclasses must implement."""
        raise NotImplementedError("Subclasses must implement 'run' method")
