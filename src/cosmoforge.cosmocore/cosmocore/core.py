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
from .basis import create_computation_basis
from .fields import (
    BaseField,
    FieldCollection,
    create_field,
)
from .in_out import (
    output_geometry,
    read_covmat,
    read_covmat_reduced,
    read_mask,
    readcl,
    write_covmat_reduced,
)
from .pixel import compute_pointings
from .settings import InputParams


def _build_fixed_spectra(
    fiducial_spectrum: dict[str, np.ndarray],
    spectra_map: dict[tuple[int, int, int], str],
    lmin_signal: list[int],
    lmin_b: int,
    lmax_b: int,
    basis_lmax: int,
) -> dict[str, np.ndarray]:
    """Filter a fiducial spectrum into the ``S_fixed`` low + high bands.

    For each spectrum label, the low band is
    ``[max(lmin_signal[i], lmin_signal[j]), lmin_b)`` and the high band is
    ``(lmax_b, basis_lmax]``. The per-pair floor guards against an
    unphysical fiducial entry below a component's spin floor leaking into
    ``S_fixed`` when ``lmin_signal`` is heterogeneous (ADR 0009).
    """
    label_to_pair = {label: (i, j) for (i, j, _), label in spectra_map.items()}
    fallback_floor = min(lmin_signal) if lmin_signal else 0
    fixed_spectra: dict[str, np.ndarray] = {}
    for label, cl_array in fiducial_spectrum.items():
        cl_fixed = np.zeros_like(cl_array)
        n = len(cl_array)
        pair = label_to_pair.get(label)
        if pair is not None:
            i, j = pair
            low_floor = max(lmin_signal[i], lmin_signal[j])
        else:
            low_floor = fallback_floor
        for ell in range(low_floor, lmin_b):
            if ell < n:
                cl_fixed[ell] = cl_array[ell]
        for ell in range(lmax_b + 1, basis_lmax + 1):
            if ell < n:
                cl_fixed[ell] = cl_array[ell]
        fixed_spectra[label] = cl_fixed
    return fixed_spectra


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

        self._normalize_lmin_signal()
        self._validate_multipole_range()

    def _normalize_lmin_signal(self) -> None:
        """Broadcast scalar ``lmin_signal`` to a per-component list.

        Spin-floor validation runs after broadcast so a scalar value below
        ``|spin|`` for any component fails with an actionable message
        rather than passing silently.
        """
        spins = self.params.spins
        lmin_signal = self.params.lmin_signal
        if isinstance(lmin_signal, int):
            lmin_signal = [lmin_signal] * len(spins)
        elif len(lmin_signal) != len(spins):
            raise ValueError(
                f"len(lmin_signal)={len(lmin_signal)} != len(spins)={len(spins)}"
            )
        for i, (lm, s) in enumerate(zip(lmin_signal, spins)):
            if lm < abs(s):
                raise ValueError(
                    f"lmin_signal[{i}]={lm} invalid for spin-{s} component "
                    f"at index {i}; minimum allowed is {abs(s)}"
                )
        self.params.lmin_signal = list(lmin_signal)

    def _validate_multipole_range(self) -> None:
        """Enforce ``max(lmin_signal) <= lmin <= lmax <= lmax_signal``.

        ``lmax_signal`` resolves to ``4*nside`` when None (matches the
        ``Fisher.lmax_signal`` / ``PICSLike.lmax_signal`` properties).
        The ``InputParams`` default ``lmax=64`` predates this validator
        and may exceed the signal-cov ceiling for small nside; in that
        case we clamp to ``lmax_signal`` so existing analyses keep
        working without an explicit YAML override. Explicit conflicts
        between ``lmin`` and ``lmax`` still raise.
        """
        lmin_signal_max = max(self.params.lmin_signal)
        lmin = self.params.lmin
        lmax = self.params.lmax
        lmax_signal = self.params.lmax_signal
        if lmax_signal is None:
            lmax_signal = 4 * self.params.nside
        if lmin < lmin_signal_max:
            raise ValueError(f"lmin={lmin} < max(lmin_signal)={lmin_signal_max}")
        if lmax is not None and lmax > lmax_signal:
            self.params.lmax = lmax_signal
            lmax = lmax_signal
        if lmax is not None and lmax < lmin:
            raise ValueError(f"lmax={lmax} < lmin={lmin}")

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
        3. Computes theta (colatitude) and phi (longitude) for computation basis
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
        self.noise_cov1 = np.empty(shape, dtype=np.float64)

        if self.params.load_reduced:
            self.noise_cov1 = (
                read_covmat_reduced(
                    self.params.covmatfile1,
                    self.noise_cov1,
                )
                * self.params.calibration**2
            )
        else:
            self.noise_cov1 = (
                read_covmat(
                    self.params.covmatfile1,
                    npix,
                    self.params.nfields,
                    concatenate_pixact,
                    self.noise_cov1,
                )
                * self.params.calibration**2
            )

        if hasattr(self.params, "outnoisecovmat1"):
            write_covmat_reduced(
                self.params.outnoisecovmat1,
                self.noise_cov1,
            )

        self.noise_cov2 = None
        if self.params.do_cross:
            self.noise_cov2 = np.empty(shape, dtype=np.float64)
            if self.params.load_reduced:
                self.noise_cov2 = (
                    read_covmat_reduced(
                        self.params.covmatfile2,
                        self.noise_cov2,
                    )
                    * self.params.calibration**2
                )
            else:
                self.noise_cov2 = (
                    read_covmat(
                        self.params.covmatfile2,
                        npix,
                        self.params.nfields,
                        concatenate_pixact,
                        self.noise_cov2,
                    )
                    * self.params.calibration**2
                )

            if hasattr(self.params, "outnoisecovmat2"):
                write_covmat_reduced(
                    self.params.outnoisecovmat2,
                    self.noise_cov2,
                )

        return self.noise_cov1, self.noise_cov2

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

    def setup_computation_basis(
        self,
        method: str = "auto",
        epsilon: float | list[float | tuple[float, float]] | None = 1e-6,
        mode_fraction: float | list[float | tuple[float, float]] | None = None,
        beam: np.ndarray | None = None,
        basis: str = "noise_weighted",
        C_ell: np.ndarray | None = None,
        lmax_signal: int | None = None,
        use_smw_optimization: bool = True,
        compress: bool = False,
        delta_m: int = 0,
        use_direct: bool = False,
    ):
        """
        Create and configure a computation basis for SMW-based operations.

        Sets up the Sherman-Morrison-Woodbury framework for efficient
        covariance operations. Geometry and covariance matrices must be
        set up first. The harmonic path enables the SMW switch
        optimisation when ``params.lmax < lmax_signal`` — signal from
        multipoles outside the inference window ``[lmin, lmax]`` is
        absorbed into an effective noise term, reducing the harmonic
        subspace dimension while preserving accuracy on the inferred
        multipoles (ADR 0009).

        Parameters
        ----------
        method : str, default "harmonic"
            Computation basis to use:
            - "harmonic": Tegmark-style direct harmonic transformation
            - "pixel": Gjerløw-style pixel-space projector with eigenvalue truncation
        epsilon : float or None, optional
            Eigenvalue threshold for pixel basis. Modes with eigenvalue
            < epsilon * max_eigenvalue are discarded. Default is 1e-6.
        lmax_signal : int or None, optional
            Signal-cov ceiling. If None, defaults to ``params.lmax_signal``
            or ``4 * nside``.
        mode_fraction : float or None, optional
            Fraction of modes to keep (between 0 and 1). Keeps the top modes
            ordered by eigenvalue. Mutually exclusive with epsilon.
        beam : numpy.ndarray or None, optional
            Beam window function B_ℓ for ℓ=0..lmax_signal. If None and
            beams have been set up via setup_beams(), the first field's
            beam is automatically extracted from the beam manager.
        basis : str, default "noise_weighted"
            Eigenvalue basis for pixel method. Options:
            "harmonic", "noise_weighted", "total_covariance", "snr".
        C_ell : numpy.ndarray or None, optional
            Power spectrum for bases that require it ("total_covariance", "snr").
        use_smw_optimization : bool, default True
            For harmonic basis, whether to absorb signal outside the
            inference window into effective noise. Reduces computation
            while preserving accuracy.
        use_direct : bool, default False
            For ``method="pixel"`` only: bypass the V-projection / eigenvalue
            truncation and run the direct pixel-space pipeline (no harmonic
            machinery, full $n_\\mathrm{pix}$-dimensional matrices). This is
            the same internal path that ``method="auto"`` selects when the
            cost model picks pixel-direct, exposed here for explicit user
            control. Ignored for ``method="harmonic"``.

        Returns
        -------
        ComputationBasis
            Configured computation basis instance.

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
        >>> # Harmonic basis (default)
        >>> cm = core.setup_computation_basis(method="harmonic")
        >>> # Pixel basis with SNR basis
        >>> cm = core.setup_computation_basis(
        ...     method="pixel",
        ...     basis="snr",
        ...     C_ell=C_ell,
        ...     epsilon=1e-4,
        ... )
        """
        if not hasattr(self, "theta") or self.theta is None:
            raise ValueError(
                "Geometry must be set up before computation basis. "
                "Call setup_geometry() first."
            )
        if not hasattr(self, "noise_cov1") or self.noise_cov1 is None:
            raise ValueError(
                "Covariance matrices must be set up before computation basis. "
                "Call setup_covariance_matrices() first."
            )

        if lmax_signal is None:
            lmax_signal = getattr(self.params, "lmax_signal", None)
        basis_lmax = lmax_signal if lmax_signal is not None else 4 * self.params.nside

        # Extract beam from field collection if not provided
        if beam is None and hasattr(self, "collection") and self.collection is not None:
            beam_dict = self.collection.beam_manager.get_beam_dict()
            first_label = self.collection.fields[0].labels[0]
            beam = beam_dict[first_label]

        # Truncate beam to match basis_lmax (beam is ℓ-indexed, length lmax+1)
        expected_beam_len = basis_lmax + 1
        if beam is not None and len(beam) > expected_beam_len:
            beam = beam[:expected_beam_len]

        # Pass theta/phi as tuples (ComputationBasis handles normalization)
        theta_arr = self.theta
        phi_arr = self.phi

        # Extract spin information from field collection
        spins = None
        if hasattr(self, "collection") and self.collection is not None:
            spins = [field.spin for field in self.collection.fields]

        # SMW optimization: signal contributions outside the inference window
        # [lmin, lmax] are absorbed into a fixed offset S_fixed (ADR 0009).
        # Both harmonic and pixel-V paths benefit — V is built only over the
        # active inference window. Pixel-direct mode operates on full
        # pixel-space matrices and skips this optimisation entirely.
        lmin_b = None
        lmax_b = None
        S_fixed = None

        if use_smw_optimization:
            params_lmin = getattr(self.params, "lmin", 2)
            params_lmax = getattr(self.params, "lmax", None)
            if params_lmax is None:
                params_lmax = basis_lmax
            fiducial_file = getattr(self.params, "fiducialfile", None) or getattr(
                self.params, "inputclfile", None
            )

            lmin_signal_min = min(self.params.lmin_signal)
            if params_lmin > lmin_signal_min or params_lmax < basis_lmax:
                lmin_b = params_lmin
                lmax_b = params_lmax

            # Compute S_fixed when the inference window is strictly narrower
            # than the signal-cov band on either side.
            if lmin_b is not None and (lmin_b > lmin_signal_min or lmax_b < basis_lmax):
                has_coll = hasattr(self, "collection") and self.collection is not None
                if fiducial_file is not None and has_coll:
                    fiducial_spectrum = readcl(
                        inputclfile=fiducial_file.strip(),
                        Params=self.params,
                        lmax=basis_lmax,
                    )

                    # ADR 0009 §"S_fixed accumulates both bands": the low
                    # band is [max(lmin_signal[i], lmin_signal[j]), lmin)
                    # per pair, the high band is (lmax, lmax_signal]. The
                    # per-pair low-band floor protects against an
                    # unphysical fiducial entry (e.g. cl_EE[1]) leaking into
                    # S_fixed when lmin_signal is heterogeneous.
                    fixed_spectra = _build_fixed_spectra(
                        fiducial_spectrum,
                        self.collection.spectra_manager._spectra_map,
                        list(self.params.lmin_signal),
                        lmin_b,
                        lmax_b,
                        basis_lmax,
                    )

                    # Save original (already beam-smoothed) spectra; restore
                    # under finally so a raise during S_fixed assembly does
                    # not leave the collection holding the zero-inside-window
                    # spectra for the rest of the analysis.
                    original_spectra_smoothed = {
                        k: v.copy()
                        for k, v in self.collection.spectra_manager._cls_dict.items()
                    }

                    try:
                        self.collection.set_cls(fixed_spectra, lmax=basis_lmax)
                        self.collection.beam_manager.apply_smoothing(
                            self.collection.spectra_manager, lmax=basis_lmax
                        )

                        from .pixel import compute_signal_matrix as _compute_signal_matrix

                        S_fixed = np.zeros_like(self.noise_cov1, dtype=np.float64)
                        _compute_signal_matrix(
                            S=S_fixed,
                            lmax=basis_lmax,
                            fields=self.collection,
                        )
                    finally:
                        # Restore original (smoothed) spectra; do NOT re-apply
                        # the beam — the saved copy was already smoothed.
                        self.collection.spectra_manager._cls_dict = (
                            original_spectra_smoothed
                        )

        # Pre-resolve method="auto" with the *same* cost model the factory
        # uses (harmonic ~ n_modes^3 vs pixel-direct ~ (n_bins+1) * n_pix^3).
        # Using a different heuristic here would let Core and the factory
        # disagree — e.g. drop lswitch/S_fixed expecting pixel-direct while
        # the factory then picks harmonic — silently corrupting the SMW path.
        from .basis import _auto_pick_method, _problem_dimensions

        n_pix, n_modes = _problem_dimensions(theta_arr, spins, basis_lmax, lmax_b)
        n_bins = (
            self.bins.nbins
            if getattr(self, "bins", None) is not None
            else max(basis_lmax - 1, 1)
        )
        if method == "auto":
            resolved_method, _, _ = _auto_pick_method(n_pix, n_modes, basis_lmax, n_bins)
        else:
            resolved_method = method

        # Pixel-direct mode operates on full pixel-space matrices and doesn't
        # need the inference-window narrowing / S_fixed (the high-ℓ signal is
        # naturally included via the pixel-space S construction). Triggered
        # when auto picks pixel-direct OR when the user explicitly requests
        # method="pixel" with use_direct=True.
        is_pixel_direct = (method == "auto" and resolved_method == "pixel") or (
            method == "pixel" and use_direct
        )
        if is_pixel_direct:
            lmin_b = None
            lmax_b = None
            S_fixed = None

        # lmin_signal is normalised to a per-component list by
        # Core._normalize_lmin_signal during params loading.
        lmin_signal = list(self.params.lmin_signal)

        # The basis takes ownership of the noise buffer end-to-end. After
        # setup, we drop our reference so any post-setup read of
        # self.noise_cov1 fails loudly (the basis has since overwritten the
        # buffer with its in-place Cholesky factor).
        self.basis_manager = create_computation_basis(
            method=method,
            N=self.noise_cov1,
            theta=theta_arr,
            phi=phi_arr,
            lmax_signal=basis_lmax,
            beam=beam,
            spins=spins,
            basis=basis,
            C_ell=C_ell,
            epsilon=epsilon,
            mode_fraction=mode_fraction,
            lmin_signal=lmin_signal,
            lmin=lmin_b,
            lmax=lmax_b,
            S_fixed=S_fixed,
            compress=compress,
            delta_m=delta_m,
            fields=getattr(self, "collection", None),
            n_bins=n_bins,
            use_direct=use_direct,
        )

        # Build harmonic operator and precompute SMW components
        self.basis_manager.setup()

        # Pixel-direct mode never factorises self._N, so noise_cov1 is still
        # the symmetric matrix and Core code can keep using it. Other modes
        # have overwritten it in place; drop the reference as a tripwire.
        if not getattr(self.basis_manager, "_use_direct", False):
            self.noise_cov1 = None

        return self.basis_manager

    # =========================================================================
    # Unified Covariance API
    # =========================================================================
    # These methods provide a basis-agnostic interface. Subclasses use
    # these methods without knowing whether a computation basis is enabled.
    # =========================================================================

    def get_total_covariance(self, C_ell: np.ndarray) -> np.ndarray:
        """
        Get total covariance matrix C = N + S.

        If a computation basis is enabled, returns the projected covariance.
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
        if hasattr(self, "basis_manager") and self.basis_manager is not None:
            return self.basis_manager.get_compressed_covariance(C_ell)
        else:
            return self.noise_cov1 + self._build_signal_matrix(C_ell)

    def get_covariance_inverse(self, C_ell: np.ndarray) -> np.ndarray:
        """
        Get inverse covariance matrix C^{-1}.

        If a computation basis is enabled, returns the projected inverse.
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
        if hasattr(self, "basis_manager") and self.basis_manager is not None:
            return self.basis_manager.get_compressed_inverse(C_ell)
        else:
            return matrix_inverse_symm(self.get_total_covariance(C_ell), overwrite=True)

    def get_covariance_logdet(self, C_ell) -> float:
        """
        Get log determinant of covariance matrix log|C|.

        If a computation basis is enabled, uses SMW formula for exact result.
        Otherwise, computes directly.

        Parameters
        ----------
        C_ell : numpy.ndarray or dict
            Power spectrum values (array) or dict with 3-tuple keys.

        Returns
        -------
        float
            Log determinant of covariance matrix.
        """
        if hasattr(self, "basis_manager") and self.basis_manager is not None:
            return self.basis_manager.get_full_logdet(C_ell)
        else:
            if isinstance(C_ell, dict):
                C_ell_arr = C_ell.get((0, 0, 0), next(iter(C_ell.values())))
            else:
                C_ell_arr = C_ell
            _, logdet = matrix_slogdet_symm(self.get_total_covariance(C_ell_arr))
            return logdet

    def get_derivative_matrix(
        self,
        ell: int,
        spectrum_idx: int = 0,
        comp_i: int | None = None,
        comp_j: int | None = None,
        mode: int = 0,
        *,
        symmetry_mode=None,
    ) -> np.ndarray:
        """
        Get derivative matrix dC/dC_ell.

        If a computation basis is enabled, returns the projected derivative.
        Otherwise, returns full pixel-space derivative.

        Parameters
        ----------
        ell : int
            Multipole for derivative.
        spectrum_idx : int
            Spectrum index for pixel-space multi-spectrum. Ignored when
            a computation basis is enabled (use comp_i/comp_j/mode instead).
        comp_i, comp_j : int or None
            Component indices for compressed multi-field. None uses
            single-field path.
        mode : int
            Spin mode encoding (auto-pair ``[EE=0, BB=1, EB=2]``; cross-pair
            spin-2 × spin-2 ``[GG=0, GC=1, CG=2, CC=3]``).
        symmetry_mode : SymmetryMode or None
            Forwarded to the basis-layer derivative builder; ``None`` keeps
            the SYMMETRIC default.
        """
        if hasattr(self, "basis_manager") and self.basis_manager is not None:
            return self.basis_manager.get_derivative_matrix(
                ell,
                comp_i=comp_i,
                comp_j=comp_j,
                mode=mode,
                symmetry_mode=symmetry_mode,
            )
        return self._build_derivative_matrix(ell, spectrum_idx=spectrum_idx)

    def set_binning(self, bins) -> None:
        """
        Configure multipole binning.

        Parameters
        ----------
        bins : Bins
            Binning specification defining multipole ranges and weights.
            Must satisfy ``bins.lmin == params.lmin`` and
            ``bins.lmax <= params.lmax``: the per-spectrum
            ``beam_smoothing`` block built by ``Fisher.run`` is keyed by
            ``params.lmin`` and has length ``params.lmax - params.lmin + 1``,
            and the binned-derivative paths index it as
            ``beam_smoothing[ell - bins.lmin]``. A binning that drifts
            below the inference floor or above the ceiling silently
            misaligns the beam weights.
        """
        if hasattr(self, "params") and self.params is not None:
            params_lmin = getattr(self.params, "lmin", None)
            params_lmax = getattr(self.params, "lmax", None)
            if params_lmin is not None and bins.lmin != params_lmin:
                raise ValueError(
                    f"bins.lmin={bins.lmin} != params.lmin={params_lmin}; "
                    "binning floor must match the inference window floor "
                    "(beam_smoothing is keyed by params.lmin)."
                )
            if params_lmax is not None and bins.lmax > params_lmax:
                raise ValueError(
                    f"bins.lmax={bins.lmax} > params.lmax={params_lmax}; "
                    "binning cannot extend above the inference ceiling."
                )
        self.bins = bins

    def get_binned_derivative_matrix(
        self,
        bin_idx: int,
        beam_smoothing: np.ndarray | None = None,
        spectrum_idx: int = 0,
        comp_i: int | None = None,
        comp_j: int | None = None,
        mode: int = 0,
        *,
        symmetry_mode=None,
    ) -> np.ndarray:
        """
        Compute binned derivative dC^b = Sum_{ell in bin} b²_ell dC^ell,

        where the sum runs over ℓ in the bin with unit weight, b²_ell
        is the beam smoothing factor, and dC^ell = dC/dC_ell is the
        per-multipole derivative matrix.

        When beam_smoothing is provided, beam window functions are
        applied per-ℓ so that the resulting Fisher matrix is in
        beam-smoothed space. Without it, the derivative is just the
        unweighted sum dC^b = Sum_{ell in bin} dC^ell.

        Parameters
        ----------
        bin_idx : int
            Index of the bin.
        beam_smoothing : np.ndarray or None
            Per-ell beam smoothing factors b²_ell (length n_ell, starting
            at ell=2). Product of beam and (optionally) pixel window
            functions for the two fields in the spectrum.
        spectrum_idx : int
            Spectrum index for pixel-space multi-spectrum.
        comp_i, comp_j : int or None
            Component indices for compressed multi-field.
        mode : int
            Spin mode (0=TT/EE/TE, 1=BB/TB, 2=EB).
        """
        # Fast path: pixel-direct mode has a batched binned derivative that
        # avoids the per-ℓ Legendre/Wigner pass when bin width is large.
        bm = getattr(self, "basis_manager", None)
        if (
            bm is not None
            and getattr(bm, "_use_direct", False)
            and hasattr(bm, "get_binned_derivative_direct")
        ):
            ci = 0 if comp_i is None else comp_i
            cj = 0 if comp_j is None else comp_j
            return bm.get_binned_derivative_direct(
                bin_idx, self.bins, beam_smoothing, ci, cj, mode
            )

        lmin_b = self.bins.lmins[bin_idx]
        lmax_b = self.bins.lmaxs[bin_idx]
        dC_b = None
        for ell in range(lmin_b, lmax_b + 1):
            dC_ell = self.get_derivative_matrix(
                ell,
                spectrum_idx=spectrum_idx,
                comp_i=comp_i,
                comp_j=comp_j,
                mode=mode,
                symmetry_mode=symmetry_mode,
            )
            weight = 1.0
            if beam_smoothing is not None:
                # beam_smoothing is the inference-range slice (ell=lmin..lmax,
                # offset-from-lmin; ADR 0009).
                weight = beam_smoothing[ell - self.params.lmin]
            if dC_b is None:
                dC_b = weight * dC_ell
            else:
                dC_b += weight * dC_ell
        return dC_b

    def compute_quadratic_form(self, data: np.ndarray, C_ell) -> float:
        """
        Compute quadratic form d^T C^{-1} d.

        If a computation basis is enabled, uses efficient SMW-based computation.
        Otherwise, computes directly with full matrices.

        Parameters
        ----------
        data : numpy.ndarray
            Data vector in pixel space.
        C_ell : numpy.ndarray or dict
            Power spectrum values (array) or dict with 3-tuple keys.

        Returns
        -------
        float
            Quadratic form value d^T C^{-1} d.
        """
        if hasattr(self, "basis_manager") and self.basis_manager is not None:
            return self.basis_manager.compute_quadratic_form(data, C_ell)
        else:
            if isinstance(C_ell, dict):
                C_ell_arr = C_ell.get((0, 0, 0), next(iter(C_ell.values())))
            else:
                C_ell_arr = C_ell
            C_inv = self.get_covariance_inverse(C_ell_arr)
            return float(data.T @ C_inv @ data)

    def _build_signal_matrix(self, C_ell: np.ndarray) -> np.ndarray:
        """Build signal covariance matrix. Subclasses must override."""
        raise NotImplementedError("Subclasses must implement _build_signal_matrix")

    def _build_derivative_matrix(self, ell: int, spectrum_idx: int = 0) -> np.ndarray:
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
