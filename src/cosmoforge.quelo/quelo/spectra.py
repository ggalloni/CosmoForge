"""
Quadratic Maximum Likelihood (QML) power spectrum estimation for cosmological analysis.

This module implements the Spectra class for optimal power spectrum estimation from
CMB observations using the Quadratic Maximum Likelihood estimator. The QML method
provides unbiased, minimum-variance power spectrum estimates that properly account
for sky cuts, instrumental noise, and pixel correlations.

The QML estimator for power spectrum amplitude q_l is given by:

q̂_l = (1/2) * x^T * C^(-1) * ∂C/∂q_l * C^(-1) * x

where x is the data vector, C is the total covariance matrix (signal + noise),
and the covariance matrix of the estimates is (F^(-1))_ll where F is the Fisher
information matrix.

Classes
-------
Spectra
    Main class for QML power spectrum estimation inheriting from cosmocore.Core.

Notes
-----
The QML implementation includes several key features:
1. Exact likelihood computation with proper noise modeling
2. MPI parallelization for computational efficiency
3. Support for auto-correlation and cross-correlation spectra
4. Noise bias computation and optional subtraction
5. Integration with Fisher matrix computation for error propagation

The method is computationally intensive, scaling as O(N_pix^3) for matrix operations,
but provides optimal statistical properties compared to pseudo-C_l estimators.

References
----------
.. [1] Tegmark, M. "How to measure CMB power spectra without losing information"
   Phys. Rev. D 55, 5895 (1997)
.. [2] Bond, J.R., Jaffe, A.H. & Knox, L. "Estimating the power spectrum of the
   cosmic microwave background" Phys. Rev. D 57, 2117 (1998)
.. [3] Oh, S.P., Spergel, D.N. & Hinshaw, G. "An efficient technique to determine
   the power spectrum from cosmic microwave background sky maps"
   Astrophys. J. 510, 551 (1999)
.. [4] Wandelt, B.D., Larson, D.L. & Lakshminarayanan, A. "Global, exact cosmic
   microwave background data analysis using Gibbs sampling"
   Phys. Rev. D 70, 083511 (2004)
"""

from __future__ import annotations

import time

import numpy as np
from mpi4py import MPI

from cosmocore import (
    CompressionManager,
    Core,
    FieldCollection,
    do_derivative_step,
    matrix_inverse_symm,
    matrix_mult,
    matrix_trace,
    read_maps,
    vec_to_cl,
    write_out_matrix,
    writecl,
)
from cosmocore.settings import InputParams
from quelo import Fisher


class Spectra(Core):
    """
    Quadratic Maximum Likelihood (QML) power spectrum estimator for CMB analysis.

    This class implements the QML method for optimal power spectrum estimation from
    cosmic microwave background observations. The QML estimator provides unbiased,
    minimum-variance estimates of angular power spectra C_l while properly accounting
    for instrumental noise, sky cuts, and pixel correlations.

    The QML estimator computes power spectrum amplitudes as:

    q̂_l = (1/2) * x^T * E_l * x

    where E_l = C^(-1) * ∂C/∂q_l * C^(-1) is the quadratic estimator matrix,
    x is the data vector, and C is the total covariance matrix.

    Parameters
    ----------
    params_file : str, optional
        Path to YAML parameter file containing analysis configuration.
    fisher : Fisher, optional
        Pre-computed Fisher matrix instance. If provided, reuses computed
        components (covariance matrices, geometry, etc.) for efficiency.
    **kwargs : dict
        Additional keyword arguments passed to the Core parent class.

    Attributes
    ----------
    comm : MPI.Comm
        MPI communicator for parallel computation.
    rank : int
        MPI process rank (0 for master process).
    size : int
        Total number of MPI processes.
    fisher_instance : Fisher
        Fisher matrix computation instance, either provided or computed.
    maps1, maps2 : numpy.ndarray
        Input map data for primary and secondary fields (cross-correlation).
    qml_results : numpy.ndarray
        QML power spectrum estimates for all simulations and multipoles.
    qml_noise_bias : numpy.ndarray
        Noise bias estimates for auto-correlation spectra.
    invfisher : numpy.ndarray
        Inverted Fisher matrix used for final spectrum normalization.
    invCov1, invCov2 : numpy.ndarray
        Inverted covariance matrices for primary and secondary datasets.
    normalization : numpy.ndarray
        Normalization factors (vecmul) for spectrum smoothing.

    Examples
    --------
    Basic QML power spectrum estimation:

    >>> from cosmoforge.quelo import Spectra
    >>> spectra = Spectra("config/qml_analysis.yaml")
    >>> spectra.run()
    >>> power_spectra = spectra.get_power_spectra()
    >>> noise_bias = spectra.get_noise_bias()

    Using pre-computed Fisher matrix:

    >>> from cosmoforge.quelo import Fisher, Spectra
    >>> fisher = Fisher("config/fisher_config.yaml")
    >>> fisher.run()
    >>> spectra = Spectra("config/qml_config.yaml", fisher=fisher)
    >>> spectra.run()  # Reuses Fisher components for efficiency

    MPI parallel execution:

    >>> # Run with: mpirun -n 8 python qml_analysis.py
    >>> spectra = Spectra("config/high_res_config.yaml")
    >>> spectra.run()  # Distributes computation across processes

    Notes
    -----
    The QML method offers several advantages over pseudo-C_l estimators:
    1. Optimal statistical properties (minimum variance, unbiased)
    2. Exact treatment of sky cuts and inhomogeneous noise
    3. Proper error propagation through Fisher matrix
    4. Natural handling of mode coupling

    However, it is computationally expensive, scaling as O(N_pix^3) due to
    matrix inversions and O(N_ell * N_pix^2) for quadratic form evaluations.

    For auto-correlation analyses, noise bias is computed and can be subtracted:
    noise_bias_l = (1/2) * Tr[N * E_l]
    where N is the noise covariance matrix.

    Cross-correlation analyses are naturally noise-bias free when using
    independent realizations of noise between maps.

    The implementation supports both temperature-only and polarization analyses
    with proper treatment of spin-2 fields and E/B mode decomposition.

    See Also
    --------
    Fisher : Fisher information matrix computation
    cosmocore.Core : Base class providing fundamental analysis infrastructure
    """

    def __init__(
        self,
        params_file: str | None = None,
        fisher: Fisher | None = None,
        compression: dict | None = None,
        **kwargs,
    ):
        """
        Initialize QML power spectrum estimation class.

        Parameters
        ----------
        params_file : str, optional
            Path to YAML configuration file containing analysis parameters.
            If None, parameters must be provided through kwargs or set later.
        fisher : Fisher, optional
            Pre-computed Fisher matrix instance. If provided, the Spectra class
            will reuse already computed components (covariance matrices, geometry,
            field collections) for computational efficiency. The Fisher instance
            must have completed its computation (run() method called).
        compression : dict, optional
            Compression configuration dictionary. If provided, enables SMW
            compression for more efficient computation. This is passed to the
            Fisher instance when created internally. Options:

            - method : str
                Compression method: "harmonic" or "pixel_projected".
                Default is "harmonic".
            - epsilon : float
                Eigenvalue threshold for mode compression.
            - basis : str
                Compression basis for pixel_projected method. Options:
                "harmonic", "noise_weighted", "total_covariance", "snr".
            - mode_fraction : float
                Alternative to epsilon: fraction of modes to keep.

        **kwargs : dict
            Additional keyword arguments passed to the Core parent class.
            Common options include 'params' for direct parameter object,
            'verbose' for logging level control.

        Raises
        ------
        TypeError
            If fisher is provided but is not an instance of Fisher class.
        ValueError
            If fisher is provided but doesn't contain a valid Fisher matrix
            (computation not completed).

        Notes
        -----
        Initialization performs several key steps:
        1. Parameter loading and validation via Core.__init__
        2. MPI communicator setup for parallel computation
        3. Fisher matrix handling - either reuse provided instance or compute new
        4. QML-specific variable initialization

        When a Fisher instance is provided, the following components are reused:

        - Field collections and geometry information
        - Covariance matrices (both noise and inverted forms)
        - Pixelization and active pixel information
        - Signal matrices and derivative computation setup

        This reuse significantly reduces computational overhead when performing
        both Fisher forecasts and QML estimation on the same dataset.

        The MPI environment must be initialized before creating Spectra instances
        for parallel computation. Each process will handle a subset of multipoles
        during the QML computation phase.

        Examples
        --------
        Initialize from configuration file:

        >>> spectra = Spectra("config/qml_analysis.yaml")

        Initialize with pre-computed Fisher matrix:

        >>> fisher = Fisher("config/fisher_config.yaml")
        >>> fisher.run()
        >>> spectra = Spectra("config/qml_config.yaml", fisher=fisher)

        Initialize with compression for faster computation:

        >>> spectra = Spectra("config/qml_analysis.yaml", compression={
        ...     "method": "harmonic",
        ...     "epsilon": 1e-4,
        ... })

        Initialize with direct parameters:

        >>> from cosmocore.settings import InputParams
        >>> params = InputParams()
        >>> spectra = Spectra(params=params)
        """
        self.params: InputParams = None
        super().__init__(params=params_file, **kwargs)

        # MPI setup
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Store compression config for Fisher creation
        self._compression_config = compression

        # Initialize Fisher matrix or compute it
        if fisher is not None:
            if not isinstance(fisher, Fisher):
                raise TypeError("fisher must be an instance of Fisher class.")
            if not hasattr(fisher, "fisher") or fisher.fisher is None:
                raise ValueError("Fisher instance must have a valid fisher matrix.")
            self.fisher_instance = fisher
            # Reuse already computed components from Fisher
            self._reuse_fisher_components()
        else:
            self.fisher_instance = self._get_fisher()
            # Also reuse components from the internally created Fisher
            self._reuse_fisher_components()

        # Initialize QML-specific variables
        self.maps1 = None
        self.maps2 = None
        self.qml_results = None
        self.qml_noise_bias = None
        self.invfisher = None

        # lmax for signal matrix computation (matches Fortran convention of 4*nside)
        self._lmax_signal = None

    @property
    def lmax_signal(self) -> int:
        """
        Maximum multipole for signal/derivative matrix computation.

        This defaults to 4*nside to match the Fortran reference implementation.
        The derivative matrices are computed up to this lmax, while the output
        power spectra use params.lmax.

        Returns
        -------
        int
            Maximum multipole for signal covariance and derivative computation.
        """
        if self._lmax_signal is not None:
            return self._lmax_signal
        return 4 * self.params.nside

    @lmax_signal.setter
    def lmax_signal(self, value: int) -> None:
        """Set custom lmax_signal value."""
        self._lmax_signal = value

    def _reuse_fisher_components(self):
        """
        Reuse computational components from a pre-computed Fisher instance.

        This method extracts and reuses expensive-to-compute components from
        an existing Fisher matrix computation, avoiding redundant calculations
        when performing QML analysis on the same dataset configuration.

        Notes
        -----
        Components reused from the Fisher instance include:

        - Field collections: Precomputed HEALPix field setup and active pixels
        - Geometry data: Pixel pointing vectors and spherical harmonic transforms
        - Covariance matrices: Both noise covariance and inverted forms
        - Signal matrices: Theoretical signal covariance from power spectra

        This reuse provides significant computational savings, especially for:
        1. High-resolution analyses where geometry setup is expensive
        2. Large datasets where covariance matrix operations dominate
        3. Combined Fisher + QML pipelines on identical configurations

        The method performs safety checks to ensure each component exists
        before attempting to copy it, gracefully handling partial Fisher
        computations or different analysis configurations.

        After copying Fisher components, covariance matrices are loaded
        from disk files to ensure consistency with the QML analysis
        requirements (inverted forms, proper normalization).

        Examples
        --------
        This method is called automatically during initialization:

        >>> fisher = Fisher("config.yaml")
        >>> fisher.run()
        >>> spectra = Spectra("config.yaml", fisher=fisher)
        # _reuse_fisher_components() called internally
        """
        # Copy already computed variables from Fisher instance
        if (
            hasattr(self.fisher_instance, "collection")
            and self.fisher_instance.collection is not None
        ):
            self.collection: FieldCollection = self.fisher_instance.collection
        if (
            hasattr(self.fisher_instance, "npixs")
            and self.fisher_instance.npixs is not None
        ):
            self.npixs = self.fisher_instance.npixs
        if (
            hasattr(self.fisher_instance, "pixact")
            and self.fisher_instance.pixact is not None
        ):
            self.pixact = self.fisher_instance.pixact
        if (
            hasattr(self.fisher_instance, "point_vectors")
            and self.fisher_instance.point_vectors is not None
        ):
            self.point_vectors = self.fisher_instance.point_vectors
        if (
            hasattr(self.fisher_instance, "NCov1")
            and self.fisher_instance.NCov1 is not None
        ):
            self.NCov1 = self.fisher_instance.NCov1
        if (
            hasattr(self.fisher_instance, "NCov2")
            and self.fisher_instance.NCov2 is not None
        ):
            self.NCov2 = self.fisher_instance.NCov2
        if hasattr(self.fisher_instance, "Sig") and self.fisher_instance.Sig is not None:
            self.Sig = self.fisher_instance.Sig

        # Copy compression manager if available
        if (
            hasattr(self.fisher_instance, "compression_manager")
            and self.fisher_instance.compression_manager is not None
        ):
            self.compression_manager: CompressionManager = (
                self.fisher_instance.compression_manager
            )
        else:
            self.compression_manager = None

        # Load covariance matrices
        self._load_covariance_matrices()

    def _load_covariance_matrices(self):
        """
        Load covariance matrices from binary files for QML computation.

        This method reads both noise covariance matrices and their inverted
        forms from disk files specified in the parameter configuration.
        The matrices are reshaped to proper 2D format for subsequent
        linear algebra operations.

        Notes
        -----
        Loads the following matrices:

        - invCov1, invCov2: Inverted total covariance matrices C^(-1)
        - NCov1, NCov2: Original noise covariance matrices N

        For cross-correlation analyses (do_cross=True), both primary and
        secondary covariance matrices are loaded. The inverted matrices
        are used directly in QML E-operator computation, while noise
        matrices are needed for noise bias calculation.

        Matrix files are expected to be in binary format (numpy.fromfile
        compatible) with total size n_active_pixels^2 elements stored
        in row-major order.

        File paths are specified in the parameter configuration:

        - outinvcovmatfile1, outinvcovmatfile2: Inverted covariance files
        - outnoisecovmat1, outnoisecovmat2: Original noise covariance files
          (created by Fisher.run())

        Raises
        ------
        FileNotFoundError
            If covariance matrix files are not found at specified paths.
        ValueError
            If matrix dimensions don't match expected active pixel counts.

        Examples
        --------
        This method is called automatically during setup:

        >>> spectra = Spectra("config.yaml")
        >>> spectra._load_covariance_matrices()  # Called internally
        """
        import os

        ntot = self.collection.total_active_pixels

        # Load inverted covariance matrices
        self.invCov1 = np.fromfile(self.params.outinvcovmatfile1).reshape(ntot, ntot)
        if self.params.do_cross:
            self.invCov2 = np.fromfile(self.params.outinvcovmatfile2).reshape(ntot, ntot)

        # Load noise covariance matrices (created by Fisher.run())
        if not os.path.exists(self.params.outnoisecovmat1):
            raise FileNotFoundError(
                f"Noise covariance file not found: {self.params.outnoisecovmat1}. "
                f"Run Fisher analysis first to generate this file."
            )
        self.NCov1 = np.fromfile(self.params.outnoisecovmat1).reshape(ntot, ntot)
        if self.params.do_cross:
            if not os.path.exists(self.params.outnoisecovmat2):
                raise FileNotFoundError(
                    f"Noise covariance file not found: {self.params.outnoisecovmat2}. "
                    f"Run Fisher analysis first to generate this file."
                )
            self.NCov2 = np.fromfile(self.params.outnoisecovmat2).reshape(ntot, ntot)

    def _get_fisher(self) -> Fisher:
        """
        Compute Fisher information matrix for QML error propagation.

        This method creates and runs a Fisher matrix computation that will
        be used for QML estimator normalization and error propagation.
        The Fisher matrix provides the optimal weighting for combining
        QML estimates across different multipoles and spectra.

        Returns
        -------
        Fisher
            Completed Fisher matrix computation instance with all necessary
            components computed and available for QML analysis.

        Notes
        -----
        The Fisher matrix computation includes:
        1. Signal covariance matrix computation from theoretical power spectra
        2. Total covariance matrix assembly (signal + noise)
        3. Matrix inversion and derivative computation setup
        4. Full Fisher matrix calculation using MPI parallelization

        The computed Fisher matrix F_ij represents the information content
        about parameter combinations θ_i, θ_j and is essential for:

        - QML estimator normalization via F^(-1)
        - Proper error bar computation on power spectrum estimates
        - Optimal combination of estimates across multipoles

        Timing information is logged for performance monitoring, especially
        important for high-resolution analyses where Fisher computation
        can be the dominant computational cost.

        The method handles MPI coordination automatically, ensuring all
        processes have consistent Fisher matrix information before
        proceeding with QML computation.

        Examples
        --------
        This method is called automatically when no Fisher instance is provided:

        >>> spectra = Spectra("config.yaml")  # No fisher parameter
        >>> # _get_fisher() called internally during initialization
        >>> spectra.run()

        See Also
        --------
        Fisher.run : Complete Fisher matrix computation pipeline
        """
        if self.rank == 0:
            self.log("Starting Fisher matrix computation...", level=1)

        start_time = time.time()

        fisher = Fisher(self.params, compression=self._compression_config)
        fisher.run()

        if self.rank == 0:
            elapsed = time.time() - start_time
            self.log(
                f"Fisher matrix computation completed in {elapsed:.2f} seconds", level=1
            )

        return fisher

    def setup_maps(self):
        """
        Read and prepare observational map data for QML analysis.

        This method loads CMB observation maps from FITS files and prepares
        them for QML power spectrum estimation. Maps are read using the
        cosmocore map reading infrastructure with proper pixel selection,
        field extraction, and calibration handling.

        Notes
        -----
        Map loading process includes:
        1. Memory allocation for map arrays based on active pixels
        2. FITS file reading with HEALPix format support
        3. Field selection (T, Q, U) based on analysis configuration
        4. Pixel masking using active pixel information
        5. Calibration factor application if specified

        For cross-correlation analyses (do_cross=True), both primary and
        secondary map sets are loaded. Maps are organized as arrays with
        dimensions (n_active_pixels, n_simulations) to support Monte Carlo
        error estimation and null testing.

        The map reading uses the cosmocore.read_maps function which handles:

        - HEALPix FITS format parsing
        - Multiple field extraction (temperature and polarization)
        - Pixel ordering conversion (RING/NESTED)
        - Calibration and unit conversion
        - Memory-efficient loading for large datasets

        Raises
        ------
        ValueError
            If pixel information is not available (setup_geometry not called).
        FileNotFoundError
            If input map files are not found at specified paths.
        RuntimeError
            If map dimensions don't match expected configuration.

        Examples
        --------
        This method is called as part of the analysis pipeline:

        >>> spectra = Spectra("config.yaml")
        >>> # Called automatically in run(), or manually:
        >>> spectra.setup_geometry()
        >>> spectra.setup_maps()
        >>> print(f"Maps shape: {spectra.maps1.shape}")

        See Also
        --------
        cosmocore.read_maps : Core map reading functionality
        setup_geometry : Pixel information setup required before map loading
        """
        if self.rank == 0:
            self.log("Reading maps", level=2)

            # Ensure we have pixel information
            if not hasattr(self, "npixs") or self.npixs is None:
                raise ValueError(
                    "Pixel information not available. Run setup_geometry first."
                )

            # Read maps using the core functionality
            ntot = sum(self.collection.n_active)
            self.maps1 = np.empty((ntot, self.params.nsims), dtype=np.float64)

            # Read maps1
            read_maps(
                maps=self.maps1,
                filename=self.params.inputmapfile1,
                pixact=self.pixact,
                field_labels=self.params.physical_labels,
                calibration=self.params.calibration,
            )

            # Read maps2 if doing cross-correlation
            if self.params.do_cross:
                self.maps2 = np.empty((ntot, self.params.nsims), dtype=np.float64)
                read_maps(
                    maps=self.maps2,
                    filename=self.params.inputmapfile2,
                    pixact=self.pixact,
                    field_labels=self.params.physical_labels,
                    calibration=self.params.calibration,
                )

    def setup_fisher_inversion(self):
        """
        Prepare inverted Fisher matrix with normalization for QML estimation.

        This method processes the Fisher information matrix to create the
        optimal weighting matrix for QML power spectrum estimates. The process
        includes normalization factor computation, Fisher matrix conditioning,
        inversion, and error bar calculation.

        Notes
        -----
        The Fisher matrix preparation involves several critical steps:

        **1. Smoothing Factor Computation:**
        Computes the "vecmul" normalization factors that account for:

        - Beam convolution effects in observed maps
        - Finite pixel size and pixelization effects
        - Mode coupling between different multipoles
        - Proper normalization for power spectrum units

        **2. Fisher Matrix Normalization:**
        Applies normalization as: F'_ij = F_ij * vecmul_i * vecmul_j
        This ensures proper weighting of different multipole contributions
        and accounts for observational effects in the final estimates.

        **3. Matrix Inversion:**
        Inverts the normalized Fisher matrix using stable algorithms:
        F_cov = (F')^(-1)
        This covariance matrix provides optimal error propagation for QML estimates.

        **4. Error Bar Computation:**
        Extracts marginal errors as: σ_i = sqrt((F_cov)_ii)
        These represent the expected 1σ uncertainties on power spectrum estimates.

        The vecmul factors are computed from beam and pixelization effects:
        vecmul_l = ∫ d²k W(k) B²(k) / ∫ d²k W(k)
        where W(k) is the pixel window and B(k) is the beam transfer function.

        All intermediate and final products are written to output files for
        verification and subsequent analysis steps.

        Raises
        ------
        ValueError
            If Fisher matrix is not available or singular.
        LinAlgError
            If Fisher matrix inversion fails due to poor conditioning.

        Examples
        --------
        This method is called during the analysis setup:

        >>> spectra = Spectra("config.yaml", fisher=fisher_instance)
        >>> spectra.setup_fisher_inversion()
        >>> cond_num = np.linalg.cond(spectra.invfisher)
        >>> print(f"Covariance matrix condition number: {cond_num}")

        See Also
        --------
        Fisher.get_fisher_matrix : Source of Fisher information matrix
        cosmocore.matrix_inverse_symm : Stable symmetric matrix inversion
        """
        if self.rank == 0:
            self.log("Reading and inverting Fisher matrix", level=2)

            # Get Fisher matrix from the Fisher instance
            fisher_matrix = self.fisher_instance.get_fisher_matrix()
            if fisher_matrix is None:
                raise ValueError("Fisher matrix not available")

            self.invfisher = fisher_matrix.copy()

            # Compute vecmul (smoothing factors) - this is critical!
            self.log("Computing smoothing factors and vecmul", level=2)
            smoothing_factors = self.collection.spectra_manager.compute_smoothing_factors(
                self.collection.beam_manager
            )

            # Create vecmul array
            nell = self.params.nspectra * (self.params.lmax - 1)
            self.normalization = np.zeros(nell, dtype=np.float64)

            # Fill vecmul array
            idx = 0
            for _, spectrum_label in enumerate(self.collection.spectra_manager.labels):
                if spectrum_label in smoothing_factors:
                    smooth_factor = smoothing_factors[spectrum_label]
                    for ell_idx in range(self.params.lmax - 1):
                        self.normalization[idx] = smooth_factor[ell_idx]
                        idx += 1
                else:
                    raise ValueError(f"No smoothing factors found for {spectrum_label}")

            # Apply vecmul normalization to Fisher matrix
            self.log("Applying vecmul normalization to Fisher matrix", level=2)
            self.invfisher = self.invfisher * np.outer(
                self.normalization, self.normalization
            )

            # Invert Fisher matrix
            self.log("Inverting normalized Fisher matrix", level=2)
            start_time = time.time()
            self.invfisher = matrix_inverse_symm(self.invfisher)

            self.log(
                f"Fisher matrix inversion time: {time.time() - start_time:.2f} seconds",
                level=3,
            )

            # Write out covariance matrix and errors
            self.log("Writing out covariance and errors", level=2)
            write_out_matrix(self.params.outcovmatfile, self.invfisher)

            # Compute and write parameter errors
            vec_error_bars = np.sqrt(np.diag(self.invfisher))

            # Convert vector to Cl format and write errors
            n_ell = self.params.lmax - 1
            nspectra = len(vec_error_bars) // n_ell
            error_bars = np.zeros((n_ell, nspectra), dtype=np.float64)
            vec_to_cl(vec_error_bars, error_bars)
            writecl(self.params.outerrfile, error_bars)

    def setup_qml_computation(self):
        """
        Initialize arrays and variables for QML power spectrum computation.

        This method allocates memory for QML estimation results and prepares
        the computational infrastructure for the main QML calculation phase.
        Arrays are sized based on the number of multipoles, spectra, and
        Monte Carlo simulations.

        Notes
        -----
        Initializes the following arrays:

        - qml_results: Storage for QML power spectrum estimates with shape
          (n_simulations, n_total_parameters) where n_total_parameters =
          n_spectra * (lmax - 1)
        - qml_noise_bias: Storage for noise bias estimates (auto-correlation only)
          with shape (n_total_parameters,)

        The qml_results array stores the raw quadratic estimates:
        q̂_l^(sim) = (1/2) * x^(sim)T * E_l * x^(sim)

        for each simulation and multipole. These raw estimates will be
        combined using the inverse Fisher matrix to produce the final
        optimally-weighted power spectrum estimates.

        For auto-correlation analyses, noise bias terms are computed as:
        bias_l = (1/2) * Tr[N * E_l]
        where N is the noise covariance matrix and E_l is the quadratic
        estimator matrix for multipole l.

        Memory allocation scales as O(n_sims * n_ell * n_spec) for results
        and O(n_ell * n_spec) for bias terms, which is typically much
        smaller than the O(n_pix^2) covariance matrix storage.

        Examples
        --------
        This method is called automatically during computation:

        >>> spectra = Spectra("config.yaml")
        >>> spectra.setup_qml_computation()
        >>> print(f"QML results shape: {spectra.qml_results.shape}")
        >>> print(f"Total parameters: {spectra.qml_results.shape[1]}")
        """
        nell = self.params.nspectra * (self.params.lmax - 1)

        # Initialize y vectors for QML estimation
        self.qml_results = np.zeros((self.params.nsims, nell), dtype=np.float64)

        if not self.params.do_cross:
            self.qml_noise_bias = np.zeros(nell, dtype=np.float64)

    def compute_e_operator(self, il: int, der_s: np.ndarray) -> np.ndarray:
        """
        Compute the QML quadratic estimator matrix E_l for a given multipole.

        This method constructs the quadratic estimator matrix used in QML
        power spectrum estimation. The E-operator encapsulates the optimal
        weighting for extracting power spectrum information from observed maps.

        Parameters
        ----------
        il : int
            Linear multipole index in the parameter vector. Related to spherical
            harmonic multipole l and spectrum type through the relationship:
            spectrum_index = il // (lmax - 1), l = (il % (lmax - 1)) + 2
        der_s : numpy.ndarray
            Derivative of the signal covariance matrix ∂S/∂C_l with respect to
            the power spectrum amplitude at the given multipole. Shape must
            match the covariance matrix dimensions (n_active_pixels, n_active_pixels).

        Returns
        -------
        numpy.ndarray
            Quadratic estimator matrix E_l with shape (n_active_pixels, n_active_pixels).
            This matrix is used to compute power spectrum estimates via the quadratic
            form q̂_l = (1/2) * x^T * E_l * x.

        Notes
        -----
        The E-operator is computed differently for auto- and cross-correlation:

        **Auto-correlation case (do_cross=False):**
        E_l = (1/2) * C^(-1) * ∂S/∂C_l * C^(-1)
        where C^(-1) is the inverted total covariance matrix.

        **Cross-correlation case (do_cross=True):**
        E_l = (1/2) * C₂^(-1) * ∂S/∂C_l * C₁^(-1)
        where C₁^(-1) and C₂^(-1) are the inverted covariance matrices for
        the two independent datasets being cross-correlated.

        The factor of 1/2 accounts for the symmetry of the quadratic form and
        ensures unbiased estimation. The E-operator has the key property that
        E[x^T * E_l * x] = C_l for the true power spectrum value C_l.

        Matrix multiplications are performed using optimized BLAS routines
        via cosmocore.matrix_mult for computational efficiency. The computation
        scales as O(n_pix^3) due to the matrix multiplications.

        Examples
        --------
        >>> der_s = np.zeros((n_pix, n_pix))
        >>> # ... fill der_s with signal derivative ...
        >>> E_l = spectra.compute_e_operator(il=15, der_s=der_s)
        >>> # Use E_l for quadratic estimation:
        >>> qml_estimate = 0.5 * np.dot(data_vector, np.dot(E_l, data_vector))

        See Also
        --------
        cosmocore.matrix_mult : Optimized matrix multiplication
        cosmocore.do_derivative_step : Signal matrix derivative computation
        """
        if self.params.do_cross:
            # E = 0.5 * invCov2^{-1} * derS * invCov1^{-1}
            E = 0.5 * matrix_mult(self.invCov2, matrix_mult(der_s, self.invCov1))
        else:
            # E = 0.5 * invCov1^{-1} * derS * invCov1^{-1}
            E = 0.5 * matrix_mult(self.invCov1, matrix_mult(der_s, self.invCov1))

        return E

    def compute_qml_spectra(self):
        """
        Execute the main parallel QML power spectrum computation.

        This method implements the core QML estimation algorithm using MPI
        parallelization to distribute multipole computations across processes.
        For each multipole, it computes the derivative matrix, E-operator,
        and quadratic estimates for all Monte Carlo simulations.

        Notes
        -----
        The computation follows this algorithmic structure:

        **1. Work Distribution:**
        Multipoles are distributed across MPI processes using round-robin:
        process_rank = multipole_index % n_processes

        **2. For each assigned multipole l:**

        - Compute signal derivative: ∂S/∂C_l using do_derivative_step
        - Construct E-operator: E_l = (1/2) * C^(-1) * ∂S/∂C_l * C^(-1)
        - Apply to all simulations: q̂_l^(sim) = x^(sim)T * E_l * x^(sim)

        **3. Noise Bias Handling (auto-correlation only):**

        - Compute bias: bias_l = (1/2) * Tr[N * E_l]
        - Optionally subtract from estimates if remove_nb=True

        **4. Cross-correlation vs Auto-correlation:**

        - Cross: q̂_l = x₂^T * E_l * x₁ (naturally noise-bias free)
        - Auto: q̂_l = x^T * E_l * x (requires noise bias computation)

        The parallelization strategy ensures good load balancing while
        minimizing memory overhead. Each process only stores matrices
        for its assigned multipoles, with final results gathered via
        MPI reduction operations.

        Computational complexity per multipole:
        - Signal derivative: O(n_pix² * l)
        - E-operator: O(n_pix³) (matrix multiplications)
        - Quadratic forms: O(n_pix² * n_sims)

        Total scaling: O(n_ell * n_pix³ + n_ell * n_pix² * n_sims)

        Examples
        --------
        This method is called as part of the computation pipeline:

        >>> spectra = Spectra("config.yaml")
        >>> spectra.setup_qml_computation()
        >>> spectra.compute_qml_spectra()
        >>> # Results available in spectra.qml_results after MPI reduction

        See Also
        --------
        compute_e_operator : E-operator construction for individual multipoles
        _reduce_qml_results : MPI reduction of results from all processes
        cosmocore.do_derivative_step : Signal matrix derivative computation
        """
        # Check if we should use compressed computation
        use_compression = (
            hasattr(self, "compression_manager") and self.compression_manager is not None
        )

        if use_compression:
            self._compute_qml_spectra_compressed()
        else:
            self._compute_qml_spectra_traditional()

    def _compute_qml_spectra_compressed(self):
        """
        Compute QML spectra using compressed representation.

        This method performs QML estimation entirely in compressed space,
        ensuring consistency with the compressed Fisher matrix computation.

        The compressed QML estimator is:
            q_l = (1/2) * w^T @ E_l @ w

        where:
            w = V @ C^{-1} @ d  (weighted compressed data via SMW)
            E_l = get_derivative_matrix(ell) (diagonal with (2ℓ+1)/(4π) at modes for ℓ)

        This is mathematically equivalent to the traditional estimator:
            q_l = (1/2) * d^T @ C^{-1} @ dC_l @ C^{-1} @ d

        The key insight is that dC_l = V^T @ E_l @ V in harmonic space, so:
            q_l = (1/2) * d^T @ C^{-1} @ V^T @ E_l @ V @ C^{-1} @ d
                = (1/2) * (V C^{-1} d)^T @ E_l @ (V C^{-1} d)
                = (1/2) * w^T @ E_l @ w
        """
        if self.rank == 0:
            self.log("Starting QML computation (compressed)", level=2)

        start_time = time.time()

        nell = self.params.nspectra * (self.params.lmax - 1)
        cm = self.compression_manager

        # Get C_ell for covariance computation
        C_ell = self.collection.spectra_manager.get_cls(0, 0, 0)

        # Compute weighted compressed data for all simulations
        # w = V @ C^{-1} @ d (using SMW formula internally)
        n_sims = self.params.nsims
        n_compressed = cm.n_kept  # n_kept is the output dimension for both methods

        # For both harmonic and pixel_projected, use get_weighted_compressed_data:
        # - Harmonic: w = V @ C^{-1} @ d (using SMW formula)
        # - Pixel_projected: w = C_c^{-1} @ U^T @ d
        #
        # For pixel_projected, precompute C_c_inv once to avoid redundant computation
        C_c_inv = None
        if cm.method == "pixel_projected":
            C_c_inv = cm.get_compressed_inverse(C_ell)

        maps1_weighted = np.zeros((n_compressed, n_sims), dtype=np.float64)
        for isim in range(n_sims):
            maps1_weighted[:, isim] = cm.get_weighted_compressed_data(
                self.maps1[:, isim], C_ell, C_c_inv=C_c_inv
            )

        if self.params.do_cross:
            maps2_weighted = np.zeros((n_compressed, n_sims), dtype=np.float64)
            for isim in range(n_sims):
                maps2_weighted[:, isim] = cm.get_weighted_compressed_data(
                    self.maps2[:, isim], C_ell, C_c_inv=C_c_inv
                )

        # For noise bias, we need to compute E[q_l|noise only]
        # E[q_l | noise] = 0.5 * Tr[E_l @ Cov(w|noise)]
        #
        # For harmonic compression, Cov(w|noise) = V @ C^{-1} @ N @ C^{-1} @ V^T
        # We compute this efficiently using SMW components:
        # V_Cinv = (I - M K^{-1}) @ V_Ninv, then Cov(w|noise) = V_Cinv @ N @ V_Cinv^T
        #
        # For diagonal N and diagonal E_l, we can compute the trace very efficiently:
        # Tr[E_l @ (V_Cinv @ N @ V_Cinv^T)] =
        # sum_a E_l[a,a] * sum_i (V_Cinv[a,i])^2 * N[i,i]
        if not self.params.do_cross:
            if cm.method == "harmonic":
                # Efficiently compute V @ C^{-1} using SMW components
                from scipy.linalg import cho_solve, cholesky

                impl = cm._impl
                Lambda_diag = impl._build_lambda_diagonal(C_ell)
                Lambda_inv_diag = np.where(Lambda_diag > 1e-30, 1.0 / Lambda_diag, 1e30)
                M = impl._V_Ninv_VT
                K = np.diag(Lambda_inv_diag) + M

                try:
                    L = cholesky(K, lower=True)
                    K_inv = cho_solve((L, True), np.eye(K.shape[0]))
                except np.linalg.LinAlgError:
                    K_inv = np.linalg.inv(K)

                # V_Cinv = (I - M @ K^{-1}) @ V_Ninv  [n_modes x n_pix]
                I_minus_MKinv = np.eye(cm.n_modes) - M @ K_inv
                V_Cinv = I_minus_MKinv @ impl._V_N_inv

                # For diagonal N, compute W = V_Cinv * sqrt(noise_var)
                # noise_var = 1 / diag(N_inv)
                # IMPORTANT: When SMW optimization is enabled, impl.N_inv is N_eff_inv
                # (includes S_fixed), but for noise bias we need the actual noise N
                if hasattr(impl, "_N_inv_original"):
                    noise_var = 1.0 / np.diag(impl._N_inv_original)
                else:
                    noise_var = 1.0 / np.diag(impl.N_inv)
                sqrt_noise = np.sqrt(noise_var)
                W = V_Cinv * sqrt_noise[np.newaxis, :]  # (n_modes, n_pix)

                # Diagonal of Cov(w|noise) = sum over columns of W^2
                noise_cov_w_diag = np.sum(W**2, axis=1)  # O(n_modes * n_pix)
            else:
                # For pixel_projected: use compressed quantities
                # Cov(w|noise) = C_c^{-1} @ (U^T @ N @ U) @ C_c^{-1}
                C_bar_inv = cm.get_compressed_inverse(C_ell)
                N_bar = cm.get_compressed_covariance(np.zeros_like(C_ell))
                noise_cov_w = C_bar_inv @ N_bar @ C_bar_inv

        # Main computation loop - distribute multipoles across processes
        for il in range(nell):
            if self.rank == il % self.size:
                _ = il // (self.params.lmax - 1)
                ell = (il % (self.params.lmax - 1)) + 2

                # Get compressed derivative matrix E_l
                E_l = cm.get_derivative_matrix(ell)

                if self.params.do_cross:
                    # Cross-correlation case
                    for isim in range(n_sims):
                        w1 = maps1_weighted[:, isim]
                        w2 = maps2_weighted[:, isim]
                        self.qml_results[isim, il] = 0.5 * w2 @ E_l @ w1
                else:
                    # Auto-correlation case
                    # Compute noise bias: E[q_l|noise] = 0.5 * Tr[E_l @ Cov(w|noise)]
                    if cm.method == "harmonic":
                        # For harmonic, E_l is diagonal - use fast diagonal trace
                        E_l_diag = np.diag(E_l)
                        tr_ne = 0.5 * np.sum(E_l_diag * noise_cov_w_diag)
                    else:
                        # For pixel_projected, E_l is full matrix - use matrix_trace
                        tr_ne = 0.5 * matrix_trace(E_l, noise_cov_w)
                    self.qml_noise_bias[il] = tr_ne

                    for isim in range(n_sims):
                        w = maps1_weighted[:, isim]
                        qml_value = 0.5 * w @ E_l @ w

                        if hasattr(self.params, "remove_nb") and self.params.remove_nb:
                            qml_value -= tr_ne

                        self.qml_results[isim, il] = qml_value

        # Synchronize all processes
        self.comm.Barrier()

        if self.rank == 0:
            self.log("QML computation done (compressed)", level=2)
            self.log(
                f"QML computation time: {time.time() - start_time:.2f} seconds", level=3
            )

        # Reduce results from all processes
        self._reduce_qml_results(nell)

    def _compute_qml_spectra_traditional(self):
        """
        Compute QML spectra using traditional pixel-space computation.

        Optimized: Precomputes y = C^{-1} @ d to avoid building full E matrix.

        The QML estimator is:
            q_l = (1/2) * d^T @ C^{-1} @ dC_l @ C^{-1} @ d
                = (1/2) * y^T @ dC_l @ y   where y = C^{-1} @ d

        This reduces complexity from 2 × O(n³) to 1 × O(n³) per multipole,
        as we only need C^{-1} @ dC for noise bias (not the full E matrix).
        """
        if self.rank == 0:
            self.log("Starting QML computation (traditional, optimized)", level=2)

        start_time = time.time()

        nell = self.params.nspectra * (self.params.lmax - 1)
        ntot = sum(self.collection.n_active)

        # Precompute weighted data: y = C^{-1} @ d for all simulations
        # This is O(n² × nsims) and avoids rebuilding for each ℓ
        y1 = matrix_mult(self.invCov1, self.maps1)  # (ntot, nsims)

        if self.params.do_cross:
            y2 = matrix_mult(self.invCov2, self.maps2)  # (ntot, nsims)

        # For noise bias: Tr[N @ E] = 0.5 * Tr[N @ C^{-1} @ dC @ C^{-1}]
        # Using cyclic trace property: = 0.5 * Tr[C^{-1} @ N @ C^{-1} @ dC]
        # Precompute C^{-1} @ N @ C^{-1} once (O(n³)), then Tr(... @ dC) per ℓ
        if not self.params.do_cross:
            Cinv_N_Cinv = matrix_mult(self.invCov1, matrix_mult(self.NCov1, self.invCov1))

        # Allocate derivative matrix
        der_s = np.zeros((ntot, ntot), dtype=np.float64)

        # Main computation loop - distribute multipoles across processes
        for il in range(nell):
            if self.rank == il % self.size:
                spectrum_idx = il // (self.params.lmax - 1)
                ell = (il % (self.params.lmax - 1)) + 2

                # Compute derivative matrix dC_l
                der_s.fill(0.0)
                do_derivative_step(
                    der_s,
                    spectrum_idx,
                    self.npixs,
                    self.params.spins,
                    ell,
                    self.collection,
                )

                # Compute dC @ y for all sims at once: O(n² × nsims)
                dC_y1 = matrix_mult(der_s, y1)

                if self.params.do_cross:
                    # Cross-correlation: q_l = 0.5 * y2^T @ dC @ y1
                    for isim in range(self.params.nsims):
                        self.qml_results[isim, il] = 0.5 * np.dot(
                            y2[:, isim], dC_y1[:, isim]
                        )
                else:
                    # Auto-correlation case
                    # Noise bias: Tr[N @ E] = 0.5 * Tr[C^{-1} @ N @ C^{-1} @ dC]
                    # Using precomputed Cinv_N_Cinv: Tr(Cinv_N_Cinv @ dC)
                    tr_ne = 0.5 * matrix_trace(Cinv_N_Cinv, der_s)
                    self.qml_noise_bias[il] = tr_ne

                    # QML values: q_l = 0.5 * y^T @ dC @ y
                    for isim in range(self.params.nsims):
                        qml_value = 0.5 * np.dot(y1[:, isim], dC_y1[:, isim])

                        if hasattr(self.params, "remove_nb") and self.params.remove_nb:
                            qml_value -= tr_ne

                        self.qml_results[isim, il] = qml_value

        # Synchronize all processes
        self.comm.Barrier()

        if self.rank == 0:
            self.log("QML computation done (traditional, optimized)", level=2)
            self.log(
                f"QML computation time: {time.time() - start_time:.2f} seconds", level=3
            )

        # Reduce results from all processes
        self._reduce_qml_results(nell)

    def _reduce_qml_results(self, nell: int):
        """
        Collect and combine QML results from all MPI processes.

        This method performs MPI reduction operations to gather partial QML
        results computed across different processes and combine them into
        complete power spectrum estimates and noise bias arrays.

        Parameters
        ----------
        nell : int
            Total number of multipole parameters (n_spectra * (lmax - 1)).
            Used to allocate reduction arrays with correct dimensions.

        Notes
        -----
        The reduction process handles two types of results:

        **1. QML Power Spectrum Estimates:**
        Each process computed estimates for a subset of multipoles.
        Results are summed across processes using MPI.SUM to combine
        contributions from all assigned multipoles.

        **2. Noise Bias Terms (auto-correlation only):**
        Noise bias values computed for each multipole are similarly
        reduced using MPI.SUM operation.

        The MPI reduction ensures that only the master process (rank 0)
        receives the complete combined results, while worker processes
        retain only their partial contributions.

        Memory allocation for reduction arrays:
        - reduced_qml_results: (n_simulations, nell) array
        - reduced_qml_noise_bias: (nell,) array (auto-correlation only)

        After reduction, the master process updates its local arrays
        with the complete results, making them available for final
        Fisher matrix multiplication and output.

        The reduction is essential for the distributed computation model
        where different processes handle different multipoles, ensuring
        all contributions are properly combined for the final estimates.

        Examples
        --------
        This method is called automatically after parallel computation:

        >>> # Called internally by compute_qml_spectra()
        >>> spectra._reduce_qml_results(nell)
        >>> # Now spectra.qml_results contains complete results on rank 0

        See Also
        --------
        compute_qml_spectra : Main computation method that calls this reduction
        MPI.Reduce : Low-level MPI reduction operation used internally
        """
        # Reduce y vectors
        reduced_qml_results = np.zeros((self.params.nsims, nell), dtype=np.float64)
        self.comm.Reduce(self.qml_results, reduced_qml_results, op=MPI.SUM, root=0)

        if not self.params.do_cross:
            # Reduce noise bias
            reduced_qml_noise_bias = np.zeros(nell, dtype=np.float64)
            self.comm.Reduce(
                self.qml_noise_bias, reduced_qml_noise_bias, op=MPI.SUM, root=0
            )

        if self.rank == 0:
            self.qml_results = reduced_qml_results
            if not self.params.do_cross:
                self.qml_noise_bias = reduced_qml_noise_bias

    def compute(self):
        """
        Execute the complete QML power spectrum computation phase.

        This method serves as the main computational entry point for QML
        estimation, coordinating the setup and execution of the quadratic
        maximum likelihood algorithm. It orchestrates both initialization
        and the parallel computation phases.

        Notes
        -----
        The computation phase consists of two main steps:

        **1. Setup Phase:**
        Initializes computational arrays and variables via setup_qml_computation():
        - Allocates memory for QML results and noise bias arrays
        - Sizes arrays based on number of multipoles and simulations
        - Prepares data structures for parallel computation

        **2. Computation Phase:**
        Executes the main QML algorithm via compute_qml_spectra():
        - Distributes multipole computations across MPI processes
        - Computes signal derivatives and E-operators for each multipole
        - Applies quadratic estimators to all Monte Carlo simulations
        - Reduces results from all processes to master process

        This method assumes that all prerequisite setup has been completed:
        - Field collections and geometry information loaded
        - Covariance matrices computed and inverted
        - Maps read and prepared for analysis
        - Fisher matrix computed and inverted with normalization

        The separation between setup and computation phases allows for
        flexibility in the analysis pipeline and clear separation of
        concerns between initialization and the computationally intensive
        QML estimation phase.

        Examples
        --------
        Called as part of the full analysis pipeline:

        >>> spectra = Spectra("config.yaml")
        >>> # ... setup phases completed in run() ...
        >>> spectra.compute()  # Execute QML computation
        >>> power_spectra = spectra.get_power_spectra()

        Or manually after setup:

        >>> spectra = Spectra("config.yaml", fisher=fisher_instance)
        >>> spectra.setup_maps()
        >>> spectra.setup_fisher_inversion()
        >>> spectra.compute()

        See Also
        --------
        setup_qml_computation : Initialize arrays for QML computation
        compute_qml_spectra : Main parallel QML estimation algorithm
        run : Complete analysis pipeline including this computation phase
        """
        # Setup QML computation variables
        self.setup_qml_computation()

        # Compute QML power spectra
        self.compute_qml_spectra()

    def run(self):
        """
        Execute the complete QML power spectrum analysis pipeline.

        This method orchestrates the entire QML analysis from initial parameter
        setup through final power spectrum computation and output. It handles
        the complex coordination between MPI processes and manages the optimal
        reuse of computational components when Fisher matrix data is available.

        Notes
        -----
        The analysis pipeline consists of several coordinated phases:

        **Phase 1 - Master Process Setup (Rank 0 only):**
        Determines whether to reuse pre-computed Fisher components or perform
        complete setup from scratch:

        - **Fisher Reuse Path:** When a Fisher instance is provided, expensive
          computational components (geometry, covariance matrices, field collections)
          are reused for efficiency
        - **Full Setup Path:** When no Fisher instance is available, complete
          setup is performed including field initialization, geometry computation,
          covariance matrix setup, power spectra loading, and beam functions

        **Phase 2 - QML-Specific Setup:**
        - Map data loading from FITS files with proper field extraction
        - Fisher matrix inversion with normalization factor computation
        - Error bar calculation and output file writing

        **Phase 3 - MPI Data Broadcasting:**
        Essential data structures are broadcast from master to all worker processes:
        - Parameter configurations and field collections
        - Geometry data (pixel vectors, active pixel information)
        - Covariance matrices (both original and inverted forms)
        - Observational maps for all simulations
        - Inverted Fisher matrix and normalization factors

        **Phase 4 - Parallel QML Computation:**
        The main computational phase where multipole estimates are computed
        in parallel across all processes, followed by MPI reduction to
        gather complete results.

        **Phase 5 - Finalization:**
        MPI synchronization ensures all processes complete successfully
        before pipeline termination.

        The method optimizes computational efficiency by:
        1. Reusing expensive Fisher computations when possible
        2. Minimizing data broadcasting through selective variable sharing
        3. Ensuring proper load balancing in parallel computation phases
        4. Managing memory efficiently across different analysis scales

        Raises
        ------
        ValueError
            If required setup components are missing or inconsistent.
        MPIError
            If MPI communication fails during broadcasting or computation phases.
        FileNotFoundError
            If input files (maps, covariance matrices) are not accessible.

        Examples
        --------
        Complete analysis with Fisher reuse:

        >>> fisher = Fisher("config/fisher_config.yaml")
        >>> fisher.run()
        >>> spectra = Spectra("config/qml_config.yaml", fisher=fisher)
        >>> spectra.run()  # Efficient reuse of Fisher components
        >>> results = spectra.get_power_spectra()

        Standalone QML analysis:

        >>> spectra = Spectra("config/standalone_config.yaml")
        >>> spectra.run()  # Complete pipeline including Fisher computation
        >>> power_spectra = spectra.get_power_spectra()
        >>> noise_bias = spectra.get_noise_bias()

        MPI parallel execution:

        >>> # Command line: mpirun -n 16 python qml_analysis.py
        >>> spectra = Spectra("config/large_scale_config.yaml")
        >>> spectra.run()  # Scales across all available processes

        See Also
        --------
        Fisher.run : Fisher matrix computation pipeline for error propagation
        compute : Main QML computation phase executed within this pipeline
        _broadcast_variables : MPI data distribution to worker processes
        """
        # Only rank 0 does the initial setup
        if self.rank == 0:
            # If we have a pre-computed Fisher instance, reuse its setup
            if hasattr(self, "collection") and self.collection is not None:
                self.log("Reusing setup from Fisher instance", level=2)
            else:
                # Setup from Core class
                self.setup_fields()
                self.setup_geometry()
                self.setup_covariance_matrices()
                # Setup Cls and beams with lmax_signal (defaults to 4*nside)
                # This matches the Fortran convention for derivative computation
                self.log(
                    f"Using lmax_signal = {self.lmax_signal} for Cls and beams", level=3
                )
                self.setup_cls(lmax=self.lmax_signal)
                self.setup_beams(lmax=self.lmax_signal)
                # Load covariance matrices for the case when not reusing Fisher instance
                self._load_covariance_matrices()

            # QML-specific setup
            self.setup_maps()
            self.setup_fisher_inversion()

        # Synchronize before broadcasting
        self.comm.Barrier()

        # Broadcast shared variables to all processes
        self._broadcast_variables()

        self.comm.Barrier()

        # Main QML computation (parallel computation)
        self.compute()

        # Finalize
        self.comm.Barrier()

    def _broadcast_variables(self):
        """
        Distribute essential computational data from master to all MPI processes.

        This method broadcasts all necessary data structures from the master process
        (rank 0) to worker processes, ensuring consistent access to shared data
        required for parallel QML computation. The broadcasting is essential for
        the distributed computation model where different processes handle different
        multipoles but need access to the same underlying data.

        Notes
        -----
        The broadcast operation distributes several categories of data:

        **Core Analysis Components:**
        - params: Complete parameter configuration for the analysis
        - collection: Field collection with HEALPix setup and active pixels
        - npixs: Pixel count information for different fields
        - pixact: Active pixel index arrays for efficient data access
        - point_vectors: Pixel pointing vectors for spherical harmonic transforms

        **Covariance Matrix Data:**
        - NCov1, NCov2: Original noise covariance matrices for bias computation
        - invCov1, invCov2: Inverted total covariance matrices for E-operators
        - Cross-correlation matrices broadcast only when do_cross=True

        **Observational Data:**
        - maps1, maps2: CMB observation maps for all Monte Carlo simulations
        - Secondary maps broadcast only for cross-correlation analyses

        **QML-Specific Data:**
        - invfisher: Inverted Fisher matrix for final spectrum normalization
        - normalization: Vecmul factors for beam and pixelization corrections

        Broadcasting Strategy:
        Each variable is broadcast using the pattern:
        ```python
        variable = comm.bcast(variable if rank == 0 else None, root=0)
        ```
        This ensures the master process provides the data while workers receive it.

        Memory Considerations:
        The broadcast operation can be memory-intensive for large datasets,
        particularly for high-resolution analyses where covariance matrices
        and maps consume significant memory. The implementation balances
        memory usage with computational efficiency.

        Examples
        --------
        This method is called automatically during the analysis pipeline:

        >>> # Called internally by run() method
        >>> spectra._broadcast_variables()
        >>> # All processes now have access to shared data structures

        See Also
        --------
        run : Main pipeline method that coordinates this broadcasting
        MPI.bcast : Low-level MPI broadcast operation used internally
        """
        # Broadcast parameters and core variables
        self.params = self.comm.bcast(self.params if self.rank == 0 else None, root=0)
        self.collection = self.comm.bcast(
            self.collection if self.rank == 0 else None, root=0
        )
        self.npixs = self.comm.bcast(self.npixs if self.rank == 0 else None, root=0)
        self.pixact = self.comm.bcast(self.pixact if self.rank == 0 else None, root=0)
        self.point_vectors = self.comm.bcast(
            self.point_vectors if self.rank == 0 else None, root=0
        )

        # Broadcast covariance matrices
        self.NCov1 = self.comm.bcast(self.NCov1 if self.rank == 0 else None, root=0)
        self.invCov1 = self.comm.bcast(self.invCov1 if self.rank == 0 else None, root=0)
        if self.params.do_cross:
            self.NCov2 = self.comm.bcast(self.NCov2 if self.rank == 0 else None, root=0)
            self.invCov2 = self.comm.bcast(
                self.invCov2 if self.rank == 0 else None, root=0
            )

        # Broadcast maps
        self.maps1 = self.comm.bcast(self.maps1 if self.rank == 0 else None, root=0)
        if self.params.do_cross:
            self.maps2 = self.comm.bcast(self.maps2 if self.rank == 0 else None, root=0)

        # Broadcast inverted Fisher matrix and vecmul
        self.invfisher = self.comm.bcast(
            self.invfisher if self.rank == 0 else None, root=0
        )
        self.normalization = self.comm.bcast(
            self.normalization if self.rank == 0 else None, root=0
        )

    def _normalize_spectra(self, spectra: np.ndarray) -> np.ndarray:
        """
        Apply Fisher matrix normalization to produce final power spectrum estimates.

        This method performs the final step in QML estimation by multiplying
        raw quadratic estimates with the inverted Fisher matrix to produce
        optimally-weighted power spectrum estimates. This normalization is
        essential for obtaining unbiased, minimum-variance estimates.

        Parameters
        ----------
        spectra : numpy.ndarray
            Raw QML estimates or noise bias terms to be normalized. Can be either:
            - 1D array with shape (n_total_parameters,) for single realization
            - 2D array with shape (n_simulations, n_total_parameters) for multiple
              Monte Carlo realizations

        Returns
        -------
        numpy.ndarray
            Normalized power spectrum estimates with the same shape as input.
            These represent the final QML power spectrum estimates with optimal
            statistical properties.

        Raises
        ------
        ValueError
            If Fisher matrix inversion or normalization factors are not available.
            This indicates setup_fisher_inversion() was not called successfully.

        Notes
        -----
        The normalization process implements the final QML transformation:

        **Mathematical Foundation:**
        For raw estimates y_i, the normalized estimates are:
        C̃_l = Σ_i,j (F^(-1))_ij * (y_i * vecmul_i)

        where F^(-1) is the inverted Fisher matrix and vecmul_i are the
        normalization factors accounting for beam and pixelization effects.

        **Two-Step Process:**
        1. **Vecmul Application:** Raw estimates are multiplied by normalization
           factors: y'_i = y_i * vecmul_i
        2. **Fisher Multiplication:** Normalized estimates computed via matrix
           multiplication: C̃ = F^(-1) * y'

        **Statistical Properties:**
        The resulting estimates have optimal properties:
        - Unbiased: E[C̃_l] = C_l^true
        - Minimum variance: Var[C̃_l] = (F^(-1))_ll
        - Proper error propagation across multipoles

        **Computational Efficiency:**
        For multiple simulations, the normalization is applied row-wise to
        optimize memory access patterns and minimize redundant computations.

        The vecmul factors account for:
        - Beam convolution effects in the observed maps
        - Finite pixel size and HEALPix pixelization
        - Mode coupling between different multipoles
        - Proper unit conversion for power spectrum measurements

        Examples
        --------
        Normalize QML results for final output:

        >>> raw_estimates = spectra.qml_results  # Shape: (n_sims, n_params)
        >>> final_spectra = spectra._normalize_spectra(raw_estimates)
        >>> print(f"Final spectra shape: {final_spectra.shape}")

        Normalize noise bias terms:

        >>> raw_bias = spectra.qml_noise_bias  # Shape: (n_params,)
        >>> normalized_bias = spectra._normalize_spectra(raw_bias)

        See Also
        --------
        setup_fisher_inversion : Prepares Fisher matrix and normalization factors
        get_power_spectra : Public interface that calls this normalization
        Fisher.get_fisher_matrix : Source of Fisher information matrix
        """
        if self.invfisher is None or self.normalization is None:
            raise ValueError("Fisher inversion and normalization must be set up first.")

        normalized_spectra = np.zeros_like(spectra)

        if spectra.ndim == 1:
            reduced_res_x_normalization = spectra * self.normalization
            normalized_spectra = np.matmul(reduced_res_x_normalization, self.invfisher)
            return normalized_spectra

        for field_idx in range(spectra.shape[0]):
            reduced_res_x_normalization = spectra[field_idx, :] * self.normalization
            normalized_spectra[field_idx, :] = np.matmul(
                reduced_res_x_normalization, self.invfisher
            )

        return normalized_spectra

    def get_power_spectra(self) -> np.ndarray | None:
        """
        Retrieve final QML power spectrum estimates with optimal normalization.

        This method returns the completed power spectrum estimates after full
        QML computation and Fisher matrix normalization. The estimates represent
        optimal, unbiased measurements of the angular power spectra C_l from
        the input CMB observations.

        Returns
        -------
        numpy.ndarray or None
            Final power spectrum estimates with shape (n_simulations, n_parameters)
            where n_parameters = n_spectra * (lmax - 1). Returns None for worker
            processes (rank != 0) or if computation has not completed.

        Notes
        -----
        The returned power spectra have several important properties:

        **Statistical Optimality:**
        - Unbiased: E[C̃_l] = C_l^true for the true underlying power spectrum
        - Minimum variance: Achieves the Cramér-Rao lower bound for estimation
        - Optimal error propagation: Uncertainties properly weighted across multipoles

        **Physical Units:**
        Power spectra are in units consistent with the input map calibration,
        typically μK² for temperature or (μK)² for polarization depending
        on the analysis configuration.

        **Array Structure:**

        - Dimension 0: Monte Carlo simulation index (0 to nsims-1)
        - Dimension 1: Parameter index combining spectrum type and multipole:
          parameter_index = spectrum_index * (lmax-1) + (l-2)
          where l ranges from 2 to lmax

        **Spectrum Ordering:**
        Parameters are ordered by spectrum type (TT, EE, BB, TE, etc.) as
        defined in the field collection configuration, with multipoles
        nested within each spectrum type.

        **Error Information:**
        Statistical uncertainties can be obtained from the Fisher matrix
        covariance via the associated Fisher instance. The diagonal elements
        of F^(-1) provide the marginal variances for each parameter.

        **Multiple Simulations:**

        When multiple Monte Carlo simulations are processed (nsims > 1),
        the mean across simulations provides the central estimate while
        the scatter enables empirical error estimation and null testing.

        Examples
        --------
        Basic power spectrum retrieval:

        >>> spectra = Spectra("config.yaml")
        >>> spectra.run()
        >>> if spectra.rank == 0:  # Only master process has results
        ...     power_spectra = spectra.get_power_spectra()
        ...     print(f"Shape: {power_spectra.shape}")
        ...     # Extract specific spectrum (e.g., TT)
        ...     tt_spectrum = power_spectra[0, :lmax-1]  # First simulation, TT only

        Statistical analysis with multiple simulations:

        >>> power_spectra = spectra.get_power_spectra()
        >>> if power_spectra is not None:
        ...     mean_spectra = np.mean(power_spectra, axis=0)
        ...     std_spectra = np.std(power_spectra, axis=0)
        ...     print(f"Mean C_2: {mean_spectra[0]:.2e}")
        ...     print(f"Standard deviation: {std_spectra[0]:.2e}")

        See Also
        --------
        get_noise_bias : Retrieve noise bias estimates for auto-correlation
        Fisher.get_error_bars : Statistical uncertainties from Fisher matrix
        _normalize_spectra : Internal normalization method called by this function
        """
        if self.rank == 0 and self.qml_results is not None:
            power_spectra = self._normalize_spectra(self.qml_results)
            return power_spectra
        return None

    def get_noise_bias(self) -> np.ndarray | None:
        """
        Retrieve noise bias estimates for auto-correlation power spectra.

        This method returns the computed noise bias terms that arise in
        auto-correlation QML analyses due to the quadratic nature of the
        estimator when applied to the same dataset. These bias terms can
        be subtracted for unbiased power spectrum estimation.

        Returns
        -------
        numpy.ndarray or None
            Noise bias estimates with shape (n_parameters,) where n_parameters =
            n_spectra * (lmax - 1). Returns None for worker processes (rank != 0),
            cross-correlation analyses (do_cross=True), or if computation has
            not completed.

        Notes
        -----
        **Noise Bias Origin:**
        In auto-correlation QML estimation, the quadratic estimator applied to
        the same dataset introduces a bias term:

        bias_l = E[x^T * E_l * x] - C_l = (1/2) * Tr[N * E_l]

        where N is the noise covariance matrix and E_l is the quadratic estimator
        for multipole l. This bias arises because the estimator correlates noise
        realizations with themselves.

        **Cross-Correlation Exemption:**
        Cross-correlation analyses using independent noise realizations are
        naturally free from this bias, so this method returns None when
        do_cross=True.

        **Bias Subtraction:**
        The bias can be removed from power spectrum estimates:
        C_l^unbiased = C_l^raw - bias_l

        Whether bias subtraction is applied during computation depends on
        the remove_nb parameter setting.

        **Physical Interpretation:**
        Noise bias represents the systematic offset introduced by instrumental
        noise when using the same data for both signal estimation and template
        construction. It scales with the noise level and is typically most
        significant at high multipoles where signal-to-noise is low.

        **Normalization:**
        Like power spectrum estimates, noise bias terms are normalized using
        the inverted Fisher matrix and vecmul factors to ensure consistent
        units and proper statistical weighting.

        **Monte Carlo Validation:**
        Noise bias can be empirically validated using Monte Carlo simulations
        with noise-only inputs. The mean QML estimate from pure noise
        realizations should match the computed bias terms.

        Examples
        --------
        Basic noise bias retrieval:

        >>> spectra = Spectra("config.yaml")  # Auto-correlation analysis
        >>> spectra.run()
        >>> if spectra.rank == 0:
        ...     noise_bias = spectra.get_noise_bias()
        ...     if noise_bias is not None:
        ...         print(f"Noise bias shape: {noise_bias.shape}")
        ...         print(f"Bias at l=2: {noise_bias[0]:.2e}")

        Bias subtraction from power spectra:

        >>> power_spectra = spectra.get_power_spectra()
        >>> noise_bias = spectra.get_noise_bias()
        >>> if power_spectra is not None and noise_bias is not None:
        ...     # Subtract bias from each simulation
        ...     corrected_spectra = power_spectra - noise_bias[np.newaxis, :]
        ...     print("Noise bias correction applied")

        Noise bias validation with simulations:

        >>> # Using noise-only simulations to verify bias computation
        >>> noise_bias_theory = spectra.get_noise_bias()
        >>> noise_only_estimates = spectra.get_power_spectra()  # From noise sims
        >>> if both arrays available:
        ...     empirical_bias = np.mean(noise_only_estimates, axis=0)
        ...     residuals = empirical_bias - noise_bias_theory
        ...     print(f"Bias validation RMS: {np.sqrt(np.mean(residuals**2)):.2e}")

        See Also
        --------
        get_power_spectra : Main power spectrum estimates (may include bias)
        _normalize_spectra : Normalization method applied to bias terms
        compute_qml_spectra : Computation method where bias is calculated
        """
        if self.rank == 0 and self.qml_noise_bias is not None:
            noise_bias = self._normalize_spectra(self.qml_noise_bias)
            return noise_bias
        return None
