"""
Pixel-based likelihood for cosmological parameter inference.

This module implements the PICSLike class for computing likelihoods directly in
pixel space, providing an alternative to harmonic-space methods. The approach
is particularly useful for handling incomplete sky coverage, non-Gaussian features,
and scenarios where pixel-space analysis offers computational or methodological
advantages.

The pixel-based likelihood function is computed as:

ln L(θ) = -1/2 * (d - s(θ))^T * C^(-1) * (d - s(θ))

where d is the observed data vector, s(θ) is the theoretical signal prediction
for parameters θ, and C is the total covariance matrix including signal and noise
contributions.

Classes
-------
PICSLike
    Main class for pixel-based likelihood computation inheriting from cosmocore.Core.

Notes
-----
The pixel-based approach offers several advantages:
1. Natural handling of masked regions without harmonic complications
2. Direct treatment of non-Gaussian signals and systematics
3. Efficient cross-correlation analysis between different maps
4. Computational efficiency for certain analysis configurations

The implementation supports MPI parallelization for efficient computation across
parameter grids and handles memory optimization for large-scale analyses.

References
----------
.. [1] Wandelt, B.D., Larson, D.L. & Lakshminarayanan, A. "Global, exact cosmic
   microwave background data analysis using Gibbs sampling"
   Phys. Rev. D 70, 083511 (2004)
.. [2] Jewell, J., Levin, S. & Anderson, C.H. "Application of Monte Carlo algorithms
   to the Bayesian analysis of the cosmic microwave background"
   Astrophys. J. 609, 1-14 (2004)
.. [3] Chu, M. et al. "Cosmological parameter constraints as derived from the Wilkinson
   Microwave Anisotropy Probe data via Gibbs sampling and the Blackwell-Rao estimator"
   Phys. Rev. D 71, 103002 (2005)
.. [4] Eriksen, H.K. et al. "Power Spectrum Estimation from High-Resolution Maps by
   Gibbs Sampling" Astrophys. J. Suppl. 155, 227-241 (2004)
.. [5] Hinshaw, G. et al. "First-Year Wilkinson Microwave Anisotropy Probe (WMAP)
   Observations: Angular Power Spectrum" Astrophys. J. Suppl. 148, 135-159 (2003)
.. [6] Planck Collaboration "Planck 2018 results. V. CMB power spectra and likelihoods"
   Astron. Astrophys. 641, A5 (2020)
.. [7] Tegmark, M. "How to measure CMB power spectra without losing information"
   Phys. Rev. D 55, 5895 (1997) - For connection between pixel-based and QML methods
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from mpi4py import MPI

from cosmocore import (
    Core,
    FieldCollection,
    compute_signal_matrix,
    matrix_inverse_symm,
    read_maps,
)

from .likelihood_result import LikelihoodResult
from .parameter_grid import ParameterGrid


class PICSLike(Core):
    """
    Pixel-based likelihood computation for cosmological parameter inference.

    This class implements pixel-space likelihood analysis for CMB observations,
    providing an alternative to harmonic-space methods. It inherits from
    cosmocore.Core and extends it with pixel-based likelihood functionality
    including parameter grid management, signal covariance computation, and
    likelihood evaluation across parameter space.

    The likelihood function computed is:

    ln L(θ) = -1/2 * (d - s(θ))^T * C^(-1) * (d - s(θ))

    where the total covariance matrix C includes both signal and noise contributions
    and is recomputed for each parameter point θ in the grid.

    Parameters
    ----------
    params_file : str, optional
        Path to YAML parameter file containing analysis configuration.
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
    maps : FieldCollection
        Collection of observed data maps.
    parameter_grid : ParameterGrid
        Grid of parameter values and corresponding theoretical spectra.
    likelihood_result : LikelihoodResult
        Container for computed likelihood values and statistics.
    data_vector : numpy.ndarray
        Flattened data vector from observed maps.
    noise_covariance : numpy.ndarray
        Noise covariance matrix.

    Examples
    --------
    Basic pixel-based likelihood analysis:

    >>> from cosmoforge.picslike import PICSLike
    >>> picslike = PICSLike("config/pixel_analysis.yaml")
    >>> picslike.setup_parameter_grid(param_ranges, theoretical_spectra)
    >>> picslike.compute_likelihood_grid()
    >>> chi2_values = picslike.get_chi_squared()
    >>> best_fit = picslike.get_best_fit()

    MPI parallel execution:

    >>> # Run with: mpirun -n 4 python pixel_analysis.py
    >>> picslike = PICSLike("config/pixel_config.yaml")
    >>> picslike.compute_likelihood_grid()  # Distributed across processes

    Notes
    -----
    The pixel-based likelihood computation is parallelized using MPI, with each
    process handling a subset of parameter grid points. The method scales well
    with the number of parameter points but can be memory-intensive for large
    maps due to covariance matrix storage and inversion.

    For analyses with many parameters or high-resolution maps, consider using
    appropriate computational resources and memory optimization strategies.

    See Also
    --------
    cosmocore.Core : Base class providing fundamental analysis infrastructure
    ParameterGrid : Helper class for parameter space management
    LikelihoodResult : Container for likelihood computation results
    """

    def __init__(
        self,
        params_file: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize pixel-based likelihood computation class.

        Parameters
        ----------
        params_file : str, optional
            Path to YAML configuration file containing analysis parameters.
            If None, parameters must be provided through kwargs or set later.
        **kwargs : Any
            Additional keyword arguments passed to the Core parent class.
            Common options include 'params' for direct parameter object,
            'verbose' for logging level control.

        Notes
        -----
        Initializes the PICSLike class by calling the parent Core constructor
        and setting up MPI communication. The MPI environment must be properly
        initialized before creating PICSLike instances.
        """
        # Initialize parent Core class
        super().__init__(params=params_file, **kwargs)

        # Initialize MPI communication
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Initialize attributes
        self.maps1 = None
        self.maps2 = None
        self.parameter_grid: ParameterGrid | None = None
        self.likelihood_result: LikelihoodResult | None = None
        self.simulation_index: int = 0  # Which simulation to use for likelihood

        if self.rank == 0:
            self.log("PICSLike initialized!")

    def compute_signal_matrix(self, param_point: tuple) -> np.ndarray:
        """
        Compute the theoretical signal covariance matrix from power spectra.

        This method computes the signal covariance matrix S that represents the
        theoretical covariances between different pixels based on the input power
        spectra. The signal matrix is essential for Fisher matrix computation as
        it defines the expected cosmological signal.

        Returns
        -------
        numpy.ndarray
            Signal covariance matrix S with shape (n_active_pixels, n_active_pixels).
            The matrix is symmetric and positive semi-definite.

        Raises
        ------
        ValueError
            If noise covariance matrices (NCov1) have not been set up prior to
            calling this method.

        Notes
        -----
        The signal matrix computation involves:
        1. Initialization of zero matrix with same shape as noise covariance
        2. Population using theoretical power spectra via compute_signal_matrix
        3. Conversion to Fortran memory layout for optimal BLAS performance

        The computation scales as O(N_pix^2 * l_max) and can be memory-intensive
        for high resolution analyses. Progress and timing information are logged
        for the master process (rank 0).

        The signal matrix elements are computed as:
        S_ij = Σ_l (2l+1)/(4π) * C_l * Y_lm(n̂_i) * Y_lm*(n̂_j)

        Examples
        --------
        >>> fisher = Fisher("config.yaml")
        >>> fisher.setup_covariance_matrices()  # Must be called first
        >>> S = fisher.compute_signal_matrix()
        >>> print(f"Signal matrix shape: {S.shape}")
        """
        if self.NCov1 is None:
            raise ValueError("Covariance matrices must be set up first")

        self.Sig = np.zeros_like(self.NCov1, dtype=np.float64)
        self.Sig = np.asfortranarray(self.Sig, dtype=np.float64)

        start_time = time.time() if self.rank == 0 else None

        spectra_dict = self.parameter_grid.get_spectrum(param_point)

        self.collection.set_cls(spectra_dict)
        self.collection.set_beams()

        compute_signal_matrix(
            S=self.Sig,
            lmax=self.params.lmax,
            fields=self.collection,
        )

        if self.rank == 0 and start_time is not None:
            elapsed = time.time() - start_time
            self.log(f"Signal matrix computed in {elapsed:.2f} seconds", level=3)
            self.log(f"Signal matrix shape: {self.Sig.shape}", level=4)
            self.log(f"Signal matrix first row: {self.Sig[0, :10]}", level=4)

        return self.Sig

    def prepare_covariance_matrix(self):
        """
        Prepare total covariance matrices and compute their inverses.

        This method combines the signal and noise covariance matrices to form
        the total covariance matrix C = S + N, then computes the matrix inverses
        required for Fisher matrix calculation. The inverted matrices are also
        written to disk for potential reuse.

        Notes
        -----
        The preparation process involves:
        1. Adding signal matrix to noise covariance: C = N + S
        2. Computing matrix inverse using Cholesky decomposition for stability
        3. Writing inverse matrices to files specified in parameters
        4. Handling cross-correlation case with secondary covariance matrix

        For cross-correlation analyses (params.do_cross = True), both primary
        and secondary covariance matrices are processed. The matrices are
        converted to Fortran memory layout for optimal linear algebra performance.

        Matrix inversion uses symmetric positive definite properties for
        numerical stability via cosmocore.matrix_inverse_symm().

        Raises
        ------
        LinAlgError
            If covariance matrices are not positive definite (e.g., due to
            insufficient noise regularization or numerical precision issues).

        Examples
        --------
        >>> fisher = Fisher("config.yaml")
        >>> fisher.setup_signal_matrix()
        >>> fisher.prepare_covariance_matrices()
        # Inverse covariance matrices are now ready for Fisher computation
        """
        # Add signal to noise covariance
        self.NCov1 = self.NCov1 + self.Sig
        self.NCov1 = np.asfortranarray(self.NCov1)
        self.log(f"Combined covariance matrix shape: {self.NCov1.shape}", level=4)

        # Compute inverse covariance matrices
        self.invCov = matrix_inverse_symm(self.NCov1)
        self.log("Computed inverse of primary covariance matrix", level=4)

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

    def set_simulation_index(self, sim_idx: int) -> None:
        """
        Set which simulation to use for likelihood computation.

        Parameters
        ----------
        sim_idx : int
            Index of the simulation to use (0-based indexing).
            Must be within the range [0, nsims-1].

        Raises
        ------
        ValueError
            If sim_idx is out of range for the available simulations.

        Notes
        -----
        In a real analysis, this would typically be 0 since you have only
        one observational dataset. For Monte Carlo studies, you might want
        to compute likelihoods for different simulations separately.
        """
        if self.maps1 is not None:
            max_sim = self.maps1.shape[1] - 1
            if sim_idx < 0 or sim_idx > max_sim:
                raise ValueError(
                    f"Simulation index {sim_idx} out of range [0, {max_sim}]"
                )

        self.simulation_index = sim_idx
        if self.rank == 0:
            self.log(f"Set simulation index to: {sim_idx}")

    def setup_parameter_grid(self) -> None:
        """
        Set up parameter grid and associated theoretical spectra.

        Parameters
        ----------
        parameter_ranges : dict[str, np.ndarray]
            Dictionary mapping parameter names to their value ranges.
            Each value should be a 1D array of parameter values to evaluate.
        theoretical_spectra : dict[tuple, np.ndarray]
            Dictionary mapping parameter value tuples to theoretical power spectra.
            Keys are tuples of parameter values, values are power spectra arrays.

        Examples
        --------
        >>> param_ranges = {
        ...     'omega_b': np.linspace(0.02, 0.025, 10),
        ...     'omega_c': np.linspace(0.10, 0.14, 10),
        ... }
        >>> theory_spectra = {
        ...     (0.022, 0.12): cl_theory_1,
        ...     (0.023, 0.12): cl_theory_2,
        ...     # ... more spectra
        ... }
        >>> picslike.setup_parameter_grid(param_ranges, theory_spectra)

        Notes
        -----
        The parameter grid defines the points in parameter space where the
        likelihood will be evaluated. Each point must have a corresponding
        theoretical power spectrum for signal covariance computation.
        """
        if self.rank == 0:
            print("Setting up parameter grid...")

        self.parameter_names = self.params.parameters.keys()

        self.parameter_ranges = {
            name: np.linspace(*self.params.parameters[name])
            for name in self.parameter_names
        }

        self.parameter_grid = ParameterGrid(
            core_params=self.params,
            parameter_ranges=self.parameter_ranges,
            root_dir=self.params.root_dir,
            root_filename=self.params.root_filename,
        )

        if self.rank == 0:
            total_points = self.parameter_grid.get_total_points()
            print(f"Parameter grid contains {total_points} points")

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
        self.collection: FieldCollection = self.comm.bcast(
            self.collection if self.rank == 0 else None, root=0
        )

        self.npixs = self.comm.bcast(self.npixs if self.rank == 0 else None, root=0)
        self.pixact = self.comm.bcast(self.pixact if self.rank == 0 else None, root=0)
        self.point_vectors = self.comm.bcast(
            self.point_vectors if self.rank == 0 else None, root=0
        )

        # Broadcast covariance matrices
        self.NCov1 = self.comm.bcast(self.NCov1 if self.rank == 0 else None, root=0)

        # Broadcast maps
        self.maps1 = self.comm.bcast(self.maps1 if self.rank == 0 else None, root=0)
        if self.params.do_cross:
            self.maps2 = self.comm.bcast(self.maps2 if self.rank == 0 else None, root=0)

    def compute(self) -> None:
        """
        Compute likelihood across the entire parameter grid for all simulations.

        Evaluates the likelihood function at each point in the parameter grid
        for each simulation, distributing the computation across MPI processes
        for efficiency. The signal covariance matrix is recomputed for each
        parameter point using the corresponding theoretical power spectrum.

        Notes
        -----
        This is the main computational routine that:
        1. Distributes parameter grid points across MPI processes
        2. For each simulation:
           - Computes signal covariance matrix for each parameter point
           - Inverts total covariance matrix (signal + noise)
           - Evaluates likelihood function
        3. Gathers results from all processes
        4. Stores collection of LikelihoodResult objects (one per simulation)

        The computation scales as O(N_sim * N_param * N_pix^3) where N_sim is
        the number of simulations, N_param is the number of parameter points
        and N_pix is the number of pixels.
        """
        # Store results for each simulation
        simulation_results = []

        # Get parameter points for this MPI process
        param_points = self.parameter_grid.get_points_for_process(self.rank, self.size)
        n_sims = self.params.nsims

        # Initialize results storage for this simulation
        local_chi2_values = np.zeros((len(param_points), n_sims))
        local_log_likelihood = np.zeros((len(param_points), n_sims))

        # Compute likelihood for each parameter point for this simulation
        for i, param_point in enumerate(param_points):
            chi2, log_like = self._compute_likelihood_point(param_point)
            local_chi2_values[i] = chi2
            local_log_likelihood[i] = log_like

        # Gather results from all processes for this simulation
        all_chi2 = self.comm.gather(local_chi2_values, root=0)
        all_log_like = self.comm.gather(local_log_likelihood, root=0)

        # if self.rank == 0:
        # Combine results from all processes for this simulation
        combined_chi2 = np.concatenate(all_chi2)
        combined_log_like = np.concatenate(all_log_like)

        # Create LikelihoodResult for this simulation
        for i in range(n_sims):
            sim_result = LikelihoodResult(
                parameter_grid=self.parameter_grid,
                chi_squared_values=combined_chi2[:, i],
                log_likelihood_values=combined_log_like[:, i],
            )
            simulation_results.append(sim_result)

        # if self.rank == 0:
        # Store the collection of results
        self.simulation_results = simulation_results

        print(self.simulation_results)

        # Compute mean likelihood result for plotting
        self.likelihood_result = self._compute_mean_likelihood_result(simulation_results)

        print(f"Likelihood computation completed for {n_sims} simulations")
        print("Mean likelihood result computed for analysis")

        # # Broadcast results to all processes if needed
        # self.likelihood_result = self.comm.bcast(self.likelihood_result, root=0)
        # if hasattr(self, "simulation_results"):
        #     self.simulation_results = self.comm.bcast(
        #         self.simulation_results if self.rank == 0 else None, root=0
        #     )

    def _compute_mean_likelihood_result(
        self, simulation_results: list[LikelihoodResult]
    ) -> LikelihoodResult:
        """
        Compute mean likelihood result from multiple simulations.

        Parameters
        ----------
        simulation_results : list[LikelihoodResult]
            List of LikelihoodResult objects, one for each simulation.

        Returns
        -------
        LikelihoodResult
            Mean likelihood result computed by averaging chi-squared and
            log-likelihood values across all simulations.

        Notes
        -----
        This method computes the ensemble average of likelihood values
        across all simulations, which provides:
        1. Reduced statistical noise in likelihood surface
        2. Better estimate of expected likelihood behavior
        3. More robust parameter estimation

        The mean is computed as:
        - chi² = mean(chi²_i) across simulations i
        - log-likelihood = mean(log L_i) across simulations i
        """
        if not simulation_results:
            raise ValueError("No simulation results provided")

        n_points = len(simulation_results[0].chi_squared_values)

        # Initialize arrays for mean computation
        mean_chi2 = np.zeros(n_points)
        mean_log_like = np.zeros(n_points)

        # Compute means across simulations
        for i in range(n_points):
            chi2_values = [result.chi_squared_values[i] for result in simulation_results]
            log_like_values = [
                result.log_likelihood_values[i] for result in simulation_results
            ]

            mean_chi2[i] = np.mean(chi2_values)
            mean_log_like[i] = np.mean(log_like_values)

        # Create mean LikelihoodResult
        mean_result = LikelihoodResult(
            parameter_grid=simulation_results[0].parameter_grid,
            chi_squared_values=mean_chi2,
            log_likelihood_values=mean_log_like,
        )

        return mean_result

    def _compute_likelihood_point(self, param_point: tuple) -> tuple[float, float]:
        """
        Compute likelihood for a single parameter point.

        Parameters
        ----------
        param_point : tuple
            Tuple of parameter values defining a point in parameter space.

        Returns
        -------
        chi_squared : float
            Chi-squared value for this parameter point.
        log_likelihood : float
            Log-likelihood value for this parameter point.

        Notes
        -----
        This method performs the core likelihood computation:
        1. Retrieves theoretical spectrum for the parameter point
        2. Computes signal covariance matrix from the spectrum
        3. Assembles total covariance matrix (signal + noise)
        4. Inverts the covariance matrix
        5. Computes chi-squared and log-likelihood

        For the simulation dimension, we use the first simulation as the "observed" data.
        In a real analysis, this would be the actual observational data.
        """
        # Compute signal covariance matrix for this parameter point
        self.compute_signal_matrix(param_point)
        self.log(f"Signal matrix computed for parameters: {param_point}", level=3)

        # Prepare total covariance matrix and its inverse
        self.prepare_covariance_matrix()
        self.log("Total covariance matrix prepared and inverted", level=3)

        # Compute chi-squared: d^T * C^-1 * d
        logdet = np.linalg.slogdet(self.NCov1)[1]
        self.log(f"Log-determinant of covariance: {logdet:.2f}", level=3)

        if self.params.do_cross:
            chi_squared = (
                np.einsum("in,ij,jn->n", self.maps1, self.invCov, self.maps2) + logdet
            )
        else:
            chi_squared = (
                np.einsum("in,ij,jn->n", self.maps1, self.invCov, self.maps1) + logdet
            )

        # Log-likelihood (excluding normalization constants)
        log_likelihood = -0.5 * chi_squared

        return chi_squared, log_likelihood

    def get_chi_squared(self) -> np.ndarray:
        """
        Get chi-squared values from likelihood computation.

        Returns
        -------
        chi_squared : numpy.ndarray
            Array of chi-squared values corresponding to parameter grid points.

        Raises
        ------
        RuntimeError
            If likelihood computation has not been performed.
        """
        if self.likelihood_result is None:
            msg = "Likelihood not computed. Call compute_likelihood_grid() first."
            raise RuntimeError(msg)

        return self.likelihood_result.chi_squared_values

    def get_log_likelihood(self) -> np.ndarray:
        """
        Get log-likelihood values from likelihood computation.

        Returns
        -------
        log_likelihood : numpy.ndarray
            Array of log-likelihood values corresponding to parameter grid points.

        Raises
        ------
        RuntimeError
            If likelihood computation has not been performed.
        """
        if self.likelihood_result is None:
            msg = "Likelihood not computed. Call compute_likelihood_grid() first."
            raise RuntimeError(msg)

        return self.likelihood_result.log_likelihood_values

    def get_best_fit(self) -> dict[str, float]:
        """
        Get best-fit parameter values from likelihood computation.

        Returns
        -------
        best_fit_params : dict[str, float]
            Dictionary mapping parameter names to their best-fit values.

        Raises
        ------
        RuntimeError
            If likelihood computation has not been performed.
        """
        if self.likelihood_result is None:
            msg = "Likelihood not computed. Call compute_likelihood_grid() first."
            raise RuntimeError(msg)

        return self.likelihood_result.get_best_fit()

    def get_simulation_results(self) -> list[LikelihoodResult]:
        """
        Get likelihood results for each individual simulation.

        Returns
        -------
        simulation_results : list[LikelihoodResult]
            List of LikelihoodResult objects, one for each simulation.

        Raises
        ------
        RuntimeError
            If likelihood computation has not been performed.

        Notes
        -----
        This method provides access to the individual likelihood results
        for each simulation, allowing for:
        1. Analysis of simulation-to-simulation variations
        2. Computation of error bars and confidence intervals
        3. Statistical studies of likelihood behavior
        """
        if not hasattr(self, "simulation_results") or self.simulation_results is None:
            msg = "Simulation results not available. Call compute() first."
            raise RuntimeError(msg)

        return self.simulation_results

    def get_mean_likelihood_result(self) -> LikelihoodResult:
        """
        Get the mean likelihood result averaged over all simulations.

        Returns
        -------
        mean_result : LikelihoodResult
            Mean likelihood result used for plotting and analysis.

        Raises
        ------
        RuntimeError
            If likelihood computation has not been performed.

        Notes
        -----
        This returns the same result as get_chi_squared(), get_log_likelihood(),
        etc., but provides direct access to the averaged LikelihoodResult object.
        """
        if self.likelihood_result is None:
            msg = "Likelihood not computed. Call compute() first."
            raise RuntimeError(msg)

        return self.likelihood_result

    def save_results(self, output_path: str) -> None:
        """
        Save likelihood computation results to file.

        Parameters
        ----------
        output_path : str
            Base path where results should be saved. Individual simulation
            results will be saved with numbered suffixes.

        Notes
        -----
        Saves both the mean likelihood results and individual simulation results:
        - output_path: Mean likelihood result (for plotting and analysis)
        - output_path_sim_XX: Individual simulation results (for error analysis)

        The mean result is used for standard likelihood analysis and plotting,
        while individual results enable Monte Carlo error estimation.
        """
        if self.likelihood_result is None:
            msg = "No results to save. Run compute() first."
            raise RuntimeError(msg)

        if self.rank == 0:
            # Save mean likelihood result
            self.likelihood_result.save(output_path)
            print(f"Mean likelihood results saved to {output_path}")

            # Save individual simulation results if available
            if (
                hasattr(self, "simulation_results")
                and self.simulation_results is not None
            ):
                from pathlib import Path

                base_path = Path(output_path)
                base_dir = base_path.parent
                base_name = base_path.stem
                base_ext = base_path.suffix

                for i, sim_result in enumerate(self.simulation_results):
                    sim_path = base_dir / f"{base_name}_sim_{i:02d}{base_ext}"
                    sim_result.save(str(sim_path))

                n_files = len(self.simulation_results)
                print(f"Individual simulation results saved ({n_files} files)")
                print("Use get_simulation_results() to access individual results")

    def run(self):
        """
        Execute the complete pixel-based likelihood analysis pipeline.

        This method implements the abstract method from Core and orchestrates
        the full analysis workflow from initialization to final results.

        Returns
        -------
        LikelihoodResult
            Final likelihood computation results.

        Notes
        -----
        The complete pipeline includes:
        1. Field and geometry setup
        2. Covariance matrix preparation
        3. Parameter grid configuration
        4. Likelihood computation across parameter space
        5. Statistical analysis and result storage

        This method provides a high-level interface for running the complete
        pixel-based likelihood analysis with minimal user intervention.
        """

        self.setup_parameter_grid()
        self.log("Starting PICSLike analysis pipeline", level=1)
        self.setup_fields()
        self.setup_geometry()
        self.setup_covariance_matrices()
        self.setup_cls()
        self.setup_beams()
        self.setup_maps()

        # if self.rank == 0:
        #     self.log("Starting PICSLike analysis pipeline", level=1)
        #     self.setup_fields()
        #     self.setup_geometry()
        #     self.setup_covariance_matrices()
        #     self.setup_cls()
        #     self.setup_beams()

        #     if self.maps1 is None:
        #         self.setup_maps()
        #     if self.params.do_cross:
        #         assert self.maps2 is not None, (
        #             "Maps2 should be loaded for cross-correlation."
        #         )

        self.comm.Barrier()

        self._broadcast_variables()

        self.comm.Barrier()

        self.compute()

        self.comm.Barrier()
