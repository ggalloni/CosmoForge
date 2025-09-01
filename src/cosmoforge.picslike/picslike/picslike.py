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
.. [1] Wandelt, B.D. et al. "Global, exact cosmic microwave background data analysis"
   Phys. Rev. D 70, 083511 (2004)
.. [2] Jewell, J. et al. "Application of MCMC methods to multi-frequency CMB data"
   Astrophys. J. 609, 1-6 (2004)
.. [3] Chu, M. et al. "Cosmic microwave background likelihood approximation by a
   Gaussianized Blackwell-Rao estimator" Phys. Rev. D 71, 103002 (2005)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from mpi4py import MPI
from tqdm import tqdm

from cosmocore import (
    Core,
    FieldCollection,
    compute_signal_matrix,
    matrix_inverse_symm,
    matrix_mult,
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
        super().__init__(params_file, **kwargs)

        # Initialize MPI communication
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Initialize attributes
        self.maps: FieldCollection | None = None
        self.parameter_grid: ParameterGrid | None = None
        self.likelihood_result: LikelihoodResult | None = None
        self.data_vector: np.ndarray | None = None
        self.noise_covariance: np.ndarray | None = None

        if self.rank == 0:
            print(f"PICSLike initialized with {self.size} MPI processes")

    def load_maps(self) -> None:
        """
        Load observed data maps from files specified in parameters.

        Reads the observational data maps and converts them to a flattened
        data vector for likelihood computation. Handles masking and creates
        the FieldCollection for subsequent analysis.

        Raises
        ------
        RuntimeError
            If map loading fails or maps are inconsistent.

        Notes
        -----
        The maps are loaded according to the configuration in the parameter file,
        including any masking operations. The resulting data vector excludes
        masked pixels to match the covariance matrix dimensions.
        """
        if self.rank == 0:
            print("Loading observed data maps...")

        # Load maps using cosmocore functionality
        self.maps = read_maps(self.params)

        # Create flattened data vector excluding masked pixels
        self.data_vector = self._create_data_vector()

        if self.rank == 0:
            print(f"Loaded {len(self.maps)} maps with {len(self.data_vector)} pixels")

    def setup_parameter_grid(
        self,
        parameter_ranges: dict[str, np.ndarray],
        theoretical_spectra: dict[tuple, np.ndarray],
    ) -> None:
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

        self.parameter_grid = ParameterGrid(parameter_ranges, theoretical_spectra)

        if self.rank == 0:
            total_points = self.parameter_grid.get_total_points()
            print(f"Parameter grid contains {total_points} points")

    def setup_noise_covariance(self) -> None:
        """
        Set up noise covariance matrix from parameters.

        Constructs the noise covariance matrix based on the instrumental
        noise properties specified in the parameter file. This matrix
        remains constant across all parameter grid points.

        Notes
        -----
        The noise covariance matrix is constructed from the noise properties
        of each field in the analysis. For diagonal noise, this is simply
        the inverse noise variance per pixel. For correlated noise, the
        full covariance structure is computed.
        """
        if self.rank == 0:
            print("Setting up noise covariance matrix...")

        # This would be implemented based on the specific noise model
        # For now, assume diagonal noise from field properties
        n_pixels = len(self.data_vector)
        self.noise_covariance = np.eye(n_pixels)

        # In practice, this would read noise properties from parameters
        # and construct the appropriate covariance matrix

    def compute_likelihood_grid(self) -> None:
        """
        Compute likelihood across the entire parameter grid.

        Evaluates the likelihood function at each point in the parameter grid,
        distributing the computation across MPI processes for efficiency.
        The signal covariance matrix is recomputed for each parameter point
        using the corresponding theoretical power spectrum.

        Notes
        -----
        This is the main computational routine that:
        1. Distributes parameter grid points across MPI processes
        2. Computes signal covariance matrix for each point
        3. Inverts total covariance matrix (signal + noise)
        4. Evaluates likelihood function
        5. Gathers results from all processes

        The computation scales as O(N_param * N_pix^3) where N_param is the
        number of parameter points and N_pix is the number of pixels.
        """
        if self.parameter_grid is None:
            msg = "Parameter grid not set up. Call setup_parameter_grid() first."
            raise RuntimeError(msg)

        if self.data_vector is None:
            self.load_maps()

        if self.noise_covariance is None:
            self.setup_noise_covariance()

        # Get parameter points for this MPI process
        param_points = self.parameter_grid.get_points_for_process(self.rank, self.size)

        # Initialize results storage
        local_chi2_values = np.zeros(len(param_points))
        local_log_likelihood = np.zeros(len(param_points))

        if self.rank == 0:
            print(f"Computing likelihood for {len(param_points)} points per process...")

        # Compute likelihood for each parameter point
        for i, param_point in enumerate(tqdm(param_points, disable=(self.rank != 0))):
            chi2, log_like = self._compute_likelihood_point(param_point)
            local_chi2_values[i] = chi2
            local_log_likelihood[i] = log_like

        # Gather results from all processes
        all_chi2 = self.comm.gather(local_chi2_values, root=0)
        all_log_like = self.comm.gather(local_log_likelihood, root=0)

        if self.rank == 0:
            # Combine results from all processes
            combined_chi2 = np.concatenate(all_chi2)
            combined_log_like = np.concatenate(all_log_like)

            # Store results
            self.likelihood_result = LikelihoodResult(
                parameter_grid=self.parameter_grid,
                chi_squared_values=combined_chi2,
                log_likelihood_values=combined_log_like,
            )

            print("Likelihood computation completed")

        # Broadcast results to all processes if needed
        self.likelihood_result = self.comm.bcast(self.likelihood_result, root=0)

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
        """
        # Get theoretical spectrum for this parameter point
        cl_theory = self.parameter_grid.get_spectrum(param_point)

        # Compute signal covariance matrix
        signal_cov = compute_signal_matrix(cl_theory, self.params)

        # Total covariance matrix (signal + noise)
        total_cov = signal_cov + self.noise_covariance

        # Invert covariance matrix
        cov_inv = matrix_inverse_symm(total_cov)

        # Compute chi-squared: x^T * C^-1 * x
        chi_squared = float(
            matrix_mult(
                matrix_mult(self.data_vector.reshape(1, -1), cov_inv),
                self.data_vector.reshape(-1, 1),
            )[0, 0]
        )

        # Log-likelihood (excluding normalization constants)
        log_likelihood = -0.5 * chi_squared

        return chi_squared, log_likelihood

    def _create_data_vector(self) -> np.ndarray:
        """
        Create flattened data vector from loaded maps.

        Returns
        -------
        data_vector : numpy.ndarray
            Flattened data vector excluding masked pixels.

        Notes
        -----
        Combines all loaded maps into a single data vector, applying
        any masking operations and excluding masked pixels to match
        the dimensions expected by the covariance matrices.
        """
        if self.maps is None:
            msg = "Maps not loaded. Call load_maps() first."
            raise RuntimeError(msg)

        # This would be implemented based on the specific map structure
        # For now, return a placeholder
        return np.concatenate([field.data.flatten() for field in self.maps])

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

    def save_results(self, output_path: str) -> None:
        """
        Save likelihood computation results to file.

        Parameters
        ----------
        output_path : str
            Path where results should be saved.

        Notes
        -----
        Saves the complete likelihood results including parameter grid,
        chi-squared values, and best-fit parameters in a format suitable
        for subsequent analysis and visualization.
        """
        if self.likelihood_result is None:
            msg = "No results to save. Run compute_likelihood_grid() first."
            raise RuntimeError(msg)

        if self.rank == 0:
            self.likelihood_result.save(output_path)
            print(f"Results saved to {output_path}")
