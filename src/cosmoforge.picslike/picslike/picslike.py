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
        compression: dict | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize pixel-based likelihood computation class.

        Parameters
        ----------
        params_file : str, optional
            Path to YAML configuration file.
        compression : dict, optional
            Compression configuration (method, epsilon, basis, mode_fraction).
        **kwargs : Any
            Additional arguments passed to Core.
        """
        # Initialize parent Core class
        super().__init__(params=params_file, **kwargs)

        # Initialize MPI communication
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Store compression config
        self._compression_config = compression

        # Initialize attributes
        self.maps1 = None
        self.maps2 = None
        self.parameter_grid: ParameterGrid | None = None
        self.likelihood_result: LikelihoodResult | None = None
        self.simulation_index: int = 0  # Which simulation to use for likelihood
        self._lmax_signal = None
        self.fiducial_spectrum: dict | None = None
        self._legendre_cache: np.ndarray | None = (
            None  # Pre-computed Legendre polynomials
        )

        if self.rank == 0:
            self.log("PICSLike initialized!")

    @property
    def lmax_signal(self) -> int:
        """Maximum multipole for signal matrix computation (defaults to 4*nside)."""
        if self._lmax_signal is not None:
            return self._lmax_signal
        return 4 * self.params.nside

    @lmax_signal.setter
    def lmax_signal(self, value: int) -> None:
        """Set custom lmax_signal value."""
        self._lmax_signal = value

    def compute_signal_matrix(self, param_point: tuple) -> np.ndarray:
        """
        Compute signal covariance matrix for a given parameter point.

        Returns
        -------
        np.ndarray
            Signal covariance matrix S, shape (n_active_pixels, n_active_pixels).

        Raises
        ------
        ValueError
            If noise covariance matrices not set up.
        """
        if self.noise_cov1 is None:
            raise ValueError("Covariance matrices must be set up first")

        self.signal_matrix = np.zeros_like(self.noise_cov1, dtype=np.float64)
        self.signal_matrix = np.asfortranarray(self.signal_matrix, dtype=np.float64)

        start_time = time.time() if self.rank == 0 else None

        spectra_dict = self.parameter_grid.get_spectrum(param_point)

        self.collection.set_cls(spectra_dict, lmax=self.lmax_signal)
        self.collection.set_beams(lmax=self.lmax_signal)

        compute_signal_matrix(
            S=self.signal_matrix,
            lmax=self.lmax_signal,
            fields=self.collection,
        )

        if self.rank == 0 and start_time is not None:
            elapsed = time.time() - start_time
            self.log(f"Signal matrix computed in {elapsed:.2f} seconds", level=3)
            self.log(f"Signal matrix shape: {self.signal_matrix.shape}", level=4)
            self.log(f"Signal matrix first row: {self.signal_matrix[0, :10]}", level=4)

        return self.signal_matrix

    def prepare_covariance_matrix(self):
        """
        Prepare total covariance C = S + N and compute its inverse.

        Modifies self.signal_matrix in-place to form total covariance, then inverts it.
        """
        # Add noise to signal IN-PLACE to form total covariance.
        # This avoids allocating a separate array (saves n_pix^2 memory).
        # Safe because self.signal_matrix is recreated fresh for each parameter point.
        self.signal_matrix += self.noise_cov1
        self.signal_matrix = np.asfortranarray(self.signal_matrix)

        # Alias for clarity in downstream code (e.g., slogdet).
        # This is just a reference, no memory copy.
        self.total_cov = self.signal_matrix
        self.log(f"Combined covariance matrix shape: {self.total_cov.shape}", level=4)

        # Compute inverse covariance matrix
        self.inv_cov = matrix_inverse_symm(self.total_cov)
        self.log("Computed inverse of primary covariance matrix", level=4)

    def setup_maps(self):
        """
        Read observational map data from FITS files.

        Loads maps with proper pixel selection, field extraction, and calibration.
        For cross-correlation (do_cross=True), loads both primary and secondary maps.
        Output shape: (n_active_pixels, n_simulations).

        Raises
        ------
        ValueError
            If pixel information not available (setup_geometry not called).
        FileNotFoundError
            If input map files not found.
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
        """Set up parameter grid from config and load associated theoretical spectra."""
        if self.rank == 0:
            self.log("Setting up parameter grid...", level=2)

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
            lmax_signal=self.lmax_signal,
        )

        if self.rank == 0:
            total_points = self.parameter_grid.get_total_points()
            self.log(f"Parameter grid contains {total_points} points", level=2)

    def _broadcast_variables(self):
        """Broadcast essential data from master to all MPI worker processes."""
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
        self.noise_cov1 = self.comm.bcast(
            self.noise_cov1 if self.rank == 0 else None, root=0
        )

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

        if self.rank == 0:
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

            # Store the collection of results
            self.simulation_results = simulation_results

            # Compute mean likelihood result for plotting
            self.likelihood_result = self._compute_mean_likelihood_result(
                simulation_results
            )

            self.log(f"Likelihood computation completed for {n_sims} simulations")
            self.log("Mean likelihood result computed for analysis", level=2)

    def _compute_mean_likelihood_result(
        self, simulation_results: list[LikelihoodResult]
    ) -> LikelihoodResult:
        """Compute mean likelihood result by averaging over simulations."""
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
        """Compute chi-squared and log-likelihood for a single parameter point."""
        # Check if compression is enabled
        use_compression = (
            hasattr(self, "compression_manager") and self.compression_manager is not None
        )

        if use_compression:
            # Compressed path: use SMW formula - O(n_modes³) instead of O(N_pix³)
            # Key optimization: we DON'T compute the full signal matrix!
            # Instead, we work directly with C_ell in harmonic space.

            # Get spectrum for this parameter point and set up C_ell
            # Note: Only the signal spectrum changes between parameter points.
            # Beams are loaded once during setup (setup_beams), but smoothing
            # must be re-applied after each set_cls call.
            spectra_dict = self.parameter_grid.get_spectrum(param_point)
            self.collection.set_cls(spectra_dict, lmax=self.lmax_signal)

            # Apply beam smoothing to the new spectra
            # This is much faster than set_beams() which re-reads beam files
            self.collection.beam_manager.apply_smoothing(
                self.collection.spectra_manager, lmax=self.lmax_signal
            )

            # Detect if we need the multi-field path (>1 components or spin-2)
            has_spin2 = any(f.spin == 2 for f in self.collection.fields)
            is_multi_field = len(self.collection.fields) > 1 or has_spin2

            if is_multi_field:
                # Multi-field/spin-2 path: use full Lambda matrix
                C_ell_dict = self._build_c_ell_dict()
                self.log(f"C_ell_dict set for parameters: {param_point}", level=3)

                # Precompute K Cholesky and logdet ONCE for this parameter point
                K_chol, _, logdet = self.compression_manager.prepare_smw(C_ell_dict)

                chi_squared = []
                for sim_idx in range(self.params.nsims):
                    if self.params.do_cross:
                        d1 = self.maps1[:, sim_idx]
                        d2 = self.maps2[:, sim_idx]
                        d1_compressed = self.compression_manager.compress_data(d1)
                        d2_compressed = self.compression_manager.compress_data(d2)
                        C_c_inv = self.compression_manager.get_compressed_inverse(
                            C_ell_dict
                        )
                        chi_sq = float(d1_compressed.T @ C_c_inv @ d2_compressed)
                    else:
                        chi_sq = self.compression_manager.quadratic_form_from_prepared(
                            self.maps1[:, sim_idx], K_chol
                        )
                    chi_squared.append(chi_sq)

                chi_squared = np.array(chi_squared)
            else:
                # Single-field path (existing code)
                C_ell = self.collection.spectra_manager.get_cls(0, 0, 0)
                self.log(f"C_ell set for parameters: {param_point}", level=3)

                chi_squared = []
                for sim_idx in range(self.params.nsims):
                    if self.params.do_cross:
                        d1 = self.maps1[:, sim_idx]
                        d2 = self.maps2[:, sim_idx]
                        d1_compressed = self.compression_manager.compress_data(d1)
                        d2_compressed = self.compression_manager.compress_data(d2)
                        C_inv = self.compression_manager.get_compressed_inverse(C_ell)
                        chi_sq = float(d1_compressed.T @ C_inv @ d2_compressed)
                    else:
                        chi_sq = self.compute_quadratic_form(
                            self.maps1[:, sim_idx], C_ell
                        )
                    chi_squared.append(chi_sq)

                chi_squared = np.array(chi_squared)
                logdet = self.get_covariance_logdet(C_ell)

            self.log(f"Log-determinant of covariance (compressed): {logdet:.2f}", level=3)
        else:
            # Non-compressed path: compute full signal matrix
            self.compute_signal_matrix(param_point)
            self.log(f"Signal matrix computed for parameters: {param_point}", level=3)
            # Non-compressed path: use existing direct matrix operations
            self.prepare_covariance_matrix()
            self.log("Total covariance matrix prepared and inverted", level=3)

            # Compute chi-squared: d^T * C^-1 * d (without logdet)
            if self.params.do_cross:
                chi_squared = np.einsum(
                    "in,ij,jn->n", self.maps1, self.inv_cov, self.maps2
                )
            else:
                chi_squared = np.einsum(
                    "in,ij,jn->n", self.maps1, self.inv_cov, self.maps1
                )

            # Get log-determinant for likelihood computation
            logdet = np.linalg.slogdet(self.total_cov)[1]
            self.log(f"Log-determinant of covariance: {logdet:.2f}", level=3)

        # Standard Gaussian log-likelihood: -0.5 * (χ² + log|C|)
        # (ignoring constant n*log(2π) term)
        log_likelihood = -0.5 * (chi_squared + logdet)

        return chi_squared, log_likelihood

    def _build_c_ell_dict(self) -> dict[tuple, np.ndarray]:
        """Build C_ell_dict from spectra_manager for compressed operations."""
        C_ell_dict, _ = self.collection.spectra_manager.build_inputs()
        return C_ell_dict

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
        list[LikelihoodResult]
            One LikelihoodResult per simulation.

        Raises
        ------
        RuntimeError
            If compute() has not been called.
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
        LikelihoodResult
            Mean likelihood result for plotting and analysis.

        Raises
        ------
        RuntimeError
            If compute() has not been called.
        """
        if self.likelihood_result is None:
            msg = "Likelihood not computed. Call compute() first."
            raise RuntimeError(msg)

        return self.likelihood_result

    def save_results(self, output_path: str) -> None:
        """
        Save likelihood results to file.

        Parameters
        ----------
        output_path : str
            Base path for saving. Individual sims saved as output_path_sim_XX.
        """
        if self.likelihood_result is None:
            msg = "No results to save. Run compute() first."
            raise RuntimeError(msg)

        if self.rank == 0:
            # Save mean likelihood result
            self.likelihood_result.save(output_path)
            self.log(f"Mean likelihood results saved to {output_path}")

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
                self.log(f"Individual simulation results saved ({n_files} files)")
                self.log(
                    "Use get_simulation_results() to access individual results",
                    level=2,
                )

    def run(self):
        """
        Execute the complete pixel-based likelihood analysis pipeline.

        Pipeline: setup fields/geometry/covariance, configure parameter grid,
        compute likelihood across parameter space, gather results.
        """

        self.setup_parameter_grid()

        if self.rank == 0:
            self.log("Starting PICSLike analysis pipeline", level=1)
            self.setup_fields()
            self.setup_geometry()
            self.setup_covariance_matrices()
            self.setup_cls(lmax=self.lmax_signal)
            self.setup_beams(lmax=self.lmax_signal)

            # Setup compression if configured
            if self._compression_config is not None:
                compression_config = self._compression_config
                self.setup_compression(
                    method=compression_config.get("method", "harmonic"),
                    epsilon=compression_config.get("epsilon"),
                    mode_fraction=compression_config.get("mode_fraction"),
                    basis=compression_config.get("basis", "noise_weighted"),
                    C_ell=compression_config.get("C_ell"),
                )
                self.log("Compression manager setup completed", level=3)

            if self.maps1 is None:
                self.setup_maps()
            if self.params.do_cross:
                assert self.maps2 is not None, (
                    "Maps2 should be loaded for cross-correlation."
                )

        self.comm.Barrier()

        self._broadcast_variables()

        self.comm.Barrier()

        self.compute()

        self.comm.Barrier()
