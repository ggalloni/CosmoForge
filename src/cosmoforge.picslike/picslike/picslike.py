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

from cosmocore import (
    Core,
    FieldCollection,
    MPISharedMemoryMixin,
    SpectrumKey,
    cholesky_solve,
    compute_signal_matrix,
    matrix_inverse_symm,
    matrix_slogdet_symm,
    read_maps,
)
from cosmocore._mpi import MPI

from .likelihood_result import LikelihoodResult
from .parameter_grid import ParameterGrid


class PICSLike(Core, MPISharedMemoryMixin):
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

    >>> from picslike import PICSLike
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
        basis=Core._UNSET,
        compression=Core._UNSET,
        **kwargs: Any,
    ) -> None:
        """
        Initialize pixel-based likelihood computation class.

        Parameters
        ----------
        params_file : str, optional
            Path to YAML configuration file.
        basis : None, False, str or dict, optional
            Computation basis selection (ADR-0018). ``None`` (default) →
            ``method="auto"``; ``False`` → traditional pixel-space path;
            ``"auto"``/``"harmonic"``/``"pixel"`` → ``{"method": …}``; a dict is
            the only form that requests compression.
        compression : dict, optional
            Deprecated alias for ``basis`` (ADR-0018): warns and is forwarded.
        **kwargs : Any
            Additional arguments passed to Core.
        """
        # Initialize parent Core class
        super().__init__(params=params_file, **kwargs)

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self._basis_config = self._resolve_basis_config(
            basis, compression, self.params.do_cross
        )

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
        """Signal-cov ceiling (ADR 0009).

        Resolution order: explicit setter, then ``params.lmax_signal``,
        then ``4 * nside``.
        """
        if self._lmax_signal is not None:
            return self._lmax_signal
        params_value = getattr(self.params, "lmax_signal", None)
        if params_value is not None:
            return params_value
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

            # Invalidate any cached parameter-independent SMW pieces; maps
            # are about to be (re)loaded.
            self._smw_data_cache = None

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
        # Python objects (small, serialization is fine)
        self.params = self.comm.bcast(self.params if self.rank == 0 else None, root=0)
        self.collection: FieldCollection = self.comm.bcast(
            self.collection if self.rank == 0 else None, root=0
        )
        self.npixs = self.comm.bcast(self.npixs if self.rank == 0 else None, root=0)
        self.pixact = self.comm.bcast(self.pixact if self.rank == 0 else None, root=0)

        # Numpy arrays via shared memory (zero-copy, no size limits).
        # Worker ranks may not have these attributes yet, so use getattr.
        point_vectors = getattr(self, "point_vectors", None)
        n_point_vectors = self.comm.bcast(
            len(point_vectors) if self.rank == 0 and point_vectors is not None else 0,
            root=0,
        )
        self.point_vectors = tuple(
            self._shared_array(
                point_vectors[i] if self.rank == 0 and point_vectors is not None else None
            )
            for i in range(n_point_vectors)
        )
        self.noise_cov1 = self._shared_array(getattr(self, "noise_cov1", None))
        self.maps1 = self._shared_array(getattr(self, "maps1", None))
        if self.params.do_cross:
            self.maps2 = self._shared_array(getattr(self, "maps2", None))

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

        mean_result = LikelihoodResult(
            parameter_grid=simulation_results[0].parameter_grid,
            chi_squared_values=mean_chi2,
            log_likelihood_values=mean_log_like,
        )

        return mean_result

    def _compute_likelihood_point(self, param_point: tuple) -> tuple[float, float]:
        """
        Compute chi-squared and log-likelihood for a single parameter point.

        ln L = -0.5 * (d^T C^{-1} d + ln|C|)

        When a computation basis is enabled, uses SMW formula in harmonic
        space — O(n_modes³) instead of O(n_pix³). Quadratic forms are
        batched across simulations for efficiency.
        """
        use_basis = hasattr(self, "basis_manager") and self.basis_manager is not None

        if use_basis:
            bm = self.basis_manager

            # Only the signal spectrum changes between parameter points.
            # Beams are loaded once during setup (setup_beams), but smoothing
            # must be re-applied after each set_cls call.
            spectra_dict = self.parameter_grid.get_spectrum(param_point)
            self.collection.set_cls(spectra_dict, lmax=self.lmax_signal)
            self.collection.beam_manager.apply_smoothing(
                self.collection.spectra_manager, lmax=self.lmax_signal
            )

            has_spin2 = any(f.spin == 2 for f in self.collection.fields)
            is_multi_field = len(self.collection.fields) > 1 or has_spin2

            if is_multi_field:
                C_ell_dict = self._build_c_ell_dict()
                C_ell_input = C_ell_dict
            else:
                C_ell_input = self.collection.spectra_manager.get_cls(0, 0, 0)

            self.log(f"C_ell set for parameters: {param_point}", level=3)

            C_ell_arg = C_ell_dict if is_multi_field else {(0, 0, 0): C_ell_input}

            if bm.method == "harmonic":
                # SMW fast path reaches into HarmonicBasis-private state
                # (``_V_N_inv``, ``_N_chol``); ``term1`` / ``projected_i``
                # depend on N and the data only, so cache once per parameter
                # scan.
                if getattr(self, "_smw_data_cache", None) is None:
                    projected1 = bm._V_N_inv @ self.maps1
                    if self.params.do_cross:
                        projected2 = bm._V_N_inv @ self.maps2
                        ninv_maps2 = cholesky_solve(bm._N_chol, self.maps2)
                        term1 = np.einsum("in,in->n", self.maps1, ninv_maps2)
                    else:
                        projected2 = projected1
                        ninv_maps1 = cholesky_solve(bm._N_chol, self.maps1)
                        term1 = np.einsum("in,in->n", self.maps1, ninv_maps1)
                    self._smw_data_cache = (projected1, projected2, term1)
                projected1, projected2, term1 = self._smw_data_cache

                # SMW identity: d1^T C^{-1} d2 = d1^T N^{-1} d2 − y1^T K^{-1} y2
                # with y_i = V N^{-1} d_i. Auto case (d1 = d2) is the
                # symmetric specialisation. ``logdet`` from prepare_for_basis
                # is the exact full pixel-space log determinant on harmonic.
                K_chol, logdet = bm.prepare_for_basis(C_ell_arg)
                kernel_inv_y2 = cholesky_solve((K_chol, True), projected2)
                term2 = np.einsum("in,in->n", projected1, kernel_inv_y2)
                chi_squared = term1 - term2
            else:
                # Polymorphic basis-space path. ``chi_squared`` and ``logdet``
                # are both basis-space, so the log-likelihood is internally
                # consistent. On pixel-direct (U = I) it equals the full
                # Gaussian; on truncated compressed pixel it is the
                # basis-space approximation admitted by the kept subspace.
                C_basis_inv = bm.get_inverse(C_ell_arg)
                d1_basis = bm.to_basis(self.maps1)
                if self.params.do_cross:
                    d2_basis = bm.to_basis(self.maps2)
                    chi_squared = np.einsum(
                        "in,ij,jn->n", d1_basis, C_basis_inv, d2_basis
                    )
                else:
                    chi_squared = np.einsum(
                        "in,ij,jn->n", d1_basis, C_basis_inv, d1_basis
                    )
                logdet = bm.get_logdet(C_ell_arg)

            self.log(f"Log-determinant: {logdet:.2f}", level=3)
        else:
            self.compute_signal_matrix(param_point)
            self.log(f"Signal matrix computed for parameters: {param_point}", level=3)
            self.prepare_covariance_matrix()
            self.log("Total covariance matrix prepared and inverted", level=3)

            if self.params.do_cross:
                chi_squared = np.einsum(
                    "in,ij,jn->n", self.maps1, self.inv_cov, self.maps2
                )
            else:
                chi_squared = np.einsum(
                    "in,ij,jn->n", self.maps1, self.inv_cov, self.maps1
                )

            logdet = matrix_slogdet_symm(self.total_cov)[1]
            self.log(f"Log-determinant: {logdet:.2f}", level=3)

        # ln L = -0.5 * (χ² + ln|C|)
        log_likelihood = -0.5 * (chi_squared + logdet)

        return chi_squared, log_likelihood

    def _build_c_ell_dict(self) -> dict[SpectrumKey, np.ndarray]:
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

            if self._basis_config is not None:
                _basis_keys = (
                    "method",
                    "epsilon",
                    "mode_fraction",
                    "compression_target",
                    "C_ell",
                )
                kwargs = {
                    k: self._basis_config[k]
                    for k in _basis_keys
                    if k in self._basis_config
                }
                self.setup_computation_basis(**kwargs)
                self.log("Computation basis setup completed", level=3)

            if self.maps1 is None:
                self.setup_maps()
            if self.params.do_cross:
                assert self.maps2 is not None, (
                    "Maps2 should be loaded for cross-correlation."
                )

        # Synchronize and broadcast shared variables to worker processes.
        # Skip broadcast for single-process runs to avoid unnecessary
        # serialization of large objects (covariance matrices can exceed 2 GB).
        self.comm.Barrier()

        if self.size > 1:
            self._broadcast_variables()
            self.comm.Barrier()

        self.compute()

        self.comm.Barrier()
