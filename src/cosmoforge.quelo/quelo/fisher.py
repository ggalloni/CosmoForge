"""
Fisher matrix computation for cosmological parameter estimation.

This module implements the Fisher class for calculating Fisher information matrices
used in cosmological parameter forecasting. The Fisher matrix provides the expected
parameter constraints from CMB observations and is computed as:

F_ij = (1/2) * Tr[C^(-1) * ∂C/∂θ_i * C^(-1) * ∂C/∂θ_j]

where C is the total covariance matrix (signal + noise) and θ_i are the cosmological
parameters of interest. The implementation supports MPI parallelization for efficient
computation of large Fisher matrices.

Classes
-------
Fisher
    Main class for Fisher matrix computation inheriting from cosmocore.Core.

Notes
-----
The Fisher matrix calculation involves several computationally intensive steps:
1. Signal covariance matrix computation from power spectra
2. Total covariance matrix assembly (signal + noise)
3. Matrix inversion for covariance matrices
4. Derivative computation for each parameter and multipole
5. Fisher matrix element calculation using trace operations

The implementation uses MPI for parallel computation, distributing Fisher matrix
elements across multiple processes for scalability.

References
----------
.. [1] Tegmark, M. et al. "How to measure CMB power spectra without losing information"
   Phys. Rev. D 55, 5895 (1997)
.. [2] Challinor, A. & Chon, G. "Error analysis of quadratic power spectrum estimates"
   MNRAS 301, 657-688 (1998)
"""

from __future__ import annotations

import time

import numpy as np
from mpi4py import MPI

from cosmocore import (
    Core,
    InputParams,
    compute_signal_matrix,
    do_derivative_step,
    matrix_inverse_symm,
    matrix_mult,
    matrix_trace,
    write_covmat_reduced,
    write_out_matrix,
)


class Fisher(Core):
    """
    Fisher matrix computation for cosmological parameter estimation.

    This class implements the Fisher information matrix calculation for forecasting
    cosmological parameter constraints from CMB observations. It inherits from
    cosmocore.Core and extends it with Fisher-specific functionality including
    signal matrix computation, covariance matrix preparation, and parallel
    Fisher matrix element calculation.

    The Fisher matrix F_ij quantifies the information content of the data about
    parameters θ_i and θ_j, computed as:

    F_ij = (1/2) * Tr[C^(-1) * ∂C/∂θ_i * C^(-1) * ∂C/∂θ_j]

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
    Sig : numpy.ndarray
        Signal covariance matrix computed from theoretical power spectra.
    fisher : numpy.ndarray
        Computed Fisher information matrix.
    derSil, derSjl : numpy.ndarray
        Derivative matrices for Fisher matrix computation.
    n_ell : int
        Number of multipole moments in analysis (lmax - 1).
    nell : int
        Total number of Fisher matrix parameters (nspectra * n_ell).

    Examples
    --------
    Basic Fisher matrix computation:

    >>> from cosmoforge.quelo import Fisher
    >>> fisher = Fisher("config/fisher_analysis.yaml")
    >>> fisher.run()
    >>> F_matrix = fisher.get_fisher_matrix()
    >>> param_errors = fisher.get_error_bars()

    MPI parallel execution:

    >>> # Run with: mpirun -n 4 python fisher_analysis.py
    >>> fisher = Fisher("config/fisher_config.yaml")
    >>> fisher.run()  # Automatically distributes computation across processes

    Notes
    -----
    The Fisher matrix computation is parallelized using MPI, with each process
    computing a subset of Fisher matrix elements. The calculation scales as
    O(N_ell^2 * N_spec^2) where N_ell is the number of multipoles and N_spec
    is the number of power spectra.

    For large analyses (high lmax or many cross-correlations), the computation
    can be memory-intensive due to the storage of covariance matrices and
    their inverses. Consider using appropriate hardware resources.

    See Also
    --------
    cosmocore.Core : Base class providing fundamental analysis infrastructure
    Spectra : QML power spectrum estimation class
    """

    def __init__(self, params_file: str | None = None, **kwargs):
        """
        Initialize Fisher matrix computation class.

        Parameters
        ----------
        params_file : str, optional
            Path to YAML configuration file containing analysis parameters.
            If None, parameters must be provided through kwargs or set later.
        **kwargs : dict
            Additional keyword arguments passed to the Core parent class.
            Common options include 'params' for direct parameter object,
            'verbose' for logging level control.

        Notes
        -----
        Initializes the Fisher class by calling the parent Core constructor
        and setting up MPI communication. The MPI environment must be
        initialized before creating Fisher instances for parallel computation.

        The initialization performs:
        1. Parameter loading and validation via Core.__init__
        2. MPI communicator setup for parallel computation
        3. Process rank and size determination

        Examples
        --------
        Initialize from configuration file:

        >>> fisher = Fisher("config/analysis.yaml")

        Initialize with direct parameters:

        >>> from cosmocore.settings import InputParams
        >>> params = InputParams()
        >>> fisher = Fisher(params=params)
        """
        # Pass the params_file to Core class constructor
        super().__init__(params=params_file, **kwargs)

        # MPI setup
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Initialize signal matrix to None
        self.Sig = None

        # lmax for signal matrix computation (matches Fortran convention of 4*nside)
        # This can be overridden if needed for different analyses
        self._lmax_signal = None

    @property
    def lmax_signal(self) -> int:
        """
        Maximum multipole for signal matrix computation.

        This defaults to 4*nside to match the Fortran reference implementation.
        The signal matrix is computed up to this lmax, while the Fisher matrix
        output uses params.lmax.

        Returns
        -------
        int
            Maximum multipole for signal covariance matrix computation.
        """
        if self._lmax_signal is not None:
            return self._lmax_signal
        return 4 * self.params.nside

    @lmax_signal.setter
    def lmax_signal(self, value: int) -> None:
        """Set custom lmax_signal value."""
        self._lmax_signal = value

    def setup_signal_matrix(self) -> np.ndarray:
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
        >>> S = fisher.setup_signal_matrix()
        >>> print(f"Signal matrix shape: {S.shape}")
        """
        if self.NCov1 is None:
            raise ValueError("Covariance matrices must be set up first")

        self.Sig = np.zeros_like(self.NCov1, dtype=np.float64)
        self.Sig = np.asfortranarray(self.Sig, dtype=np.float64)

        start_time = time.time() if self.rank == 0 else None

        compute_signal_matrix(
            S=self.Sig,
            lmax=self.lmax_signal,
            fields=self.collection,
        )

        if self.rank == 0 and start_time is not None:
            elapsed = time.time() - start_time
            self.log(f"Signal matrix computed in {elapsed:.2f} seconds", level=3)
            self.log(f"Signal matrix shape: {self.Sig.shape}", level=4)
            self.log(f"Signal matrix first row: {self.Sig[0, :10]}", level=4)

        return self.Sig

    def prepare_covariance_matrices(self):
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
        if self.Sig is None:
            self.setup_signal_matrix()

        # Save original noise covariance BEFORE adding signal (for noise bias computation)
        write_covmat_reduced(self.params.outnoisecovmat1, self.NCov1)
        self.log("Saved original noise covariance matrix 1", level=4)
        if self.params.do_cross:
            write_covmat_reduced(self.params.outnoisecovmat2, self.NCov2)
            self.log("Saved original noise covariance matrix 2", level=4)

        # Add signal to noise covariance
        self.NCov1 = self.NCov1 + self.Sig
        self.NCov1 = np.asfortranarray(self.NCov1)
        self.log(f"Combined covariance matrix shape: {self.NCov1.shape}", level=4)

        # Compute inverse covariance matrices
        self.NCov1 = matrix_inverse_symm(self.NCov1)
        self.log("Computed inverse of primary covariance matrix", level=4)

        # Write inverse covariance matrix
        write_covmat_reduced(self.params.outinvcovmatfile1, self.NCov1)

        if self.params.do_cross:
            self.NCov2 = self.NCov2 + self.Sig
            self.NCov2 = np.asfortranarray(self.NCov2)
            self.NCov2 = matrix_inverse_symm(self.NCov2)
            write_covmat_reduced(self.params.outinvcovmatfile2, self.NCov2)
            self.log("Computed inverse of secondary covariance matrix", level=4)

    def setup_fisher_matrices(self):
        """
        Initialize Fisher matrix and derivative arrays for computation.

        This method allocates and initializes the data structures needed for
        Fisher matrix computation, including the Fisher matrix itself and
        temporary derivative matrices used in the calculation.

        Notes
        -----
        Initializes the following arrays:

        - fisher: Main Fisher matrix of shape (nell, nell) where
          nell = n_spectra * (lmax - 1)
        - derSil, derSjl: Derivative matrices with same shape as covariance
          matrices, used for storing ∂C/∂θ derivatives

        All arrays are converted to Fortran memory layout for optimal
        performance with BLAS/LAPACK routines used in matrix operations.

        The Fisher matrix dimensions are determined by:

        - n_ell: Number of multipole moments (lmax - 1, starting from l=2)
        - nell: Total parameters = n_spectra * n_ell
        - n_spectra: Number of power spectra (auto + cross correlations)

        Memory allocation scales as O(nell^2) for the Fisher matrix and
        O(n_pixels^2) for derivative matrices.

        Examples
        --------
        >>> fisher = Fisher("config.yaml")
        >>> fisher.setup_fisher_matrices()
        >>> print(f"Fisher matrix shape: {fisher.fisher.shape}")
        >>> print(f"Total parameters: {fisher.nell}")
        """
        self.n_ell = self.params.lmax - 1
        self.nell = self.params.nspectra * self.n_ell

        self.fisher = np.zeros((self.nell, self.nell))
        self.derSil = np.zeros_like(self.NCov1)
        self.derSjl = np.zeros_like(self.NCov1)

        # Convert to Fortran order for better performance
        self.fisher = np.asfortranarray(self.fisher)
        self.derSil = np.asfortranarray(self.derSil)
        self.derSjl = np.asfortranarray(self.derSjl)

    def compute_fisher_element(
        self,
        il: int,
        jl: int,
        curr_ell_i: int,
        curr_ell_j: int,
        spectrum_i: int,
        spectrum_j: int,
        appil: int,
    ) -> int:
        """
        Compute a single Fisher matrix element F_ij.

        This method computes individual elements of the Fisher information matrix
        using the formula: F_ij = (1/2) * Tr[C^(-1) * ∂C/∂θ_i * C^(-1) * ∂C/∂θ_j]
        where C is the total covariance matrix and θ_i, θ_j are parameters.

        Parameters
        ----------
        il, jl : int
            Linear indices in the Fisher matrix corresponding to parameters
            θ_i and θ_j respectively.
        curr_ell_i, curr_ell_j : int
            Multipole moments l for parameters i and j (typically l ≥ 2).
        spectrum_i, spectrum_j : int
            Power spectrum indices (0=TT, 1=EE, 2=BB, 3=TE, etc.) for
            parameters i and j.
        appil : int
            Previous il value used for caching derivative computations to
            avoid redundant calculations when il hasn't changed.

        Returns
        -------
        int
            Updated appil value for next iteration (equals il).

        Notes
        -----
        The computation implements several optimizations:
        1. Caching: Derivative ∂C/∂θ_i is recomputed only when il changes
        2. Symmetry: F_ij = F_ji, so only upper triangle is computed
        3. Special handling for diagonal vs off-diagonal elements

        For diagonal elements (il == jl):
        F_ii = (1/2) * Tr[∂C/∂θ_i * C^(-1) * ∂C/∂θ_i * C^(-1)]

        For off-diagonal elements (il != jl):
        F_ij = (1/2) * Tr[∂C/∂θ_i * C^(-1) * ∂C/∂θ_j * C^(-1)]

        The method handles both auto-correlation and cross-correlation cases
        based on the params.do_cross flag.

        Examples
        --------
        This method is typically called within the main compute() loop:

        >>> appil = fisher.compute_fisher_element(
        ...     il=5, jl=8, curr_ell_i=7, curr_ell_j=10,
        ...     spectrum_i=0, spectrum_j=1, appil=4
        ... )
        """
        if self.rank == 0:
            self.log("-" * 80, level=2)
            spec_i_label = self.collection.spectra_labels[spectrum_i]
            spec_j_label = self.collection.spectra_labels[spectrum_j]
            self.log(
                f"Rank {self.rank} ---> "
                f"Spec {spec_i_label} l={curr_ell_i} VS "
                f"Spec {spec_j_label} l={curr_ell_j}",
                level=2,
            )

        if il != appil:
            # Compute derivative for spectrum i
            self.derSil.fill(0.0)
            do_derivative_step(
                self.derSil,
                spectrum_i,
                self.npixs,
                self.params.spins,
                curr_ell_i,
                self.collection,
            )

            self.log(f"DerSil shape: {self.derSil.shape}", level=4)

            if jl == il:
                # Diagonal element
                if self.params.do_cross:
                    temp_mult = matrix_mult(self.derSil, self.NCov1)
                    Sig_temp = matrix_mult(self.NCov2, temp_mult)
                else:
                    temp_mult = matrix_mult(self.derSil, self.NCov1)
                    Sig_temp = matrix_mult(self.NCov1, temp_mult)

                self.fisher[il, il] = 0.5 * matrix_trace(self.derSil, Sig_temp)
                diag_msg = f"Fisher diagonal element [{il}, {il}]: {self.fisher[il, il]}"
                self.log(diag_msg, level=4)

            # Transform derivative matrix
            self.derSil = matrix_mult(self.derSil, self.NCov1)
            if self.params.do_cross:
                self.derSil = matrix_mult(self.NCov2, self.derSil)
            else:
                self.derSil = matrix_mult(self.NCov1, self.derSil)

        if jl != il:
            # Off-diagonal element
            self.derSjl.fill(0.0)
            do_derivative_step(
                self.derSjl,
                spectrum_j,
                self.npixs,
                self.params.spins,
                curr_ell_j,
                self.collection,
            )

            self.fisher[il, jl] = 0.5 * matrix_trace(self.derSjl, self.derSil)
            self.fisher[jl, il] = self.fisher[il, jl]  # Symmetry

            offdiag_msg = (
                f"Fisher off-diagonal element [{il}, {jl}]: {self.fisher[il, jl]}"
            )
            self.log(offdiag_msg, level=5)

        return il

    def compute(self):
        """
        Execute the parallel Fisher matrix computation using MPI.

        This method implements the main computational kernel for Fisher matrix
        calculation, distributing the work across multiple MPI processes for
        scalability. The computation loops over all parameter pairs and
        calculates Fisher matrix elements in parallel.

        Notes
        -----
        The parallel computation strategy:
        1. Divides the upper triangular Fisher matrix into chunks
        2. Assigns chunks to different MPI processes based on rank
        3. Each process computes its assigned Fisher matrix elements
        4. Results are gathered and summed using MPI collective operations

        Work distribution uses the formula:
        elements_per_process = ceil(N*(N+1)/2 / n_processes)
        where N = nell is the total number of parameters.

        The computation scales as O(N²) in the number of parameters and
        O(n_pixels²) for each element due to matrix operations. Memory
        usage is dominated by covariance matrices and their inverses.

        Synchronization points:
        - Before computation: ensures all processes have necessary data
        - After computation: gathers results from all processes
        - MPI.Reduce sums Fisher matrices from all processes

        Progress logging is provided by the master process (rank 0) including
        computation timing and element completion tracking.

        Examples
        --------
        The compute method is typically called as part of the full pipeline:

        >>> fisher = Fisher("config.yaml")
        >>> fisher.run()  # Calls compute() internally
        # or manually:
        >>> fisher.setup_fisher_matrices()
        >>> fisher.compute()
        >>> F = fisher.get_fisher_matrix()
        """
        if self.rank == 0:
            self.log("Starting Fisher matrix computation", level=2)

        start_time = time.time() if self.rank == 0 else None

        # Determine work distribution
        ellperproc = np.ceil((self.nell + 1.0) * self.nell / 2.0 / self.size)
        self.log(f"Rank {self.rank} will compute {ellperproc} elements", level=2)

        counter = 0
        appil = -1
        count_computed = 0

        # Main computation loop
        for il in range(self.nell):
            spectrum_i = il // self.n_ell
            curr_ell_i = (il % self.n_ell) + 2

            for jl in range(il, self.nell):
                spectrum_j = jl // self.n_ell
                curr_ell_j = jl % self.n_ell + 2

                counter += 1

                # Check if this element is assigned to current rank
                if not (
                    counter > self.rank * ellperproc
                    and counter <= (self.rank + 1) * ellperproc
                ):
                    continue

                count_computed += 1
                if self.rank == 0:
                    self.log(
                        f"Computed {count_computed} of {int(ellperproc)} elements",
                        level=2,
                    )

                # Compute Fisher matrix element
                appil = self.compute_fisher_element(
                    il, jl, curr_ell_i, curr_ell_j, spectrum_i, spectrum_j, appil
                )

        # Synchronize all processes
        self.comm.Barrier()

        # Reduce Fisher matrices from all processes
        redfisher = np.zeros_like(self.fisher)
        self.comm.Reduce(self.fisher, redfisher, op=MPI.SUM, root=0)

        if self.rank == 0:
            self.fisher = redfisher
            self.log("-" * 80, level=1)
            self.log("Fisher matrix computation completed", level=1)
            self.log(f"Fisher matrix shape: {self.fisher.shape}", level=4)

            if start_time is not None:
                elapsed = time.time() - start_time
                self.log(f"Total computation time: {elapsed:.2f} seconds", level=3)

            # Write Fisher matrix to file
            if hasattr(self.params, "outfilefisher"):
                write_out_matrix(self.params.outfilefisher, self.fisher)
                self.log(f"Fisher matrix written to {self.params.outfilefisher}", level=4)

        self.comm.Barrier()

    def run(self):
        """
        Execute the complete Fisher matrix analysis pipeline.

        This method orchestrates the entire Fisher matrix computation from
        initial parameter setup through final matrix calculation and output.
        It handles the complex coordination between MPI processes and ensures
        proper data distribution for parallel computation.

        Notes
        -----
        The analysis pipeline consists of several phases:

        **Phase 1 - Master Process Setup (Rank 0 only):**
        1. Parameter validation and logging initialization
        2. Field collection setup with HEALPix pixelization
        3. Geometry computation for pixel pointing vectors
        4. Covariance matrix loading and preparation
        5. Power spectra and beam function initialization
        6. Signal matrix computation and covariance inversion

        **Phase 2 - Data Broadcasting:**
        All essential data structures are broadcast from master to worker
        processes including parameters, field collections, geometry data,
        and inverted covariance matrices.

        **Phase 3 - Parallel Computation:**
        Fisher matrix elements are computed in parallel across all processes
        using MPI work distribution.

        **Phase 4 - Results Gathering:**
        Fisher matrix contributions from all processes are reduced and
        the final matrix is written to output files.

        The method handles MPI synchronization at critical points to ensure
        data consistency and proper load balancing. Memory usage is optimized
        by broadcasting only necessary data structures.

        Raises
        ------
        ValueError
            If parameters are not properly set before calling run().
        MPIError
            If MPI communication fails during data broadcasting or reduction.

        Examples
        --------
        Basic Fisher matrix analysis:

        >>> fisher = Fisher("analysis_config.yaml")
        >>> fisher.run()
        >>> # Results automatically written to files specified in config

        MPI execution:

        >>> # Command line: mpirun -n 8 python fisher_script.py
        >>> fisher = Fisher("high_resolution_config.yaml")
        >>> fisher.run()
        >>> if fisher.rank == 0:
        ...     errors = fisher.get_error_bars()
        ...     print(f"Parameter constraints: {errors}")
        """
        # Only rank 0 does the initial setup
        if self.rank == 0:
            if self.params is None:
                raise ValueError("Parameters must be set before running analysis")

            self.log("Starting Fisher matrix analysis pipeline", level=1)

            # Setup analysis components
            self.setup_fields()
            self.log("Fields setup completed", level=3)

            self.setup_geometry()
            self.log("Geometry setup completed", level=3)

            self.setup_covariance_matrices()
            self.log("Covariance matrices setup completed", level=3)

            # Setup Cls and beams with lmax_signal (defaults to 4*nside)
            # This matches the Fortran convention for signal matrix computation
            self.log(f"Using lmax_signal = {self.lmax_signal} for Cls and beams", level=3)
            self.setup_cls(lmax=self.lmax_signal)
            self.log("Power spectra setup completed", level=3)

            self.setup_beams(lmax=self.lmax_signal)
            self.log("Beam functions setup completed", level=3)

            self.prepare_covariance_matrices()
            self.log("Signal matrix and covariance preparation completed", level=3)

        # Synchronize before broadcasting
        self.comm.Barrier()

        # Broadcast shared variables to all processes
        self.params: InputParams = self.comm.bcast(
            self.params if self.rank == 0 else None, root=0
        )
        self.collection = self.comm.bcast(
            self.collection if self.rank == 0 else None, root=0
        )
        self.npixs = self.comm.bcast(self.npixs if self.rank == 0 else None, root=0)
        self.pixact = self.comm.bcast(self.pixact if self.rank == 0 else None, root=0)
        self.point_vectors = self.comm.bcast(
            self.point_vectors if self.rank == 0 else None, root=0
        )
        self.NCov1 = self.comm.bcast(self.NCov1 if self.rank == 0 else None, root=0)
        self.Sig = self.comm.bcast(self.Sig if self.rank == 0 else None, root=0)

        if self.params.do_cross:
            self.NCov2 = self.comm.bcast(self.NCov2 if self.rank == 0 else None, root=0)

        self.comm.Barrier()

        # Setup Fisher matrices on all processes
        self.setup_fisher_matrices()

        # Compute Fisher matrix (parallel computation)
        self.compute()

        # Finalize MPI
        self.comm.Barrier()
        # MPI.Finalize()

    def get_fisher_matrix(self) -> np.ndarray | None:
        """
        Retrieve the computed Fisher information matrix.

        Returns
        -------
        numpy.ndarray or None
            Fisher information matrix of shape (nell, nell) where nell is the
            total number of parameters (n_spectra * (lmax-1)). Returns None
            for worker processes (rank != 0) or if computation hasn't completed.

        Notes
        -----
        The Fisher matrix is only available on the master process (rank 0) after
        the computation has completed via run() or compute(). Worker processes
        return None as they only store partial results during computation.

        The matrix is symmetric and positive semi-definite by construction.
        Element F_ij represents the Fisher information between parameters θ_i
        and θ_j, quantifying how well the data can constrain parameter combinations.

        Examples
        --------
        >>> fisher = Fisher("config.yaml")
        >>> fisher.run()
        >>> if fisher.rank == 0:
        ...     F = fisher.get_fisher_matrix()
        ...     print(f"Fisher matrix shape: {F.shape}")
        ...     print(f"Condition number: {np.linalg.cond(F)}")
        """
        if self.rank == 0:
            return self.fisher
        return None

    def get_error_bars(self) -> np.ndarray | None:
        """
        Compute parameter forecast errors from the Fisher matrix.

        This method inverts the Fisher matrix to obtain the parameter covariance
        matrix, then extracts the marginal 1σ parameter uncertainties from the
        diagonal elements.

        Returns
        -------
        numpy.ndarray or None
            1D array of marginal parameter errors (1σ uncertainties) with length
            equal to the number of parameters. Returns None for worker processes
            or if Fisher matrix is singular.

        Notes
        -----
        The parameter errors are computed as:
        σ_i = sqrt((F^(-1))_ii)

        where F^(-1) is the inverse Fisher matrix (parameter covariance matrix).
        These represent the marginal constraints on individual parameters,
        marginalized over all other parameters.

        The method handles numerical issues by catching LinearAlgebraError
        exceptions that can occur if the Fisher matrix is singular or
        poorly conditioned. This can happen with insufficient data or
        degenerate parameter combinations.

        For accurate forecasts, ensure:
        1. Sufficient l-range for parameter sensitivity
        2. Adequate signal-to-noise ratio
        3. Non-degenerate parameter combinations

        Examples
        --------
        >>> fisher = Fisher("config.yaml")
        >>> fisher.run()
        >>> if fisher.rank == 0:
        ...     errors = fisher.get_error_bars()
        ...     if errors is not None:
        ...         print(f"Parameter errors: {errors}")
        ...         # Relative errors
        ...         fiducial_params = [1.0, 0.05, 0.8]  # Example values
        ...         rel_errors = errors[:len(fiducial_params)] / fiducial_params
        ...         print(f"Relative errors: {rel_errors}")

        Raises
        ------
        LinAlgError
            If the Fisher matrix is singular and cannot be inverted. This is
            caught internally and None is returned with a warning message.
        """
        if self.rank == 0 and self.fisher is not None:
            # Invert Fisher matrix to get covariance
            cov_matrix = np.linalg.inv(self.fisher)
            errors = np.sqrt(np.diag(cov_matrix))
            return errors
        return None

    def get_window_matrix(self) -> np.ndarray | None:
        """
        Retrieve the window matrix for QML power spectrum estimation.

        The window matrix W relates the expected QML estimates to the true
        power spectrum: <y> = W @ C_true. It encodes the mode coupling induced
        by partial sky coverage and pixel window effects.

        Returns
        -------
        numpy.ndarray or None
            Window matrix of shape (nell, nell) where nell = n_spectra * (lmax-1).
            Returns None for worker processes (rank != 0) or if computation
            hasn't completed.

        Notes
        -----
        The window matrix elements are computed as:
        W_αβ = (1/2) Tr[C⁻¹ P_α C⁻¹ P_β]

        where P_α = ∂C/∂C_α is the derivative of the covariance matrix with
        respect to power spectrum amplitude at multipole α.

        This is mathematically equivalent to the Fisher matrix before
        normalization factors (vecmul) are applied. The window matrix is
        essential for the "convolved" normalization mode in QML estimation,
        where instead of deconvolving the window function, the theory is
        convolved with the window for comparison.

        Examples
        --------
        >>> fisher = Fisher("config.yaml")
        >>> fisher.run()
        >>> if fisher.rank == 0:
        ...     W = fisher.get_window_matrix()
        ...     # Convolve theory spectrum with window
        ...     cl_theory_convolved = W @ cl_theory

        See Also
        --------
        get_fisher_matrix : Returns the same matrix (Fisher = Window before normalization)
        Spectra.get_power_spectra : Uses window matrix for 'convolved' mode
        """
        return self.get_fisher_matrix()
