"""
Fisher matrix computation for cosmological parameter estimation.

This module implements the Fisher class for calculating Fisher information matrices
used in cosmological parameter forecasting. The Fisher matrix provides the expected
parameter constraints from CMB observations and is computed as:

F_ij = (1/2) * Tr[C^(-1) * ∂C/∂θ_i * C^(-1) * ∂C/∂θ_j]

where C is the total covariance matrix (signal + noise) and θ_i are the cosmological
parameters of interest. The implementation supports MPI parallelization for efficient
computation of large Fisher matrices.

For single-spectrum (temperature-only) analyses, the unified API from Core is used,
which transparently handles compression if enabled. Multi-spectrum analyses use
traditional pixel-space computation.

Classes
-------
Fisher
    Main class for Fisher matrix computation inheriting from cosmocore.Core.

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
    cosmocore.Core and extends it with Fisher-specific functionality.

    The Fisher matrix F_ij quantifies the information content of the data about
    parameters θ_i and θ_j, computed as:

    F_ij = (1/2) * Tr[C^(-1) * ∂C/∂θ_i * C^(-1) * ∂C/∂θ_j]

    For single-spectrum analyses (nspectra=1), the computation uses Core's unified
    API which transparently handles compression. Multi-spectrum analyses use
    traditional pixel-space computation.

    Parameters
    ----------
    params_file : str, optional
        Path to YAML parameter file containing analysis configuration.
    compression : dict, optional
        Compression configuration. Only supported for single-spectrum analyses.
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
    fisher : numpy.ndarray
        Computed Fisher information matrix.
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

    With compression (single-spectrum only):

    >>> fisher = Fisher("config.yaml", compression={"method": "harmonic"})
    >>> fisher.run()  # Uses compressed computation transparently
    """

    def __init__(
        self,
        params_file: str | None = None,
        compression: dict | None = None,
        **kwargs,
    ):
        """
        Initialize Fisher matrix computation class.

        Parameters
        ----------
        params_file : str, optional
            Path to YAML configuration file.
        compression : dict, optional
            Compression configuration dictionary. Only supported for
            single-spectrum (temperature-only) analyses. Options:

            - method : str ("harmonic" or "pixel_projected")
            - epsilon : float (eigenvalue threshold)
            - basis : str (for pixel_projected: "harmonic", "noise_weighted", etc.)
            - mode_fraction : float (alternative to epsilon)

        **kwargs : dict
            Additional keyword arguments passed to Core.
        """
        super().__init__(params=params_file, **kwargs)

        # MPI setup
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Compression config
        self._compression_config = compression

        # Initialize attributes
        self.Sig = None
        self._lmax_signal = None

    @property
    def lmax_signal(self) -> int:
        """Maximum multipole for signal matrix computation (defaults to 4*nside)."""
        if self._lmax_signal is not None:
            return self._lmax_signal
        return 4 * self.params.nside

    @lmax_signal.setter
    def lmax_signal(self, value: int) -> None:
        self._lmax_signal = value

    # =========================================================================
    # Setup Methods
    # =========================================================================

    def setup_signal_matrix(self) -> np.ndarray:
        """Compute the theoretical signal covariance matrix from power spectra."""
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

        return self.Sig

    def prepare_covariance_matrices(self):
        """Prepare total covariance matrices and compute their inverses."""
        if self.Sig is None:
            self.setup_signal_matrix()

        # Save original noise covariance BEFORE adding signal
        write_covmat_reduced(self.params.outnoisecovmat1, self.NCov1)
        if self.params.do_cross:
            write_covmat_reduced(self.params.outnoisecovmat2, self.NCov2)

        # Add signal to noise covariance: C = N + S
        self.NCov1 = self.NCov1 + self.Sig
        self.NCov1 = np.asfortranarray(self.NCov1)

        # Compute inverse: C^{-1}
        self.NCov1 = matrix_inverse_symm(self.NCov1)
        write_covmat_reduced(self.params.outinvcovmatfile1, self.NCov1)

        if self.params.do_cross:
            self.NCov2 = self.NCov2 + self.Sig
            self.NCov2 = np.asfortranarray(self.NCov2)
            self.NCov2 = matrix_inverse_symm(self.NCov2)
            write_covmat_reduced(self.params.outinvcovmatfile2, self.NCov2)

    def setup_fisher_matrices(self):
        """Initialize Fisher matrix and derivative arrays."""
        self.n_ell = self.params.lmax - 1
        self.nell = self.params.nspectra * self.n_ell

        self.fisher = np.zeros((self.nell, self.nell))
        self.derSil = np.zeros_like(self.NCov1)
        self.derSjl = np.zeros_like(self.NCov1)

        # Fortran order for BLAS performance
        self.fisher = np.asfortranarray(self.fisher)
        self.derSil = np.asfortranarray(self.derSil)
        self.derSjl = np.asfortranarray(self.derSjl)

    # =========================================================================
    # Single-Spectrum Computation (Unified API - Compression Agnostic)
    # =========================================================================

    def _compute_single_spectrum(self):
        """
        Compute Fisher matrix for single-spectrum analysis.

        This method adapts its algorithm based on whether compression is enabled:
        - With compression: Uses optimized compute_fisher_matrix() which precomputes
          V C^{-1} V^T once for O(ℓ²) speedup
        - Without compression: Uses precomputed C^{-1} and builds derivatives

        The Fisher matrix element is computed as:
            F_ij = (1/2) Tr[C^{-1} dC_i C^{-1} dC_j]
        """
        use_compression = (
            hasattr(self, "compression_manager") and self.compression_manager is not None
        )

        if self.rank == 0:
            mode = "compressed (optimized)" if use_compression else "traditional"
            self.log(f"Starting single-spectrum Fisher computation ({mode})", level=2)

        start_time = time.time() if self.rank == 0 else None

        n_ell = self.params.lmax - 1  # ell from 2 to lmax

        if use_compression:
            # Optimized path: use compute_fisher_matrix() which precomputes
            # V C^{-1} V^T once and reuses for all elements
            C_ell = self._get_theory_cl_vector()

            if self.rank == 0:
                # Only rank 0 computes the full matrix (no MPI distribution needed
                # since the optimized method is already efficient)
                self.fisher = self.compression_manager.compute_fisher_matrix(
                    C_ell, ell_min=2, ell_max=self.params.lmax
                )
                self.log("-" * 80, level=1)
                self.log("Fisher matrix computation completed", level=1)

                if start_time is not None:
                    elapsed = time.time() - start_time
                    self.log(f"Total computation time: {elapsed:.2f} seconds", level=3)

                if hasattr(self.params, "outfilefisher"):
                    write_out_matrix(self.params.outfilefisher, self.fisher)
            else:
                self.fisher = np.zeros((n_ell, n_ell))

            # Broadcast result to all ranks
            self.comm.Barrier()
            self.fisher = self.comm.bcast(self.fisher, root=0)
        else:
            # Traditional path with MPI distribution
            total_elements = n_ell * (n_ell + 1) // 2
            elements_per_proc = int(np.ceil(total_elements / self.size))

            if self.rank == 0:
                self.log(
                    f"Rank {self.rank} will compute ~{elements_per_proc} elements",
                    level=2,
                )

            # Initialize Fisher matrix
            fisher_local = np.zeros((n_ell, n_ell))

            # NCov1 is already C^{-1} after prepare_covariance_matrices()
            C_inv = self.NCov1

            # Precompute all derivative matrices
            dC_matrices = {}
            for ell in range(2, self.params.lmax + 1):
                dC = np.zeros_like(self.NCov1)
                dC = np.asfortranarray(dC, dtype=np.float64)
                do_derivative_step(
                    dC,
                    0,  # spectrum_idx = 0 for single-spectrum
                    self.npixs,
                    self.params.spins,
                    ell,
                    self.collection,
                )
                dC_matrices[ell] = dC

            # Main computation loop over ell pairs
            counter = 0
            for i, ell_i in enumerate(range(2, self.params.lmax + 1)):
                for j, ell_j in enumerate(range(ell_i, self.params.lmax + 1)):
                    counter += 1

                    # Check if this element is assigned to current rank
                    if not (
                        counter > self.rank * elements_per_proc
                        and counter <= (self.rank + 1) * elements_per_proc
                    ):
                        continue

                    # Traditional: F_ij = 0.5 * Tr[C_inv @ dC_i @ C_inv @ dC_j]
                    # Use matrix_trace(A, B) = Tr(A @ B) which is O(n²) vs O(n³)
                    dC_i = dC_matrices[ell_i]
                    dC_j = dC_matrices[ell_j]
                    Cinv_dCi = matrix_mult(C_inv, dC_i)
                    Cinv_dCj = matrix_mult(C_inv, dC_j)
                    fisher_val = 0.5 * matrix_trace(Cinv_dCi, Cinv_dCj)

                    idx_i = ell_i - 2
                    idx_j = ell_j - 2

                    fisher_local[idx_i, idx_j] = fisher_val
                    if idx_i != idx_j:
                        fisher_local[idx_j, idx_i] = fisher_val  # Symmetry

            # Synchronize and reduce
            self.comm.Barrier()
            redfisher = np.zeros_like(fisher_local)
            self.comm.Reduce(fisher_local, redfisher, op=MPI.SUM, root=0)

            if self.rank == 0:
                self.fisher = redfisher
                self.log("-" * 80, level=1)
                self.log("Fisher matrix computation completed", level=1)

                if start_time is not None:
                    elapsed = time.time() - start_time
                    self.log(f"Total computation time: {elapsed:.2f} seconds", level=3)

                if hasattr(self.params, "outfilefisher"):
                    write_out_matrix(self.params.outfilefisher, self.fisher)

            self.comm.Barrier()

    def _get_theory_cl_vector(self) -> np.ndarray:
        """Get theoretical C_ell values from the field collection."""
        cls = self.collection.spectra_manager.get_cls(0, 0, 0)
        return cls

    # =========================================================================
    # Multi-Spectrum Computation (Traditional - No Compression Support)
    # =========================================================================

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
        """Compute a single Fisher matrix element F_ij (traditional method)."""
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
            self.derSil.fill(0.0)
            do_derivative_step(
                self.derSil,
                spectrum_i,
                self.npixs,
                self.params.spins,
                curr_ell_i,
                self.collection,
            )

            if jl == il:
                if self.params.do_cross:
                    temp_mult = matrix_mult(self.derSil, self.NCov1)
                    Sig_temp = matrix_mult(self.NCov2, temp_mult)
                else:
                    temp_mult = matrix_mult(self.derSil, self.NCov1)
                    Sig_temp = matrix_mult(self.NCov1, temp_mult)

                self.fisher[il, il] = 0.5 * matrix_trace(self.derSil, Sig_temp)

            self.derSil = matrix_mult(self.derSil, self.NCov1)
            if self.params.do_cross:
                self.derSil = matrix_mult(self.NCov2, self.derSil)
            else:
                self.derSil = matrix_mult(self.NCov1, self.derSil)

        if jl != il:
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
            self.fisher[jl, il] = self.fisher[il, jl]

        return il

    def _compute_multi_spectrum(self):
        """
        Compute Fisher matrix for multi-spectrum analysis (traditional method).

        This uses direct pixel-space operations and does not support compression.
        """
        if self.rank == 0:
            self.log("Starting multi-spectrum Fisher computation (traditional)", level=2)

        start_time = time.time() if self.rank == 0 else None

        ellperproc = np.ceil((self.nell + 1.0) * self.nell / 2.0 / self.size)
        self.log(f"Rank {self.rank} will compute {ellperproc} elements", level=2)

        counter = 0
        appil = -1
        count_computed = 0

        for il in range(self.nell):
            spectrum_i = il // self.n_ell
            curr_ell_i = (il % self.n_ell) + 2

            for jl in range(il, self.nell):
                spectrum_j = jl // self.n_ell
                curr_ell_j = jl % self.n_ell + 2

                counter += 1

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

                appil = self.compute_fisher_element(
                    il, jl, curr_ell_i, curr_ell_j, spectrum_i, spectrum_j, appil
                )

        self.comm.Barrier()

        redfisher = np.zeros_like(self.fisher)
        self.comm.Reduce(self.fisher, redfisher, op=MPI.SUM, root=0)

        if self.rank == 0:
            self.fisher = redfisher
            self.log("-" * 80, level=1)
            self.log("Fisher matrix computation completed", level=1)

            if start_time is not None:
                elapsed = time.time() - start_time
                self.log(f"Total computation time: {elapsed:.2f} seconds", level=3)

            if hasattr(self.params, "outfilefisher"):
                write_out_matrix(self.params.outfilefisher, self.fisher)

        self.comm.Barrier()

    # =========================================================================
    # Main Entry Points
    # =========================================================================

    def compute(self):
        """
        Execute Fisher matrix computation.

        Automatically selects the appropriate method:
        - Single-spectrum: Uses unified API (compression agnostic)
        - Multi-spectrum: Uses traditional computation
        """
        if self.params.nspectra == 1:
            self._compute_single_spectrum()
        else:
            self._compute_multi_spectrum()

    def run(self) -> None:
        """
        Execute the complete Fisher matrix analysis pipeline.

        For single-spectrum analyses, compression can be enabled via the
        compression parameter. Multi-spectrum analyses do not support
        compression and will raise an error if attempted.
        """
        # Validate compression config
        if self._compression_config is not None and self.params.nspectra > 1:
            raise ValueError(
                "Compression is only supported for single-spectrum (temperature-only) "
                "analyses. For multi-spectrum analyses, set compression=None."
            )

        # Setup phase (rank 0 only)
        if self.rank == 0:
            if self.params is None:
                raise ValueError("Parameters must be set before running analysis")

            self.log("Starting Fisher matrix analysis pipeline", level=1)

            self.setup_fields()
            self.log("Fields setup completed", level=3)

            self.setup_geometry()
            self.log("Geometry setup completed", level=3)

            self.setup_covariance_matrices()
            self.log("Covariance matrices setup completed", level=3)

            self.log(f"Using lmax_signal = {self.lmax_signal} for Cls and beams", level=3)
            self.setup_cls(lmax=self.lmax_signal)
            self.log("Power spectra setup completed", level=3)

            self.setup_beams(lmax=self.lmax_signal)
            self.log("Beam functions setup completed", level=3)

            # Setup compression if configured (single-spectrum only)
            if self._compression_config is not None:
                config = self._compression_config
                self.setup_compression(
                    method=config.get("method", "harmonic"),
                    lmax=config.get("lmax"),
                    epsilon=config.get("epsilon"),
                    mode_fraction=config.get("mode_fraction"),
                    basis=config.get("basis", "noise_weighted"),
                    C_ell=config.get("C_ell"),
                )
                self.log(
                    f"Compression enabled: {self.compression_manager.n_kept} modes "
                    f"({self.compression_manager.compression_ratio:.1%})",
                    level=2,
                )
                # Also prepare covariance matrices for Spectra compatibility
                # (Spectra needs the inverted covariance files even with compression)
                self.prepare_covariance_matrices()
                self.log(
                    "Covariance matrices prepared for Spectra compatibility", level=3
                )
            else:
                # Traditional: prepare covariance matrices (signal + inverse)
                self.prepare_covariance_matrices()
                self.log("Signal matrix and covariance preparation completed", level=3)

        # Synchronize before broadcasting
        self.comm.Barrier()

        # Broadcast shared variables
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

        if self._compression_config is not None:
            self.compression_manager = self.comm.bcast(
                self.compression_manager if self.rank == 0 else None, root=0
            )
        else:
            self.Sig = self.comm.bcast(self.Sig if self.rank == 0 else None, root=0)
            if self.params.do_cross:
                self.NCov2 = self.comm.bcast(
                    self.NCov2 if self.rank == 0 else None, root=0
                )

        self.comm.Barrier()

        # Setup Fisher matrices dimensions
        self.n_ell = self.params.lmax - 1
        self.nell = self.params.nspectra * self.n_ell
        self.fisher = np.zeros((self.nell, self.nell))

        # For multi-spectrum, also need derivative arrays
        if self.params.nspectra > 1:
            self.setup_fisher_matrices()

        # Compute Fisher matrix
        self.compute()

        self.comm.Barrier()

    # =========================================================================
    # Result Retrieval
    # =========================================================================

    def get_fisher_matrix(self) -> np.ndarray | None:
        """Retrieve the computed Fisher information matrix."""
        if self.rank == 0:
            return self.fisher
        return None

    def get_error_bars(self) -> np.ndarray | None:
        """Compute parameter forecast errors from the Fisher matrix."""
        if self.rank == 0 and self.fisher is not None:
            cov_matrix = np.linalg.inv(self.fisher)
            errors = np.sqrt(np.diag(cov_matrix))
            return errors
        return None
