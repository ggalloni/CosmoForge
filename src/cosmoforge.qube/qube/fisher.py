"""
Fisher matrix computation for cosmological parameter estimation.

This module implements the Fisher class for calculating Fisher information matrices
used in cosmological parameter forecasting. The Fisher matrix provides the expected
parameter constraints from observations of spin-0 and spin-2 fields on the sphere
(e.g. CMB temperature and polarization, cosmic shear) and is computed as:

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
    Bins,
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
    cosmological parameter constraints from observations of spin-0 and spin-2 fields
    on the sphere. It inherits from cosmocore.Core and extends it with
    Fisher-specific functionality.

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
        Compression configuration. Supports multiple spin-0 fields (Phase 1).
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

    >>> from cosmoforge.qube import Fisher
    >>> fisher = Fisher("config/fisher_analysis.yaml")
    >>> fisher.run()
    >>> F_matrix = fisher.get_fisher_matrix()

    With compression (spin-0 fields only):

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

    def _build_derivative_matrix(self, ell: int) -> np.ndarray:
        """Build pixel-space derivative matrix dC/dC_ell for single spectrum."""
        dC = np.zeros_like(self.NCov1, dtype=np.float64)
        dC = np.asfortranarray(dC)
        do_derivative_step(dC, 0, self.npixs, self.params.spins, ell, self.collection)
        return dC

    # =========================================================================
    # Single-Spectrum Computation (Unified API - Compression Agnostic)
    # =========================================================================

    def _compute_single_spectrum(self):
        """Compute Fisher matrix for single-spectrum analysis (compression agnostic)."""
        use_compression = (
            hasattr(self, "compression_manager") and self.compression_manager is not None
        )

        if self.rank == 0:
            mode = "compressed" if use_compression else "traditional"
            self.log(f"Starting single-spectrum Fisher computation ({mode})", level=2)

        start_time = time.time() if self.rank == 0 else None

        nbins = self.bins.nbins
        total_elements = nbins * (nbins + 1) // 2
        elements_per_proc = int(np.ceil(total_elements / self.size))

        if self.rank == 0:
            self.log(
                f"Rank {self.rank} will compute ~{elements_per_proc} elements "
                f"({nbins} bins)",
                level=2,
            )

        fisher_local = np.zeros((nbins, nbins))

        if use_compression:
            C_ell = self.collection.spectra_manager.get_cls(0, 0, 0)
            C_inv = self.compression_manager.get_projected_inverse(C_ell)
        else:
            C_inv = self.NCov1

        # Precompute C⁻¹ dC^b for each bin
        cinv_times_dcb = {}
        for bin_idx in range(nbins):
            dC_b = self.get_binned_derivative_matrix(
                bin_idx, beam_smoothing=self.beam_smoothing[: self.n_ell]
            )
            cinv_times_dcb[bin_idx] = matrix_mult(C_inv, dC_b)

        # F_{bb'} = (1/2) Tr[(C⁻¹ dC^b)(C⁻¹ dC^{b'})]
        counter = 0
        for bi in range(nbins):
            for bj in range(bi, nbins):
                counter += 1
                if not (
                    counter > self.rank * elements_per_proc
                    and counter <= (self.rank + 1) * elements_per_proc
                ):
                    continue

                fisher_val = 0.5 * matrix_trace(cinv_times_dcb[bi], cinv_times_dcb[bj])
                fisher_local[bi, bj] = fisher_val
                if bi != bj:
                    fisher_local[bj, bi] = fisher_val

        self.comm.Barrier()
        reduced_fisher = np.zeros_like(fisher_local)
        self.comm.Reduce(fisher_local, reduced_fisher, op=MPI.SUM, root=0)

        if self.rank == 0:
            self.fisher = reduced_fisher
            self.log("-" * 80, level=1)
            self.log("Fisher matrix computation completed", level=1)

            if start_time is not None:
                elapsed = time.time() - start_time
                self.log(f"Total computation time: {elapsed:.2f} seconds", level=3)

            if hasattr(self.params, "outfilefisher"):
                write_out_matrix(self.params.outfilefisher, self.fisher)

        self.comm.Barrier()

    def _build_multi_spectrum_inputs(self):
        """Build C_ell_dict and spectra_list for multi-spectrum compressed Fisher."""
        return self.collection.spectra_manager.build_inputs()

    # =========================================================================
    # Multi-Spectrum Computation
    # =========================================================================

    def _get_binned_derivative_multi(
        self, bin_idx: int, spectrum_idx: int, spectra_list=None
    ) -> np.ndarray:
        """Compute beam-smoothed binned derivative for multi-spectrum.

        Handles both pixel-space and compressed paths. Beam smoothing
        factors b²_ell are absorbed into the binning weights.
        """
        use_compression = (
            hasattr(self, "compression_manager") and self.compression_manager is not None
        )

        n_ell = self.n_ell
        w_matrix, _ = self.bins._bin_operators()
        lmin_b = self.bins.lmins[bin_idx]
        lmax_b = self.bins.lmaxs[bin_idx]
        beam_offset = spectrum_idx * n_ell
        dC_b = None

        for ell in range(lmin_b, lmax_b + 1):
            weight = w_matrix[bin_idx, ell] * self.beam_smoothing[beam_offset + ell - 2]

            if use_compression:
                comp_i, comp_j, mode = spectra_list[spectrum_idx]
                dC_ell = self.compression_manager.get_derivative_matrix(
                    ell, comp_i, comp_j, mode
                )
            else:
                dC_ell = np.zeros_like(self.NCov1)
                dC_ell = np.asfortranarray(dC_ell, dtype=np.float64)
                do_derivative_step(
                    dC_ell,
                    spectrum_idx,
                    self.npixs,
                    self.params.spins,
                    ell,
                    self.collection,
                )

            if dC_b is None:
                dC_b = weight * dC_ell
            else:
                dC_b += weight * dC_ell

        return dC_b

    def _compute_multi_spectrum(self):
        """Compute Fisher matrix for multi-spectrum analysis (compression supported)."""
        use_compression = (
            hasattr(self, "compression_manager") and self.compression_manager is not None
        )

        if self.rank == 0:
            mode = "compressed" if use_compression else "traditional"
            self.log(f"Starting multi-spectrum Fisher computation ({mode})", level=2)

        start_time = time.time() if self.rank == 0 else None

        nbins = self.bins.nbins
        nspectra = self.params.nspectra
        n_params = nspectra * nbins

        total_elements = n_params * (n_params + 1) // 2
        elements_per_proc = int(np.ceil(total_elements / self.size))

        if self.rank == 0:
            self.log(
                f"Rank {self.rank} will compute ~{elements_per_proc} elements "
                f"({nspectra} spectra x {nbins} bins)",
                level=2,
            )

        fisher_local = np.zeros((n_params, n_params))

        spectra_list = None
        if use_compression:
            C_ell_dict, spectra_list = self._build_multi_spectrum_inputs()
            C_inv = self.compression_manager.get_projected_inverse(C_ell_dict)
        else:
            if self.params.do_cross:
                C_inv1 = self.NCov1
                C_inv2 = self.NCov2
            else:
                C_inv = self.NCov1

        # Precompute binned derivatives and C⁻¹ dC^b products
        binned_derivatives = {}
        cinv_times_dcb = {}
        for param_idx in range(n_params):
            spectrum_idx = param_idx // nbins
            bin_idx = param_idx % nbins
            binned_deriv = self._get_binned_derivative_multi(
                bin_idx, spectrum_idx, spectra_list
            )
            binned_derivatives[param_idx] = binned_deriv

            if use_compression or not self.params.do_cross:
                cinv_times_dcb[param_idx] = matrix_mult(C_inv, binned_deriv)
            else:
                cinv_times_dcb[param_idx] = matrix_mult(
                    C_inv2, matrix_mult(binned_deriv, C_inv1)
                )

        # F_{ij} = (1/2) Tr[(C⁻¹ dC^i)(C⁻¹ dC^j)]
        counter = 0
        for param_i in range(n_params):
            for param_j in range(param_i, n_params):
                counter += 1
                if not (
                    counter > self.rank * elements_per_proc
                    and counter <= (self.rank + 1) * elements_per_proc
                ):
                    continue

                if use_compression or not self.params.do_cross:
                    fisher_val = 0.5 * matrix_trace(
                        cinv_times_dcb[param_i], cinv_times_dcb[param_j]
                    )
                else:
                    fisher_val = 0.5 * matrix_trace(
                        binned_derivatives[param_j], cinv_times_dcb[param_i]
                    )

                fisher_local[param_i, param_j] = fisher_val
                if param_i != param_j:
                    fisher_local[param_j, param_i] = fisher_val

        self.comm.Barrier()
        reduced_fisher = np.zeros_like(fisher_local)
        self.comm.Reduce(fisher_local, reduced_fisher, op=MPI.SUM, root=0)

        if self.rank == 0:
            self.fisher = reduced_fisher
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

        Compression can be enabled via the compression parameter for
        spin-0 field analyses. Spin-2 field compression is planned for Phase 2.
        """
        # Note: Multi-field compression validation is done in Core.setup_compression()
        # Phase 1 supports multiple spin-0 fields; spin-2 support is planned for Phase 2

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

            # Setup compression if configured (spin-0 fields only for Phase 1)
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

        # Setup binning: from user set_binning() > config delta_ell > default (1)
        if not hasattr(self, "bins") or self.bins is None:
            delta_ell = getattr(self.params, "delta_ell", 1)
            self.set_binning(Bins.fromdeltal(2, self.params.lmax, delta_ell))

        # Warn if bins don't cover the full ell range
        if self.bins.lmax < self.params.lmax:
            self.log(
                f"Binning covers ell up to {self.bins.lmax}, "
                f"but lmax={self.params.lmax}. "
                f"Multipoles {self.bins.lmax + 1}..{self.params.lmax} "
                f"are excluded.",
                level=1,
            )

        # Beam smoothing factors b²_ell for each spectrum (product of beam
        # and pixel window functions). Flat vector: [spec0_ell2, ..., spec0_ellmax,
        # spec1_ell2, ..., spec1_ellmax, ...].
        self.n_ell = self.params.lmax - 1
        smoothing_dict = self.collection.spectra_manager.compute_smoothing_factors(
            self.collection.beam_manager
        )
        self.beam_smoothing = np.zeros(
            self.params.nspectra * self.n_ell, dtype=np.float64
        )
        idx = 0
        for label in self.collection.spectra_manager.labels:
            self.beam_smoothing[idx : idx + self.n_ell] = smoothing_dict[label]
            idx += self.n_ell

        # Setup Fisher matrices dimensions
        self.nell = self.params.nspectra * self.bins.nbins
        self.fisher = np.zeros((self.nell, self.nell))

        # Compute Fisher matrix
        self.compute()

        self.comm.Barrier()

    # =========================================================================
    # Result Retrieval
    # =========================================================================

    def get_fisher_matrix(self) -> np.ndarray | None:
        """Retrieve the beam-smoothed Fisher information matrix."""
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

    def get_window_matrix(self) -> np.ndarray | None:
        """
        Retrieve the window matrix for QML power spectrum estimation.

        The window matrix W relates the expected QML estimates to the
        beam-smoothed theory spectrum: <q> = W @ C_theory. It encodes
        the mode coupling induced by partial sky coverage, beam, and
        pixel window effects.

        Returns
        -------
        numpy.ndarray or None
            Window matrix of shape (n_params, n_params) where
            n_params = n_spectra * n_bins. Returns None for worker
            processes or if computation hasn't completed.

        Notes
        -----
        The window matrix is the beam-smoothed Fisher matrix:

            W_{bb'} = (1/2) Tr[C⁻¹ dC^b C⁻¹ dC^{b'}]

        where dC^b = Sum_ell w_{b,ell} b²_ell dC^ell includes the
        binning weights and beam smoothing factors.

        Used by the "convolved" normalization mode, where instead of
        deconvolving the window, the theory is convolved for comparison.
        """
        return self.get_fisher_matrix()
