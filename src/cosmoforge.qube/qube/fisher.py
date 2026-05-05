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

The unified API from Core transparently handles both harmonic and pixel
computation bases when configured, falling back to traditional pixel-space
computation otherwise.

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

import os
import time
from contextlib import contextmanager, nullcontext

import numpy as np
from mpi4py import MPI

try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
except ImportError:  # graceful degradation; setup just runs at OMP_NUM_THREADS
    _threadpool_limits = None


@contextmanager
def _wide_threadpool():
    """Widen BLAS threads to the visible CPU set for rank-0 setup work."""
    if _threadpool_limits is None:
        yield
        return
    n_visible = len(os.sched_getaffinity(0))
    with _threadpool_limits(limits=n_visible):
        yield


from cosmocore import (
    Bins,
    Core,
    InputParams,
    MPISharedMemoryMixin,
    compute_signal_matrix,
    do_derivative_step,
    matrix_inverse_symm,
    matrix_mult,
    matrix_trace,
    write_covmat_reduced,
    write_out_matrix,
)

_WEIGHT_ZERO_THRESHOLD = 1e-30  # Skip derivative terms with negligible weight


def _basis_path_label(basis_manager) -> str:
    """Human-readable label for the active computation-basis path."""
    if basis_manager is None:
        return "traditional"
    if basis_manager.method == "pixel":
        return (
            "pixel-direct"
            if getattr(basis_manager, "_use_direct", False)
            else "pixel-truncated"
        )
    return "harmonic"


class Fisher(Core, MPISharedMemoryMixin):
    """
    Fisher matrix computation for cosmological parameter estimation.

    This class implements the Fisher information matrix calculation for forecasting
    cosmological parameter constraints from observations of spin-0 and spin-2 fields
    on the sphere. It inherits from cosmocore.Core and extends it with
    Fisher-specific functionality.

    The Fisher matrix F_ij quantifies the information content of the data about
    parameters θ_i and θ_j, computed as:

    F_ij = (1/2) * Tr[C^(-1) * ∂C/∂θ_i * C^(-1) * ∂C/∂θ_j]

    The computation uses Core's unified API which transparently handles
    harmonic and pixel computation bases when configured.

    Parameters
    ----------
    params_file : str, optional
        Path to YAML parameter file containing analysis configuration.
    compression : dict, optional
        Computation basis configuration (method, epsilon, basis, mode_fraction).
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
    n_params : int
        Total number of Fisher matrix parameters (nspectra * n_ell).

    Examples
    --------
    Basic Fisher matrix computation:

    >>> from cosmoforge.qube import Fisher
    >>> fisher = Fisher("config/fisher_analysis.yaml")
    >>> fisher.run()
    >>> F_matrix = fisher.get_fisher_matrix()

    With harmonic basis:

    >>> fisher = Fisher("config.yaml", compression={"method": "harmonic"})
    >>> fisher.run()  # Uses harmonic basis transparently
    """

    def __init__(
        self,
        params_file: str | None = None,
        compression: dict | None = None,
        cache_derivatives: bool = False,
        **kwargs,
    ):
        """
        Initialize Fisher matrix computation class.

        Parameters
        ----------
        params_file : str, optional
            Path to YAML configuration file.
        compression : dict, optional
            Computation basis configuration dictionary. Options:

            - method : str ("harmonic" or "pixel")
            - epsilon : float (eigenvalue threshold for pixel basis)
            - basis : str (for pixel: "harmonic", "noise_weighted", etc.)
            - mode_fraction : float (alternative to epsilon)

        cache_derivatives : bool, optional
            Whether to cache binned derivative matrices for Spectra to
            reuse. Default is False — Spectra recomputes via the
            pixel-direct fast path (single Legendre/Wigner pass per bin),
            which is amortized across MPI ranks. Setting True keeps the
            dense per-bin cache, which can cost n_params * n_pix^2 of
            memory (e.g. ~14 GB at QU_nside64 fsky=0.1) but skips the
            recompute. The harmonic-fast path always caches its small
            COO triplets regardless of this flag.
        **kwargs : dict
            Additional keyword arguments passed to Core.
        """
        super().__init__(params=params_file, **kwargs)

        # MPI setup
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Computation basis config
        self._basis_config = compression

        # Derivative caching
        self._cache_derivatives = cache_derivatives

        # Initialize attributes
        self.signal_matrix = None
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
    # Profiling hook (no-op unless ``_profiler`` is set externally)
    # =========================================================================

    def _stage(self, name: str):
        """Return the profiler's stage context, or a nullcontext when none.

        Hosts (e.g. benchmark scripts) attach a ``StageProfiler`` instance
        by setting ``fisher._profiler``; otherwise the wrapping ``with``
        block is a zero-cost no-op.
        """
        profiler = getattr(self, "_profiler", None)
        return profiler.stage(name) if profiler is not None else nullcontext()

    # =========================================================================
    # Setup Methods
    # =========================================================================

    def setup_signal_matrix(self) -> np.ndarray:
        """
        Compute signal covariance S = Sum_ell C_ell P_ell from fiducial spectra.

        Returns
        -------
        numpy.ndarray
            Signal covariance matrix of shape (n_pix, n_pix).
        """
        if self.noise_cov1 is None:
            raise ValueError("Covariance matrices must be set up first")

        self.signal_matrix = np.zeros_like(self.noise_cov1, dtype=np.float64)
        self.signal_matrix = np.asfortranarray(self.signal_matrix, dtype=np.float64)

        start_time = time.time() if self.rank == 0 else None

        compute_signal_matrix(
            S=self.signal_matrix,
            lmax=self.lmax_signal,
            fields=self.collection,
        )

        if self.rank == 0 and start_time is not None:
            elapsed = time.time() - start_time
            self.log(f"Signal matrix computed in {elapsed:.2f} seconds", level=3)

        return self.signal_matrix

    def prepare_covariance_matrices(self):
        """
        Build total covariance C = N + S, then compute C^{-1}.

        After this method, noise_cov1 (and noise_cov2 if cross-correlation)
        hold the inverted total covariance, not the original noise.
        """
        if self.signal_matrix is None:
            self.setup_signal_matrix()

        # Save original noise covariance BEFORE adding signal
        write_covmat_reduced(self.params.outnoisecovmat1, self.noise_cov1)
        if self.params.do_cross:
            write_covmat_reduced(self.params.outnoisecovmat2, self.noise_cov2)

        # Add signal to noise covariance: C = N + S
        self.noise_cov1 = self.noise_cov1 + self.signal_matrix
        self.noise_cov1 = np.asfortranarray(self.noise_cov1)

        # Compute inverse: C^{-1}
        self.noise_cov1 = matrix_inverse_symm(self.noise_cov1)
        write_covmat_reduced(self.params.outinvcovmatfile1, self.noise_cov1)

        if self.params.do_cross:
            self.noise_cov2 = self.noise_cov2 + self.signal_matrix
            self.noise_cov2 = np.asfortranarray(self.noise_cov2)
            self.noise_cov2 = matrix_inverse_symm(self.noise_cov2)
            write_covmat_reduced(self.params.outinvcovmatfile2, self.noise_cov2)

    def _build_derivative_matrix(self, ell: int, spectrum_idx: int = 0) -> np.ndarray:
        """Build pixel-space derivative matrix dC/dC_ell."""
        dC = np.zeros_like(self.noise_cov1, dtype=np.float64)
        dC = np.asfortranarray(dC)
        do_derivative_step(
            dC, spectrum_idx, self.npixs, self.params.spins, ell, self.collection
        )
        return dC

    def _build_multi_spectrum_inputs(self):
        """Build C_ell_dict and spectra_list for multi-spectrum compressed Fisher."""
        return self.collection.spectra_manager.build_inputs()

    # =========================================================================
    # Main Entry Points
    # =========================================================================

    def compute(self):
        """
        Compute Fisher matrix F_{ij} = (1/2) Tr[C⁻¹ dC^i C⁻¹ dC^j].

        Supports single- and multi-spectrum, with or without computation basis.
        For harmonic basis, exploits sparse derivative structure
        for O(nnz_i × nnz_j) trace computation instead of O(n_modes²).
        """
        use_basis = hasattr(self, "basis_manager") and self.basis_manager is not None
        use_harmonic_fast = use_basis and self.basis_manager.method == "harmonic"

        nbins = self.bins.nbins
        nspectra = self.params.nspectra
        n_params = nspectra * nbins

        if self.rank == 0:
            path_label = (
                _basis_path_label(self.basis_manager) if use_basis else "traditional"
            )
            self.log(
                f"Starting Fisher computation (path: {path_label}, "
                f"{nspectra} spectra x {nbins} bins)",
                level=2,
            )

        start_time = time.time() if self.rank == 0 else None
        total_elements = n_params * (n_params + 1) // 2
        elements_per_proc = int(np.ceil(total_elements / self.size))

        fisher_local = np.zeros((n_params, n_params))

        with self._stage("fisher.compute.derivative_cache"):
            # --- Setup: get C_inv or V_Cinv_VT, spectra_list ---
            spectra_list = None
            C_ell_dict = None
            if use_basis:
                C_ell_dict, spectra_list = self._build_multi_spectrum_inputs()
            elif self.params.do_cross:
                C_inv1 = self.noise_cov1
                C_inv2 = self.noise_cov2
            else:
                C_inv = self.noise_cov1

            # For pixel-space single-spectrum, build a dummy spectra_list
            if spectra_list is None:
                spectra_list = [(0, 0, 0)] * nspectra

            # --- Precompute derivatives ---
            # Harmonic fast path: sparse COO triplets (rows, cols, vals)
            # Generic path: dense C⁻¹ dC^b products
            sparse_coo_data = {}
            cinv_times_dcb = {}
            binned_derivatives = {}
            V_Cinv_VT = None

            deriv_start = time.time()

            if use_harmonic_fast:
                bm = self.basis_manager
                field_groups = bm._detect_field_blocks(C_ell_dict)
                V_Cinv_VT = bm.get_projected_inverse(
                    C_ell_dict, field_groups=field_groups
                )

                for param_idx in range(n_params):
                    spectrum_idx = param_idx // nbins
                    bin_idx = param_idx % nbins
                    key = (spectrum_idx, bin_idx)
                    comp_i, comp_j = spectra_list[spectrum_idx][0:2]
                    spec_mode = (
                        spectra_list[spectrum_idx][2]
                        if len(spectra_list[spectrum_idx]) == 3
                        else 0
                    )
                    beam_offset = spectrum_idx * self.n_ell

                    spin_i = self.collection.fields[comp_i].spin
                    spin_j = self.collection.fields[comp_j].spin
                    is_diagonal = comp_i == comp_j and (
                        (spin_i == 0 and spin_j == 0)
                        or (spin_i == 2 and spin_j == 2 and spec_mode in (0, 1))
                    )

                    if is_diagonal:
                        # dC^b diagonal: E_b = sum_ell beam * E_diag
                        E_b_diag = np.zeros(bm.n_kept, dtype=np.float64)
                        for ell in range(
                            self.bins.lmins[bin_idx], self.bins.lmaxs[bin_idx] + 1
                        ):
                            weight = self.beam_smoothing[beam_offset + ell - 2]
                            E_b_diag += weight * bm._get_derivative_diagonal(
                                ell, comp_i, comp_j, spec_mode
                            )
                        nz = np.nonzero(E_b_diag)[0]
                        sparse_coo_data[key] = (nz, nz, E_b_diag[nz])
                    else:
                        # Cross-spectrum (EB/TE/TB): sparse off-diagonal COO
                        all_rows = []
                        all_cols = []
                        all_vals = []
                        for ell in range(
                            self.bins.lmins[bin_idx], self.bins.lmaxs[bin_idx] + 1
                        ):
                            weight = self.beam_smoothing[beam_offset + ell - 2]
                            if abs(weight) < _WEIGHT_ZERO_THRESHOLD:
                                continue
                            dC_ell = bm.get_derivative_matrix(
                                ell, comp_i, comp_j, spec_mode
                            )
                            r, c = np.nonzero(dC_ell)
                            all_rows.append(r)
                            all_cols.append(c)
                            all_vals.append(dC_ell[r, c] * weight)

                        if all_rows:
                            rows = np.concatenate(all_rows)
                            cols = np.concatenate(all_cols)
                            vals = np.concatenate(all_vals)
                            rc_pairs = rows * bm.n_kept + cols
                            unique_pairs, inverse = np.unique(
                                rc_pairs, return_inverse=True
                            )
                            combined_vals = np.zeros(len(unique_pairs))
                            np.add.at(combined_vals, inverse, vals)
                            sparse_coo_data[key] = (
                                unique_pairs // bm.n_kept,
                                unique_pairs % bm.n_kept,
                                combined_vals,
                            )
                        else:
                            sparse_coo_data[key] = (
                                np.array([], dtype=int),
                                np.array([], dtype=int),
                                np.array([], dtype=float),
                            )
            else:
                # Generic path: pixel basis or traditional pixel-space
                if use_basis:
                    C_inv = self.basis_manager.get_projected_inverse(C_ell_dict)

                for param_idx in range(n_params):
                    spectrum_idx = param_idx // nbins
                    bin_idx = param_idx % nbins
                    key = (spectrum_idx, bin_idx)
                    beam_offset = spectrum_idx * self.n_ell
                    beam = self.beam_smoothing[beam_offset : beam_offset + self.n_ell]

                    dC_b = self.get_binned_derivative_matrix(
                        bin_idx,
                        beam_smoothing=beam,
                        spectrum_idx=spectrum_idx,
                        comp_i=spectra_list[spectrum_idx][0] if use_basis else None,
                        comp_j=spectra_list[spectrum_idx][1] if use_basis else None,
                        mode=(
                            spectra_list[spectrum_idx][2]
                            if use_basis and len(spectra_list[spectrum_idx]) == 3
                            else 0
                        ),
                    )
                    # Retain dC when the trace consumes it OR when the user
                    # opted in to caching for Spectra to reuse.
                    trace_needs_dC = not use_basis and self.params.do_cross
                    if trace_needs_dC or self._cache_derivatives:
                        binned_derivatives[key] = dC_b

                    if use_basis or not self.params.do_cross:
                        cinv_times_dcb[key] = matrix_mult(C_inv, dC_b)
                    else:
                        cinv_times_dcb[key] = matrix_mult(
                            C_inv2, matrix_mult(dC_b, C_inv1)
                        )

            # Harmonic-fast COO triplets are small (~2l+1 nonzeros per bin)
            # and Spectra uses them on the hot path, so they're always cached.
            # Dense per-bin matrices can be huge (n_params * n_pix^2) and are
            # only cached when the user explicitly opts in.
            if use_harmonic_fast:
                self._cached_binned_derivatives = None
                self._cached_sparse_coo_data = sparse_coo_data
            elif self._cache_derivatives:
                self._cached_binned_derivatives = binned_derivatives
                self._cached_sparse_coo_data = None
            else:
                self._cached_binned_derivatives = None
                self._cached_sparse_coo_data = None

            if self.rank == 0:
                trace_path = (
                    "sparse-COO harmonic" if use_harmonic_fast else "dense matmul"
                )
                self.log(
                    f"All {n_params} derivatives precomputed in "
                    f"{time.time() - deriv_start:.1f}s (trace path: {trace_path})",
                    level=3,
                )

        with self._stage("fisher.compute.trace_loop"):
            # --- Compute Fisher traces with MPI distribution ---
            # F_{ij} = (1/2) Tr[C⁻¹ dC^i C⁻¹ dC^j]
            trace_start = time.time()
            counter = 0
            for param_i in range(n_params):
                key_i = (param_i // nbins, param_i % nbins)
                for param_j in range(param_i, n_params):
                    counter += 1
                    if not (
                        counter > self.rank * elements_per_proc
                        and counter <= (self.rank + 1) * elements_per_proc
                    ):
                        continue

                    key_j = (param_j // nbins, param_j % nbins)

                    if (
                        use_harmonic_fast
                        and key_i in sparse_coo_data
                        and key_j in sparse_coo_data
                    ):
                        r_i, c_i, v_i = sparse_coo_data[key_i]
                        r_j, c_j, v_j = sparse_coo_data[key_j]
                        M1 = V_Cinv_VT[np.ix_(c_j, r_i)]
                        M2 = V_Cinv_VT[np.ix_(c_i, r_j)]
                        fisher_val = 0.5 * np.einsum("ji,ij,i,j->", M1, M2, v_i, v_j)
                    elif use_basis or not self.params.do_cross:
                        fisher_val = 0.5 * matrix_trace(
                            cinv_times_dcb[key_i], cinv_times_dcb[key_j]
                        )
                    else:
                        fisher_val = 0.5 * matrix_trace(
                            binned_derivatives[key_j], cinv_times_dcb[key_i]
                        )

                    fisher_local[param_i, param_j] = fisher_val
                    if param_i != param_j:
                        fisher_local[param_j, param_i] = fisher_val

            if self.rank == 0:
                self.log(
                    f"Fisher traces done in {time.time() - trace_start:.1f}s "
                    f"({total_elements} elements)",
                    level=3,
                )

            # --- MPI reduce and output ---
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

    def run(self) -> None:
        """
        Execute the complete Fisher matrix analysis pipeline.

        A computation basis can be enabled via the compression parameter.
        """
        # Setup phase (rank 0 only). Wrap the BLAS-heavy LAPACK work in a
        # widened threadpool so rank 0 can use the full visible CPU set
        # while ranks 1..N-1 idle on the subsequent broadcast.
        if self.rank == 0:
            if self.params is None:
                raise ValueError("Parameters must be set before running analysis")

            self.log("Starting Fisher matrix analysis pipeline", level=1)

            with self._stage("fisher.setup_static"):
                self.setup_fields()
                self.log("Fields setup completed", level=3)

                self.setup_geometry()
                self.log("Geometry setup completed", level=3)

            with self._stage("fisher.covariance_setup"):
                with _wide_threadpool():
                    self.setup_covariance_matrices()
                self.log("Covariance matrices setup completed", level=3)

            with self._stage("fisher.setup_cls_beams"):
                self.log(
                    f"Using lmax_signal = {self.lmax_signal} for Cls and beams",
                    level=3,
                )
                self.setup_cls(lmax=self.lmax_signal)
                self.log("Power spectra setup completed", level=3)

                self.setup_beams(lmax=self.lmax_signal)
                self.log("Beam functions setup completed", level=3)

            # Setup computation basis if configured. Only forward keys that
            # are explicitly set in the config dict so the function defaults
            # (e.g. epsilon=1e-6) take effect when the user omits a key.
            if self._basis_config is not None:
                _basis_keys = (
                    "method",
                    "lmax",
                    "epsilon",
                    "mode_fraction",
                    "basis",
                    "C_ell",
                    "use_direct",
                )
                kwargs = {
                    k: self._basis_config[k]
                    for k in _basis_keys
                    if k in self._basis_config
                }
                with self._stage("fisher.basis_setup"):
                    with _wide_threadpool():
                        self.setup_computation_basis(**kwargs)
                bm = self.basis_manager
                path_label = _basis_path_label(bm)
                if bm.method == "harmonic":
                    size_desc = f"{bm.n_kept} modes ({bm.compression_ratio:.1%} kept)"
                else:
                    size_desc = f"{bm.n_kept} pixels"
                self.log(
                    f"Computation basis: {path_label} — {size_desc}",
                    level=2,
                )
                # Fisher's harmonic QML path reads only V_N_inv and V_Ninv_VT
                # after setup. Drop V to free n_modes × n_pix.
                bm.release_pixel_projector()
                # The basis manager handles covariance inversion internally
                # (SMW for harmonic, direct/truncated solve for pixel), so the
                # traditional explicit C = N+S build and inversion is skipped.
                self.log(
                    f"Skipping explicit pixel-space C inversion "
                    f"(handled by {path_label} basis)",
                    level=3,
                )
            else:
                # Traditional: prepare covariance matrices (signal + inverse)
                with self._stage("fisher.prepare_covariance"):
                    with _wide_threadpool():
                        self.prepare_covariance_matrices()
                self.log("Signal matrix and covariance preparation completed", level=3)

        # Synchronize and broadcast shared variables to worker processes.
        # Skip broadcast for single-process runs to avoid unnecessary
        # serialization of large objects (basis_manager can exceed 2 GB).
        self.comm.Barrier()

        with self._stage("fisher.mpi_distribution"):
            if self.size > 1:
                # Python objects (small, serialization is fine)
                self.params: InputParams = self.comm.bcast(
                    self.params if self.rank == 0 else None, root=0
                )
                self.collection = self.comm.bcast(
                    self.collection if self.rank == 0 else None, root=0
                )
                self.npixs = self.comm.bcast(
                    self.npixs if self.rank == 0 else None, root=0
                )
                self.pixact = self.comm.bcast(
                    self.pixact if self.rank == 0 else None, root=0
                )

                # Numpy arrays via shared memory (zero-copy, no size limits).
                # Worker ranks may not have these attributes yet (setup is rank-0 only),
                # so use getattr to pass None for non-root ranks.
                point_vectors = getattr(self, "point_vectors", None)
                n_point_vectors = self.comm.bcast(
                    len(point_vectors)
                    if self.rank == 0 and point_vectors is not None
                    else 0,
                    root=0,
                )
                self.point_vectors = tuple(
                    self._shared_array(
                        point_vectors[i]
                        if self.rank == 0 and point_vectors is not None
                        else None
                    )
                    for i in range(n_point_vectors)
                )

                if self._basis_config is not None:
                    self.basis_manager = self.comm.bcast(
                        self.basis_manager if self.rank == 0 else None, root=0
                    )
                else:
                    # Traditional path needs pixel-space matrices
                    self.noise_cov1 = self._shared_array(
                        getattr(self, "noise_cov1", None)
                    )
                    self.signal_matrix = self._shared_array(
                        getattr(self, "signal_matrix", None)
                    )
                    if self.params.do_cross:
                        self.noise_cov2 = self._shared_array(
                            getattr(self, "noise_cov2", None)
                        )

                self.comm.Barrier()

        with self._stage("fisher.binning_beam"):
            # Binning: set_binning() > config bin_lmins/lmaxs > config delta_ell
            if not hasattr(self, "bins") or self.bins is None:
                bin_lmins = getattr(self.params, "bin_lmins", None)
                bin_lmaxs = getattr(self.params, "bin_lmaxs", None)
                if bin_lmins is not None and bin_lmaxs is not None:
                    self.set_binning(Bins(bin_lmins, bin_lmaxs))
                else:
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
            self.n_params = self.params.nspectra * self.bins.nbins
            self.fisher = np.zeros((self.n_params, self.n_params))

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

    def get_bandpower_window_function(
        self,
        per_ell_fisher: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """
        Compute the bandpower window function for binned QML inference.

        The window matrix W maps a per-ℓ theory spectrum to the expected
        binned bandpower estimates from the deconvolved QML output:

            <C_hat_b> = W @ C_ell_theory

        Use this in a Gaussian likelihood for parameter inference:

            mu_b(θ) = W @ C_ell(θ)
            -2 ln L = (d - mu)ᵀ F_b (d - mu)

        where d = ``Spectra.get_power_spectra(mode="deconvolved")`` and
        F_b = ``Fisher.get_fisher_matrix()``.

        This differs from :meth:`get_window_matrix`, which returns the
        bin-to-bin Fisher used in "convolved" mode and assumes a constant
        spectrum within each bin. The bandpower window function is the
        correct quantity for inference when bins are wide enough that
        the theory varies appreciably within them.

        Parameters
        ----------
        per_ell_fisher : np.ndarray, optional
            Pre-computed per-ℓ Fisher matrix of shape ``(n_ell, n_ell)``
            (single spectrum). If None, the per-ℓ Fisher is computed on
            demand by re-running the Fisher pipeline with delta_ell=1
            using the current configuration. The result is cached for
            subsequent calls.

        Returns
        -------
        W : np.ndarray of shape (n_bins, n_ell), or None
            Bandpower window matrix. Rows correspond to bin indices,
            columns to multipoles ℓ=2..lmax. Returns None on worker
            processes (rank != 0).

        Notes
        -----
        The window function encodes:

        - Mode coupling from sky cuts.
        - Beam smoothing (b²_ℓ).
        - Pixel window functions.
        - The Fisher-weighting that QML naturally applies within each bin.

        The theory C_ℓ should be the **unbeamed** physical spectrum
        (standard CAMB/CLASS output), since beam² is already absorbed
        into the derivatives used to build the window.

        For best statistical accuracy near lmax, use the buffer approach:
        estimate to ``lmax_buffer = lmax_science + margin`` (a few bin
        widths or ~1/fsky multipoles), then drop the buffer rows from
        ``W``, ``d`` and ``F_b`` before forming the likelihood. This
        absorbs mode coupling from ℓ > lmax_science into discarded
        buffer bins.

        Currently supports single-spectrum analysis only. Multi-spectrum
        extension is planned.

        Examples
        --------
        Standard inference workflow:

        >>> fisher = Fisher("config.yaml", compression={"method": "harmonic"})
        >>> fisher.set_binning(bins)
        >>> fisher.run()
        >>> W = fisher.get_bandpower_window_function()
        >>> F_b = fisher.get_fisher_matrix()
        >>> # In the MCMC loop:
        >>> mu = W @ cl_theory_per_ell        # expected bandpower
        >>> r = data_bandpower - mu           # residual
        >>> neg2lnL = r @ F_b @ r             # quadratic form

        See Also
        --------
        get_window_matrix : Bin-to-bin window for convolved mode.
        get_fisher_matrix : Inverse covariance for the bandpower likelihood.
        """
        # _compute_per_ell_fisher() is collective; all ranks must
        # enter together. Only rank 0 returns W; workers return None.
        if not hasattr(self, "bins") or self.bins is None:
            if self.rank == 0:
                raise ValueError(
                    "Bandpower window requires binning. Call set_binning() before run()."
                )
            return None

        if self.params.nspectra > 1:
            if self.rank == 0:
                raise NotImplementedError(
                    "Bandpower window function currently supports "
                    "single-spectrum analysis only. "
                    "Multi-spectrum extension is planned."
                )
            return None

        # Broadcast rank 0's decisions so all ranks branch identically.
        cached = getattr(self, "_bandpower_window", None)
        provided = bool(self.comm.bcast(per_ell_fisher is not None, root=0))
        has_cached = bool(self.comm.bcast(cached is not None, root=0)) and not provided
        if has_cached:
            return cached if self.rank == 0 else None

        if not provided:
            per_ell_fisher = self._compute_per_ell_fisher()

        # Only rank 0 has self.fisher and the reduced per_ell_fisher
        # populated; workers return None.
        if self.rank != 0 or self.fisher is None:
            return None

        # Build Q (sum-over-ℓ-in-bin operator)
        n_ell = self.params.lmax - 1  # ell indices 2..lmax
        Q = np.zeros((self.bins.nbins, n_ell), dtype=np.float64)
        for b, (lo, hi) in enumerate(zip(self.bins.lmins, self.bins.lmaxs)):
            Q[b, lo - 2 : hi - 2 + 1] = 1.0

        # W = F_b^{-1} @ Q @ F_perell
        W = np.linalg.solve(self.fisher, Q @ per_ell_fisher)
        self._bandpower_window = W
        return W

    def _compute_per_ell_fisher(self) -> np.ndarray:
        """
        Compute the per-ℓ Fisher matrix using the current configuration.

        Triggers a Fisher computation with delta_ell=1, reusing the
        existing setup (basis, beams, covariance) by re-invoking the
        compute() phase with binning temporarily set to per-ℓ.

        Returns
        -------
        F_perell : np.ndarray, shape (n_ell, n_ell)
            Per-ℓ Fisher for ell=2..lmax.
        """
        self.log("Computing per-ℓ Fisher for bandpower window function...", level=2)
        saved_bins = self.bins
        saved_fisher = self.fisher
        saved_n_params = self.n_params
        saved_cached = getattr(self, "_cached_binned_derivatives", None)
        saved_coo = getattr(self, "_cached_sparse_coo_data", None)
        try:
            self.bins = Bins.fromdeltal(2, self.params.lmax, 1)
            self.n_params = self.params.nspectra * self.bins.nbins
            self.fisher = np.zeros((self.n_params, self.n_params))
            # Drop binned-derivative cache so per-ℓ run rebuilds its own
            self._cached_binned_derivatives = None
            self._cached_sparse_coo_data = None
            self.compute()
            F_perell = self.fisher.copy()
        finally:
            self.bins = saved_bins
            self.fisher = saved_fisher
            self.n_params = saved_n_params
            self._cached_binned_derivatives = saved_cached
            self._cached_sparse_coo_data = saved_coo
        return F_perell
