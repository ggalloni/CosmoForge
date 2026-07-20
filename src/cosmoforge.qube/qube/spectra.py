"""
Quadratic Maximum Likelihood (QML) power spectrum estimation for cosmological analysis.

This module implements the Spectra class for optimal power spectrum estimation from
observations of spin-0 and spin-2 fields on the sphere using the Quadratic Maximum
Likelihood estimator. The QML method provides unbiased, minimum-variance power spectrum
estimates that properly account for sky cuts, instrumental noise, and pixel correlations.
Applications include CMB temperature and polarization, cosmic shear, and any other signal
described by angular power spectra.

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

import os
import time
import typing
from contextlib import nullcontext

import numpy as np

from cosmocore import (
    Bins,
    Core,
    MPISharedMemoryMixin,
    SpectrumKey,
    SpectrumKind,
    SymmetryMode,
    do_derivative_step,
    matrix_inverse_symm,
    matrix_mult,
    matrix_trace,
    vec_to_cl,
    write_out_matrix,
    writecl,
)
from cosmocore._mpi import MPI
from cosmocore.basics import eigh
from cosmocore.settings import InputParams
from qube import Fisher
from qube.fisher import _basis_path_label


class Spectra(Core, MPISharedMemoryMixin):
    """
    Quadratic Maximum Likelihood (QML) power spectrum estimator.

    This class implements the QML method for optimal power spectrum estimation from
    observations of spin-0 and spin-2 fields on the sphere (e.g. CMB temperature
    and polarization, cosmic shear). The QML estimator provides unbiased,
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
        Inverse of the beam-smoothed Fisher matrix.
    inv_cov1, inv_cov2 : numpy.ndarray
        Inverted covariance matrices for primary and secondary datasets.

    Examples
    --------
    Basic QML power spectrum estimation:

    >>> from qube import Spectra
    >>> spectra = Spectra("config/qml_analysis.yaml")
    >>> spectra.run()
    >>> power_spectra = spectra.get_power_spectra()
    >>> noise_bias = spectra.get_noise_bias()

    Using pre-computed Fisher matrix:

    >>> from qube import Fisher, Spectra
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
        basis: dict | str | bool | None = Core._UNSET,
        compression: dict | str | bool | None = Core._UNSET,
        mask: np.ndarray | None = None,
        noise_cov1: np.ndarray | None = None,
        noise_cov2: np.ndarray | None = None,
        cls_data: dict | np.ndarray | None = None,
        fiducial_cls: dict | np.ndarray | None = None,
        beam: np.ndarray | None = None,
        maps1: np.ndarray | None = None,
        maps2: np.ndarray | None = None,
        **kwargs,
    ):
        """
        Initialize QML power spectrum estimation class.

        Parameters
        ----------
        params_file : str, optional
            Path to YAML configuration file.
        fisher : Fisher, optional
            Pre-computed Fisher instance. If provided, reuses computed components
            (covariance matrices, geometry, field collections) for efficiency.
        basis : None, False, str or dict, optional
            Computation basis selection (ADR-0018). ``None`` (default) →
            ``method="auto"``; ``False`` → traditional pixel-space path;
            ``"auto"``/``"harmonic"``/``"pixel"`` → ``{"method": …}``; a dict is
            the only form that requests compression.
        compression : dict, optional
            Deprecated alias for ``basis`` (ADR-0018): warns and is forwarded.
        mask : numpy.ndarray, optional
            In-memory mask injected in place of ``params.maskfile``; see
            :meth:`Core.__init__` for the contract (ADR-0017).
        noise_cov1, noise_cov2 : numpy.ndarray, optional
            In-memory noise covariances injected in place of
            ``params.covmatfile1``/``covmatfile2``; see :meth:`Core.__init__`
            for the contract (ADR-0017). Forwarded into the internally-built
            ``Fisher``; cannot be combined with ``fisher=``.
        cls_data, fiducial_cls : dict or numpy.ndarray, optional
            In-memory power spectra injected in place of
            ``params.inputclfile``/``params.fiducialfile``; see
            :meth:`Core.__init__` for the contract (ADR-0017). Forwarded into
            the internally-built ``Fisher``; cannot be combined with ``fisher=``.
        beam : numpy.ndarray, optional
            In-memory beam injected in place of ``params.beam_file``; see
            :meth:`Core.__init__` for the contract (ADR-0017). Forwarded into
            the internally-built ``Fisher``; cannot be combined with ``fisher=``.
        maps1, maps2 : numpy.ndarray, optional
            In-memory map data injected in place of
            ``params.inputmapfile1``/``inputmapfile2`` (ADR-0017). Exactly what
            :func:`read_maps` would have returned: the reduced
            ``(n_active, n_sims)`` float64 array, pixel-ordered per the
            concatenated active-pixel index. Used as-is. ``maps2`` is consumed
            only when ``params.do_cross``. Independent of ``fisher=`` (maps are
            not part of the Fisher reuse), so both may be given together.
        **kwargs : dict
            Additional arguments passed to Core.

        Raises
        ------
        TypeError
            If fisher is not a Fisher instance.
        ValueError
            If fisher doesn't contain a valid Fisher matrix, or if ``mask=`` /
            ``noise_cov1=`` / ``noise_cov2=`` is supplied together with
            ``fisher=``.
        """
        self.params: InputParams = None
        super().__init__(
            params=params_file,
            mask=mask,
            noise_cov1=noise_cov1,
            noise_cov2=noise_cov2,
            cls_data=cls_data,
            fiducial_cls=fiducial_cls,
            beam=beam,
            **kwargs,
        )

        # MPI setup
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Store computation basis config for Fisher creation (ADR-0018).
        self._basis_config = self._resolve_basis_config(
            basis, compression, self.params.do_cross
        )

        # Initialize Fisher matrix or compute it
        if fisher is not None:
            if not isinstance(fisher, Fisher):
                raise TypeError("fisher must be an instance of Fisher class.")
            if not hasattr(fisher, "fisher") or fisher.fisher is None:
                raise ValueError("Fisher instance must have a valid fisher matrix.")
            if self._injected_mask is not None:
                raise ValueError(
                    "mask= cannot be used together with fisher= because Spectra "
                    "reuses the Fisher geometry and mask."
                )
            if (
                self._injected_noise_cov1 is not None
                or self._injected_noise_cov2 is not None
            ):
                raise ValueError(
                    "noise_cov1= or noise_cov2= cannot be used together with "
                    "fisher= because Spectra reuses the Fisher covariances."
                )
            if (
                self._injected_cls_data is not None
                or self._injected_fiducial_cls is not None
            ):
                raise ValueError(
                    "cls_data= or fiducial_cls= cannot be used together with "
                    "fisher= because Spectra reuses the Fisher spectra setup."
                )
            if self._injected_beam is not None:
                raise ValueError(
                    "beam= cannot be used together with fisher= because Spectra "
                    "reuses the Fisher beam setup."
                )
            self.fisher_instance = fisher
            # Reuse already computed components from Fisher
            self._reuse_fisher_components()
        else:
            self.fisher_instance = self._get_fisher()
            # Also reuse components from the internally created Fisher
            self._reuse_fisher_components()

        # SymmetryMode is owned by Fisher (ADR-0011); Spectra inherits it
        # so the two cannot drift apart and produce dimension-mismatched
        # spectra lists.
        self.symmetry_mode = self.fisher_instance.symmetry_mode

        # Initialize QML-specific variables
        self.maps1 = None
        self.maps2 = None
        # Injected map arrays (ADR-0017, A3); consumed by setup_maps. Stored
        # here rather than forwarded to Core: maps are a Spectra-only seam.
        self._injected_maps1 = maps1
        self._injected_maps2 = maps2
        self.qml_results = None
        self.qml_noise_bias = None
        self.invfisher = None
        # Populated lazily in compute_qml_spectra so _get_binned_derivative
        # (Core's keyed API consumer) can resolve spectrum_idx → SpectrumKey
        # without threading lists through method signatures.
        self.spectra_list: list | None = None

        # Normalization mode support
        self.inv_fisher_sqrt: np.ndarray | None = None  # F^(-1/2) for decorrelated mode
        self.fisher_normalized: np.ndarray | None = (
            None  # Normalized Fisher (for convolved covariance)
        )

        # lmax for signal matrix computation (matches Fortran convention of 4*nside)
        self._lmax_signal = None

    def _stage(self, name: str):
        """Return the profiler's stage context, or a nullcontext when none.

        Hosts (e.g. benchmark scripts) attach a ``StageProfiler`` instance
        by setting ``spectra._profiler``; otherwise the wrapping ``with``
        block is a zero-cost no-op.
        """
        profiler = getattr(self, "_profiler", None)
        return profiler.stage(name) if profiler is not None else nullcontext()

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
        """Copy computational components from a pre-computed Fisher instance."""
        attrs = [
            "collection",
            "npixs",
            "pixact",
            "point_vectors",
            "signal_matrix",
            "bins",
            "beam_smoothing",
            "basis_manager",
        ]
        for attr in attrs:
            val = getattr(self.fisher_instance, attr, None)
            if val is not None:
                setattr(self, attr, val)

        if not hasattr(self, "basis_manager"):
            self.basis_manager = None

        # Harmonic basis uses SMW formula — pixel-space covariance matrices
        # (inv_cov, noise_cov) are not needed for the compressed QML path.
        if self.basis_manager is None:
            self._resolve_covariance_matrices()

    def _resolve_covariance_matrices(self):
        """Obtain inv_cov (C^{-1}) and noise_cov (reduced N) for the traditional path.

        Priority (ADR-0016): (i) alias the live Fisher's in-memory arrays;
        (ii) read the ``out*`` files (opt-in two-job transport); (iii) raise.
        After ``run()``, ``fisher.noise_cov1`` is C^{-1} while N lives on
        ``fisher.reduced_noise_cov1``; both are aliased (read-only downstream).
        """
        fisher = self.fisher_instance
        if getattr(fisher, "reduced_noise_cov1", None) is not None:
            self.inv_cov1 = fisher.noise_cov1
            self.noise_cov1 = fisher.reduced_noise_cov1
            if self.params.do_cross:
                self.inv_cov2 = fisher.noise_cov2
                self.noise_cov2 = fisher.reduced_noise_cov2
            return

        disk_paths = [self.params.outinvcovmatfile1, self.params.outnoisecovmat1]
        if self.params.do_cross:
            disk_paths += [self.params.outinvcovmatfile2, self.params.outnoisecovmat2]
        if all(disk_paths):
            self._load_covariance_matrices()
            return

        required = (
            "outinvcovmatfile1/2 and outnoisecovmat1/2"
            if self.params.do_cross
            else "outinvcovmatfile1 and outnoisecovmat1"
        )
        raise ValueError(
            "Cannot obtain covariance matrices for the traditional QML path: the "
            "Fisher instance holds no in-memory covariances (reduced_noise_cov1 is "
            f"None) and not all disk-adapter paths are set (do_cross="
            f"{self.params.do_cross} requires {required}). Pass a run() Fisher via "
            "fisher=, or set those paths to load from a prior job."
        )

    def _load_covariance_matrices(self):
        """Load noise and inverted covariance matrices from disk files (adapter ii)."""
        ntot = self.collection.total_active_pixels

        # Load inverted covariance matrices
        self.inv_cov1 = np.fromfile(self.params.outinvcovmatfile1).reshape(ntot, ntot)
        if self.params.do_cross:
            self.inv_cov2 = np.fromfile(self.params.outinvcovmatfile2).reshape(ntot, ntot)

        # Load noise covariance matrices (created by Fisher.run())
        if not os.path.exists(self.params.outnoisecovmat1):
            raise FileNotFoundError(
                f"Noise covariance file not found: {self.params.outnoisecovmat1}. "
                f"Run Fisher analysis first to generate this file."
            )
        self.noise_cov1 = np.fromfile(self.params.outnoisecovmat1).reshape(ntot, ntot)
        if self.params.do_cross:
            if not os.path.exists(self.params.outnoisecovmat2):
                raise FileNotFoundError(
                    f"Noise covariance file not found: {self.params.outnoisecovmat2}. "
                    f"Run Fisher analysis first to generate this file."
                )
            self.noise_cov2 = np.fromfile(self.params.outnoisecovmat2).reshape(ntot, ntot)

    def _get_fisher(self) -> Fisher:
        """
        Compute Fisher information matrix for QML normalization.

        Returns
        -------
        Fisher
            Completed Fisher instance with all components computed.
        """
        if self.rank == 0:
            self.log("Starting Fisher matrix computation...", level=1)

        start_time = time.time()

        # Forward the already-resolved config losslessly: None (traditional) →
        # basis=False, dict → the dict. Avoids a second resolution flipping the
        # traditional sentinel back to auto.
        fisher = Fisher(
            self.params,
            basis=(False if self._basis_config is None else self._basis_config),
            mask=self._injected_mask,
            noise_cov1=self._injected_noise_cov1,
            noise_cov2=self._injected_noise_cov2,
            cls_data=self._injected_cls_data,
            fiducial_cls=self._injected_fiducial_cls,
            beam=self._injected_beam,
        )
        fisher.run()

        if self.rank == 0:
            elapsed = time.time() - start_time
            self.log(
                f"Fisher matrix computation completed in {elapsed:.2f} seconds", level=1
            )

        return fisher

    def setup_maps(self):
        """
        Read observational map data from FITS files.

        Loads maps with proper pixel selection and field extraction.
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
            self.maps1 = self._resolve_maps(
                self._injected_maps1, self.params.inputmapfile1, ntot
            )
            if self.params.do_cross:
                self.maps2 = self._resolve_maps(
                    self._injected_maps2, self.params.inputmapfile2, ntot
                )

    def setup_fisher_inversion(self):
        """
        Prepare inverted Fisher matrix for QML estimation.

        The Fisher matrix is already beam-smoothed (beam window functions
        absorbed into derivatives). This method computes F⁻¹, F⁻¹/²,
        and writes results to output files.

        Raises
        ------
        ValueError
            If Fisher matrix is not available or singular.
        LinAlgError
            If Fisher matrix inversion fails.
        """
        if self.rank == 0:
            self.log("Reading and inverting Fisher matrix", level=2)

            # Get Fisher matrix from the Fisher instance
            fisher_matrix = self.fisher_instance.get_fisher_matrix()
            if fisher_matrix is None:
                raise ValueError("Fisher matrix not available")

            self.invfisher = fisher_matrix.copy()

            # Store beam-smoothed Fisher for convolved mode covariance
            self.fisher_normalized = self.invfisher.copy()

            # F^(-1/2) for the decorrelated mode is computed lazily on first
            # request — it costs an O(n_params^3) eigendecomposition that only
            # decorrelated mode consumes.

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
            nbins = self.bins.nbins
            nspectra = len(vec_error_bars) // nbins
            error_bars = np.zeros((nbins, nspectra), dtype=np.float64)
            vec_to_cl(vec_error_bars, error_bars)
            writecl(self.params.outerrfile, error_bars)

    def _compute_inv_fisher_sqrt(self, fisher: np.ndarray) -> None:
        """
        Compute F^(-1/2) using eigendecomposition for decorrelated mode.

        This method computes the matrix square root of the inverse Fisher matrix
        using eigendecomposition: F = V Λ V^T → F^(-1/2) = V Λ^(-1/2) V^T.
        This is used for the "decorrelated" normalization mode which produces
        uncorrelated bandpower estimates.

        Parameters
        ----------
        fisher : numpy.ndarray
            Normalized Fisher information matrix of shape (n_params, n_params).
            Must be symmetric positive semi-definite.

        Notes
        -----
        The computation uses eigendecomposition for numerical stability:

        1. Decompose: F = V Λ V^T where V are eigenvectors, Λ are eigenvalues
        2. Compute inverse square root of eigenvalues: Λ^(-1/2)
        3. Reconstruct: F^(-1/2) = V @ diag(Λ^(-1/2)) @ V^T

        **Ill-conditioning handling:**
        - Eigenvalues below 10^(-12) × max eigenvalue are set to zero
        - A warning is logged if condition number exceeds 10^10
        - This truncation prevents numerical instability from near-zero modes

        The resulting F^(-1/2) matrix is stored in self.inv_fisher_sqrt and
        used by the decorrelated mode to produce estimates with identity
        covariance matrix.

        Examples
        --------
        This method is called automatically during setup_fisher_inversion():

        >>> spectra = Spectra("config.yaml")
        >>> spectra.run()
        >>> # inv_fisher_sqrt is now available for decorrelated mode
        >>> cl_decorr = spectra.get_power_spectra(mode="decorrelated")

        See Also
        --------
        get_power_spectra : Uses inv_fisher_sqrt for 'decorrelated' mode
        setup_fisher_inversion : Calls this method during setup
        """
        eigenvalues, eigenvectors = eigh(fisher)

        # Check conditioning
        min_eigenvalue = (
            eigenvalues[eigenvalues > 0].min() if np.any(eigenvalues > 0) else 1e-300
        )
        max_eigenvalue = eigenvalues.max()
        cond = max_eigenvalue / min_eigenvalue if min_eigenvalue > 0 else np.inf

        if cond > 1e10:
            self.log(
                f"Warning: Fisher matrix poorly conditioned (cond={cond:.2e}). "
                "Decorrelated mode may have inflated errors.",
                level=1,
            )

        # Compute Λ^(-1/2), handling small eigenvalues
        inv_sqrt_eigenvalues = np.zeros_like(eigenvalues)
        threshold = max_eigenvalue * 1e-12
        valid = eigenvalues > threshold
        inv_sqrt_eigenvalues[valid] = 1.0 / np.sqrt(eigenvalues[valid])

        # F^(-1/2) = V @ diag(Λ^(-1/2)) @ V^T
        self.inv_fisher_sqrt = (
            eigenvectors @ np.diag(inv_sqrt_eigenvalues) @ eigenvectors.T
        )

        self.log(
            f"Computed F^(-1/2) for decorrelated mode "
            f"({np.sum(valid)}/{len(eigenvalues)} modes valid)",
            level=3,
        )

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
        n_params = self.params.nspectra * self.bins.nbins

        # Initialize y vectors for QML estimation
        self.qml_results = np.zeros((self.params.nsims, n_params), dtype=np.float64)

        if not self.params.do_cross:
            self.qml_noise_bias = np.zeros(n_params, dtype=np.float64)

    def compute_qml_spectra(self):
        """
        Execute parallel QML power spectrum computation.

        Distributes multipole computations across MPI processes (round-robin).
        For each multipole: computes signal derivative, E-operator, and
        quadratic estimates for all simulations. Selects compressed or
        traditional method based on basis_manager availability.
        """
        # Resolve spectra_list once so the per-bin SpectrumKey lookup inside
        # _get_binned_derivative works for both compressed and traditional
        # paths. Mirrors Fisher.compute() which populates
        # ``self.fisher_instance.spectra_list`` for both paths too. Fall
        # back to building locally if Fisher hasn't populated it yet
        # (test fixtures that bypass Fisher.compute()).
        if self.spectra_list is None:
            fisher_list = getattr(self.fisher_instance, "spectra_list", None)
            if fisher_list is not None:
                self.spectra_list = list(fisher_list)
            else:
                _, self.spectra_list = self._build_multi_spectrum_inputs()

        # Check if we should use compressed computation
        use_basis = hasattr(self, "basis_manager") and self.basis_manager is not None

        if use_basis:
            self._compute_qml_spectra_compressed()
        else:
            self._compute_qml_spectra_traditional()

    def _build_multi_spectrum_inputs(self):
        """Build C_ell_dict and spectra_list keyed by SpectrumKey."""
        # Fall back to SYMMETRIC for callers that constructed Spectra via
        # __new__ (test fixtures bypass __init__).
        symmetry_mode = getattr(self, "symmetry_mode", SymmetryMode.SYMMETRIC)
        return self.collection.spectra_manager.build_inputs(symmetry_mode=symmetry_mode)

    def _build_derivative_matrix(self, ell: int, spectrum_idx: int = 0) -> np.ndarray:
        """Build pixel-space derivative matrix dC/dC_ell (no-basis fallback)."""
        ntot = sum(self.collection.n_active)
        dC = np.zeros((ntot, ntot), dtype=np.float64, order="F")
        do_derivative_step(dC, spectrum_idx, current_ell=ell, fields=self.collection)
        return dC

    def _get_binned_derivative(self, bin_idx: int, spectrum_idx: int = 0) -> np.ndarray:
        """
        Compute binned derivative dC^b = Sum_{ell in bin} b²_ell dC^ell.

        Uses Fisher's cache when available, otherwise delegates to
        :meth:`Core.get_binned_derivative_matrix` with the SpectrumKey
        resolved from ``self.spectra_list[spectrum_idx]``.
        """
        if hasattr(self, "fisher_instance") and self.fisher_instance is not None:
            cache = getattr(self.fisher_instance, "_cached_binned_derivatives", None)
            if cache is not None:
                cache_key = (spectrum_idx, bin_idx)
                if cache_key in cache:
                    return cache[cache_key]

        # beam_smoothing per-spectrum blocks hold the inference-range beams
        # (ell=lmin..lmax, offset-from-lmin; ADR 0009).
        n_ell = self.params.lmax - self.params.lmin + 1
        beam_offset = spectrum_idx * n_ell
        beam = self.beam_smoothing[beam_offset : beam_offset + n_ell]

        key = self.spectra_list[spectrum_idx]
        return self.get_binned_derivative_matrix(
            bin_idx,
            key,
            beam_smoothing=beam,
            symmetry_mode=self.symmetry_mode,
        )

    def _compute_noise_cov_compressed(
        self,
        bm,
        stable_inner_inv,
        *,
        full_matrix=False,
    ) -> np.ndarray:
        """Compute noise covariance in compressed space.

        ``Cov(w|noise) = V C^{-1} N C^{-1} V^T = A T A^T`` with
        ``A = (I + Lambda M)^{-T} = stable_inner_inv.T`` and
        ``T = V N_eff^{-1} N N_eff^{-1} V^T`` (sourced from the basis
        via :meth:`ComputationBasis.get_noise_for_bias`).
        """
        A = stable_inner_inv.T
        AT = A @ bm.get_noise_for_bias()
        if full_matrix:
            return AT @ A.T
        return np.einsum("ij,ij->i", AT, A)

    def _compute_qml_spectra_compressed(self):
        """
        Compute QML spectra using a computation basis (harmonic or pixel).

        Performs QML estimation entirely in basis space, consistent with the
        basis-aware Fisher matrix computation. The estimator is

            q_b = (1/2) * w^T @ E_b @ w

        where w = V C^{-1} d is the basis-projected weighted data (built via
        SMW for the harmonic basis) and E_b is the binned derivative matrix.
        This is mathematically equivalent to the traditional pixel-space form
        q_b = (1/2) * d^T C^{-1} dC_b C^{-1} d, since dC_b = V^T E_b V.

        Inner-loop paths
        ----------------
        - **Sparse-COO** (preferred when ``basis_manager.method == "harmonic"``
          and Fisher cached the COO triplets): for each bin the per-ℓ
          derivative is stored as ``(rows, cols, vals)`` with O(2ℓ+1) nonzeros.
          q and the noise-bias trace reduce to einsum/contractions over the
          nonzero pattern (~200x-800x faster than dense E_b @ w at production
          scale; same trick the Fisher trace path uses).
        - **Dense fallback**: pixel basis, or harmonic basis when the COO
          cache is missing. Builds the full E_b matrix on demand and uses
          the original dense E_b @ w / matrix_trace path.

        Supports single-field, multi-field, and cross-correlation
        (``do_cross=True``) configurations on both paths.
        """
        if self.rank == 0:
            path_label = _basis_path_label(self.basis_manager)
            self.log(f"Starting QML computation (path: {path_label})", level=2)

        start_time = time.time()

        bm = self.basis_manager
        n_sims = self.params.nsims
        dim = bm.dim

        # Multi-field path is needed when >1 components or spin-2 (spin-2 has
        # multiple spectra EE/BB/EB even for a single field)
        has_spin2 = any(f.spin == 2 for f in self.collection.fields)
        is_multi_field = bm.n_components > 1 or has_spin2

        # Build C_ell or C_ell_dict depending on multi-field
        if is_multi_field:
            C_ell_dict, spectra_list = self._build_multi_spectrum_inputs()
            C_ell = None  # Not used for multi-field
        else:
            C_ell = self.collection.spectra_manager.get_cls(0, 0, 0)
            C_ell_dict = None
            spins = tuple(f.spin for f in self.collection.fields)
            spectra_list = [SpectrumKey(0, 0, SpectrumKind.SS, spins=spins)]
        self.spectra_list = spectra_list

        # Compute weighted compressed data for all simulations
        # w = V @ C^{-1} @ d (using SMW formula internally)
        # For multi-field harmonic, precompute kernel_inv once — reused for
        # data weighting, cross-correlation weighting, and noise bias.
        # Stable SMW algebra: w = V C^{-1} d = (I + Lambda M)^{-1} (V N^{-1} d).
        # Replaces the legacy w = y - M K^{-1} y, which loses precision in
        # the cosmic-variance-limited regime via catastrophic cancellation.
        # The same matrix (I + Lambda M)^{-1} reappears as the noise-bias
        # matrix A = I - M K^{-1}, so we cache it once and reuse below.
        stable_inner_inv = None
        maps1_weighted = np.zeros((dim, n_sims), dtype=np.float64)
        C_arg = C_ell_dict if is_multi_field else C_ell
        if bm.method == "harmonic":
            stable_inner_inv = bm.prepare_stable_inner_inv(C_arg)
            maps1_weighted = bm.get_weighted_data(
                self.maps1, C_arg, stable_inner_inv=stable_inner_inv
            )
        else:
            C_c_inv = bm.get_inverse(C_arg)
            if is_multi_field:
                # pixel multi-field: explicit compress-then-multiply
                d_c = bm.to_basis(self.maps1)
                maps1_weighted = C_c_inv @ d_c
            else:
                maps1_weighted = bm.get_weighted_data(self.maps1, C_ell, C_c_inv=C_c_inv)

        if self.params.do_cross:
            maps2_weighted = np.zeros((dim, n_sims), dtype=np.float64)
            if bm.method == "harmonic":
                maps2_weighted = bm.get_weighted_data(
                    self.maps2, C_arg, stable_inner_inv=stable_inner_inv
                )
            elif is_multi_field:
                d_c2 = bm.to_basis(self.maps2)
                maps2_weighted = C_c_inv @ d_c2
            else:
                maps2_weighted = bm.get_weighted_data(self.maps2, C_ell, C_c_inv=C_c_inv)

        # For noise bias computation (only for non-cross)
        noise_cov_w_diag = None
        noise_cov_w = None
        if not self.params.do_cross:
            if bm.method == "harmonic":
                # Non-diagonal derivatives (cross-field spin-0, EB, TE, TB)
                # need the full noise covariance for correct Tr[E_b @ Cov].
                # Only single-spectrum (pure TT/EE/BB) can use diagonal alone.
                need_full = self.params.nspectra > 1
                if need_full:
                    noise_cov_w = self._compute_noise_cov_compressed(
                        bm, stable_inner_inv, full_matrix=True
                    )
                    noise_cov_w_diag = np.diag(noise_cov_w)
                else:
                    noise_cov_w_diag = self._compute_noise_cov_compressed(
                        bm, stable_inner_inv
                    )
            else:
                # For pixel: use compressed quantities. Noise bias requires
                # the raw N (not N_eff with absorbed high-ℓ signal), so use
                # the dedicated get_noise_for_bias() method.
                if is_multi_field:
                    C_bar_inv = bm.get_inverse(C_ell_dict)
                else:
                    C_bar_inv = bm.get_inverse(C_ell)
                N_bar = bm.get_noise_for_bias()
                noise_cov_w = C_bar_inv @ N_bar @ C_bar_inv

        # Harmonic-basis binned derivatives are extremely sparse (~2ℓ+1 nonzeros
        # in an n_modes×n_modes layout), so Fisher caches them in COO form. When
        # available, replace dense E_b @ w with einsum over the nonzero pattern.
        # The dense path remains for pixel-basis QML, which has no comparable
        # sparsity, and for the cache-less case used by some tests.
        coo_cache = (
            getattr(self.fisher_instance, "_cached_sparse_coo_data", None)
            if getattr(self, "fisher_instance", None) is not None
            else None
        )
        use_sparse = coo_cache is not None
        self._qml_path_used = "sparse" if use_sparse else "dense"

        # Main computation loop - distribute bins across processes
        nbins = self.bins.nbins
        n_params = self.params.nspectra * nbins
        for il in range(n_params):
            if self.rank == il % self.size:
                spectrum_idx = il // nbins
                bin_idx = il % nbins

                if use_sparse:
                    rows, cols, vals = coo_cache[(spectrum_idx, bin_idx)]
                    if vals.size == 0:
                        if not self.params.do_cross:
                            self.qml_noise_bias[il] = 0.0
                        self.qml_results[:, il] = 0.0
                        continue

                    if self.params.do_cross:
                        # q = 0.5 * sum_k v_k * w2[r_k, s] * w1[c_k, s]
                        self.qml_results[:, il] = 0.5 * np.einsum(
                            "k,ks,ks->s",
                            vals,
                            maps2_weighted[rows],
                            maps1_weighted[cols],
                        )
                    else:
                        q_per_sim = 0.5 * np.einsum(
                            "k,ks,ks->s",
                            vals,
                            maps1_weighted[rows],
                            maps1_weighted[cols],
                        )

                        # Noise bias: 0.5 * Tr[E_b @ Cov] = 0.5 * sum_k v * Cov[c,r]
                        if noise_cov_w is not None:
                            tr_ne = 0.5 * float(np.sum(vals * noise_cov_w[cols, rows]))
                        else:
                            # Diagonal-only noise cov is sufficient when E_b is
                            # diagonal (rows == cols for harmonic auto-spectra).
                            tr_ne = 0.5 * float(np.sum(vals * noise_cov_w_diag[cols]))
                        self.qml_noise_bias[il] = tr_ne

                        if hasattr(self.params, "remove_nb") and self.params.remove_nb:
                            q_per_sim -= tr_ne
                        self.qml_results[:, il] = q_per_sim
                    continue

                # Get binned derivative: 1D diagonal vector (harmonic auto-
                # spectra) or 2D dense matrix (cross-spectra / pixel-space).
                E_b = self._get_binned_derivative(bin_idx, spectrum_idx)

                # Exploit diagonal structure when available:
                # diag * maps is O(n × n_sims) vs dense @ maps O(n² × n_sims)
                is_diag = E_b.ndim == 1
                if is_diag:
                    E_b_times_w1 = E_b[:, np.newaxis] * maps1_weighted
                else:
                    E_b_times_w1 = E_b @ maps1_weighted  # (n_modes, n_sims)

                if self.params.do_cross:
                    # q = 0.5 * w2^T @ E_b @ w1 for all sims
                    self.qml_results[:, il] = 0.5 * np.sum(
                        maps2_weighted * E_b_times_w1, axis=0
                    )
                else:
                    # Noise bias: 0.5 * Tr[E_b @ Cov(w|noise)]. The dense-E_b /
                    # diag-only-noise branch handles the single-spectrum cache-miss
                    # recompute path where _get_binned_derivative returns dense.
                    if is_diag and noise_cov_w_diag is not None:
                        tr_ne = 0.5 * np.sum(E_b * noise_cov_w_diag)
                    elif is_diag:
                        tr_ne = 0.5 * np.sum(E_b * np.diag(noise_cov_w))
                    elif noise_cov_w is not None:
                        tr_ne = 0.5 * matrix_trace(E_b, noise_cov_w)
                    else:
                        tr_ne = 0.5 * np.sum(np.diagonal(E_b) * noise_cov_w_diag)
                    self.qml_noise_bias[il] = tr_ne

                    # q = 0.5 * w^T @ E_b @ w for all sims
                    qml_values = 0.5 * np.sum(maps1_weighted * E_b_times_w1, axis=0)

                    if hasattr(self.params, "remove_nb") and self.params.remove_nb:
                        qml_values -= tr_ne

                    self.qml_results[:, il] = qml_values

        # Synchronize all processes
        self.comm.Barrier()

        if self.rank == 0:
            path_label = _basis_path_label(self.basis_manager)
            self.log(f"QML computation done (path: {path_label})", level=2)
            self.log(
                f"QML computation time: {time.time() - start_time:.2f} seconds", level=3
            )

        # Reduce results from all processes
        self._reduce_qml_results(n_params)

    def _compute_qml_spectra_traditional(self):
        """
        Compute QML spectra using traditional pixel-space computation.

        Optimized: Precomputes y = C^{-1} @ d to avoid building full E matrix.

        The QML estimator is:
            q_b = (1/2) * d^T @ C^{-1} @ dC_b @ C^{-1} @ d
                = (1/2) * y^T @ dC_b @ y   where y = C^{-1} @ d
        """
        if self.rank == 0:
            self.log("Starting QML computation (path: traditional)", level=2)

        start_time = time.time()

        nbins = self.bins.nbins
        nspectra = self.params.nspectra
        n_params = nspectra * nbins

        # Precompute weighted data: y = C⁻¹ d for all simulations
        weighted_maps1 = matrix_mult(self.inv_cov1, self.maps1)

        if self.params.do_cross:
            weighted_maps2 = matrix_mult(self.inv_cov2, self.maps2)

        # For noise bias: C⁻¹ N C⁻¹ (precomputed once)
        if not self.params.do_cross:
            cinv_noise_cinv = matrix_mult(
                self.inv_cov1, matrix_mult(self.noise_cov1, self.inv_cov1)
            )

        for param_idx in range(n_params):
            if self.rank == param_idx % self.size:
                spectrum_idx = param_idx // nbins
                bin_idx = param_idx % nbins

                binned_deriv = self._get_binned_derivative(bin_idx, spectrum_idx)
                deriv_times_y1 = matrix_mult(binned_deriv, weighted_maps1)

                if self.params.do_cross:
                    self.qml_results[:, param_idx] = 0.5 * np.sum(
                        weighted_maps2 * deriv_times_y1, axis=0
                    )
                else:
                    noise_bias = 0.5 * matrix_trace(cinv_noise_cinv, binned_deriv)
                    self.qml_noise_bias[param_idx] = noise_bias

                    qml_values = 0.5 * np.sum(weighted_maps1 * deriv_times_y1, axis=0)
                    if hasattr(self.params, "remove_nb") and self.params.remove_nb:
                        qml_values -= noise_bias

                    self.qml_results[:, param_idx] = qml_values

        self.comm.Barrier()

        if self.rank == 0:
            self.log("QML computation done (path: traditional)", level=2)
            self.log(
                f"QML computation time: {time.time() - start_time:.2f} seconds",
                level=3,
            )

        self._reduce_qml_results(n_params)

    def _reduce_qml_results(self, n_params: int):
        """Gather and combine QML results from all MPI processes."""
        # Reduce y vectors
        reduced_qml_results = np.zeros((self.params.nsims, n_params), dtype=np.float64)
        self.comm.Reduce(self.qml_results, reduced_qml_results, op=MPI.SUM, root=0)

        if not self.params.do_cross:
            # Reduce noise bias
            reduced_qml_noise_bias = np.zeros(n_params, dtype=np.float64)
            self.comm.Reduce(
                self.qml_noise_bias, reduced_qml_noise_bias, op=MPI.SUM, root=0
            )

        if self.rank == 0:
            self.qml_results = reduced_qml_results
            if not self.params.do_cross:
                self.qml_noise_bias = reduced_qml_noise_bias

    def compute(self):
        """
        Execute QML power spectrum computation.

        Calls setup_qml_computation() to initialize arrays, then
        compute_qml_spectra() for the main parallel computation.
        """
        with self._stage("spectra.compute.setup"):
            self.setup_qml_computation()

        # Compute QML power spectra
        with self._stage("spectra.compute.qml_spectra"):
            self.compute_qml_spectra()

    def run(self):
        """
        Execute the complete QML power spectrum analysis pipeline.

        Pipeline phases:

        1. Master setup: fields, geometry, covariance, spectra, beams
           (reuses Fisher components if provided)
        2. QML setup: maps, Fisher inversion
        3. MPI broadcast of shared data to workers
        4. Parallel QML computation
        5. MPI synchronization

        Raises
        ------
        ValueError
            If required setup components are missing.
        FileNotFoundError
            If input files not accessible.
        """
        # Only rank 0 does the initial setup
        if self.rank == 0:
            with self._stage("spectra.setup_static"):
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
                        f"Using lmax_signal = {self.lmax_signal} for Cls and beams",
                        level=3,
                    )
                    self.setup_cls(lmax=self.lmax_signal)
                    self.setup_beams(lmax=self.lmax_signal)
                    # Load covariance matrices when not reusing a Fisher instance
                    self._load_covariance_matrices()

            with self._stage("spectra.binning"):
                # Setup binning: Fisher > set_binning() > config > default
                if not hasattr(self, "bins") or self.bins is None:
                    delta_ell = getattr(self.params, "delta_ell", 1)
                    self.set_binning(
                        Bins.fromdeltal(self.params.lmin, self.params.lmax, delta_ell)
                    )

            with self._stage("spectra.setup_maps"):
                self.setup_maps()

            with self._stage("spectra.setup_fisher_inversion"):
                self.setup_fisher_inversion()

        # Synchronize and broadcast shared variables to worker processes.
        # Skip broadcast for single-process runs to avoid unnecessary
        # serialization of large objects (covariance matrices can exceed 2 GB).
        self.comm.Barrier()

        with self._stage("spectra.mpi_distribution"):
            if self.size > 1:
                self._broadcast_variables()
                self.comm.Barrier()

        # Main QML computation (parallel computation)
        self.compute()

        # Finalize
        self.comm.Barrier()

    def _broadcast_variables(self):
        """Broadcast essential data from master to all MPI worker processes."""
        # Python objects (small, serialization is fine)
        self.params = self.comm.bcast(self.params if self.rank == 0 else None, root=0)
        self.collection = self.comm.bcast(
            self.collection if self.rank == 0 else None, root=0
        )
        self.bins = self.comm.bcast(self.bins if self.rank == 0 else None, root=0)
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

        use_basis = hasattr(self, "basis_manager") and self.basis_manager is not None

        # Pixel-space covariance matrices only needed for traditional path
        if not use_basis:
            self.noise_cov1 = self._shared_array(getattr(self, "noise_cov1", None))
            self.inv_cov1 = self._shared_array(getattr(self, "inv_cov1", None))
            if self.params.do_cross:
                self.noise_cov2 = self._shared_array(getattr(self, "noise_cov2", None))
                self.inv_cov2 = self._shared_array(getattr(self, "inv_cov2", None))

        # Maps (can be large: n_pix × n_sims)
        self.maps1 = self._shared_array(getattr(self, "maps1", None))
        if self.params.do_cross:
            self.maps2 = self._shared_array(getattr(self, "maps2", None))

        # Fisher-related (small but still read-only)
        self.invfisher = self._shared_array(getattr(self, "invfisher", None))
        self.beam_smoothing = self._shared_array(getattr(self, "beam_smoothing", None))
        self.inv_fisher_sqrt = self._shared_array(getattr(self, "inv_fisher_sqrt", None))
        self.fisher_normalized = self._shared_array(
            getattr(self, "fisher_normalized", None)
        )

    def _normalize_spectra(self, spectra: np.ndarray) -> np.ndarray:
        """
        Apply inverse Fisher to raw QML estimates: Ĉ = F⁻¹ q.

        Parameters
        ----------
        spectra : np.ndarray
            Raw QML estimates, shape (n_params,) or (n_sims, n_params).

        Returns
        -------
        np.ndarray
            Deconvolved power spectrum estimates, same shape as input.
        """
        if self.invfisher is None:
            raise ValueError("Fisher inversion must be set up first.")
        return spectra @ self.invfisher

    def get_power_spectra(
        self, mode: str = "deconvolved", *, as_dict: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, typing.Callable] | dict | None:
        """
        Retrieve power spectrum estimates in specified normalization mode.

        This method returns power spectrum estimates with three different
        normalization prescriptions, allowing users to choose the output format
        best suited to their analysis needs.

        Parameters
        ----------
        mode : str, optional
            Normalization mode for output (default: "deconvolved"):

            - "deconvolved": F⁻¹y - estimates of true C_ℓ with correlated errors.
              This is the standard QML output that attempts to recover the true
              underlying spectrum by inverting the window function.

            - "decorrelated": F⁻¹/²y - uncorrelated bandpower estimates with
              identity covariance matrix. Useful when independent error bars
              are needed.

            - "convolved": Raw y estimates with window matrix W for theory
              comparison. Instead of deconvolving the window, compare with
              window-convolved theory: <y> = W @ C_true.

        as_dict : bool, optional
            If ``False`` (default, back-compat), return the flat numpy
            array described below. If ``True``, return a dict keyed by
            physical labels (``"TT"``, ``"EE"``, ``"TE"``, ...) whose
            values have shape ``(n_simulations, n_bins)``. In
            ``"convolved"`` mode, the dict applies to the raw estimates
            ``y`` only; the window matrix ``W`` and ``convolve_theory``
            callable are unchanged (use :meth:`Fisher.get_bandpower_slices`
            on ``self.fisher_instance`` to navigate ``W``).

        Returns
        -------
        numpy.ndarray or tuple or dict or None
            For "deconvolved" or "decorrelated" modes with ``as_dict=False``,
            an array of shape ``(n_simulations, n_parameters)`` where
            ``n_parameters = n_spectra * n_bins``.

            For "convolved" mode with ``as_dict=False``, a tuple
            ``(y, W, convolve_theory_func)`` where ``y`` is the raw estimates
            of shape ``(n_simulations, n_parameters)``, ``W`` is the window
            matrix of shape ``(n_parameters, n_parameters)``, and
            ``convolve_theory_func`` is a callable that applies
            ``W @ theory``.

            With ``as_dict=True``, a dict for the per-spectrum arrays
            (and, in convolved mode, a 3-tuple ``(y_dict, W, convolve)``).

            Returns ``None`` for worker processes (rank != 0) or if
            computation has not completed.

        **Layout (flat-array shape)**:
        The columns of the returned array are organised as ``n_spectra``
        contiguous blocks of ``n_bins`` columns each. Spectrum order
        follows :attr:`Fisher.spectra_list` on
        ``self.fisher_instance`` — auto-spectra first (per component,
        in the order each component emits them), then cross-spectra in
        ``(i, j)`` order with ``i < j``. For TQU (``spins=[0, 2]``,
        ``labels=["T", "E", "B"]``) under ``SymmetryMode.SYMMETRIC``
        with ``nbins=10``, ``n_parameters = 60`` and the columns are::

            cols  0..9   : TT
            cols 10..19  : EE
            cols 20..29  : BB
            cols 30..39  : EB
            cols 40..49  : TE
            cols 50..59  : TB

        Use :meth:`Fisher.get_bandpower_slices` (on
        ``self.fisher_instance``) or pass ``as_dict=True`` to look up
        per-spectrum slices without computing these offsets by hand.

        Raises
        ------
        ValueError
            If mode is not one of "deconvolved", "decorrelated", "convolved".

        Notes
        -----
        **Mode comparison:**

        - **deconvolved** -- formula ``F⁻¹y``, covariance ``F⁻¹`` (correlated),
          standard analysis.
        - **decorrelated** -- formula ``F⁻¹/²y``, covariance ``I`` (identity),
          independent errors.
        - **convolved** -- formula ``y``, covariance ``F``, theory comparison.

        **Deconvolved Mode (default):**
        The standard QML output that inverts the window function to recover
        estimates of the true C_ℓ. Errors are correlated between multipoles.

        **Decorrelated Mode:**
        Produces bandpower estimates with uncorrelated errors (unit variance).
        Useful for plotting error bars or when independent measurements are
        required. Note that information content is preserved but redistributed.

        **Convolved Mode:**
        Returns raw QML estimates without deconvolution, along with the window
        matrix for comparing with convolved theoretical spectra. This avoids
        numerical issues from inverting poorly-conditioned window matrices.

        **Output Convention:**
        When ``output_convention`` is set to ``"Dl"`` in the configuration,
        the estimator itself works in D_ℓ = ℓ(ℓ+1)/(2π) C_ℓ (ADR-0019): the
        bandpower shape weight enters the binned derivative, so the returned
        spectra, covariance and window are D_ℓ quantities natively — there is
        no post-hoc scalar. In convolved mode the returned
        ``convolve_theory_func`` therefore expects D_ℓ input. ``Dl`` and ``Cl``
        estimate genuinely different observables (``D_output ≠ scalar · C_output``
        for wide bins); it must be fixed before ``run()``.

        Examples
        --------
        Default (flat array) — back-compat::

            >>> spectra = Spectra("config.yaml")
            >>> spectra.run()
            >>> cl_deconv = spectra.get_power_spectra()        # shape (n_sims, 60)
            >>> ee_slice = cl_deconv[:, 10:20]                 # bins 0..9 of EE

        Label-keyed dict (recommended for human-readable code)::

            >>> cl = spectra.get_power_spectra(as_dict=True)
            >>> cl["EE"].shape                                 # (n_sims, n_bins)
            >>> cl["TE"]                                       # cross-spectrum

        Decorrelated bandpowers::

            >>> cl_decorr = spectra.get_power_spectra(mode="decorrelated")
            >>> # Errors are all 1.0 by construction

        Convolved mode for theory comparison::

            >>> y, W, convolve = spectra.get_power_spectra(mode="convolved")
            >>> cl_theory_convolved = convolve(cl_theory)
            >>> # Compare: y should match cl_theory_convolved

            >>> # With as_dict=True, the y component becomes a dict; W and
            >>> # the convolve callable are unchanged.
            >>> y_dict, W, convolve = spectra.get_power_spectra(
            ...     mode="convolved", as_dict=True
            ... )

        See Also
        --------
        get_covariance : Get covariance matrix for specified mode
        get_error_bars : Get 1-sigma error bars for specified mode
        convolve_theory : Apply window matrix to theoretical spectrum
        Fisher.get_bandpower_slices : Slice mapping for navigating the flat layout.
        """
        valid_modes = ("deconvolved", "decorrelated", "convolved")
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got '{mode}'")

        if self.rank != 0 or self.qml_results is None:
            return None

        if mode == "deconvolved":
            result = self._get_deconvolved()
        elif mode == "decorrelated":
            result = self._get_decorrelated()
        else:  # convolved
            result = self._get_convolved()

        if as_dict and result is not None:
            from cosmocore.conventions.cmb import to_label_dict

            labels = list(self.params.labels)
            spins = tuple(self.params.spins)
            spectra_list = self.fisher_instance.spectra_list
            n_bins = self.bins.nbins
            if mode == "convolved":
                y, W, convolve = result
                y_dict = to_label_dict(
                    y,
                    labels=labels,
                    spins=spins,
                    spectra_list=spectra_list,
                    n_bins=n_bins,
                )
                return y_dict, W, convolve
            return to_label_dict(
                result,
                labels=labels,
                spins=spins,
                spectra_list=spectra_list,
                n_bins=n_bins,
            )

        return result

    def _spectrum_labels(self) -> list[str] | None:
        """Physical spectrum labels in flat spectrum-major order.

        Returns ``None`` when ``spectra_list`` is unavailable (i.e. before
        ``fisher.run()``), so dict-keyed inputs cannot be assembled.
        """
        from cosmocore.conventions.cmb import spectrum_key_to_label

        spectra_list = self.fisher_instance.spectra_list
        if spectra_list is None:
            return None
        labels_per_slot = list(self.params.labels)
        spins = tuple(self.params.spins)
        return [
            spectrum_key_to_label(k, labels=labels_per_slot, spins=spins)
            for k in spectra_list
        ]

    def _make_convolve_theory(self, W: np.ndarray):
        """Build a ``convolve_theory`` callable that accepts either a flat
        ``cl_theory`` of length ``n_params`` or a label-keyed dict mapping
        physical labels (e.g. ``"TT"``, ``"EE"``) to per-bin arrays.

        Both forms return a flat numpy array of length ``n_params`` so the
        downstream comparison code does not have to branch on shape.

        Convention
        ----------
        The input values must be in the configuration's active output
        convention. In ``Dl`` mode the window ``W`` is itself built from the
        shape-weighted derivative (ADR-0019), so it already maps D_ℓ theory to
        the D_b estimate — pass D_ℓ-binned theory; in ``Cl`` mode pass
        C_ℓ-binned theory. Mixing conventions yields silently-wrong predictions.
        """
        # Pre-compute the spectrum order in physical labels so dict inputs
        # can be assembled into the flat vector without re-deriving labels
        # on every call.
        spectrum_labels = self._spectrum_labels()

        def convolve_theory(cl_theory) -> np.ndarray:
            if isinstance(cl_theory, dict):
                if spectrum_labels is None:
                    raise RuntimeError(
                        "convolve_theory dict input requires spectra_list "
                        "(call fisher.run() first)."
                    )
                missing = [lab for lab in spectrum_labels if lab not in cl_theory]
                if missing:
                    raise KeyError(
                        f"convolve_theory dict missing entries for: {missing}. "
                        f"Expected keys: {spectrum_labels}"
                    )
                cl_flat = np.concatenate(
                    [np.asarray(cl_theory[lab]) for lab in spectrum_labels]
                )
            else:
                cl_flat = np.asarray(cl_theory)
            return W @ cl_flat

        return convolve_theory

    def _get_deconvolved(self) -> np.ndarray:
        """
        Compute deconvolved power spectrum estimates (F⁻¹y).

        This is the standard QML output that multiplies raw estimates by the
        inverse Fisher matrix to recover estimates of the true C_ℓ.

        Returns
        -------
        numpy.ndarray
            Deconvolved power spectrum estimates with shape (nsims, n_params).
        """
        return self._normalize_spectra(self.qml_results)

    def _get_decorrelated(self) -> np.ndarray:
        """
        Compute decorrelated bandpower estimates (F⁻¹/²y).

        Produces uncorrelated estimates with identity covariance matrix.
        F^(-1/2) is computed lazily on first call.

        Returns
        -------
        numpy.ndarray
            Decorrelated power spectrum estimates with shape (nsims, n_params).
        """
        if self.inv_fisher_sqrt is None:
            self._compute_inv_fisher_sqrt(self.fisher_normalized)

        decorrelated = self.qml_results @ self.inv_fisher_sqrt

        return decorrelated

    def _get_convolved(self) -> tuple[np.ndarray, np.ndarray, typing.Callable]:
        """
        Get raw QML estimates with window matrix for theory comparison.

        Returns raw y estimates along with the window matrix W, allowing
        comparison with window-convolved theoretical spectra.

        Returns
        -------
        tuple
            ``(y, W, convolve_theory_func)`` where:

            - ``y``: Raw normalized estimates, shape ``(nsims, n_params)``
            - ``W``: Window matrix, shape ``(n_params, n_params)``
            - ``convolve_theory_func``: Callable accepting either a flat
              ``cl_theory`` array of length ``n_params`` (ordered per
              :meth:`Fisher.get_bandpower_slices`) or a label-keyed dict
              like ``{"TT": [...], "EE": [...], ...}``. Returns a flat
              array of length ``n_params``.

        Notes
        -----
        With binning enabled, ``n_params = nspectra * nbins``. The theory
        input to ``convolve_theory`` must be binned (one value per bin),
        matching the in-bin shape declared by the binning operator
        (ADR-0019). For the flat form, the entries must be ordered the
        same way as the Fisher matrix columns; the dict form bypasses
        the ordering question entirely.
        """
        y = self.qml_results

        W = self.fisher_instance.get_window_matrix()
        if W is None:
            raise ValueError("Window matrix not available from Fisher instance.")

        return (y, W, self._make_convolve_theory(W))

    def get_effective_ells(self, use_midpoint: bool = False) -> np.ndarray | None:
        """
        Return the effective multipole for each bin.

        By default this is the inverse-variance-weighted window centroid
        (ADR-0019), delegated to :meth:`Fisher.get_effective_ells` — it needs a
        completed Fisher run and differs per spectrum. Pass ``use_midpoint=True``
        for the cheap opt-in that returns the bin midpoint (``Bins.lmid``) with
        no Fisher run.

        Parameters
        ----------
        use_midpoint : bool, optional
            If True, return the bin midpoint instead of the window centroid.

        Returns
        -------
        np.ndarray or None
            Effective multipoles. Shape ``(nspectra·nbins,)`` for the window
            centroid, ``(nbins,)`` for the midpoint. None for workers.
        """
        if use_midpoint:
            return self.bins.lmid if self.rank == 0 else None
        # Delegate on every rank — Fisher.get_effective_ells is collective and
        # must not be fronted by a worker early-return (ADR-0019 MPI note).
        return self.fisher_instance.get_effective_ells()

    def get_noise_bias(self) -> np.ndarray | None:
        """
        Retrieve noise bias estimates for auto-correlation spectra.

        Returns
        -------
        np.ndarray or None
            Shape (n_params,). Returns None for workers, cross-correlation
            analyses (do_cross=True), or if computation incomplete.

        Notes
        -----
        Noise bias: (1/2) * Tr[N * E_l]. Cross-correlations are bias-free.
        """
        if self.rank == 0 and self.qml_noise_bias is not None:
            # The noise bias is built from the binned derivative E_b, which
            # already carries the bandpower shape weight in Dl mode (ADR-0019),
            # so it comes out in the estimated quantity's own space — no
            # post-hoc scalar.
            return self._normalize_spectra(self.qml_noise_bias)
        return None

    def convolve_theory_for_inference(
        self, cl_theory: np.ndarray | dict
    ) -> np.ndarray | None:
        """
        Apply the QML bandpower window to a per-ℓ theory spectrum.

        Returns the expected binned bandpower for the given theory, in
        the same convention as ``get_power_spectra(mode="deconvolved")``.
        Use this in a Gaussian likelihood for parameter inference::

            mu_b(theta) = convolve_theory_for_inference(C_ell(theta))
            -2 ln L = (d - mu)^T F_b (d - mu)

        Parameters
        ----------
        cl_theory : np.ndarray or dict
            Per-ℓ theory C_ℓ. Accepted formats:

            - shape ``(n_ell,)`` for ℓ=lmin..lmax (length lmax-lmin+1),
              single-spectrum only.
            - shape ``(lmax+1,)`` starting at ℓ=0 (entries below ``lmin``
              ignored), single-spectrum only.
            - shape ``(nspectra·n_ell,)`` block-ordered spectrum-major
              (matching :meth:`Fisher.get_bandpower_slices`), for a
              multi-spectrum run.
            - a label-keyed dict ``{"TT": [...], "EE": [...], ...}`` of
              per-ℓ arrays (each ``n_ell`` or ``lmax+1`` long), which
              bypasses the ordering question entirely.

            Must be **unbeamed** physical C_ℓ — beam² is already absorbed
            into the window.

        Returns
        -------
        cl_binned : np.ndarray of shape (nspectra·n_bins,), or None
            Expected binned bandpower for the given theory. If the
            output convention is "Dl", the result is converted to D_ℓ
            using the bin effective ells. Returns None on worker
            processes (rank != 0).

        Notes
        -----
        This delegates to :meth:`Fisher.get_bandpower_window_function`
        on the underlying Fisher instance. The first call triggers a
        per-ℓ Fisher computation (cached for subsequent calls).

        See :meth:`Fisher.get_bandpower_window_function` for details on
        the buffer approach and the multi-spectrum window.

        Examples
        --------
        >>> spectra = Spectra("config.yaml", fisher=fisher, basis="harmonic")
        >>> spectra.set_binning(bins)
        >>> spectra.run()
        >>> cl_th = compute_camb_cl(theta)        # ell=0..lmax
        >>> mu_b = spectra.convolve_theory_for_inference(cl_th)
        >>> data = spectra.get_power_spectra(mode="deconvolved").mean(0)
        >>> F_b = spectra.fisher_instance.get_fisher_matrix()
        >>> neg2lnL = (data - mu_b) @ F_b @ (data - mu_b)

        See Also
        --------
        Fisher.get_bandpower_window_function : Underlying window matrix.
        get_power_spectra : Get the data bandpower (deconvolved mode).
        get_covariance : Get the covariance matrix (= F_b⁻¹).
        """
        if self.rank != 0:
            return None

        if not hasattr(self, "fisher_instance") or self.fisher_instance is None:
            raise ValueError(
                "convolve_theory_for_inference requires an attached Fisher "
                "instance. Pass fisher=fisher to Spectra() at construction."
            )
        if getattr(self.fisher_instance, "fisher", None) is None:
            raise ValueError(
                "Fisher matrix has not been computed. Call fisher.run() "
                "(and spectra.run()) before convolve_theory_for_inference()."
            )

        lmin = self.params.lmin
        lmax = self.params.lmax
        n_ell = lmax - lmin + 1
        nspectra = self.params.nspectra

        def _to_perell(arr: np.ndarray, what: str) -> np.ndarray:
            """Trim one per-ℓ array to ℓ=lmin..lmax (accepts n_ell or lmax+1)."""
            if arr.ndim != 1:
                raise ValueError(f"{what} must be 1D, got shape {arr.shape}")
            if arr.size == n_ell:
                return arr
            if arr.size >= lmax + 1:
                return arr[lmin : lmax + 1]
            raise ValueError(
                f"{what} length {arr.size} does not match expected "
                f"{n_ell} (ℓ=lmin..lmax) or {lmax + 1} (ℓ=0..lmax)"
            )

        if isinstance(cl_theory, dict):
            labels = self._spectrum_labels()
            if labels is None:
                raise RuntimeError(
                    "cl_theory dict input requires spectra_list "
                    "(call fisher.run() first)."
                )
            missing = [lab for lab in labels if lab not in cl_theory]
            if missing:
                raise KeyError(
                    f"cl_theory dict missing entries for: {missing}. "
                    f"Expected keys: {labels}"
                )
            cl_vec = np.concatenate(
                [
                    _to_perell(np.asarray(cl_theory[lab], dtype=np.float64), lab)
                    for lab in labels
                ]
            )
        else:
            cl = np.asarray(cl_theory, dtype=np.float64)
            if cl.ndim != 1:
                raise ValueError(f"cl_theory must be 1D, got shape {cl.shape}")
            if nspectra > 1:
                if cl.size != nspectra * n_ell:
                    raise ValueError(
                        f"cl_theory length {cl.size} does not match expected "
                        f"{nspectra * n_ell} (nspectra·n_ell, spectrum-major) "
                        f"for a {nspectra}-spectrum run; or pass a label-keyed "
                        f"dict."
                    )
                cl_vec = cl
            else:
                cl_vec = _to_perell(cl, "cl_theory")

        W = self.fisher_instance.get_bandpower_window_function()
        if W is None:
            return None

        # In Dl mode the window W already maps per-ℓ theory to D_b (its
        # derivatives carry the shape weight; ADR-0019), so no post-hoc scalar.
        return W @ cl_vec

    def get_covariance(self, mode: str = "deconvolved") -> np.ndarray | None:
        """
        Get covariance matrix for power spectrum estimates in specified mode.

        Returns the covariance matrix appropriate for the normalization mode,
        which quantifies the statistical uncertainties and correlations between
        different multipole estimates.

        Parameters
        ----------
        mode : str, optional
            Normalization mode (default: "deconvolved"):

            - "deconvolved": Returns F⁻¹, the inverse Fisher matrix with
              correlated errors between multipoles.

            - "decorrelated": Returns identity matrix I, as decorrelated
              estimates have unit variance by construction.

            - "convolved": Returns F, the normalized Fisher matrix, which
              is the covariance of raw y estimates.

        Returns
        -------
        numpy.ndarray or None
            Covariance matrix of shape (n_params, n_params). Returns None for worker
            processes (rank != 0) or if matrices are not available.

        Raises
        ------
        ValueError
            If mode is not one of "deconvolved", "decorrelated", "convolved".

        Notes
        -----
        **Mode-specific covariances:**

        - **deconvolved** -- ``F⁻¹``, correlated errors on C_ℓ estimates.
        - **decorrelated** -- ``I``, unit variance (uncorrelated).
        - **convolved** -- ``F``, covariance of raw y estimates.

        The covariance matrix is essential for:

        - Computing chi-square statistics
        - Parameter estimation from power spectra
        - Constructing likelihood functions
        - Understanding multipole correlations

        Examples
        --------
        Get covariance for chi-square computation:

        >>> cov = spectra.get_covariance(mode="deconvolved")
        >>> cl_data = spectra.get_power_spectra(mode="deconvolved")
        >>> residuals = np.mean(cl_data, axis=0) - cl_theory
        >>> chi2 = residuals @ np.linalg.inv(cov) @ residuals

        Decorrelated mode has identity covariance:

        >>> cov_decorr = spectra.get_covariance(mode="decorrelated")
        >>> # cov_decorr is identity matrix

        See Also
        --------
        get_power_spectra : Get power spectrum estimates
        get_error_bars : Get 1-sigma error bars (sqrt of diagonal)
        """
        valid_modes = ("deconvolved", "decorrelated", "convolved")
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got '{mode}'")

        if self.rank != 0:
            return None

        if mode == "deconvolved":
            if self.invfisher is None:
                return None
            cov = self.invfisher.copy()
        elif mode == "decorrelated":
            if self.invfisher is None:
                return None
            n_params = self.invfisher.shape[0]
            cov = np.eye(n_params)
        else:  # convolved
            if self.fisher_normalized is None:
                return None
            cov = self.fisher_normalized.copy()

        # F_b (hence F_b⁻¹ and the normalised Fisher) is built from the
        # shape-weighted derivative in Dl mode, so the covariance is already in
        # the estimated quantity's own space (ADR-0019) — no post-hoc scalar.
        return cov

    def get_error_bars(self, mode: str = "deconvolved") -> np.ndarray | None:
        """
        Get 1-sigma error bars for power spectrum estimates.

        Returns the marginal standard deviations for each multipole estimate,
        computed from the diagonal of the covariance matrix.

        Parameters
        ----------
        mode : str, optional
            Normalization mode (default: "deconvolved"):

            - "deconvolved": sqrt(diag(F⁻¹)) - standard QML errors
            - "decorrelated": all ones (unit variance by construction)
            - "convolved": sqrt(diag(F)) - errors on raw y estimates

        Returns
        -------
        numpy.ndarray or None
            1D array of error bars with shape (n_params,). Returns None for
            worker processes (rank != 0) or if covariance not available.

        Raises
        ------
        ValueError
            If mode is not one of "deconvolved", "decorrelated", "convolved".

        Notes
        -----
        The error bars are computed as:
        σ_i = sqrt(Cov_ii)

        where Cov is the mode-appropriate covariance matrix. These represent
        marginal uncertainties on individual estimates, marginalized over all
        other multipoles.

        **Important:** For deconvolved and convolved modes, errors are
        correlated between multipoles. The full covariance matrix (from
        get_covariance) should be used for proper statistical analysis.

        For decorrelated mode, error bars are all 1.0 by construction,
        making them suitable for simple error bar plots.

        Examples
        --------
        Get error bars for plotting:

        >>> errors = spectra.get_error_bars(mode="deconvolved")
        >>> cl = np.mean(spectra.get_power_spectra(), axis=0)
        >>> plt.errorbar(ell, cl, yerr=errors)

        Decorrelated mode has unit errors:

        >>> errors_decorr = spectra.get_error_bars(mode="decorrelated")
        >>> # All elements are 1.0

        See Also
        --------
        get_covariance : Get full covariance matrix
        get_power_spectra : Get power spectrum estimates
        """
        cov = self.get_covariance(mode)
        if cov is None:
            return None
        return np.sqrt(np.diag(cov))

    def convolve_theory(self, cl_theory: np.ndarray) -> np.ndarray | None:
        """
        Apply window matrix to theoretical power spectrum.

        Convolves a theoretical power spectrum with the QML window matrix,
        producing the expected value of the raw QML estimates y for that
        theory: <y> = W @ C_theory.

        Parameters
        ----------
        cl_theory : numpy.ndarray
            Theoretical power spectrum values. Should be a 1D array with
            shape (n_params,) matching the QML output dimensions. When
            ``output_convention="Dl"``, this should be D_ℓ values.

        Returns
        -------
        numpy.ndarray or None
            Window-convolved theoretical spectrum with shape (n_params,),
            in the same convention as the input. Returns None if window
            matrix is not available.

        Notes
        -----
        This method is useful for comparing QML estimates with theoretical
        predictions when using the "convolved" normalization mode. Instead
        of deconvolving the window function from the data (which can be
        numerically unstable), the window is applied to the theory.

        The relationship is:
        <y> = W @ C_true

        where y are the raw QML estimates and W is the window matrix.
        When ``output_convention="Dl"``, the window is built from the
        shape-weighted derivative (ADR-0019), so it already maps D_ℓ theory to
        the D_ℓ estimate — pass D_ℓ input; there is no separate transform.

        Examples
        --------
        Compare convolved estimates with theory:

        >>> y, W, _ = spectra.get_power_spectra(mode="convolved")
        >>> cl_theory_convolved = spectra.convolve_theory(cl_theory)
        >>> residuals = np.mean(y, axis=0) - cl_theory_convolved

        See Also
        --------
        get_power_spectra : Get power spectra (convolved mode returns window)
        """
        if self.rank != 0:
            return None

        W = self.fisher_instance.get_window_matrix()
        if W is None:
            return None

        # In Dl mode the window is built from the shape-weighted derivative, so
        # it already maps D_ℓ theory to <y> in D-space (ADR-0019) — no scalar.
        return W @ cl_theory

    def write_power_spectra(
        self,
        mode: str = "deconvolved",
        filename: str | None = None,
        include_errors: bool = True,
    ) -> None:
        """
        Write power spectra to file in specified normalization mode.

        Outputs power spectrum estimates and optionally error bars to text
        files. For convolved mode, also writes the window matrix.

        Parameters
        ----------
        mode : str, optional
            Normalization mode (default: "deconvolved"):
            "deconvolved", "decorrelated", or "convolved".
        filename : str or None, optional
            Output filename. If None, auto-generates based on mode:
            "{outclfile_base}_{mode}.{ext}"
        include_errors : bool, optional
            If True, writes error bars to a separate file (default: True).
            For convolved mode, this parameter is ignored.

        Notes
        -----
        **File outputs by mode:**

        - deconvolved/decorrelated: Writes mean spectrum across simulations
          and optionally error bars from diagonal of covariance.

        - convolved: Writes mean raw y estimates and the window matrix W
          to separate files (filename and filename_window).

        Files are only written on the master process (rank 0).

        Examples
        --------
        Write all modes:

        >>> spectra.write_power_spectra(mode="deconvolved")
        >>> spectra.write_power_spectra(mode="decorrelated")
        >>> spectra.write_power_spectra(mode="convolved")

        Custom filename:

        >>> spectra.write_power_spectra(
        ...     mode="deconvolved",
        ...     filename="my_results.dat"
        ... )

        See Also
        --------
        get_power_spectra : Get power spectrum estimates
        get_error_bars : Get 1-sigma error bars
        """
        if self.rank != 0:
            return

        valid_modes = ("deconvolved", "decorrelated", "convolved")
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got '{mode}'")

        # Generate default filename
        if filename is None:
            if hasattr(self.params, "outclfile") and self.params.outclfile:
                parts = self.params.outclfile.rsplit(".", 1)
                base = parts[0]
                ext = parts[1] if len(parts) > 1 else "dat"
                filename = f"{base}_{mode}.{ext}"
            else:
                filename = f"spectra_{mode}.dat"

        if mode == "convolved":
            self._write_convolved(filename)
        else:
            spectra = self.get_power_spectra(mode=mode)
            if spectra is None:
                self.log(f"Cannot write {mode} spectra: not available", level=1)
                return

            mean_spectra = np.mean(spectra, axis=0)

            # Convert to Cl format and write
            n_ell = self.params.lmax - self.params.lmin + 1
            nspectra = len(mean_spectra) // n_ell
            cl_array = np.zeros((n_ell, nspectra), dtype=np.float64)
            vec_to_cl(mean_spectra, cl_array)
            writecl(filename, cl_array)
            self.log(f"Wrote {mode} power spectra to {filename}", level=2)

            if include_errors:
                errors = self.get_error_bars(mode=mode)
                if errors is not None:
                    error_filename = filename.rsplit(".", 1)
                    error_filename = (
                        f"{error_filename[0]}_errors.{error_filename[1]}"
                        if len(error_filename) > 1
                        else f"{filename}_errors"
                    )
                    error_array = np.zeros((n_ell, nspectra), dtype=np.float64)
                    vec_to_cl(errors, error_array)
                    writecl(error_filename, error_array)
                    self.log(f"Wrote {mode} error bars to {error_filename}", level=2)

    def _write_convolved(self, filename: str) -> None:
        """
        Write convolved mode outputs: y vector and window matrix.

        Parameters
        ----------
        filename : str
            Base filename for outputs.
        """
        result = self.get_power_spectra(mode="convolved")
        if result is None:
            self.log("Cannot write convolved spectra: not available", level=1)
            return

        y, W, _ = result
        mean_y = np.mean(y, axis=0)

        # Write y vector
        np.savetxt(
            filename,
            mean_y,
            header="Raw QML estimates (y vector, mean across simulations)",
        )
        self.log(f"Wrote convolved y vector to {filename}", level=2)

        # Write window matrix
        parts = filename.rsplit(".", 1)
        window_filename = (
            f"{parts[0]}_window.{parts[1]}" if len(parts) > 1 else f"{filename}_window"
        )
        np.savetxt(
            window_filename,
            W,
            header="Window matrix W: <y> = W @ C_true",
        )
        self.log(f"Wrote window matrix to {window_filename}", level=2)
