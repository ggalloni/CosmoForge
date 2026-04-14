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

import time
import typing

import numpy as np
from mpi4py import MPI

from cosmocore import (
    BaseCompression,
    Bins,
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
from qube import Fisher


class Spectra(Core):
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
        Inverted Fisher matrix used for final spectrum normalization.
    invCov1, invCov2 : numpy.ndarray
        Inverted covariance matrices for primary and secondary datasets.
    normalization : numpy.ndarray
        Normalization factors (vecmul) for spectrum smoothing.

    Examples
    --------
    Basic QML power spectrum estimation:

    >>> from cosmoforge.qube import Spectra
    >>> spectra = Spectra("config/qml_analysis.yaml")
    >>> spectra.run()
    >>> power_spectra = spectra.get_power_spectra()
    >>> noise_bias = spectra.get_noise_bias()

    Using pre-computed Fisher matrix:

    >>> from cosmoforge.qube import Fisher, Spectra
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
            Path to YAML configuration file.
        fisher : Fisher, optional
            Pre-computed Fisher instance. If provided, reuses computed components
            (covariance matrices, geometry, field collections) for efficiency.
        compression : dict, optional
            Compression configuration (method, epsilon, basis, mode_fraction).
        **kwargs : dict
            Additional arguments passed to Core.

        Raises
        ------
        TypeError
            If fisher is not a Fisher instance.
        ValueError
            If fisher doesn't contain a valid Fisher matrix.
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

        # Normalization mode support
        self.inv_fisher_sqrt: np.ndarray | None = None  # F^(-1/2) for decorrelated mode
        self.fisher_normalized: np.ndarray | None = (
            None  # Normalized Fisher (for convolved covariance)
        )

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
        """Copy computational components from a pre-computed Fisher instance."""
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

        # Copy binning and vecmul if available
        if (
            hasattr(self.fisher_instance, "bins")
            and self.fisher_instance.bins is not None
        ):
            self.bins = self.fisher_instance.bins
        if (
            hasattr(self.fisher_instance, "vecmul_per_ell")
            and self.fisher_instance.vecmul_per_ell is not None
        ):
            self.vecmul_per_ell = self.fisher_instance.vecmul_per_ell

        # Copy compression manager if available
        if (
            hasattr(self.fisher_instance, "compression_manager")
            and self.fisher_instance.compression_manager is not None
        ):
            self.compression_manager: BaseCompression = (
                self.fisher_instance.compression_manager
            )
        else:
            self.compression_manager = None

        # Load covariance matrices
        self._load_covariance_matrices()

    def _load_covariance_matrices(self):
        """Load noise and inverted covariance matrices from disk files."""
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
        Compute Fisher information matrix for QML normalization.

        Returns
        -------
        Fisher
            Completed Fisher instance with all components computed.
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

    def setup_fisher_inversion(self):
        """
        Prepare inverted Fisher matrix with normalization for QML estimation.

        Computes smoothing factors (vecmul), applies normalization
        F'_ij = F_ij * vecmul_i * vecmul_j,
        inverts the Fisher matrix, and writes results to output files.

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

            # Vecmul is already absorbed into the derivative matrices
            # during Fisher/QML computation, so normalization is identity.
            nbins = self.bins.nbins
            nell = self.params.nspectra * nbins
            self.normalization = np.ones(nell, dtype=np.float64)

            # Store normalized Fisher matrix for convolved mode covariance
            self.fisher_normalized = self.invfisher.copy()

            # Compute F^(-1/2) for decorrelated mode
            self._compute_inv_fisher_sqrt(self.invfisher)

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
            Normalized Fisher information matrix of shape (nell, nell).
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
        eigenvalues, eigenvectors = np.linalg.eigh(fisher)

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
        nell = self.params.nspectra * self.bins.nbins

        # Initialize y vectors for QML estimation
        self.qml_results = np.zeros((self.params.nsims, nell), dtype=np.float64)

        if not self.params.do_cross:
            self.qml_noise_bias = np.zeros(nell, dtype=np.float64)

    def compute_qml_spectra(self):
        """
        Execute parallel QML power spectrum computation.

        Distributes multipole computations across MPI processes (round-robin).
        For each multipole: computes signal derivative, E-operator, and
        quadratic estimates for all simulations. Selects compressed or
        traditional method based on compression_manager availability.
        """
        # Check if we should use compressed computation
        use_compression = (
            hasattr(self, "compression_manager") and self.compression_manager is not None
        )

        if use_compression:
            self._compute_qml_spectra_compressed()
        else:
            self._compute_qml_spectra_traditional()

    def _build_multi_spectrum_inputs(self):
        """Build C_ell_dict and spectra_list for multi-spectrum."""
        return self.collection.spectra_manager.build_inputs()

    def _get_binned_derivative(
        self, bin_idx: int, spectrum_idx: int = 0, spectra_list=None
    ) -> np.ndarray:
        """Get vecmul-weighted binned derivative for QML computation.

        Handles pixel-space (do_derivative_step) and compressed
        (cm.get_derivative_matrix) paths, single and multi-spectrum.
        Vecmul is absorbed into the binning weights.
        """
        use_compression = (
            hasattr(self, "compression_manager") and self.compression_manager is not None
        )

        P, _ = self.bins._bin_operators()
        lmin_b = self.bins.lmins[bin_idx]
        lmax_b = self.bins.lmaxs[bin_idx]
        n_ell = self.params.lmax - 1
        vm_offset = spectrum_idx * n_ell
        dC_b = None

        for ell in range(lmin_b, lmax_b + 1):
            w = P[bin_idx, ell] * self.vecmul_per_ell[vm_offset + ell - 2]

            if use_compression:
                cm = self.compression_manager
                if spectra_list is not None:
                    comp_i, comp_j, mode = spectra_list[spectrum_idx]
                    dC_ell = cm.get_derivative_matrix(ell, comp_i, comp_j, mode)
                else:
                    dC_ell = cm.get_derivative_matrix(ell)
            else:
                ntot = sum(self.collection.n_active)
                dC_ell = np.zeros((ntot, ntot), dtype=np.float64)
                do_derivative_step(
                    dC_ell,
                    spectrum_idx,
                    self.npixs,
                    self.params.spins,
                    ell,
                    self.collection,
                )

            if dC_b is None:
                dC_b = w * dC_ell
            else:
                dC_b += w * dC_ell

        return dC_b

    def _compute_noise_cov_diag_compressed(
        self, cm, C_ell, C_ell_dict, is_multi_field
    ) -> np.ndarray:
        """
        Compute diagonal of noise covariance in compressed space.

        For harmonic compression:
            Cov(w|noise) = V @ C^{-1} @ N @ C^{-1} @ V^T

        Returns the diagonal for efficient trace computation.
        """
        from scipy.linalg import cho_solve, cholesky

        from cosmocore.basics import matrix_inverse_symm

        if is_multi_field:
            # Build full Lambda and its inverse (auto-detects key format)
            Lambda_full = cm._build_lambda_full(C_ell_dict)
            Lambda_reg = Lambda_full + np.eye(Lambda_full.shape[0]) * 1e-20
            Lambda_inv = matrix_inverse_symm(np.asfortranarray(Lambda_reg))

            M = cm._V_Ninv_VT
            K = Lambda_inv + M
            K_inv = matrix_inverse_symm(np.asfortranarray(K))

            # V_Cinv = (I - M @ K^{-1}) @ V_Ninv
            n_modes = cm.n_modes_total
            I_minus_MKinv = np.eye(n_modes) - M @ K_inv
            V_Cinv = I_minus_MKinv @ cm._V_N_inv
        else:
            Lambda_diag = cm._build_lambda_diagonal(C_ell)
            Lambda_inv_diag = np.where(Lambda_diag > 1e-30, 1.0 / Lambda_diag, 1e30)
            M = cm._V_Ninv_VT
            K = np.diag(Lambda_inv_diag) + M

            try:
                L = cholesky(K, lower=True)
                K_inv = cho_solve((L, True), np.eye(K.shape[0]))
            except np.linalg.LinAlgError:
                K_inv = np.linalg.inv(K)

            # V_Cinv = (I - M @ K^{-1}) @ V_Ninv
            I_minus_MKinv = np.eye(cm.n_modes) - M @ K_inv
            V_Cinv = I_minus_MKinv @ cm._V_N_inv

        # For diagonal N, compute noise_cov_w_diag
        if hasattr(cm, "_N_inv_original"):
            noise_var = 1.0 / np.diag(cm._N_inv_original)
        else:
            noise_var = 1.0 / np.diag(cm.N_inv)
        sqrt_noise = np.sqrt(noise_var)
        W = V_Cinv * sqrt_noise[np.newaxis, :]

        # Diagonal of Cov(w|noise) = sum over columns of W^2
        return np.sum(W**2, axis=1)

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

        Supports both single-field and multi-field compression.
        """
        if self.rank == 0:
            self.log("Starting QML computation (compressed)", level=2)

        start_time = time.time()

        cm = self.compression_manager
        n_sims = self.params.nsims
        n_compressed = cm.n_kept

        # Multi-field path is needed when >1 components or spin-2 (spin-2 has
        # multiple spectra EE/BB/EB even for a single field)
        has_spin2 = any(f.spin == 2 for f in self.collection.fields)
        is_multi_field = cm.n_components > 1 or has_spin2

        # Build C_ell or C_ell_dict depending on multi-field
        if is_multi_field:
            C_ell_dict, spectra_list = self._build_multi_spectrum_inputs()
            C_ell = None  # Not used for multi-field
        else:
            C_ell = self.collection.spectra_manager.get_cls(0, 0, 0)
            C_ell_dict = None
            spectra_list = [(0, 0)]

        # Compute weighted compressed data for all simulations
        # w = V @ C^{-1} @ d (using SMW formula internally)
        # For multi-field, precompute K_inv once to avoid repeated matrix inversions
        maps1_weighted = np.zeros((n_compressed, n_sims), dtype=np.float64)
        if is_multi_field:
            if cm.method == "harmonic":
                # Precompute SMW matrices once
                from cosmocore.basics import matrix_inverse_symm

                Lambda_full = cm._build_lambda_full(C_ell_dict)
                Lambda_reg = Lambda_full + np.eye(Lambda_full.shape[0]) * 1e-20
                Lambda_inv = matrix_inverse_symm(np.asfortranarray(Lambda_reg))
                K = Lambda_inv + cm._V_Ninv_VT
                K_inv = matrix_inverse_symm(np.asfortranarray(K))
                M_K_inv = cm._V_Ninv_VT @ K_inv

                # Compute weighted data for all sims using precomputed matrices
                # w = y - M @ K^{-1} @ y where y = V @ N^{-1} @ d
                Y1 = cm._V_N_inv @ self.maps1  # (n_modes, n_sims)
                maps1_weighted = Y1 - M_K_inv @ Y1
            else:
                # pixel_projected: use compressed-space weighted data
                C_c_inv = cm.get_compressed_inverse(C_ell_dict)
                d_c = cm.compress_data(self.maps1)
                maps1_weighted = C_c_inv @ d_c
        else:
            C_c_inv = None
            if cm.method == "pixel_projected":
                C_c_inv = cm.get_compressed_inverse(C_ell)
            for isim in range(n_sims):
                maps1_weighted[:, isim] = cm.get_weighted_compressed_data(
                    self.maps1[:, isim], C_ell, C_c_inv=C_c_inv
                )

        if self.params.do_cross:
            maps2_weighted = np.zeros((n_compressed, n_sims), dtype=np.float64)
            if is_multi_field:
                if cm.method == "harmonic":
                    # Use precomputed matrices
                    Y2 = cm._V_N_inv @ self.maps2
                    maps2_weighted = Y2 - M_K_inv @ Y2
                else:
                    d_c2 = cm.compress_data(self.maps2)
                    maps2_weighted = C_c_inv @ d_c2
            else:
                for isim in range(n_sims):
                    maps2_weighted[:, isim] = cm.get_weighted_compressed_data(
                        self.maps2[:, isim], C_ell, C_c_inv=C_c_inv
                    )

        # For noise bias computation (only for non-cross)
        noise_cov_w_diag = None
        noise_cov_w = None
        if not self.params.do_cross:
            if cm.method == "harmonic":
                noise_cov_w_diag = self._compute_noise_cov_diag_compressed(
                    cm, C_ell, C_ell_dict, is_multi_field
                )
            else:
                # For pixel_projected: use compressed quantities
                if is_multi_field:
                    C_bar_inv = cm.get_compressed_inverse(C_ell_dict)
                    zero_dict = {k: np.zeros_like(v) for k, v in C_ell_dict.items()}
                    N_bar = cm.get_compressed_covariance(zero_dict)
                else:
                    C_bar_inv = cm.get_compressed_inverse(C_ell)
                    N_bar = cm.get_compressed_covariance(np.zeros_like(C_ell))
                noise_cov_w = C_bar_inv @ N_bar @ C_bar_inv

        # Main computation loop - distribute bins across processes
        nbins = self.bins.nbins
        nell = self.params.nspectra * nbins
        for il in range(nell):
            if self.rank == il % self.size:
                spectrum_idx = il // nbins
                bin_idx = il % nbins

                # Get binned compressed derivative matrix
                E_b = self._get_binned_derivative(
                    bin_idx,
                    spectrum_idx,
                    spectra_list if is_multi_field else None,
                )

                if self.params.do_cross:
                    for isim in range(n_sims):
                        w1 = maps1_weighted[:, isim]
                        w2 = maps2_weighted[:, isim]
                        self.qml_results[isim, il] = 0.5 * w2 @ E_b @ w1
                else:
                    # Noise bias: 0.5 * Tr[E_b @ Cov(w|noise)]
                    if cm.method == "harmonic" and noise_cov_w_diag is not None:
                        E_b_diag = np.diag(E_b)
                        tr_ne = 0.5 * np.sum(E_b_diag * noise_cov_w_diag)
                    else:
                        tr_ne = 0.5 * matrix_trace(E_b, noise_cov_w)
                    self.qml_noise_bias[il] = tr_ne

                    for isim in range(n_sims):
                        w = maps1_weighted[:, isim]
                        qml_value = 0.5 * w @ E_b @ w

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
            q_b = (1/2) * d^T @ C^{-1} @ dC_b @ C^{-1} @ d
                = (1/2) * y^T @ dC_b @ y   where y = C^{-1} @ d
        """
        if self.rank == 0:
            self.log("Starting QML computation (traditional, optimized)", level=2)

        start_time = time.time()

        nbins = self.bins.nbins
        nspectra = self.params.nspectra
        nell = nspectra * nbins

        # Precompute weighted data: y = C^{-1} @ d for all simulations
        y1 = matrix_mult(self.invCov1, self.maps1)

        if self.params.do_cross:
            y2 = matrix_mult(self.invCov2, self.maps2)

        # For noise bias: precompute C^{-1} @ N @ C^{-1}
        if not self.params.do_cross:
            Cinv_N_Cinv = matrix_mult(self.invCov1, matrix_mult(self.NCov1, self.invCov1))

        # Main computation loop - distribute bins across processes
        for il in range(nell):
            if self.rank == il % self.size:
                spectrum_idx = il // nbins
                bin_idx = il % nbins

                # Compute binned derivative matrix
                der_s = self._get_binned_derivative(bin_idx, spectrum_idx)

                # Compute dC @ y for all sims at once
                dC_y1 = matrix_mult(der_s, y1)

                if self.params.do_cross:
                    for isim in range(self.params.nsims):
                        self.qml_results[isim, il] = 0.5 * np.dot(
                            y2[:, isim], dC_y1[:, isim]
                        )
                else:
                    # Noise bias
                    tr_ne = 0.5 * matrix_trace(Cinv_N_Cinv, der_s)
                    self.qml_noise_bias[il] = tr_ne

                    for isim in range(self.params.nsims):
                        qml_value = 0.5 * np.dot(y1[:, isim], dC_y1[:, isim])

                        if hasattr(self.params, "remove_nb") and self.params.remove_nb:
                            qml_value -= tr_ne

                        self.qml_results[isim, il] = qml_value

        self.comm.Barrier()

        if self.rank == 0:
            self.log("QML computation done (traditional, optimized)", level=2)
            self.log(
                f"QML computation time: {time.time() - start_time:.2f} seconds", level=3
            )

        self._reduce_qml_results(nell)

    def _reduce_qml_results(self, nell: int):
        """Gather and combine QML results from all MPI processes."""
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
        Execute QML power spectrum computation.

        Calls setup_qml_computation() to initialize arrays, then
        compute_qml_spectra() for the main parallel computation.
        """
        # Setup QML computation variables
        self.setup_qml_computation()

        # Compute QML power spectra
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

            # Setup binning: Fisher > set_binning() > config > default
            if not hasattr(self, "bins") or self.bins is None:
                delta_ell = getattr(self.params, "delta_ell", 1)
                self.set_binning(Bins.fromdeltal(2, self.params.lmax, delta_ell))

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
        """Broadcast essential data from master to all MPI worker processes."""
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

        # Broadcast binning and vecmul
        self.bins = self.comm.bcast(self.bins if self.rank == 0 else None, root=0)
        self.vecmul_per_ell = self.comm.bcast(
            self.vecmul_per_ell if self.rank == 0 else None, root=0
        )

        # Broadcast normalization mode support matrices
        self.inv_fisher_sqrt = self.comm.bcast(
            self.inv_fisher_sqrt if self.rank == 0 else None, root=0
        )
        self.fisher_normalized = self.comm.bcast(
            self.fisher_normalized if self.rank == 0 else None, root=0
        )

    def _normalize_spectra(self, spectra: np.ndarray) -> np.ndarray:
        """
        Apply Fisher matrix normalization to raw QML estimates.

        Parameters
        ----------
        spectra : np.ndarray
            Raw QML estimates, shape (n_params,) or (n_sims, n_params).

        Returns
        -------
        np.ndarray
            Normalized power spectrum estimates with same shape as input.

        Raises
        ------
        ValueError
            If Fisher inversion or normalization factors not available.
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

    def get_power_spectra(
        self, mode: str = "deconvolved"
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, typing.Callable] | None:
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

        Returns
        -------
        numpy.ndarray or tuple or None
            For "deconvolved" or "decorrelated" modes:
                Array of shape (n_simulations, n_parameters) containing
                normalized power spectrum estimates.

            For "convolved" mode:
                Tuple of (y, W, convolve_theory_func) where:
                - y: Raw estimates array of shape (n_simulations, n_parameters)
                - W: Window matrix of shape (n_parameters, n_parameters)
                - convolve_theory_func: Callable that applies W @ theory

            Returns None for worker processes (rank != 0) or if computation
            has not completed.

        Raises
        ------
        ValueError
            If mode is not one of "deconvolved", "decorrelated", "convolved".

        Notes
        -----
        **Mode Comparison:**

        | Mode | Formula | Covariance | Use Case |
        |------|---------|------------|----------|
        | deconvolved | F⁻¹y | F⁻¹ (correlated) | Standard analysis |
        | decorrelated | F⁻¹/²y | I (identity) | Independent errors |
        | convolved | y | F | Theory comparison |

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
        all returned spectra are converted from C_ℓ to D_ℓ = ℓ(ℓ+1)/(2π) C_ℓ.
        In convolved mode, the window matrix is also transformed so that
        the returned ``convolve_theory_func`` expects D_ℓ input.

        Examples
        --------
        Default (deconvolved) mode - backwards compatible:

        >>> spectra = Spectra("config.yaml")
        >>> spectra.run()
        >>> cl_deconv = spectra.get_power_spectra()  # Default mode

        Decorrelated bandpowers:

        >>> cl_decorr = spectra.get_power_spectra(mode="decorrelated")
        >>> # Errors are all 1.0 by construction

        Convolved mode for theory comparison:

        >>> y, W, convolve = spectra.get_power_spectra(mode="convolved")
        >>> cl_theory_convolved = convolve(cl_theory)
        >>> # Compare: y should match cl_theory_convolved

        See Also
        --------
        get_covariance : Get covariance matrix for specified mode
        get_error_bars : Get 1-sigma error bars for specified mode
        convolve_theory : Apply window matrix to theoretical spectrum
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

        if self._output_is_dl() and result is not None:
            result = self._apply_output_convention(result, mode)
        return result

    def _output_is_dl(self) -> bool:
        """Check if output convention is Dl (case-insensitive)."""
        value = getattr(self.params, "output_convention", "Cl")
        key = value.strip().lower()
        if key not in ("cl", "dl"):
            raise ValueError(
                f"Unknown spectra convention '{value}'. Must be 'Cl' or 'Dl'."
            )
        return key == "dl"

    def _dl_factor(self) -> np.ndarray:
        """Return the Cl->Dl factor tiled over all spectra."""
        ell = self.bins.lbin.astype(np.float64)
        return np.tile(ell * (ell + 1) / (2 * np.pi), self.params.nspectra)

    def _apply_output_convention(self, result, mode):
        """Apply Cl->Dl conversion to output power spectra."""
        d = self._dl_factor()

        if mode in ("deconvolved", "decorrelated"):
            return result * d[np.newaxis, :]
        else:  # convolved
            y, W, convolve_cl = result
            # W_Dl = D @ W_Cl @ D^{-1} so that <y_Dl> = W_Dl @ C_Dl
            W_dl = W * np.outer(d, 1.0 / d)

            def convolve_theory_dl(cl_theory_dl: np.ndarray) -> np.ndarray:
                return W_dl @ cl_theory_dl

            return (y * d[np.newaxis, :], W_dl, convolve_theory_dl)

    def _get_deconvolved(self) -> np.ndarray:
        """
        Compute deconvolved power spectrum estimates (F⁻¹y).

        This is the standard QML output that multiplies raw estimates by the
        inverse Fisher matrix to recover estimates of the true C_ℓ.

        Returns
        -------
        numpy.ndarray
            Deconvolved power spectrum estimates with shape (nsims, nell).
        """
        return self._normalize_spectra(self.qml_results)

    def _get_decorrelated(self) -> np.ndarray:
        """
        Compute decorrelated bandpower estimates (F⁻¹/²y).

        Produces uncorrelated estimates with identity covariance matrix.

        Returns
        -------
        numpy.ndarray
            Decorrelated power spectrum estimates with shape (nsims, nell).

        Raises
        ------
        ValueError
            If F^(-1/2) was not computed (e.g., due to ill-conditioning).
        """
        if self.inv_fisher_sqrt is None:
            raise ValueError(
                "F^(-1/2) not computed. Check Fisher matrix conditioning or "
                "ensure setup_fisher_inversion() was called."
            )

        # Vectorized: broadcast normalization and apply matrix multiplication
        decorrelated = (self.qml_results * self.normalization) @ self.inv_fisher_sqrt

        return decorrelated

    def _get_convolved(self) -> tuple[np.ndarray, np.ndarray, typing.Callable]:
        """
        Get raw QML estimates with window matrix for theory comparison.

        Returns raw y estimates along with the window matrix W, allowing
        comparison with window-convolved theoretical spectra.

        Returns
        -------
        tuple
            (y, W, convolve_theory_func) where:
            - y: Raw normalized estimates, shape (nsims, nell)
            - W: Window matrix, shape (nell, nell)
            - convolve_theory_func: Callable to apply W @ theory
        """
        # Raw estimates multiplied by normalization
        y = self.qml_results * self.normalization

        # Window matrix from Fisher instance
        W = self.fisher_instance.get_window_matrix()
        if W is None:
            raise ValueError("Window matrix not available from Fisher instance.")

        # Apply normalization to window matrix to match y units
        W_normalized = W * np.outer(self.normalization, self.normalization)

        def convolve_theory(cl_theory: np.ndarray) -> np.ndarray:
            """Apply window matrix to theoretical power spectrum."""
            return W_normalized @ cl_theory

        return (y, W_normalized, convolve_theory)

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
            noise_bias = self._normalize_spectra(self.qml_noise_bias)
            if self._output_is_dl():
                noise_bias = noise_bias * self._dl_factor()
            return noise_bias
        return None

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
            Covariance matrix of shape (nell, nell). Returns None for worker
            processes (rank != 0) or if matrices are not available.

        Raises
        ------
        ValueError
            If mode is not one of "deconvolved", "decorrelated", "convolved".

        Notes
        -----
        **Mode-specific covariances:**

        | Mode | Covariance | Interpretation |
        |------|------------|----------------|
        | deconvolved | F⁻¹ | Correlated errors on C_ℓ estimates |
        | decorrelated | I | Unit variance (uncorrelated) |
        | convolved | F | Covariance of raw y estimates |

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
            nell = self.invfisher.shape[0]
            cov = np.eye(nell)
        else:  # convolved
            if self.fisher_normalized is None:
                return None
            cov = self.fisher_normalized.copy()

        if self._output_is_dl():
            d = self._dl_factor()
            cov = cov * np.outer(d, d)
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
            1D array of error bars with shape (nell,). Returns None for
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
            shape (nell,) matching the QML output dimensions. When
            ``output_convention="Dl"``, this should be D_ℓ values.

        Returns
        -------
        numpy.ndarray or None
            Window-convolved theoretical spectrum with shape (nell,),
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
        When ``output_convention="Dl"``, the window matrix is internally
        transformed so the input and output are both in D_ℓ convention.

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

        # Apply normalization to window matrix
        W_normalized = W * np.outer(self.normalization, self.normalization)

        if self._output_is_dl():
            d = self._dl_factor()
            # W_Dl = D @ W_Cl @ D^{-1}, input theory is Dl so convert first
            return d * (W_normalized @ (cl_theory / d))

        return W_normalized @ cl_theory

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
            n_ell = self.params.lmax - 1
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
