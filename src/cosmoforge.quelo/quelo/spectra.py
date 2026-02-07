"""
Quadratic Maximum Likelihood (QML) power spectrum estimation for cosmological analysis.

This module implements the Spectra class for optimal power spectrum estimation from
CMB observations using the Quadratic Maximum Likelihood estimator. The QML method
provides unbiased, minimum-variance power spectrum estimates that properly account
for sky cuts, instrumental noise, and pixel correlations.

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

import numpy as np
from mpi4py import MPI

from cosmocore import (
    CompressionManager,
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
from quelo import Fisher


class Spectra(Core):
    """
    Quadratic Maximum Likelihood (QML) power spectrum estimator for CMB analysis.

    This class implements the QML method for optimal power spectrum estimation from
    cosmic microwave background observations. The QML estimator provides unbiased,
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

    >>> from cosmoforge.quelo import Spectra
    >>> spectra = Spectra("config/qml_analysis.yaml")
    >>> spectra.run()
    >>> power_spectra = spectra.get_power_spectra()
    >>> noise_bias = spectra.get_noise_bias()

    Using pre-computed Fisher matrix:

    >>> from cosmoforge.quelo import Fisher, Spectra
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

        # Copy compression manager if available
        if (
            hasattr(self.fisher_instance, "compression_manager")
            and self.fisher_instance.compression_manager is not None
        ):
            self.compression_manager: CompressionManager = (
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

            # Compute vecmul (smoothing factors) - this is critical!
            self.log("Computing smoothing factors and vecmul", level=2)
            smoothing_factors = self.collection.spectra_manager.compute_smoothing_factors(
                self.collection.beam_manager
            )

            # Create vecmul array
            nell = self.params.nspectra * (self.params.lmax - 1)
            self.normalization = np.zeros(nell, dtype=np.float64)

            # Fill vecmul array
            idx = 0
            for _, spectrum_label in enumerate(self.collection.spectra_manager.labels):
                if spectrum_label in smoothing_factors:
                    smooth_factor = smoothing_factors[spectrum_label]
                    for ell_idx in range(self.params.lmax - 1):
                        self.normalization[idx] = smooth_factor[ell_idx]
                        idx += 1
                else:
                    raise ValueError(f"No smoothing factors found for {spectrum_label}")

            # Apply vecmul normalization to Fisher matrix
            self.log("Applying vecmul normalization to Fisher matrix", level=2)
            self.invfisher = self.invfisher * np.outer(
                self.normalization, self.normalization
            )

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
            n_ell = self.params.lmax - 1
            nspectra = len(vec_error_bars) // n_ell
            error_bars = np.zeros((n_ell, nspectra), dtype=np.float64)
            vec_to_cl(vec_error_bars, error_bars)
            writecl(self.params.outerrfile, error_bars)

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
        nell = self.params.nspectra * (self.params.lmax - 1)

        # Initialize y vectors for QML estimation
        self.qml_results = np.zeros((self.params.nsims, nell), dtype=np.float64)

        if not self.params.do_cross:
            self.qml_noise_bias = np.zeros(nell, dtype=np.float64)

    def compute_e_operator(self, il: int, der_s: np.ndarray) -> np.ndarray:
        """
        Compute QML quadratic estimator matrix E_l.

        Parameters
        ----------
        il : int
            Linear multipole index: spectrum_idx = il // (lmax-1), l = (il % (lmax-1)) + 2
        der_s : np.ndarray
            Signal covariance derivative ∂S/∂C_l, shape (n_pix, n_pix).

        Returns
        -------
        np.ndarray
            E_l matrix for quadratic estimation: q̂_l = (1/2) * x^T * E_l * x

        Notes
        -----
        Auto: E_l = (1/2) * C^{-1} * ∂S/∂C_l * C^{-1}
        Cross: E_l = (1/2) * C₂^{-1} * ∂S/∂C_l * C₁^{-1}
        """
        if self.params.do_cross:
            # E = 0.5 * invCov2^{-1} * derS * invCov1^{-1}
            E = 0.5 * matrix_mult(self.invCov2, matrix_mult(der_s, self.invCov1))
        else:
            # E = 0.5 * invCov1^{-1} * derS * invCov1^{-1}
            E = 0.5 * matrix_mult(self.invCov1, matrix_mult(der_s, self.invCov1))

        return E

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

    def _build_multi_spectrum_inputs_spectra(
        self,
    ) -> tuple[dict[tuple, np.ndarray], list[tuple]]:
        """
        Build C_ell_dict and spectra_list for multi-spectrum compressed QML.

        Always uses 3-tuple keys (field_i, field_j, mode).

        Returns
        -------
        C_ell_dict : dict
            Dictionary mapping (field_i, field_j, mode) to C_ell arrays.
        spectra_list : list
            List of 3-tuples in same order as spectra_labels.
        """
        sm = self.collection.spectra_manager

        C_ell_dict = {}
        spectra_list = []

        for fi, fj, mode in sm._spectra_map:
            C_ell_dict[(fi, fj, mode)] = sm.get_cls(fi, fj, mode)
            spectra_list.append((fi, fj, mode))

        return C_ell_dict, spectra_list

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

        impl = cm._impl

        if is_multi_field:
            # Build full Lambda and its inverse (auto-detects key format)
            Lambda_full = impl._build_lambda_full(C_ell_dict)
            Lambda_reg = Lambda_full + np.eye(Lambda_full.shape[0]) * 1e-20
            Lambda_inv = matrix_inverse_symm(np.asfortranarray(Lambda_reg))

            M = impl._V_Ninv_VT
            K = Lambda_inv + M
            K_inv = matrix_inverse_symm(np.asfortranarray(K))

            # V_Cinv = (I - M @ K^{-1}) @ V_Ninv
            n_modes = impl.n_modes_total
            I_minus_MKinv = np.eye(n_modes) - M @ K_inv
            V_Cinv = I_minus_MKinv @ impl._V_N_inv
        else:
            Lambda_diag = impl._build_lambda_diagonal(C_ell)
            Lambda_inv_diag = np.where(Lambda_diag > 1e-30, 1.0 / Lambda_diag, 1e30)
            M = impl._V_Ninv_VT
            K = np.diag(Lambda_inv_diag) + M

            try:
                L = cholesky(K, lower=True)
                K_inv = cho_solve((L, True), np.eye(K.shape[0]))
            except np.linalg.LinAlgError:
                K_inv = np.linalg.inv(K)

            # V_Cinv = (I - M @ K^{-1}) @ V_Ninv
            I_minus_MKinv = np.eye(cm.n_modes) - M @ K_inv
            V_Cinv = I_minus_MKinv @ impl._V_N_inv

        # For diagonal N, compute noise_cov_w_diag
        if hasattr(impl, "_N_inv_original"):
            noise_var = 1.0 / np.diag(impl._N_inv_original)
        else:
            noise_var = 1.0 / np.diag(impl.N_inv)
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

        n_ell = self.params.lmax - 1
        nell = self.params.nspectra * n_ell
        cm = self.compression_manager
        n_sims = self.params.nsims
        n_compressed = cm.n_kept

        # Multi-field path is needed when >1 components or spin-2 (spin-2 has
        # multiple spectra EE/BB/EB even for a single field)
        has_spin2 = any(f.spin == 2 for f in self.collection.fields)
        is_multi_field = cm.n_components > 1 or has_spin2

        # Build C_ell or C_ell_dict depending on multi-field
        if is_multi_field:
            C_ell_dict, spectra_list = self._build_multi_spectrum_inputs_spectra()
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

                impl = cm._impl
                Lambda_full = impl._build_lambda_full(C_ell_dict)
                Lambda_reg = Lambda_full + np.eye(Lambda_full.shape[0]) * 1e-20
                Lambda_inv = matrix_inverse_symm(np.asfortranarray(Lambda_reg))
                K = Lambda_inv + impl._V_Ninv_VT
                K_inv = matrix_inverse_symm(np.asfortranarray(K))
                M_K_inv = impl._V_Ninv_VT @ K_inv

                # Compute weighted data for all sims using precomputed matrices
                # w = y - M @ K^{-1} @ y where y = V @ N^{-1} @ d
                Y1 = impl._V_N_inv @ self.maps1  # (n_modes, n_sims)
                maps1_weighted = Y1 - M_K_inv @ Y1
            else:
                # pixel_projected: use compressed-space weighted data
                C_c_inv = cm.get_compressed_inverse_multi(C_ell_dict)
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
                    Y2 = impl._V_N_inv @ self.maps2
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
                    C_bar_inv = cm.get_compressed_inverse_multi(C_ell_dict)
                    zero_dict = {k: np.zeros_like(v) for k, v in C_ell_dict.items()}
                    N_bar = cm.get_compressed_covariance_multi(zero_dict)
                else:
                    C_bar_inv = cm.get_compressed_inverse(C_ell)
                    N_bar = cm.get_compressed_covariance(np.zeros_like(C_ell))
                noise_cov_w = C_bar_inv @ N_bar @ C_bar_inv

        # Main computation loop - distribute multipoles across processes
        for il in range(nell):
            if self.rank == il % self.size:
                spectrum_idx = il // n_ell
                ell = (il % n_ell) + 2

                # Get compressed derivative matrix E_l
                if is_multi_field:
                    comp_i, comp_j, mode = spectra_list[spectrum_idx]
                    E_l = cm._impl.get_derivative_matrix_multi(ell, comp_i, comp_j, mode)
                else:
                    E_l = cm.get_derivative_matrix(ell)

                if self.params.do_cross:
                    # Cross-correlation case
                    for isim in range(n_sims):
                        w1 = maps1_weighted[:, isim]
                        w2 = maps2_weighted[:, isim]
                        self.qml_results[isim, il] = 0.5 * w2 @ E_l @ w1
                else:
                    # Auto-correlation case
                    # Compute noise bias: E[q_l|noise] = 0.5 * Tr[E_l @ Cov(w|noise)]
                    if cm.method == "harmonic":
                        # For harmonic, E_l is diagonal - use fast diagonal trace
                        E_l_diag = np.diag(E_l)
                        tr_ne = 0.5 * np.sum(E_l_diag * noise_cov_w_diag)
                    else:
                        # For pixel_projected, E_l is full matrix - use matrix_trace
                        tr_ne = 0.5 * matrix_trace(E_l, noise_cov_w)
                    self.qml_noise_bias[il] = tr_ne

                    for isim in range(n_sims):
                        w = maps1_weighted[:, isim]
                        qml_value = 0.5 * w @ E_l @ w

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
            q_l = (1/2) * d^T @ C^{-1} @ dC_l @ C^{-1} @ d
                = (1/2) * y^T @ dC_l @ y   where y = C^{-1} @ d

        This reduces complexity from 2 × O(n³) to 1 × O(n³) per multipole,
        as we only need C^{-1} @ dC for noise bias (not the full E matrix).
        """
        if self.rank == 0:
            self.log("Starting QML computation (traditional, optimized)", level=2)

        start_time = time.time()

        nell = self.params.nspectra * (self.params.lmax - 1)
        ntot = sum(self.collection.n_active)

        # Precompute weighted data: y = C^{-1} @ d for all simulations
        # This is O(n² × nsims) and avoids rebuilding for each ℓ
        y1 = matrix_mult(self.invCov1, self.maps1)  # (ntot, nsims)

        if self.params.do_cross:
            y2 = matrix_mult(self.invCov2, self.maps2)  # (ntot, nsims)

        # For noise bias: Tr[N @ E] = 0.5 * Tr[N @ C^{-1} @ dC @ C^{-1}]
        # Using cyclic trace property: = 0.5 * Tr[C^{-1} @ N @ C^{-1} @ dC]
        # Precompute C^{-1} @ N @ C^{-1} once (O(n³)), then Tr(... @ dC) per ℓ
        if not self.params.do_cross:
            Cinv_N_Cinv = matrix_mult(self.invCov1, matrix_mult(self.NCov1, self.invCov1))

        # Allocate derivative matrix
        der_s = np.zeros((ntot, ntot), dtype=np.float64)

        # Main computation loop - distribute multipoles across processes
        for il in range(nell):
            if self.rank == il % self.size:
                spectrum_idx = il // (self.params.lmax - 1)
                ell = (il % (self.params.lmax - 1)) + 2

                # Compute derivative matrix dC_l
                der_s.fill(0.0)
                do_derivative_step(
                    der_s,
                    spectrum_idx,
                    self.npixs,
                    self.params.spins,
                    ell,
                    self.collection,
                )

                # Compute dC @ y for all sims at once: O(n² × nsims)
                dC_y1 = matrix_mult(der_s, y1)

                if self.params.do_cross:
                    # Cross-correlation: q_l = 0.5 * y2^T @ dC @ y1
                    for isim in range(self.params.nsims):
                        self.qml_results[isim, il] = 0.5 * np.dot(
                            y2[:, isim], dC_y1[:, isim]
                        )
                else:
                    # Auto-correlation case
                    # Noise bias: Tr[N @ E] = 0.5 * Tr[C^{-1} @ N @ C^{-1} @ dC]
                    # Using precomputed Cinv_N_Cinv: Tr(Cinv_N_Cinv @ dC)
                    tr_ne = 0.5 * matrix_trace(Cinv_N_Cinv, der_s)
                    self.qml_noise_bias[il] = tr_ne

                    # QML values: q_l = 0.5 * y^T @ dC @ y
                    for isim in range(self.params.nsims):
                        qml_value = 0.5 * np.dot(y1[:, isim], dC_y1[:, isim])

                        if hasattr(self.params, "remove_nb") and self.params.remove_nb:
                            qml_value -= tr_ne

                        self.qml_results[isim, il] = qml_value

        # Synchronize all processes
        self.comm.Barrier()

        if self.rank == 0:
            self.log("QML computation done (traditional, optimized)", level=2)
            self.log(
                f"QML computation time: {time.time() - start_time:.2f} seconds", level=3
            )

        # Reduce results from all processes
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

    def get_power_spectra(self) -> np.ndarray | None:
        """
        Retrieve final normalized QML power spectrum estimates.

        Returns
        -------
        np.ndarray or None
            Shape (n_sims, n_params) where n_params = n_spectra * (lmax-1).
            Parameters ordered by spectrum type, multipoles nested within.
            Returns None for worker processes or if computation incomplete.
        """
        if self.rank == 0 and self.qml_results is not None:
            power_spectra = self._normalize_spectra(self.qml_results)
            return power_spectra
        return None

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
            return noise_bias
        return None
