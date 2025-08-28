"""
QML power spectrum estimation class inheriting from Core.

This module contains the Spectra class which implements Quadratic Maximum Likelihood
power spectrum estimation for cosmological analysis.
"""

import time

import numpy as np
from mpi4py import MPI

from cosmocore import (
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
    QML power spectrum estimation class for cosmological analysis.

    This class inherits from Core and implements the Quadratic Maximum Likelihood
    estimation pipeline including map reading, E operator computation, and
    power spectrum estimation using MPI for parallel computation.
    """

    def __init__(
        self, params_file: str | None = None, fisher: Fisher | None = None, **kwargs
    ):
        """
        Initialize Spectra class.

        Parameters:
        -----------
        params_file : str, optional
            Path to parameter file
        fisher : Fisher, optional
            Precomputed Fisher instance
        **kwargs : dict
            Additional parameters for Core class
        """
        self.params: InputParams = None
        super().__init__(params=params_file, **kwargs)

        # MPI setup
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

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

        # Initialize QML-specific variables
        self.maps1 = None
        self.maps2 = None
        self.y = None
        self.ynb = None
        self.invfisher = None

    def _reuse_fisher_components(self):
        """
        Reuse components already computed by the Fisher instance.
        """
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

        # Load covariance matrices
        self._load_covariance_matrices()

    def _load_covariance_matrices(self):
        """
        Load covariance matrices from files.
        """
        ntot = sum(self.collection.n_active)

        # Load inverted covariance matrices
        self.invCov1 = np.fromfile(self.params.outinvcovmatfile1).reshape(ntot, ntot)
        if self.params.do_cross:
            self.invCov2 = np.fromfile(self.params.outinvcovmatfile2).reshape(ntot, ntot)

        # Load noise covariance matrices
        self.NCov1 = np.fromfile(self.params.covmatfile1).reshape(ntot, ntot)
        if self.params.do_cross:
            self.NCov2 = np.fromfile(self.params.covmatfile2).reshape(ntot, ntot)

    def _get_fisher(self) -> Fisher:
        """
        Run the Fisher matrix computation and return the Fisher instance.

        This method orchestrates the entire Fisher matrix computation process.
        """
        if self.rank == 0:
            self.log("Starting Fisher matrix computation...", level=1)

        start_time = time.time()

        fisher = Fisher(self.params)
        fisher.run()

        if self.rank == 0:
            elapsed = time.time() - start_time
            self.log(
                f"Fisher matrix computation completed in {elapsed:.2f} seconds", level=1
            )

        return fisher

    def setup_maps(self):
        """
        Read and setup map data for QML analysis.
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
        Read and invert the Fisher matrix for QML estimation.
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
            self.vecmul = np.zeros(nell, dtype=np.float64)

            # Fill vecmul array
            idx = 0
            for ispec, spectrum_label in enumerate(
                self.collection.spectra_manager.labels
            ):
                if spectrum_label in smoothing_factors:
                    smooth_factor = smoothing_factors[spectrum_label]
                    for ell_idx in range(self.params.lmax - 1):
                        self.vecmul[idx] = smooth_factor[ell_idx]
                        idx += 1
                else:
                    # Fill with ones if no smoothing factors available
                    for ell_idx in range(self.params.lmax - 1):
                        self.vecmul[idx] = 1.0
                        idx += 1

            # Apply vecmul normalization to Fisher matrix
            self.log("Applying vecmul normalization to Fisher matrix", level=2)
            self.invfisher = self.invfisher * np.outer(self.vecmul, self.vecmul)

            # Invert Fisher matrix
            self.log("Inverting normalized Fisher matrix", level=2)
            start_time = time.time()
            self.invfisher = matrix_inverse_symm(self.invfisher)

            if self.params.feedback > 3:
                elapsed = time.time() - start_time
                self.log(f"Fisher matrix inversion time: {elapsed:.2f} seconds", level=4)

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
        Setup variables for QML computation.
        """
        nell = self.params.nspectra * (self.params.lmax - 1)

        # Initialize y vectors for QML estimation
        self.y = np.zeros((self.params.nsims, nell), dtype=np.float64)

        if not self.params.do_cross:
            self.ynb = np.zeros(nell, dtype=np.float64)

    def compute_e_operator(self, il: int, der_s: np.ndarray) -> np.ndarray:
        """
        Compute the E operator for a given multipole.

        Parameters:
        -----------
        il : int
            Multipole index
        der_s : np.ndarray
            Derivative of signal matrix

        Returns:
        --------
        np.ndarray
            E operator matrix
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
        Main computation method for QML power spectrum estimation.

        This method implements the parallel QML computation using MPI.
        """
        if self.rank == 0:
            self.log("Starting QML computation", level=2)

        start_time = time.time()

        nell = self.params.nspectra * (self.params.lmax - 1)
        ntot = sum(self.collection.n_active)

        # Allocate derivative matrix
        der_s = np.zeros((ntot, ntot), dtype=np.float64)
        E = np.zeros((ntot, ntot), dtype=np.float64)

        # Main computation loop - distribute multipoles across processes
        for il in range(nell):
            if self.rank == il % self.size:
                # Compute derivative of signal matrix for this multipole
                spectrum_idx = il // (self.params.lmax - 1)
                ell = (il % (self.params.lmax - 1)) + 2

                # Compute derivative step
                do_derivative_step(
                    der_s,
                    spectrum_idx,
                    self.npixs,
                    self.params.spins,
                    ell,
                    self.collection,
                )

                # Compute E operator
                E = self.compute_e_operator(il, der_s)

                if self.params.do_cross:
                    # Cross-correlation case
                    for isim in range(self.params.nsims):
                        self.y[isim, il] = matrix_mult(
                            self.maps2[:, isim].T, matrix_mult(E, self.maps1[:, isim])
                        )
                else:
                    # Auto-correlation case
                    # Compute noise bias
                    tr_ne = matrix_trace(self.NCov1, E)
                    self.ynb[il] = tr_ne

                    for isim in range(self.params.nsims):
                        qml_value = matrix_mult(
                            self.maps1[:, isim].T, matrix_mult(E, self.maps1[:, isim])
                        )

                        if hasattr(self.params, "remove_nb") and self.params.remove_nb:
                            qml_value -= tr_ne

                        self.y[isim, il] = qml_value

        # Synchronize all processes
        self.comm.Barrier()

        if self.rank == 0:
            self.log("QML computation done", level=2)

        if self.params.feedback > 3 and self.rank == 0:
            elapsed = time.time() - start_time
            self.log(f"QML computation time: {elapsed:.2f} seconds", level=4)

        # Reduce results from all processes
        self._reduce_qml_results(nell)

    def _reduce_qml_results(self, nell: int):
        """
        Reduce QML results from all MPI processes.

        Parameters:
        -----------
        nell : int
            Number of multipole bins
        """
        # Reduce y vectors
        red_y = np.zeros((self.params.nsims, nell), dtype=np.float64)
        self.comm.Reduce(self.y, red_y, op=MPI.SUM, root=0)

        if not self.params.do_cross:
            # Reduce noise bias
            red_ynb = np.zeros(nell, dtype=np.float64)
            self.comm.Reduce(self.ynb, red_ynb, op=MPI.SUM, root=0)

        if self.rank == 0:
            self.y = red_y
            if not self.params.do_cross:
                self.ynb = red_ynb

    def _write_cl(self, filename: str, cl_array: np.ndarray):
        """
        Write Cl array to file.

        Parameters:
        -----------
        filename : str
            Output filename
        cl_array : np.ndarray
            Cl array to write
        """
        # Use simple numpy save for now
        # Could be enhanced to use a more sophisticated format
        np.savetxt(filename, cl_array)

    def compute(self):
        """
        Main computation method for QML power spectrum estimation.
        """
        # Setup QML computation variables
        self.setup_qml_computation()

        # Compute QML power spectra
        self.compute_qml_spectra()

    def run(self):
        """
        Run the complete QML power spectrum analysis pipeline.

        This method orchestrates the entire analysis from parameter reading
        to final power spectrum computation and output.
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
                self.setup_cls()
                self.setup_beams()
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
        """
        Broadcast variables from rank 0 to all other processes.
        """
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
        self.vecmul = self.comm.bcast(self.vecmul if self.rank == 0 else None, root=0)

    def get_power_spectra(self) -> np.ndarray | None:
        """
        Get the computed power spectra with final Fisher matrix multiplication.

        Returns:
        --------
        np.ndarray or None
            Power spectra (only available on rank 0 after computation)
        """
        if self.rank == 0 and self.y is not None:
            # Apply final Fisher matrix multiplication as in the notebook
            power_spectra = np.zeros_like(self.y)
            for field_idx in range(self.params.nsims):
                # Element-wise multiplication first, then matrix multiplication
                redy_times_vecmul = self.y[field_idx, :] * self.vecmul
                power_spectra[field_idx, :] = np.matmul(redy_times_vecmul, self.invfisher)
            return power_spectra
        return None

    def get_noise_bias(self) -> np.ndarray | None:
        """
        Get the computed noise bias.

        Returns:
        --------
        np.ndarray or None
            Noise bias (only available on rank 0 after computation, auto-correlation only)
        """
        if self.rank == 0 and self.ynb is not None:
            # Apply final Fisher matrix multiplication to noise bias
            redynb_times_vecmul = self.ynb * self.vecmul
            noise_bias = np.matmul(redynb_times_vecmul, self.invfisher)
            return noise_bias
        return None
