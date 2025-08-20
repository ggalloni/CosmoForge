"""
Fisher matrix computation class inheriting from Core.

This module contains the Fisher class which implements Fisher matrix
calculations for cosmological parameter estimation.
"""

import time

import numpy as np
from cosmocore import (
    compute_signal_matrix,
    do_derivative_step,
    matrix_inverse_symm,
    matrix_mult,
    matrix_trace,
    write_covmat_reduced,
    write_out_matrix,
)
from cosmocore.core import Core
from mpi4py import MPI


class Fisher(Core):
    """
    Fisher matrix computation class for cosmological parameter estimation.

    This class inherits from Core and implements the Fisher matrix calculation
    pipeline including signal matrix computation, derivative steps, and
    Fisher matrix assembly using MPI for parallel computation.
    """

    def __init__(self, params_file: str | None = None, **kwargs):
        """
        Initialize Fisher class.

        Parameters:
        -----------
        params_file : str, optional
            Path to parameter file
        **kwargs : dict
            Additional parameters for Core class
        """
        # Pass the params_file to Core class constructor
        super().__init__(params=params_file, **kwargs)

        # MPI setup
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Fisher-specific attributes
        self.fisher = None
        self.derSil = None
        self.derSjl = None
        self.n_ell = None
        self.nell = None

        # No need to call read_params again since Core.__init__ already does it

    def setup_signal_matrix(self) -> np.ndarray:
        """
        Set up and compute the signal covariance matrix.

        Returns:
        --------
        np.ndarray
            Signal covariance matrix
        """
        if self.NCov1 is None:
            raise ValueError("Covariance matrices must be set up first")

        self.Sig = np.zeros_like(self.NCov1, dtype=np.float64)
        self.Sig = np.asfortranarray(self.Sig, dtype=np.float64)

        start_time = time.time() if self.rank == 0 else None

        compute_signal_matrix(
            S=self.Sig,
            lmax=self.params.lmax,
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
        Prepare covariance matrices by adding signal and computing inverses.
        """
        if self.Sig is None:
            self.setup_signal_matrix()

        # Add signal to noise covariance
        self.NCov1 = self.NCov1 + self.Sig
        self.log(f"Combined covariance matrix shape: {self.NCov1.shape}", level=4)

        # Save combined covariance if not doing cross-correlation
        if not self.params.do_cross:
            write_covmat_reduced(self.params.outnoisecovmat1, self.NCov1)

        # Compute inverse covariance matrices
        self.NCov1 = matrix_inverse_symm(self.NCov1)
        self.log("Computed inverse of primary covariance matrix", level=4)

        # Write inverse covariance matrix
        write_covmat_reduced(self.params.outinvcovmatfile1, self.NCov1)

        if self.params.do_cross:
            self.NCov2 = np.asfortranarray(self.NCov2)
            self.NCov2 = matrix_inverse_symm(self.NCov2)
            write_covmat_reduced(self.params.outinvcovmatfile2, self.NCov2)
            self.log("Computed inverse of secondary covariance matrix", level=4)

    def setup_fisher_matrices(self):
        """Set up Fisher matrix and derivative matrices."""
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
        Compute a single Fisher matrix element.

        Parameters:
        -----------
        il, jl : int
            Matrix indices
        curr_ell_i, curr_ell_j : int
            Current multipole moments
        spectrum_i, spectrum_j : int
            Spectrum indices
        appil : int
            Previous il value for caching

        Returns:
        --------
        int
            Updated appil value
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
        Main computation method for Fisher matrix calculation.

        This method implements the parallel Fisher matrix computation using MPI.
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

            if start_time is not None:
                elapsed = time.time() - start_time
                self.log(f"Total computation time: {elapsed:.2f} seconds", level=3)

            # Write Fisher matrix to file
            write_out_matrix(self.params.outfilefisher, self.fisher)
            self.log(f"Fisher matrix written to {self.params.outfilefisher}", level=4)
            self.log(f"Fisher matrix shape: {self.fisher.shape}", level=4)

        self.comm.Barrier()

    def run(self):
        """
        Run the complete Fisher matrix analysis pipeline.

        This method orchestrates the entire analysis from parameter reading
        to final Fisher matrix computation and output.
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

            self.setup_cls()
            self.log("Power spectra setup completed", level=3)

            self.setup_beams()
            self.log("Beam functions setup completed", level=3)

            self.setup_signal_matrix()
            self.prepare_covariance_matrices()
            self.log("Signal matrix and covariance preparation completed", level=3)

        # Synchronize before broadcasting
        self.comm.Barrier()

        # Broadcast shared variables to all processes
        self.params = self.comm.bcast(self.params if self.rank == 0 else None, root=0)
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
        MPI.Finalize()

    def get_fisher_matrix(self) -> np.ndarray | None:
        """
        Get the computed Fisher matrix.

        Returns:
        --------
        np.ndarray or None
            Fisher matrix (only available on rank 0 after computation)
        """
        if self.rank == 0:
            return self.fisher
        return None

    def get_parameter_errors(self) -> np.ndarray | None:
        """
        Get parameter errors from Fisher matrix diagonal.

        Returns:
        --------
        np.ndarray or None
            Parameter errors (only available on rank 0 after computation)
        """
        if self.rank == 0 and self.fisher is not None:
            # Invert Fisher matrix to get covariance
            try:
                cov_matrix = np.linalg.inv(self.fisher)
                errors = np.sqrt(np.diag(cov_matrix))
                return errors
            except np.linalg.LinAlgError:
                self.log(
                    "Warning: Fisher matrix is singular, cannot compute errors", level=1
                )
                return None
        return None
