from abc import ABC, abstractmethod

import healpy as hp
import numpy as np

from .fields import (
    BaseField,
    FieldCollection,
    create_field,
)
from .in_out import output_geometry, read_covmat, read_mask, write_covmat_reduced
from .pixel import compute_pointings
from .settings import InputParams


class Core(ABC):
    """
    Abstract base class for cosmological analysis tools.

    This class provides common functionality for Fisher matrix calculations,
    power spectrum analysis, and other cosmological computations.
    """

    def __init__(
        self,
        params: InputParams | str | dict | None = None,
    ):
        self.read_params(params)

    def read_params(self, params: InputParams | str | dict):
        if isinstance(params, InputParams):
            self.params = params
        elif isinstance(params, str):
            self.params = InputParams.read_parameter_file(params)
        elif isinstance(params, dict):
            self.params = InputParams()
            self.params.update(params)
        else:
            msg = (
                "params must be an instance of InputParams, "
                "a string with the path to a parameter file, "
                "or a dictionary with parameters."
            )
            raise TypeError(msg)

    def setup_fields(self) -> FieldCollection:
        """
        Set up fields using the new design with clean architecture.

        Returns:
        --------
        LogicalFieldCollection
            Collection of fields using the new design
        """
        npix = hp.nside2npix(self.params.nside)
        mask = np.empty((self.params.nfields, npix), dtype=np.float64)
        mask = read_mask(self.params.maskfile, mask)

        if len(mask.shape) == 1:
            mask = mask[:, np.newaxis]

        # Create fields using the new factory function
        fields: list[BaseField] = []
        counter = 0
        for spin in self.params.spins:
            if spin == 0:
                labels = self.params.labels[counter]
                counter += 1
            else:
                labels = [self.params.labels[counter], self.params.labels[counter + 1]]
                counter += 2

            # Use new factory function for type-safe field creation
            field = create_field(
                spin=spin,
                nside=self.params.nside,
                lmax=self.params.lmax,
                mask=mask[:, counter - 1],
                labels=labels,
            )
            fields.append(field)

        # Create collection using new design
        self.collection = FieldCollection(self.params, fields)

        return self.collection

    def setup_geometry(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Set up geometry including active pixels and pointing vectors.

        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Active pixels and pointing vectors
        """
        if self.collection is None:
            raise ValueError("Fields must be set up before geometry")

        # Get active pixels
        self.npixs = []
        for lf in self.collection.fields:
            self.npixs += lf.n_active

        self.point_vectors = tuple(
            np.empty((self.npixs[i], 3), dtype=np.float64) for i in range(len(self.npixs))
        )

        self.pixact = self.collection.get_active_pixels()

        # Compute pointing vectors
        self.point_vectors = compute_pointings(
            self.params.nside,
            self.npixs,
            self.point_vectors,
            self.pixact,
            self.params.ordering,
        )

        # Set pointing vectors in collection
        self.collection.set_pointing_vectors(self.point_vectors)

        # Output geometry if requested
        if hasattr(self.params, "output_geometry_file"):
            output_geometry(
                self.params.output_geometry_file,
                self.npixs,
                self.point_vectors,
                self.pixact,
            )

        return self.pixact, self.point_vectors

    def setup_covariance_matrices(self) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Set up noise covariance matrices.

        Returns:
        --------
        Tuple[np.ndarray, Optional[np.ndarray]]
            Primary and optionally secondary covariance matrices
        """
        if self.pixact is None:
            raise ValueError("Geometry must be set up before covariance matrices")

        npix = hp.nside2npix(self.params.nside)
        concatenate_pixact = np.concatenate(
            [self.pixact[i] + i * npix for i in range(len(self.pixact))]
        )

        shape = (concatenate_pixact.shape[0], concatenate_pixact.shape[0])
        self.NCov1 = np.empty(shape, dtype=np.float64)

        self.NCov1 = (
            read_covmat(
                self.params.covmatfile1,
                npix,
                self.params.nfields,
                concatenate_pixact,
                self.NCov1,
            )
            * self.params.calibration**2
        )

        if hasattr(self.params, "outnoisecovmat1"):
            write_covmat_reduced(
                self.params.outnoisecovmat1,
                self.NCov1,
            )

        self.NCov2 = None
        if self.params.do_cross:
            self.NCov2 = np.empty(shape, dtype=np.float64)
            self.NCov2 = (
                read_covmat(
                    self.params.covmatfile2,
                    npix,
                    self.params.nfields,
                    concatenate_pixact,
                    self.NCov2,
                )
                * self.params.calibration**2
            )

            if hasattr(self.params, "outnoisecovmat2"):
                write_covmat_reduced(
                    self.params.outnoisecovmat2,
                    self.NCov2,
                )

        return self.NCov1, self.NCov2

    def setup_cls(self):
        """
        Set up power spectra using the new field design.
        """
        if self.collection is None:
            raise ValueError("Fields must be set up before Cls and beams")
        self.collection.set_cls()

    def setup_beams(self):
        """
        Set up beam functions for each field using the new design.
        """
        if self.collection is None:
            raise ValueError("Fields must be set up before Cls and beams")
        self.collection.set_beams()

    def log(self, message: str, level: int = 1):
        """
        Log a message based on feedback level.

        Parameters:
        -----------
        message : str
            Message to log
        level : int
            Minimum feedback level required to display message
        """
        if hasattr(self.params, "feedback") and self.params.feedback >= level:
            print(message)

    @abstractmethod
    def compute(self):
        """
        Abstract method for performing the main computation.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def run(self):
        """
        Abstract method for running the full analysis pipeline.
        Must be implemented by subclasses.
        """
        pass
