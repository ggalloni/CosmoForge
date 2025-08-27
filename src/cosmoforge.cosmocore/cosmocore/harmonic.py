from typing import TYPE_CHECKING

import numpy as np
from numba import njit


if TYPE_CHECKING:
    from cosmocore import BaseField, InputParams


@njit(cache=True)
def cl_to_vec(cl, vec):
    lmax = cl.shape[0] + 1
    n_spec = cl.shape[1]
    counter = 0
    for ispec in range(n_spec):
        for il in range(2, lmax):
            vec[counter] = cl[il - 2, ispec]
            counter += 1


@njit(cache=True)
def vec_to_cl(vec, cl):
    lmax = cl.shape[0] + 1
    n_spec = cl.shape[1]
    counter = 0
    for ispec in range(n_spec):
        for il in range(2, lmax):
            cl[il - 2, ispec] = vec[counter]
            counter += 1


def coswinbeam(nside):
    L = 4 * nside + 1
    beam = np.zeros(L, dtype=np.float64)
    # flat top
    beam[: nside + 1] = 1.0
    # cosine roll-off
    ell = np.arange(nside + 1, 3 * nside + 1)
    beam[nside + 1 : 3 * nside + 1] = 0.5 * (
        1.0 + np.cos((ell - nside) * np.pi / (2.0 * nside))
    )
    return beam


class SpectraManager:
    """Manages power spectra for a collection of fields."""

    def __init__(self, fields: list["BaseField"]):
        self.fields = fields
        self._spectra_labels = []
        self._spectra_map = {}
        self._cls_dict = {}
        self._cls_matrix = None

        self._build_spectra_structure()

    def _build_spectra_structure(self) -> None:
        """Build the mapping between field pairs and spectrum labels."""
        # Auto-spectra
        for i, field in enumerate(self.fields):
            labels = field.get_spectrum_labels()
            for mode, label in enumerate(labels):
                self._spectra_labels.append(label)
                self._spectra_map[(i, i, mode)] = label

        # Cross-spectra
        for i in range(len(self.fields)):
            for j in range(i + 1, len(self.fields)):
                labels = self.fields[i].get_cross_spectrum_labels(self.fields[j])
                for mode, label in enumerate(labels):
                    self._spectra_labels.append(label)
                    self._spectra_map[(i, j, mode)] = label

    @property
    def labels(self) -> list[str]:
        return self._spectra_labels.copy()

    @property
    def n_spectra(self) -> int:
        return len(self._spectra_labels)

    def get_spectrum_label(self, field_i: int, field_j: int, mode: int = 0) -> str:
        """Get spectrum label for given field pair and mode."""
        return self._spectra_map.get((field_i, field_j, mode))

    def read_cls_from_file(self, inputclfile: str, params) -> dict[str, np.ndarray]:
        """
        Read power spectra from file.

        Parameters:
        -----------
        inputclfile : str
            Path to power spectra file
        params : InputParams
            Parameter object containing configuration

        Returns:
        --------
        dict[str, np.ndarray]
            Dictionary mapping spectrum labels to power spectra arrays
        """
        from .in_out import readcl

        return readcl(inputclfile, params)

    def set_cls_from_file(self, inputclfile: str, params) -> None:
        """Set power spectra by reading from file."""
        cls_data = self.read_cls_from_file(inputclfile, params)
        self.set_cls(cls_data)

    def set_cls(self, cls_data: dict[str, np.ndarray] | np.ndarray) -> None:
        """Set power spectra from dictionary or matrix."""
        if isinstance(cls_data, dict):
            self._cls_dict = cls_data.copy()
            # Build matrix from dictionary
            lmax = self.fields[0].lmax
            self._cls_matrix = np.zeros((lmax - 1, self.n_spectra))
            for idx, label in enumerate(self._spectra_labels):
                if label not in cls_data:
                    raise ValueError(f"Missing power spectrum for {label}")
                self._cls_matrix[:, idx] = cls_data[label][: lmax - 1]

        elif isinstance(cls_data, np.ndarray):
            if cls_data.shape[1] != self.n_spectra:
                raise ValueError(
                    f"Expected {self.n_spectra} spectra columns, got {cls_data.shape[1]}"
                )
            lmax = self.fields[0].lmax
            self._cls_matrix = cls_data[: lmax - 1].copy()
            # Build dictionary from matrix
            self._cls_dict = {
                label: self._cls_matrix[:, idx]
                for idx, label in enumerate(self._spectra_labels)
            }

    def get_cls(self, field_i: int, field_j: int, mode: int = 0) -> np.ndarray:
        """Get power spectrum for field pair and mode."""
        label = self.get_spectrum_label(field_i, field_j, mode)
        if label not in self._cls_dict:
            raise ValueError(f"No power spectrum found for {label}")
        return self._cls_dict[label]

    def apply_normalization(self) -> None:
        """Apply normalization factors based on spin combinations."""
        # Use precomputed normalization factors
        normalization_factors = self.compute_normalization_factors()

        for idx, label in enumerate(self._spectra_labels):
            if label in normalization_factors:
                # Apply normalization factor
                self._cls_matrix[:, idx] *= normalization_factors[label]
                # Update dictionary
                self._cls_dict[label] = self._cls_matrix[:, idx]

    def compute_normalization_factors(self) -> dict[str, np.ndarray]:
        """
        Compute normalization factors for all spectrum labels.

        Returns:
        --------
        dict[str, np.ndarray]
            Dictionary mapping spectrum labels to normalization factor arrays
        """
        lmax = self.fields[0].lmax
        ell = np.arange(2, lmax + 1, dtype=np.float64)

        # Base factors for different spin combinations
        factor2 = 1 / ((ell + 2) * (ell + 1) * ell * (ell - 1))
        factor = np.sqrt(factor2)
        chngconv = (2 * ell + 1) / (4 * np.pi)

        normalization_factors = {}

        for label in self._spectra_labels:
            # Find the field pair for this spectrum
            for (i, j, mode), spec_label in self._spectra_map.items():
                if spec_label == label:
                    spin_i = self.fields[i].spin
                    spin_j = self.fields[j].spin

                    # Compute normalization factor based on spin combination
                    norm_factor = chngconv.copy()
                    if spin_i == 2 and spin_j == 2:
                        norm_factor *= factor2
                    elif (spin_i, spin_j) in [(0, 2), (2, 0)]:
                        norm_factor *= factor
                    # spin_i == 0 and spin_j == 0: no additional factor

                    normalization_factors[label] = norm_factor
                    break

        return normalization_factors

    def compute_smoothing_factors(
        self, beam_manager: "BeamManager"
    ) -> dict[str, np.ndarray]:
        """
        Compute smoothing factors for all spectrum labels.

        Parameters:
        -----------
        beam_manager : BeamManager
            BeamManager instance for smoothing factors

        Returns:
        --------
        dict[str, np.ndarray]
            Dictionary mapping spectrum labels to smoothing factor arrays
        """
        lmax = self.fields[0].lmax
        ell = np.arange(2, lmax + 1, dtype=np.float64)
        chngconv_smooth = 2 * np.pi / (ell * (ell + 1.0))

        # Get beam dictionary
        beam_dict = beam_manager.get_beam_dict()
        smoothing_factors = {}

        for label in self._spectra_labels:
            # Compute smoothing factor (beam + conversion)
            smooth_factor = chngconv_smooth.copy()

            # Extract field labels from spectrum label
            # (e.g., "TE" -> "T", "E")
            label1, label2 = label[0], label[1]
            if label1 in beam_dict and label2 in beam_dict:
                beam_factor = beam_dict[label1] * beam_dict[label2]
                smooth_factor *= beam_factor

            smoothing_factors[label] = smooth_factor

        return smoothing_factors


class BeamManager:
    """Manages beam functions for a collection of fields."""

    def __init__(self, fields: list["BaseField"]):
        self.fields = fields
        self._beam_dict = {}

    def compute_beams(
        self, lmax: int, nside: int, smoothtype: int, fwhmarcmin: float, beam_file: str
    ) -> dict[str, np.ndarray]:
        """
        Compute beam functions based on smoothing type.

        Parameters:
        -----------
        lmax : int
            Maximum multipole
        nside : int
            HEALPix nside parameter
        smoothtype : int
            Type of smoothing (0=none, 1=Gaussian, 2=cosine, 3=file)
        fwhmarcmin : float
            FWHM in arcminutes for Gaussian beam
        beam_file : str
            Path to beam file for smoothtype=3

        Returns:
        --------
        Dict[str, np.ndarray]
            Dictionary with beam functions for T, E, B
        """
        import healpy as hp

        if smoothtype == 0:
            beam = np.ones((3, lmax - 1), dtype=np.float64)
        elif smoothtype == 1:
            # fwhmarcmin in arcminutes → fwhm_rad
            beam = np.array(
                hp.gauss_beam(np.deg2rad(fwhmarcmin / 60.0), lmax=lmax + 1, pol=True)[
                    2 : lmax + 1, :-1
                ],
                dtype=np.float64,
            ).T
        elif smoothtype == 2:
            b = coswinbeam(nside)[2 : lmax + 1]
            beam = np.column_stack([b] * 3).T
        elif smoothtype == 3:
            # assume beam_file contains three columns of ell-window
            # healpy.read_cl returns a tuple of arrays when multiple fields
            bls = hp.read_cl(beam_file.strip()).astype(np.float64)
            beam = np.column_stack([bls[i][2 : lmax + 1] for i in range(bls.shape[0])]).T
        else:
            raise ValueError(f"Unknown smoothtype={smoothtype}")

        if beam.shape[0] != 3 or beam.shape[1] != lmax - 1:
            raise ValueError(
                f"Beam shape mismatch: expected (3, {lmax - 1}), got {beam.shape}"
            )

        return {
            "T": beam[0, :],
            "E": beam[1, :],
            "B": beam[2, :],
        }

    def set_beams_from_params(self, params: "InputParams") -> None:
        """Set beams for all fields using parameter configuration."""
        beam_dict = self.compute_beams(
            lmax=params.lmax,
            nside=params.nside,
            smoothtype=params.smoothing_type,
            fwhmarcmin=params.fwhmarcmin,
            beam_file=params.beam_file,
        )

        # Set beams for each field
        for field in self.fields:
            if field.spin == 0:
                # Scalar field - single beam
                beam_label = field.labels[0]
                if beam_label in beam_dict:
                    field.set_beam(beam_dict[beam_label])
                else:
                    # Fallback to generic beam if label not found
                    field.set_beam(beam_dict.get("T", list(beam_dict.values())[0]))
            elif field.spin == 2:
                # Polarization field - E and B beams
                e_beam = beam_dict.get(
                    "E", beam_dict.get("P", list(beam_dict.values())[0])
                )
                b_beam = beam_dict.get(
                    "B", beam_dict.get("P", list(beam_dict.values())[0])
                )
                beam_array = np.column_stack([e_beam, b_beam])
                field.set_beam(beam_array)

    def get_beam_dict(self) -> dict[str, np.ndarray]:
        """Get dictionary mapping field labels to beam functions."""
        beam_dict = {}
        for field in self.fields:
            if field.beam is None:
                raise ValueError(f"Beam not set for field with labels {field.labels}")

            if field.spin == 0:
                beam_dict[field.labels[0]] = field.beam
            elif field.spin == 2:
                # Polarization fields have beam shape (lmax-1, 2)
                beam_dict[field.labels[0]] = field.beam[:, 0]
                beam_dict[field.labels[1]] = field.beam[:, 1]

        return beam_dict

    def apply_smoothing(self, spectra_manager: SpectraManager) -> None:
        """Apply beam smoothing to power spectra."""
        # Use precomputed smoothing factors
        smoothing_factors = spectra_manager.compute_smoothing_factors(self)

        for label in spectra_manager.labels:
            if label not in spectra_manager._cls_dict:
                print(f"Warning: No power spectrum found for {label}...")
                continue

            if label in smoothing_factors:
                # Apply smoothing factor
                spectra_manager._cls_dict[label] *= smoothing_factors[label]

        # Update matrix from dictionary
        for idx, label in enumerate(spectra_manager._spectra_labels):
            spectra_manager._cls_matrix[:, idx] = spectra_manager._cls_dict[label]


@njit(cache=True)
def normcl(cl, lmax):
    """
    Compute normalization factor for power spectra.

    Parameters:
    -----------
    cl : ndarray
        Power spectrum array (lmax-1, n_spec)
    lmax : int
        Maximum multipole

    Returns:
    --------
    ndarray
        Normalization factors
    """
    ell = np.arange(2, lmax + 1, dtype=np.float64)
    norm_factor = 2 * np.pi / (ell * (ell + 1.0))
    n_spec = cl.shape[1]

    # Create normalization matrix
    norm_matrix = np.zeros_like(cl)
    for ispec in range(n_spec):
        norm_matrix[:, ispec] = norm_factor

    return norm_matrix


@njit(cache=True)
def smoothcl(cl, beam, lmax):
    """
    Apply beam smoothing to power spectra.

    Parameters:
    -----------
    cl : ndarray
        Power spectrum array (lmax-1, n_spec)
    beam : ndarray
        Beam function array
    lmax : int
        Maximum multipole

    Returns:
    --------
    ndarray
        Smoothed power spectra
    """
    smoothed_cl = np.zeros_like(cl)
    n_spec = cl.shape[1]

    for ispec in range(n_spec):
        smoothed_cl[:, ispec] = cl[:, ispec] * beam

    return smoothed_cl
