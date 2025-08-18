from dataclasses import dataclass

import healpy as hp
import numpy as np

from .basics import cross_index


def preprocess_fields(npixs, spins):
    logical_npixs = []
    logical_spins = []
    logical_to_orig = []  # list of lists of original indices for each logical field
    i = 0
    n = len(spins)
    while i < n:
        if spins[i] == 0:
            logical_npixs.append(npixs[i])
            logical_spins.append(0)
            logical_to_orig.append([i])
            i += 1
        elif spins[i] == 2:
            if i + 1 >= n or spins[i + 1] != 2:
                msg = (
                    f"Spin-2 field at index {i} is not paired with another spin-2 field."
                )
                raise ValueError(msg)
            if npixs[i] != npixs[i + 1]:
                msg = f"Spin-2 pair at indices {i},{i + 1} have different npixs."
                raise ValueError(msg)
            logical_npixs.append(npixs[i] * 2)  # size per Q/U, total block will be 2*npix
            logical_spins.append(2)
            logical_to_orig.append([i, i + 1])
            i += 2
        else:
            msg = f"Unknown spin value {spins[i]} at index {i}."
            raise ValueError(msg)
    return logical_npixs, logical_spins, logical_to_orig


def organize_spectra_by_spin(Cls, logical_spins, logical_to_orig):
    """
    Organize spectra into subgroups based on logical spins and mapping.
    Returns a dict with keys:
      - ('auto', i): list of auto spectra for logical field i
      - ('cross', i, j): list of cross spectra between logical fields i and j
    """
    n_orig = sum(len(x) for x in logical_to_orig)
    n_logical = len(logical_spins)
    spectra = {}

    # Auto spectra
    for i, spin in enumerate(logical_spins):
        orig = logical_to_orig[i]
        if spin == 0:
            spectra[("auto", i)] = [Cls[:, orig[0]]]
        elif spin == 2:
            idx1, idx2 = orig
            auto1 = Cls[:, idx1]
            auto2 = Cls[:, idx2]
            cross = Cls[:, n_orig + cross_index(idx1, idx2, n_orig)]
            spectra[("auto", i)] = [auto1, auto2, cross]
        else:
            msg = f"Unknown spin {spin}"
            raise ValueError(msg)

    # Cross spectra
    for i in range(n_logical):
        for j in range(i + 1, n_logical):
            spin_i, spin_j = logical_spins[i], logical_spins[j]
            orig_i, orig_j = logical_to_orig[i], logical_to_orig[j]
            cross_list = []
            if spin_i == 0 and spin_j == 0:
                cross_list.append(
                    Cls[:, n_orig + cross_index(orig_i[0], orig_j[0], n_orig)]
                )
            elif spin_i == 0 and spin_j == 2:
                cross_list.append(
                    Cls[:, n_orig + cross_index(orig_i[0], orig_j[0], n_orig)]
                )
                cross_list.append(
                    Cls[:, n_orig + cross_index(orig_i[0], orig_j[1], n_orig)]
                )
            elif spin_i == 2 and spin_j == 0:
                cross_list.append(
                    Cls[:, n_orig + cross_index(orig_j[0], orig_i[0], n_orig)]
                )
                cross_list.append(
                    Cls[:, n_orig + cross_index(orig_j[0], orig_i[1], n_orig)]
                )
            elif spin_i == 2 and spin_j == 2:
                for a in orig_i:
                    for b in orig_j:
                        cross_list.append(Cls[:, n_orig + cross_index(a, b, n_orig)])
            spectra[("cross", i, j)] = cross_list

    orig_to_logical = {}
    for logical_idx, orig_list in enumerate(logical_to_orig):
        for pos, orig_idx in enumerate(orig_list):
            orig_to_logical[orig_idx] = (logical_idx, pos)

    return spectra, orig_to_logical


@dataclass
class LogicalField:
    spin: int
    nside: int
    lmax: int
    mask: np.ndarray
    maps_label: str | list[str]  # "X" or ["X", "Y"]
    beam: np.ndarray | None = None
    fiducial_cl: np.ndarray | None = None
    point_vectors: np.ndarray | None = None
    active: np.ndarray | None = None

    def __post_init__(self):
        if self.spin not in (0, 2):
            raise ValueError(f"Invalid spin {self.spin}. Must be 0 or 2.")
        if self.mask.ndim != 1:
            raise ValueError("Mask must be a 1D array.")
        if self.lmax < 2:
            raise ValueError("lmax must be at least 2.")
        elif self.lmax > self.nside * 4:
            raise ValueError("lmax is too large for the given nside.")

        self.npix = hp.nside2npix(self.nside)

        if self.spin == 0:
            self.N_maps = 1
            self.N_spectra = 1
        elif self.spin == 2:
            self.N_maps = 2
            self.N_spectra = 3

        if isinstance(self.maps_label, str):
            self.maps_label = [self.maps_label]

        if len(self.maps_label) != self.N_maps:
            raise ValueError(
                f"maps_label must have {self.N_maps} entries, got {len(self.maps_label)}"
            )

        self.cl_label = []
        for label_i in self.maps_label:
            self.cl_label.append(f"{label_i.upper()}{label_i.upper()}")
        for i in range(self.N_maps):
            for j in range(i + 1, self.N_maps):
                self.cl_label.append(
                    f"{self.maps_label[i].upper()}{self.maps_label[j].upper()}"
                )

        self.mapping = {self.cl_label[i]: i for i in range(len(self.cl_label))}

        self.compute_active_pixels()

    def compute_active_pixels(self):
        n_active = np.count_nonzero(self.mask)
        if self.spin == 2:
            # For spin-2 fields, both Q and U use the same mask
            self.N_active = [n_active, n_active]
        else:
            # For spin-0 fields, just one component
            self.N_active = [n_active]
        self.active = np.where(self.mask > 0.5)[0]

    def set_active_pixels(self, active):
        self.N_active = len(active)
        self.active = active

    def set_pointing_vectors(self, point_vectors: np.ndarray):
        if point_vectors.shape[0] != self.N_active[0]:
            raise ValueError(
                f"Point vectors must have {self.N_active[0]} rows, "
                f"got {point_vectors.shape[0]}"
            )
        self.point_vectors = point_vectors

    def set_fiducial_cl(self, fiducial_cl: np.ndarray):
        if fiducial_cl.shape[0] != self.lmax - 1:
            raise ValueError(
                f"Fiducial CL must have {self.lmax - 1} rows, got {fiducial_cl.shape[0]}"
            )
        if fiducial_cl.shape[1] != self.N_spectra:
            raise ValueError(
                f"Fiducial CL must have {self.N_spectra} columns, "
                f"got {fiducial_cl.shape[1]}"
            )
        self.fiducial_cl = fiducial_cl

    def set_beam(self, beam: np.ndarray):
        if beam.shape[0] != self.lmax - 1:
            raise ValueError(f"Beam must have {self.lmax - 1} rows, got {beam.shape[0]}")
        self.beam = beam


class LogicalFieldCollection:
    def __init__(self, logical_fields):
        """
        logical_fields: list of LogicalField objects
        """
        self.logical_fields: list[LogicalField] = logical_fields
        self.n_fields = len(logical_fields)
        self.field_labels: list[list[str]] = [lf.maps_label for lf in logical_fields]
        self.spectra_labels: list[str] = []
        self.spectra_map: dict[tuple[int, int, int], str] = {}
        for i, lf_i in enumerate(logical_fields):
            for mode in range(logical_fields[i].N_spectra):
                label = logical_fields[i].cl_label[mode]
                self.spectra_labels.append(label)
                self.spectra_map[(i, i, mode)] = label

        for i in range(self.n_fields):
            for j in range(i + 1, self.n_fields):
                lf_i, lf_j = logical_fields[i], logical_fields[j]
                grid = np.array(
                    np.meshgrid(
                        range(len(lf_i.maps_label)),
                        range(len(lf_j.maps_label)),
                        indexing="ij",
                    )
                ).squeeze()
                if len(grid.shape) == 1:
                    grid = grid.reshape(-1, 2)
                for mode, (gi, gj) in enumerate(grid):
                    label = f"{lf_i.maps_label[gi].upper()}{lf_j.maps_label[gj].upper()}"
                    self.spectra_labels.append(label)
                    self.spectra_map[(i, j, mode)] = label

        self.n_spectra = len(self.spectra_labels)
        self.spin = [lf.spin for lf in logical_fields]
        self.active = self.get_active_pixels()
        self.N_active = self.get_N_active()

        self.lmax = logical_fields[0].lmax
        self.nside = logical_fields[0].nside
        for i, lf_i in enumerate(logical_fields):
            assert lf_i.lmax == self.lmax, (
                f"Logical field {i} has lmax {lf_i.lmax}, expected {self.lmax}"
            )
            assert lf_i.nside == self.nside, (
                f"Logical field {i} has nside {lf_i.nside}, expected {self.nside}"
            )

    def get_spectrum_label(self, i, j, mode=0):
        return self.spectra_map.get((i, j, mode))

    def get_field(self, idx):
        return self.logical_fields[idx]

    def get_auto_spectra(self):
        """
        Returns a list of (label, field_i, field_j, mode) for all auto spectra.
        """
        auto_list = []
        for i, lf in enumerate(self.logical_fields):
            for mode in range(lf.N_spectra):
                label = self.get_spectrum_label(i, i, mode)
                auto_list.append((label, i, i, mode))
        return auto_list

    def get_cross_spectra(self):
        """
        Returns a list of (label, field_i, field_j, mode) for all cross spectra (i < j).
        """
        cross_list = []
        for i in range(self.n_fields):
            for j in range(i + 1, self.n_fields):
                lf_i, lf_j = self.logical_fields[i], self.logical_fields[j]
                grid = np.array(
                    np.meshgrid(
                        range(len(lf_i.maps_label)),
                        range(len(lf_j.maps_label)),
                        indexing="ij",
                    )
                ).squeeze()
                for mode, (gi, gj) in enumerate(grid):
                    label = f"{lf_i.maps_label[gi].upper()}{lf_j.maps_label[gj].upper()}"
                    cross_list.append((label, i, j, mode))
        return cross_list

    def get_spectra_matrix(self):
        """
        Returns a 2D matrix (list of lists) of spectra labels for all (i, j) pairs.
        Only i <= j blocks are filled; matrix[j][i] is always empty for j > i.
        """
        return self.get_auto_spectra() + self.get_cross_spectra()

    def get_active_pixels(self):
        """
        Returns a 2D array of active pixels for each logical field.
        For spin-0 fields, returns one entry per field.
        For spin-2 fields, returns two entries (Q and U components) with same
        active pixels.
        """
        active_pixels = []
        for lf in self.logical_fields:
            active_pixels.append(lf.active)
            if lf.spin == 2:
                active_pixels.append(lf.active)
        return np.array(active_pixels, dtype=object)

    def get_beam(self):
        """
        Returns a 2D array of beam patterns for each logical field.
        Each row corresponds to a logical field, and contains the beam pattern.
        """
        beams = {}
        for lf in self.logical_fields:
            if lf.spin == 0:
                beams[lf.maps_label[0]] = lf.beam
            elif lf.spin == 2:
                # For spin-2 fields, we need to duplicate the beam
                beams[lf.maps_label[0]] = lf.beam[:, 0]
                beams[lf.maps_label[1]] = lf.beam[:, 1]
        return beams

    def get_N_active(self):
        """
        Store the number of active pixels for each logical field.
        Returns a 1D array with the number of active pixels for each field.
        """
        n_active = []
        for lf in self.logical_fields:
            n_active += lf.N_active
        return np.array(n_active, dtype=int)

    def set_cls(self, Cls):
        """
        cls_dict: dict mapping spectra label (e.g. 'TT', 'TE', 'EE', ...)
        to Cls column (1D array)
        """
        if isinstance(Cls, dict):
            self.cls_dict = Cls
            self.n_cls = len(Cls)
            self.cls_matrix = np.zeros((self.lmax - 1, self.n_cls))
            # Optionally, build a matrix or array for fast access:
            for idx, label in enumerate(self.spectra_labels):
                if label not in Cls:
                    raise ValueError(f"Missing Cls for spectrum label {label}")
                self.cls_matrix[:, idx] = Cls[label]
            # You can now access Cls by label, or by (i, j, mode) using self.spectra_map
        elif isinstance(Cls, np.ndarray):
            assert Cls.shape[1] == self.n_spectra, (
                f"Cls must have {self.n_spectra} columns, got {Cls.shape[1]}"
            )
            self.cls_matrix = Cls[: self.lmax - 1]
            self.cls_dict = {
                label: self.cls_matrix[:, idx]
                for idx, label in enumerate(self.spectra_labels)
            }

        self.apply_normalization()

    def apply_normalization(self):
        ell = np.arange(2, self.lmax + 1, dtype=np.float64)
        factor2 = 1 / ((ell + 2) * (ell + 1) * ell * (ell - 1))
        factor = np.sqrt(factor2)
        chngconv = (2 * ell + 1) / (4 * np.pi)

        for idx, label in enumerate(self.spectra_labels):
            found = False
            for key, lbl in self.spectra_map.items():
                if lbl == label:
                    i, j, mode = key
                    spin_i = self.logical_fields[i].spin
                    spin_j = self.logical_fields[j].spin

                    self.cls_matrix[:, idx] *= chngconv
                    if spin_i == 2 and spin_j == 2:
                        self.cls_matrix[:, idx] *= factor2
                    elif (spin_i, spin_j) in [(0, 2), (2, 0)]:
                        self.cls_matrix[:, idx] *= factor
                    elif (spin_i, spin_j) == (0, 0):
                        pass
                    found = True
                    break
            if not found:
                raise ValueError(f"Could not infer spin for label {label}")
            self.cls_dict[label] = self.cls_matrix[:, idx]

    def apply_smoothing(self):
        ell = np.arange(2, self.lmax + 1, dtype=np.float64)
        chngconv = 2 * np.pi / (ell * (ell + 1.0))
        beams = self.get_beam()
        for i, label in enumerate(self.spectra_labels):
            if label not in self.cls_dict:
                raise ValueError(f"Missing Cls for spectrum label {label}")
            self.cls_dict[label] *= beams[label[0]] * beams[label[1]]
            self.cls_matrix[:, i] *= chngconv

    def get_cls(self, i, j, mode=0):
        label = self.get_spectrum_label(i, j, mode)
        return self.cls_dict[label]

    def set_pointing_vectors(self, point_vectors):
        """
        Set the pointing vectors for each logical field.
        """
        for i, lf in enumerate(self.logical_fields):
            lf.set_pointing_vectors(point_vectors[i])
