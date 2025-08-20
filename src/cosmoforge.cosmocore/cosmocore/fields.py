from abc import ABC, abstractmethod
from dataclasses import dataclass

import healpy as hp
import numpy as np

from cosmocore.harmonic import BeamManager, SpectraManager
from cosmocore.in_out import readcl
from cosmocore.settings import InputParams


@dataclass
class FieldConfig:
    """Configuration for a cosmological field."""

    spin: int
    nside: int
    lmax: int
    mask: np.ndarray
    labels: str | list[str]  # "T" for spin-0, ["E", "B"] for spin-2

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.spin not in (0, 2):
            raise ValueError(f"Invalid spin {self.spin}. Must be 0 or 2.")
        if self.mask.ndim != 1:
            raise ValueError("Mask must be a 1D array.")
        if self.lmax < 2:
            raise ValueError("lmax must be at least 2.")
        if self.lmax > self.nside * 4:
            raise ValueError("lmax is too large for the given nside.")

        # Normalize labels to list format
        if isinstance(self.labels, str):
            self.labels = [self.labels]

        # Validate label count matches spin
        expected_labels = 1 if self.spin == 0 else 2
        if len(self.labels) != expected_labels:
            raise ValueError(
                f"Spin-{self.spin} field requires {expected_labels} labels, "
                f"got {len(self.labels)}"
            )


class BaseField(ABC):
    """Abstract base class for cosmological fields."""

    def __init__(self, config: FieldConfig):
        self.config = config
        self._active_pixels = None
        self._point_vectors = None
        self._beam = None

        # Derived properties
        self.npix = hp.nside2npix(config.nside)
        self.n_components = 1 if config.spin == 0 else 2

    @property
    def spin(self) -> int:
        return self.config.spin

    @property
    def nside(self) -> int:
        return self.config.nside

    @property
    def lmax(self) -> int:
        return self.config.lmax

    @property
    def labels(self) -> list[str]:
        return self.config.labels

    @property
    def maps_label(self) -> str | list[str]:
        """Backward compatibility property for legacy code."""
        if len(self.labels) == 1:
            return self.labels[0]
        return self.labels

    @property
    def mask(self) -> np.ndarray:
        return self.config.mask

    @property
    def active_pixels(self) -> np.ndarray:
        if self._active_pixels is None:
            self._active_pixels = np.where(self.mask > 0.5)[0]
        return self._active_pixels

    @property
    def n_active(self) -> list[int]:
        return [len(self.active_pixels)]

    @property
    def point_vectors(self) -> np.ndarray | None:
        return self._point_vectors

    @property
    def beam(self) -> np.ndarray | None:
        return self._beam

    def set_point_vectors(self, vectors: np.ndarray) -> None:
        """Set pointing vectors for active pixels."""
        if vectors.shape[0] != self.n_active[0]:
            raise ValueError(
                f"Expected {self.n_active} point vectors, got {vectors.shape[0]}"
            )
        if vectors.shape[1] != 3:
            raise ValueError("Point vectors must have 3 components (x,y,z)")
        self._point_vectors = vectors.copy()

    def set_beam(self, beam: np.ndarray) -> None:
        """Set beam function."""
        expected_rows = self.lmax - 1
        if beam.shape[0] != expected_rows:
            raise ValueError(f"Beam must have {expected_rows} rows")
        self._beam = beam.copy()

    @abstractmethod
    def get_spectrum_labels(self) -> list[str]:
        """Get labels for all auto-spectra of this field."""
        pass

    @abstractmethod
    def get_cross_spectrum_labels(self, other: "BaseField") -> list[str]:
        """Get labels for cross-spectra with another field."""
        pass


class ScalarField(BaseField):
    """Spin-0 cosmological field (e.g., temperature)."""

    def __init__(self, config: FieldConfig):
        if config.spin != 0:
            raise ValueError("ScalarField requires spin=0")
        super().__init__(config)

    def get_spectrum_labels(self) -> list[str]:
        """Returns ['TT'] for temperature field."""
        label = self.labels[0].upper()
        return [f"{label}{label}"]

    def get_cross_spectrum_labels(self, other: "BaseField") -> list[str]:
        """Cross-spectra with another field."""
        self_label = self.labels[0].upper()
        if isinstance(other, ScalarField):
            other_label = other.labels[0].upper()
            return [f"{self_label}{other_label}"]
        elif isinstance(other, PolarizationField):
            return [
                f"{self_label}{other.labels[0].upper()}",
                f"{self_label}{other.labels[1].upper()}",
            ]
        else:
            raise TypeError(f"Unknown field type: {type(other)}")


class PolarizationField(BaseField):
    """Spin-2 cosmological field (e.g., polarization E/B modes)."""

    def __init__(self, config: FieldConfig):
        if config.spin != 2:
            raise ValueError("PolarizationField requires spin=2")
        super().__init__(config)

    def get_spectrum_labels(self) -> list[str]:
        """Returns ['EE', 'BB', 'EB'] for polarization field."""
        e_label, b_label = (label.upper() for label in self.labels)
        return [f"{e_label}{e_label}", f"{b_label}{b_label}", f"{e_label}{b_label}"]

    def get_cross_spectrum_labels(self, other: "BaseField") -> list[str]:
        """Cross-spectra with another field."""
        if isinstance(other, ScalarField):
            other_label = other.labels[0].upper()
            return [
                f"{other_label}{self.labels[0].upper()}",
                f"{other_label}{self.labels[1].upper()}",
            ]
        elif isinstance(other, PolarizationField):
            cross_labels = []
            for self_label in self.labels:
                for other_label in other.labels:
                    cross_labels.append(f"{self_label.upper()}{other_label.upper()}")
            return cross_labels
        else:
            raise TypeError(f"Unknown field type: {type(other)}")


class FieldCollection:
    """New redesigned collection class with clean separation of concerns."""

    def __init__(self, core_params: InputParams, fields: list[BaseField]):
        self.core_params = core_params
        self.fields = fields
        self.n_fields = len(fields)

        # Validate consistency
        self._validate_fields()

        # Initialize managers
        self.spectra_manager = SpectraManager(fields)
        self.beam_manager = BeamManager(fields)

        # Cache commonly used properties
        self._lmax = fields[0].lmax
        self._nside = fields[0].nside

    def _validate_fields(self) -> None:
        """Ensure all fields have consistent lmax and nside."""
        if not self.fields:
            raise ValueError("Must provide at least one field")

        reference = self.fields[0]
        for i, field in enumerate(self.fields[1:], 1):
            if field.lmax != reference.lmax:
                raise ValueError(
                    f"Field {i} has lmax {field.lmax}, expected {reference.lmax}"
                )
            if field.nside != reference.nside:
                raise ValueError(
                    f"Field {i} has nside {field.nside}, expected {reference.nside}"
                )

    @property
    def lmax(self) -> int:
        return self._lmax

    @property
    def nside(self) -> int:
        return self._nside

    @property
    def spin(self) -> list[int]:
        return [field.spin for field in self.fields]

    @property
    def n_active(self) -> list[int]:
        """Number of active pixels per field component."""
        n_active = []
        for field in self.fields:
            n_active += field.n_active
            if field.spin == 2:
                n_active += field.n_active
        return n_active

    def get_active_pixels(self) -> np.ndarray:
        """Get active pixels for each field component (backward compatible format)."""
        active_list = []
        for field in self.fields:
            active_list.append(field.active_pixels)
            if field.spin == 2:
                active_list.append(field.active_pixels)  # Same for Q and U
        return np.array(active_list, dtype=object)

    def set_pointing_vectors(self, point_vectors: list[np.ndarray]) -> None:
        """Set pointing vectors for all fields."""

        for field, vectors in zip(self.fields, point_vectors):
            field.set_point_vectors(vectors)

    def set_cls(
        self,
        cls_data: dict[str, np.ndarray] | np.ndarray | None = None,
    ) -> None:
        """Set power spectra and apply normalization."""
        if cls_data is None:
            cls_data = readcl(self.core_params.inputclfile, self.core_params)

        self.spectra_manager.set_cls(cls_data)
        self.spectra_manager.apply_normalization()

    def get_cls(self, field_i: int, field_j: int, mode: int = 0) -> np.ndarray:
        """Get power spectrum for field pair."""
        return self.spectra_manager.get_cls(field_i, field_j, mode)

    def set_beams(self) -> None:
        """Set beam functions for all fields."""

        self.beam_manager.set_beams_from_params(self.core_params)
        self.beam_manager.apply_smoothing(self.spectra_manager)

    @property
    def fields(self) -> list[BaseField]:
        """Provide access to fields for backward compatibility."""
        return self._fields

    @fields.setter
    def fields(self, value: list[BaseField]) -> None:
        """Setter to allow modification of fields."""
        self._fields = value
        self._validate_fields()

    @property
    def spectra_labels(self) -> list[str]:
        """Backward compatibility property for Fisher class."""
        return self.spectra_manager.labels

    @property
    def n_spectra(self) -> int:
        """Backward compatibility property for number of spectra."""
        return self.spectra_manager.n_spectra


def create_field(
    spin: int, nside: int, lmax: int, mask: np.ndarray, labels: str | list[str]
) -> BaseField:
    """Factory function to create appropriate field type."""
    config = FieldConfig(spin=spin, nside=nside, lmax=lmax, mask=mask, labels=labels)

    if spin == 0:
        return ScalarField(config)
    elif spin == 2:
        return PolarizationField(config)
    else:
        raise ValueError(f"Unsupported spin value: {spin}")
