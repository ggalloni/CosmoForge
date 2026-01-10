"""
Parameter grid management for pixel-based likelihood analysis.

This module provides the ParameterGrid class for managing parameter spaces
and associated theoretical spectra in pixel-based likelihood computations.
It handles parameter range definitions, grid point generation, and efficient
distribution of computation across MPI processes.

Classes
-------
ParameterGrid
    Manager for parameter grids and theoretical spectra.

Notes
-----
The ParameterGrid class supports both regular grids defined by parameter ranges
and irregular grids defined by explicit parameter combinations. It provides
efficient methods for distributing computation across MPI processes and
retrieving theoretical spectra for specific parameter combinations.
"""

from __future__ import annotations

import itertools
from copy import deepcopy
from typing import Any

import numpy as np

from cosmocore import InputParams, readcl


class ParameterGrid:
    """
    Manager for parameter grids and theoretical spectra.

    This class handles the organization of parameter spaces for likelihood
    analysis, including grid generation, spectrum management, and efficient
    distribution of parameter points across parallel processes.

    Parameters
    ----------
    parameter_ranges : dict[str, np.ndarray]
        Dictionary mapping parameter names to their value ranges.
        Each value should be a 1D array of parameter values.
    theoretical_spectra : dict[tuple, np.ndarray]
        Dictionary mapping parameter value tuples to theoretical power spectra.
        Keys are tuples of parameter values in the same order as parameter_ranges.

    Attributes
    ----------
    parameter_names : list[str]
        Names of parameters in the grid.
    parameter_ranges : dict[str, np.ndarray]
        Parameter ranges defining the grid.
    theoretical_spectra : dict[tuple, np.ndarray]
        Theoretical spectra for each parameter combination.
    grid_points : list[tuple]
        All parameter combinations in the grid.

    Examples
    --------
    Regular parameter grid:

    >>> param_ranges = {
    ...     'omega_b': np.linspace(0.02, 0.025, 5),
    ...     'omega_c': np.linspace(0.10, 0.14, 5),
    ... }
    >>> spectra = {}  # Filled with theoretical spectra
    >>> grid = ParameterGrid(param_ranges, spectra)
    >>> total_points = grid.get_total_points()  # Returns 25

    MPI process distribution:

    >>> points_for_process = grid.get_points_for_process(rank=0, size=4)
    >>> # Returns subset of points for MPI rank 0 out of 4 processes

    Notes
    -----
    The class assumes that theoretical spectra are provided for all parameter
    combinations in the grid. Missing spectra will raise errors during
    likelihood computation.
    """

    def __init__(
        self,
        core_params: InputParams,
        parameter_ranges: dict[str, np.ndarray],
        theoretical_spectra: dict[tuple, np.ndarray] | None = None,
        root_dir: str | None = None,
        root_filename: str | None = None,
    ) -> None:
        """
        Initialize parameter grid.

        Parameters
        ----------
        parameter_ranges : dict[str, np.ndarray]
            Dictionary mapping parameter names to value arrays.
        theoretical_spectra : dict[tuple, np.ndarray]
            Dictionary mapping parameter tuples to power spectra.

        Raises
        ------
        ValueError
            If parameter ranges are empty or inconsistent with spectra.
        """

        self.core_params = core_params

        self.parameter_names = list(parameter_ranges.keys())
        self.parameter_ranges = deepcopy(parameter_ranges)

        # Generate all parameter combinations
        self.grid_points = self._generate_grid_points()

        if theoretical_spectra is not None:
            self.theoretical_spectra = deepcopy(theoretical_spectra)
        else:
            assert root_dir is not None and root_filename is not None, (
                "Either theoretical_spectra or both root_dir and root_filename "
                "must be provided to read spectra from files."
            )
            self.root_dir = root_dir
            self.root_filename = root_filename
            self.read_theoretical_spectra()

        # Validate that all grid points have corresponding spectra
        self._validate_spectra()

    def _generate_grid_points(self) -> list[tuple]:
        """
        Generate all parameter combinations in the grid.

        Returns
        -------
        grid_points : list[tuple]
            List of all parameter combinations as tuples.

        Notes
        -----
        Uses itertools.product to generate the Cartesian product of all
        parameter ranges, creating a complete grid of parameter combinations.
        """
        if not self.parameter_ranges:
            return []

        # Get parameter values in the order of parameter_names
        param_values = [self.parameter_ranges[name] for name in self.parameter_names]

        # Generate Cartesian product of all parameter ranges
        return list(itertools.product(*param_values))

    def read_theoretical_spectra(self) -> None:
        """
        Placeholder for reading theoretical spectra.

        Notes
        -----
        This method should be implemented to load theoretical spectra from
        files or other sources. Currently, it initializes an empty dictionary.
        """
        self.theoretical_spectra = {}

        root = self.root_dir + "/" + self.root_filename

        for point in self.grid_points:
            filename = (
                root
                + "_"
                + "_".join(
                    f"{self.parameter_names[i]}{val:.6g}" for i, val in enumerate(point)
                )
                + ".txt"
            )

            self.theoretical_spectra[point] = readcl(
                inputclfile=filename, Params=self.core_params
            )

    def _validate_spectra(self) -> None:
        """
        Validate that theoretical spectra exist for all grid points.

        Raises
        ------
        ValueError
            If any grid point lacks a corresponding theoretical spectrum.

        Notes
        -----
        Checks that every parameter combination in the grid has an associated
        theoretical power spectrum. This validation prevents runtime errors
        during likelihood computation.
        """
        missing_spectra = []

        for point in self.grid_points:
            if point not in self.theoretical_spectra:
                missing_spectra.append(point)

        if missing_spectra:
            msg = f"Missing theoretical spectra for {len(missing_spectra)} grid points"
            raise ValueError(msg)

    def get_total_points(self) -> int:
        """
        Get total number of points in the parameter grid.

        Returns
        -------
        total_points : int
            Total number of parameter combinations in the grid.
        """
        return len(self.grid_points)

    def get_points_for_process(self, rank: int, size: int) -> list[tuple]:
        """
        Get parameter points assigned to a specific MPI process.

        Parameters
        ----------
        rank : int
            MPI process rank (0-indexed).
        size : int
            Total number of MPI processes.

        Returns
        -------
        process_points : list[tuple]
            Parameter points assigned to this process.

        Notes
        -----
        Distributes parameter grid points as evenly as possible across MPI
        processes. Uses simple round-robin assignment to ensure load balance.
        """
        if size <= 0:
            msg = f"Invalid MPI size: {size}"
            raise ValueError(msg)

        if rank < 0 or rank >= size:
            msg = f"Invalid MPI rank {rank} for size {size}"
            raise ValueError(msg)

        # Distribute points using round-robin assignment
        return [self.grid_points[i] for i in range(rank, len(self.grid_points), size)]

    def get_spectrum(self, param_point: tuple) -> np.ndarray:
        """
        Get theoretical spectrum for a parameter point.

        Parameters
        ----------
        param_point : tuple
            Parameter values as a tuple in the order of parameter_names.

        Returns
        -------
        spectrum : numpy.ndarray
            Theoretical power spectrum for the given parameters.

        Raises
        ------
        KeyError
            If no spectrum exists for the given parameter point.

        Notes
        -----
        Retrieves the pre-computed theoretical power spectrum corresponding
        to the specified parameter combination. The spectrum is used for
        signal covariance matrix computation in likelihood analysis.
        """
        try:
            return self.theoretical_spectra[param_point]
        except KeyError as exc:
            msg = f"No theoretical spectrum found for parameters: {param_point}"
            raise KeyError(msg) from exc

    def get_parameter_dict(self, param_point: tuple) -> dict[str, Any]:
        """
        Convert parameter tuple to named dictionary.

        Parameters
        ----------
        param_point : tuple
            Parameter values as a tuple.

        Returns
        -------
        param_dict : dict[str, Any]
            Dictionary mapping parameter names to values.

        Examples
        --------
        >>> grid = ParameterGrid(param_ranges, spectra)
        >>> point = (0.022, 0.12)  # (omega_b, omega_c)
        >>> param_dict = grid.get_parameter_dict(point)
        >>> # Returns {'omega_b': 0.022, 'omega_c': 0.12}
        """
        if len(param_point) != len(self.parameter_names):
            msg = (
                f"Parameter point has {len(param_point)} values, "
                f"expected {len(self.parameter_names)}"
            )
            raise ValueError(msg)

        return dict(zip(self.parameter_names, param_point))

    def get_grid_shape(self) -> tuple[int, ...]:
        """
        Get shape of the parameter grid.

        Returns
        -------
        grid_shape : tuple[int, ...]
            Shape of the grid with dimensions corresponding to each parameter.

        Notes
        -----
        Returns the shape of the regular grid formed by the parameter ranges.
        Useful for reshaping results into multidimensional arrays for
        visualization and analysis.
        """
        return tuple(len(self.parameter_ranges[name]) for name in self.parameter_names)

    def get_parameter_index(self, param_point: tuple) -> int:
        """
        Get flat index of a parameter point in the grid.

        Parameters
        ----------
        param_point : tuple
            Parameter values as a tuple.

        Returns
        -------
        index : int
            Flat index of the parameter point in the grid.

        Raises
        ------
        ValueError
            If the parameter point is not found in the grid.

        Notes
        -----
        Converts multidimensional parameter coordinates to a flat index,
        useful for indexing into result arrays and mapping between parameter
        space and result storage.
        """
        try:
            return self.grid_points.index(param_point)
        except ValueError as exc:
            msg = f"Parameter point {param_point} not found in grid"
            raise ValueError(msg) from exc

    def add_spectrum(self, param_point: tuple, spectrum: np.ndarray) -> None:
        """
        Add theoretical spectrum for a parameter point.

        Parameters
        ----------
        param_point : tuple
            Parameter values as a tuple.
        spectrum : numpy.ndarray
            Theoretical power spectrum to add.

        Notes
        -----
        Allows dynamic addition of theoretical spectra to the grid.
        Useful for cases where spectra are computed on-the-fly or
        loaded incrementally.
        """
        self.theoretical_spectra[param_point] = deepcopy(spectrum)

    def __len__(self) -> int:
        """Return total number of points in the grid."""
        return len(self.grid_points)

    def __iter__(self):
        """Iterate over parameter points in the grid."""
        return iter(self.grid_points)

    def __contains__(self, param_point: tuple) -> bool:
        """Check if parameter point exists in the grid."""
        return param_point in self.grid_points

    def __eq__(self, other):
        """Check equality with another ParameterGrid."""
        if not isinstance(other, ParameterGrid):
            return False
        return (
            self.parameter_names == other.parameter_names
            and self.parameter_ranges == other.parameter_ranges
            and self.theoretical_spectra == other.theoretical_spectra
            and self.grid_points == other.grid_points
        )
