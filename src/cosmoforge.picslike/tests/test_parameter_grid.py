"""
Tests for ParameterGrid class.

This module contains unit tests for the ParameterGrid class functionality
including grid generation, parameter management, and MPI distribution.
"""

import pytest

from picslike.parameter_grid import ParameterGrid


class TestParameterGrid:
    """Test suite for ParameterGrid class."""

    def test_grid_initialization(
        self, minimal_params, sample_parameter_ranges, sample_theoretical_spectra
    ):
        """Test basic grid initialization."""
        grid = ParameterGrid(
            core_params=minimal_params,
            parameter_ranges=sample_parameter_ranges,
            theoretical_spectra=sample_theoretical_spectra,
        )

        assert grid.parameter_names == ["omega_b", "omega_c"]
        assert len(grid.grid_points) == 9  # 3 * 3 combinations
        assert grid.get_total_points() == 9

    def test_grid_points_generation(
        self, minimal_params, sample_parameter_ranges, sample_theoretical_spectra
    ):
        """Test that all expected grid points are generated."""
        grid = ParameterGrid(
            core_params=minimal_params,
            parameter_ranges=sample_parameter_ranges,
            theoretical_spectra=sample_theoretical_spectra,
        )

        expected_points = [
            (0.020, 0.10),
            (0.020, 0.12),
            (0.020, 0.14),
            (0.022, 0.10),
            (0.022, 0.12),
            (0.022, 0.14),
            (0.024, 0.10),
            (0.024, 0.12),
            (0.024, 0.14),
        ]

        assert len(grid.grid_points) == len(expected_points)
        for point in expected_points:
            assert point in grid.grid_points

    def test_spectrum_retrieval(
        self, minimal_params, sample_parameter_ranges, sample_theoretical_spectra
    ):
        """Test spectrum retrieval for parameter points."""
        grid = ParameterGrid(
            core_params=minimal_params,
            parameter_ranges=sample_parameter_ranges,
            theoretical_spectra=sample_theoretical_spectra,
        )

        # Test retrieval of existing spectrum
        test_point = (0.022, 0.12)
        spectrum = grid.get_spectrum(test_point)
        assert isinstance(spectrum, dict)
        assert "TT" in spectrum

        # Test error for non-existing spectrum
        with pytest.raises(KeyError):
            grid.get_spectrum((0.999, 0.999))

    def test_parameter_dict_conversion(
        self, minimal_params, sample_parameter_ranges, sample_theoretical_spectra
    ):
        """Test conversion of parameter tuples to dictionaries."""
        grid = ParameterGrid(
            core_params=minimal_params,
            parameter_ranges=sample_parameter_ranges,
            theoretical_spectra=sample_theoretical_spectra,
        )

        test_point = (0.022, 0.12)
        param_dict = grid.get_parameter_dict(test_point)

        expected_dict = {"omega_b": 0.022, "omega_c": 0.12}
        assert param_dict == expected_dict

    def test_mpi_distribution(
        self, minimal_params, sample_parameter_ranges, sample_theoretical_spectra
    ):
        """Test MPI process distribution of grid points."""
        grid = ParameterGrid(
            core_params=minimal_params,
            parameter_ranges=sample_parameter_ranges,
            theoretical_spectra=sample_theoretical_spectra,
        )

        # Test distribution across 3 processes
        size = 3
        all_points = []

        for rank in range(size):
            points = grid.get_points_for_process(rank, size)
            all_points.extend(points)

        # Check that all points are distributed exactly once
        assert len(all_points) == grid.get_total_points()
        assert len(set(all_points)) == len(all_points)  # No duplicates

    def test_grid_shape(
        self, minimal_params, sample_parameter_ranges, sample_theoretical_spectra
    ):
        """Test grid shape computation."""
        grid = ParameterGrid(
            core_params=minimal_params,
            parameter_ranges=sample_parameter_ranges,
            theoretical_spectra=sample_theoretical_spectra,
        )

        expected_shape = (3, 3)  # 3 omega_b values, 3 omega_c values
        assert grid.get_grid_shape() == expected_shape

    def test_parameter_index(
        self, minimal_params, sample_parameter_ranges, sample_theoretical_spectra
    ):
        """Test parameter point indexing."""
        grid = ParameterGrid(
            core_params=minimal_params,
            parameter_ranges=sample_parameter_ranges,
            theoretical_spectra=sample_theoretical_spectra,
        )

        test_point = (0.022, 0.12)
        index = grid.get_parameter_index(test_point)

        # Check that the index is valid and corresponds to the right point
        assert 0 <= index < grid.get_total_points()
        assert grid.grid_points[index] == test_point

    def test_spectrum_addition(self, minimal_params, sample_parameter_ranges):
        """Test dynamic spectrum addition."""
        # Start with empty spectra dictionary
        empty_spectra = {}

        # This should raise an error during validation
        with pytest.raises(ValueError):
            ParameterGrid(
                core_params=minimal_params,
                parameter_ranges=sample_parameter_ranges,
                theoretical_spectra=empty_spectra,
            )

    def test_invalid_mpi_parameters(
        self, minimal_params, sample_parameter_ranges, sample_theoretical_spectra
    ):
        """Test error handling for invalid MPI parameters."""
        grid = ParameterGrid(
            core_params=minimal_params,
            parameter_ranges=sample_parameter_ranges,
            theoretical_spectra=sample_theoretical_spectra,
        )

        # Test invalid size
        with pytest.raises(ValueError):
            grid.get_points_for_process(0, 0)

        # Test invalid rank
        with pytest.raises(ValueError):
            grid.get_points_for_process(-1, 4)

        with pytest.raises(ValueError):
            grid.get_points_for_process(4, 4)

    def test_empty_parameter_ranges(self, minimal_params):
        """Test handling of empty parameter ranges."""
        empty_ranges = {}
        empty_spectra = {}

        grid = ParameterGrid(
            core_params=minimal_params,
            parameter_ranges=empty_ranges,
            theoretical_spectra=empty_spectra,
        )
        assert grid.get_total_points() == 0
        assert len(grid.grid_points) == 0

    def test_contains_method(
        self, minimal_params, sample_parameter_ranges, sample_theoretical_spectra
    ):
        """Test __contains__ method."""
        grid = ParameterGrid(
            core_params=minimal_params,
            parameter_ranges=sample_parameter_ranges,
            theoretical_spectra=sample_theoretical_spectra,
        )

        assert (0.022, 0.12) in grid
        assert (0.999, 0.999) not in grid

    def test_len_method(
        self, minimal_params, sample_parameter_ranges, sample_theoretical_spectra
    ):
        """Test __len__ method."""
        grid = ParameterGrid(
            core_params=minimal_params,
            parameter_ranges=sample_parameter_ranges,
            theoretical_spectra=sample_theoretical_spectra,
        )

        assert len(grid) == 9

    def test_iter_method(
        self, minimal_params, sample_parameter_ranges, sample_theoretical_spectra
    ):
        """Test __iter__ method."""
        grid = ParameterGrid(
            core_params=minimal_params,
            parameter_ranges=sample_parameter_ranges,
            theoretical_spectra=sample_theoretical_spectra,
        )

        points_from_iter = list(grid)
        assert len(points_from_iter) == grid.get_total_points()
        assert set(points_from_iter) == set(grid.grid_points)

    def test_parameter_dict_wrong_length(
        self, minimal_params, sample_parameter_ranges, sample_theoretical_spectra
    ):
        """Test get_parameter_dict with wrong number of values."""
        grid = ParameterGrid(
            core_params=minimal_params,
            parameter_ranges=sample_parameter_ranges,
            theoretical_spectra=sample_theoretical_spectra,
        )

        # Too few parameters
        with pytest.raises(ValueError, match="Parameter point has"):
            grid.get_parameter_dict((0.022,))

        # Too many parameters
        with pytest.raises(ValueError, match="Parameter point has"):
            grid.get_parameter_dict((0.022, 0.12, 0.5))

    def test_get_parameter_index_not_found(
        self, minimal_params, sample_parameter_ranges, sample_theoretical_spectra
    ):
        """Test get_parameter_index with non-existing point."""
        grid = ParameterGrid(
            core_params=minimal_params,
            parameter_ranges=sample_parameter_ranges,
            theoretical_spectra=sample_theoretical_spectra,
        )

        with pytest.raises(ValueError, match="not found in grid"):
            grid.get_parameter_index((0.999, 0.999))

    def test_add_spectrum(
        self, minimal_params, sample_parameter_ranges, sample_theoretical_spectra
    ):
        """Test dynamic spectrum addition."""
        import numpy as np

        grid = ParameterGrid(
            core_params=minimal_params,
            parameter_ranges=sample_parameter_ranges,
            theoretical_spectra=sample_theoretical_spectra,
        )

        # Add a new spectrum for an existing point (update)
        new_spectrum = {"TT": np.ones(100)}
        grid.add_spectrum((0.022, 0.12), new_spectrum)
        retrieved = grid.get_spectrum((0.022, 0.12))
        assert "TT" in retrieved
        np.testing.assert_array_equal(retrieved["TT"], new_spectrum["TT"])

    def test_equality_with_non_grid(
        self, minimal_params, sample_parameter_ranges, sample_theoretical_spectra
    ):
        """Test __eq__ method with non-ParameterGrid objects."""
        grid = ParameterGrid(
            core_params=minimal_params,
            parameter_ranges=sample_parameter_ranges,
            theoretical_spectra=sample_theoretical_spectra,
        )

        # Different type should not be equal
        assert grid != "not a grid"
        assert grid != 42
        assert grid != None
