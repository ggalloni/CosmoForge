"""
Tests for PICSLike class.

This module contains unit and integration tests for the PICSLike class
functionality including initialization, signal matrix computation,
likelihood computation, and result management.

Note: Many PICSLike operations require MPI. These tests are designed to
run in single-process mode (mpirun -n 1) or with MPI mocked where necessary.
"""

import numpy as np
import pytest

from picslike import LikelihoodResult, PICSLike


class TestPICSLikeInit:
    """Test suite for PICSLike initialization."""

    def test_initialization_from_config(self, config_path):
        """Test PICSLike initialization from config file."""
        picslike = PICSLike(params_file=config_path)

        assert picslike is not None
        assert picslike.rank == 0  # Single process
        assert picslike.size == 1  # Single process
        assert picslike.maps1 is None  # Not loaded yet
        assert picslike.parameter_grid is None  # Not set up yet

    def test_initialization_attributes(self, config_path):
        """Test that initialization sets expected attributes."""
        picslike = PICSLike(params_file=config_path)

        assert hasattr(picslike, "comm")
        assert hasattr(picslike, "rank")
        assert hasattr(picslike, "size")
        assert hasattr(picslike, "maps1")
        assert hasattr(picslike, "maps2")
        assert hasattr(picslike, "parameter_grid")
        assert hasattr(picslike, "likelihood_result")
        assert hasattr(picslike, "simulation_index")

    def test_simulation_index_default(self, config_path):
        """Test default simulation index is 0."""
        picslike = PICSLike(params_file=config_path)
        assert picslike.simulation_index == 0


class TestSetSimulationIndex:
    """Test suite for set_simulation_index method."""

    def test_set_valid_simulation_index(self, config_path):
        """Test setting a valid simulation index."""
        picslike = PICSLike(params_file=config_path)
        picslike.setup_fields()
        picslike.setup_geometry()
        picslike.setup_maps()

        # Should work for valid index
        picslike.set_simulation_index(5)
        assert picslike.simulation_index == 5

    def test_set_simulation_index_out_of_range(self, config_path):
        """Test error handling for out-of-range simulation index."""
        picslike = PICSLike(params_file=config_path)
        picslike.setup_fields()
        picslike.setup_geometry()
        picslike.setup_maps()

        # Should raise ValueError for invalid index
        with pytest.raises(ValueError, match="out of range"):
            picslike.set_simulation_index(9999)

    def test_set_negative_simulation_index(self, config_path):
        """Test error handling for negative simulation index."""
        picslike = PICSLike(params_file=config_path)
        picslike.setup_fields()
        picslike.setup_geometry()
        picslike.setup_maps()

        with pytest.raises(ValueError, match="out of range"):
            picslike.set_simulation_index(-1)


class TestSetupParameterGrid:
    """Test suite for parameter grid setup."""

    def test_setup_parameter_grid(self, config_path):
        """Test parameter grid setup from config."""
        picslike = PICSLike(params_file=config_path)
        picslike.setup_parameter_grid()

        assert picslike.parameter_grid is not None
        assert picslike.parameter_grid.get_total_points() > 0

    def test_parameter_names_set(self, config_path):
        """Test that parameter names are extracted from config."""
        picslike = PICSLike(params_file=config_path)
        picslike.setup_parameter_grid()

        # Check that parameter names are available
        assert hasattr(picslike, "parameter_names")

    def test_parameter_ranges_set(self, config_path):
        """Test that parameter ranges are set from config."""
        picslike = PICSLike(params_file=config_path)
        picslike.setup_parameter_grid()

        assert hasattr(picslike, "parameter_ranges")
        assert len(picslike.parameter_ranges) > 0


class TestSetupMaps:
    """Test suite for map setup."""

    def test_setup_maps_loads_data(self, config_path):
        """Test that setup_maps loads map data."""
        picslike = PICSLike(params_file=config_path)
        picslike.setup_fields()
        picslike.setup_geometry()
        picslike.setup_maps()

        assert picslike.maps1 is not None
        assert picslike.maps1.shape[1] == picslike.params.nsims

    def test_setup_maps_before_geometry_raises(self, config_path):
        """Test that setup_maps fails if geometry not set up."""
        picslike = PICSLike(params_file=config_path)

        with pytest.raises(ValueError, match="Pixel information not available"):
            picslike.setup_maps()


class TestComputeSignalMatrix:
    """Test suite for signal matrix computation (requires full pipeline)."""

    def test_compute_signal_matrix_requires_covariance(self, fast_config_path):
        """Test that compute_signal_matrix requires covariance setup."""
        picslike = PICSLike(params_file=fast_config_path)
        picslike.setup_fields()
        picslike.setup_geometry()
        picslike.setup_parameter_grid()

        # Without covariance matrices, should raise AttributeError or ValueError
        param_point = list(picslike.parameter_grid.grid_points)[0]
        with pytest.raises((ValueError, AttributeError)):
            picslike.compute_signal_matrix(param_point)

    def test_compute_signal_matrix_shape(self, fast_config_path):
        """Test that signal matrix has correct shape."""
        picslike = PICSLike(params_file=fast_config_path)
        picslike.setup_fields()
        picslike.setup_geometry()
        picslike.setup_covariance_matrices()
        picslike.setup_cls()
        picslike.setup_beams()
        picslike.setup_parameter_grid()

        param_point = list(picslike.parameter_grid.grid_points)[0]
        signal_matrix = picslike.compute_signal_matrix(param_point)

        # Signal matrix should be square
        assert signal_matrix.shape[0] == signal_matrix.shape[1]

        # Should match noise covariance shape
        assert signal_matrix.shape == picslike.NCov1.shape


class TestPrepareCovariance:
    """Test suite for covariance preparation (requires full pipeline)."""

    def test_prepare_covariance_creates_inverse(self, fast_config_path):
        """Test that prepare_covariance creates inverse covariance."""
        picslike = PICSLike(params_file=fast_config_path)
        picslike.setup_fields()
        picslike.setup_geometry()
        picslike.setup_covariance_matrices()
        picslike.setup_cls()
        picslike.setup_beams()
        picslike.setup_parameter_grid()

        param_point = list(picslike.parameter_grid.grid_points)[0]
        picslike.compute_signal_matrix(param_point)
        picslike.prepare_covariance_matrix()

        assert hasattr(picslike, "invCov")
        assert picslike.invCov is not None


class TestGetters:
    """Test suite for getter methods before computation."""

    def test_get_chi_squared_before_compute_raises(self, config_path):
        """Test that get_chi_squared raises error if not computed."""
        picslike = PICSLike(params_file=config_path)

        with pytest.raises(RuntimeError, match="Likelihood not computed"):
            picslike.get_chi_squared()

    def test_get_log_likelihood_before_compute_raises(self, config_path):
        """Test that get_log_likelihood raises error if not computed."""
        picslike = PICSLike(params_file=config_path)

        with pytest.raises(RuntimeError, match="Likelihood not computed"):
            picslike.get_log_likelihood()

    def test_get_best_fit_before_compute_raises(self, config_path):
        """Test that get_best_fit raises error if not computed."""
        picslike = PICSLike(params_file=config_path)

        with pytest.raises(RuntimeError, match="Likelihood not computed"):
            picslike.get_best_fit()

    def test_get_simulation_results_before_compute_raises(self, config_path):
        """Test that get_simulation_results raises error if not computed."""
        picslike = PICSLike(params_file=config_path)

        with pytest.raises(RuntimeError, match="Simulation results not available"):
            picslike.get_simulation_results()

    def test_get_mean_likelihood_result_before_compute_raises(self, config_path):
        """Test that get_mean_likelihood_result raises error if not computed."""
        picslike = PICSLike(params_file=config_path)

        with pytest.raises(RuntimeError, match="Likelihood not computed"):
            picslike.get_mean_likelihood_result()


class TestSaveResults:
    """Test suite for save_results method."""

    def test_save_results_before_compute_raises(self, config_path, temp_output_dir):
        """Test that save_results raises error if not computed."""
        picslike = PICSLike(params_file=config_path)

        with pytest.raises(RuntimeError, match="No results to save"):
            picslike.save_results(str(temp_output_dir / "results.pkl"))


class TestComputeMeanLikelihoodResult:
    """Test suite for mean likelihood computation."""

    def test_compute_mean_empty_list_raises(self, config_path):
        """Test that empty results list raises error."""
        picslike = PICSLike(params_file=config_path)

        with pytest.raises(ValueError, match="No simulation results provided"):
            picslike._compute_mean_likelihood_result([])

    def test_compute_mean_single_result(
        self,
        config_path,
        sample_likelihood_result,
    ):
        """Test mean computation with single result."""
        picslike = PICSLike(params_file=config_path)

        mean_result = picslike._compute_mean_likelihood_result([sample_likelihood_result])

        # Mean of single result should equal the result
        np.testing.assert_array_equal(
            mean_result.chi_squared_values,
            sample_likelihood_result.chi_squared_values,
        )

    def test_compute_mean_multiple_results(
        self,
        config_path,
        sample_parameter_grid,
    ):
        """Test mean computation with multiple results."""
        picslike = PICSLike(params_file=config_path)

        n_points = sample_parameter_grid.get_total_points()

        # Create two results with different chi-squared
        chi2_1 = np.ones(n_points) * 100
        chi2_2 = np.ones(n_points) * 200

        result1 = LikelihoodResult(
            parameter_grid=sample_parameter_grid,
            chi_squared_values=chi2_1,
            log_likelihood_values=-0.5 * chi2_1,
        )
        result2 = LikelihoodResult(
            parameter_grid=sample_parameter_grid,
            chi_squared_values=chi2_2,
            log_likelihood_values=-0.5 * chi2_2,
        )

        mean_result = picslike._compute_mean_likelihood_result([result1, result2])

        # Mean should be 150
        expected_mean = np.ones(n_points) * 150
        np.testing.assert_array_almost_equal(
            mean_result.chi_squared_values,
            expected_mean,
        )


class TestIntegration:
    """Integration tests for full pipeline (requires test data).

    These tests use fast_config with a 2x2 grid and 10 simulations for speed.
    """

    def test_full_pipeline_setup(self, fast_config_path):
        """Test full pipeline setup without compute."""
        picslike = PICSLike(params_file=fast_config_path)

        # Setup pipeline
        picslike.setup_parameter_grid()
        picslike.setup_fields()
        picslike.setup_geometry()
        picslike.setup_covariance_matrices()
        picslike.setup_cls()
        picslike.setup_beams()
        picslike.setup_maps()

        # Verify setup completed
        assert picslike.parameter_grid is not None
        assert picslike.maps1 is not None
        assert picslike.NCov1 is not None
        assert picslike.collection is not None

    def test_single_point_likelihood(self, fast_config_path):
        """Test likelihood computation at a single parameter point."""
        picslike = PICSLike(params_file=fast_config_path)

        # Setup pipeline
        picslike.setup_parameter_grid()
        picslike.setup_fields()
        picslike.setup_geometry()
        picslike.setup_covariance_matrices()
        picslike.setup_cls()
        picslike.setup_beams()
        picslike.setup_maps()

        # Compute likelihood at first point
        param_point = list(picslike.parameter_grid.grid_points)[0]
        chi2, log_like = picslike._compute_likelihood_point(param_point)

        # Should return arrays (one per simulation)
        assert len(chi2) == picslike.params.nsims
        assert len(log_like) == picslike.params.nsims

        # Chi-squared should be positive
        assert np.all(chi2 > 0)

        # Log-likelihood should be negative
        assert np.all(log_like < 0)

        # Log-likelihood = -0.5 * chi2
        np.testing.assert_array_almost_equal(log_like, -0.5 * chi2)

    def test_compute_and_getters(self, fast_config_path):
        """Test full compute method and getter methods."""
        picslike = PICSLike(params_file=fast_config_path)

        # Setup pipeline
        picslike.setup_parameter_grid()
        picslike.setup_fields()
        picslike.setup_geometry()
        picslike.setup_covariance_matrices()
        picslike.setup_cls()
        picslike.setup_beams()
        picslike.setup_maps()

        # Run compute
        picslike.compute()

        # Test that likelihood_result is set
        assert picslike.likelihood_result is not None

        # Test getters after computation
        chi2 = picslike.get_chi_squared()
        assert chi2 is not None
        assert len(chi2) == picslike.parameter_grid.get_total_points()

        log_like = picslike.get_log_likelihood()
        assert log_like is not None
        assert len(log_like) == picslike.parameter_grid.get_total_points()

        best_fit = picslike.get_best_fit()
        assert isinstance(best_fit, dict)
        assert len(best_fit) > 0

        # Test simulation results getter
        sim_results = picslike.get_simulation_results()
        assert sim_results is not None
        assert len(sim_results) == picslike.params.nsims

        # Test mean likelihood result getter
        mean_result = picslike.get_mean_likelihood_result()
        assert mean_result is not None
        assert mean_result == picslike.likelihood_result

    def test_save_results(self, fast_config_path, temp_output_dir):
        """Test saving likelihood results to file."""
        picslike = PICSLike(params_file=fast_config_path)

        # Setup and run pipeline
        picslike.setup_parameter_grid()
        picslike.setup_fields()
        picslike.setup_geometry()
        picslike.setup_covariance_matrices()
        picslike.setup_cls()
        picslike.setup_beams()
        picslike.setup_maps()
        picslike.compute()

        # Save results
        output_path = temp_output_dir / "test_results.npz"
        picslike.save_results(str(output_path))

        # Verify main file was created
        assert output_path.exists()

        # Verify simulation result files were created
        n_sims = picslike.params.nsims
        for i in range(n_sims):
            sim_path = temp_output_dir / f"test_results_sim_{i:02d}.npz"
            assert sim_path.exists()

    def test_run_method(self, fast_config_path):
        """Test the full run() pipeline method."""
        picslike = PICSLike(params_file=fast_config_path)

        # Run the full pipeline
        picslike.run()

        # Verify results are computed
        assert picslike.likelihood_result is not None
        assert picslike.simulation_results is not None
        assert picslike.parameter_grid is not None


class TestMPIDistribution:
    """Test suite for MPI point distribution."""

    def test_points_distribution_single_process(self, config_path):
        """Test that single process gets all points."""
        picslike = PICSLike(params_file=config_path)
        picslike.setup_parameter_grid()

        # In single process mode, rank=0, size=1
        points = picslike.parameter_grid.get_points_for_process(0, 1)

        assert len(points) == picslike.parameter_grid.get_total_points()

    def test_points_distribution_multiple_processes_simulation(self, config_path):
        """Test simulated multi-process distribution."""
        picslike = PICSLike(params_file=config_path)
        picslike.setup_parameter_grid()

        # Simulate 4 processes
        size = 4
        all_points = []

        for rank in range(size):
            points = picslike.parameter_grid.get_points_for_process(rank, size)
            all_points.extend(points)

        # All points should be covered exactly once
        total_points = picslike.parameter_grid.get_total_points()
        assert len(all_points) == total_points
        assert len(set(all_points)) == total_points  # No duplicates
