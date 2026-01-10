"""
Tests for LikelihoodResult class.

This module contains unit tests for the LikelihoodResult class functionality
including initialization, best-fit extraction, confidence intervals, and I/O.
"""

import numpy as np
import pytest

from picslike import LikelihoodResult


class TestLikelihoodResultInit:
    """Test suite for LikelihoodResult initialization."""

    def test_basic_initialization(
        self,
        sample_parameter_grid,
        sample_chi_squared_values,
        sample_log_likelihood_values,
    ):
        """Test basic initialization of LikelihoodResult."""
        result = LikelihoodResult(
            parameter_grid=sample_parameter_grid,
            chi_squared_values=sample_chi_squared_values,
            log_likelihood_values=sample_log_likelihood_values,
        )

        assert result.parameter_grid is not None
        assert len(result.chi_squared_values) == sample_parameter_grid.get_total_points()
        assert (
            len(result.log_likelihood_values) == sample_parameter_grid.get_total_points()
        )
        assert len(result.likelihood_values) == sample_parameter_grid.get_total_points()

    def test_chi_squared_dimension_mismatch(
        self,
        sample_parameter_grid,
        sample_log_likelihood_values,
    ):
        """Test error handling for mismatched chi-squared dimensions."""
        wrong_size_chi2 = np.zeros(5)  # Wrong size

        with pytest.raises(ValueError, match="Chi-squared array length"):
            LikelihoodResult(
                parameter_grid=sample_parameter_grid,
                chi_squared_values=wrong_size_chi2,
                log_likelihood_values=sample_log_likelihood_values,
            )

    def test_log_likelihood_dimension_mismatch(
        self,
        sample_parameter_grid,
        sample_chi_squared_values,
    ):
        """Test error handling for mismatched log-likelihood dimensions."""
        wrong_size_loglike = np.zeros(5)  # Wrong size

        with pytest.raises(ValueError, match="Log-likelihood array length"):
            LikelihoodResult(
                parameter_grid=sample_parameter_grid,
                chi_squared_values=sample_chi_squared_values,
                log_likelihood_values=wrong_size_loglike,
            )

    def test_likelihood_values_computed(self, sample_likelihood_result):
        """Test that likelihood values are computed correctly."""
        # Likelihood should be positive and normalized
        assert np.all(sample_likelihood_result.likelihood_values >= 0)

        # Maximum likelihood should be 1 (after normalization)
        assert np.isclose(np.max(sample_likelihood_result.likelihood_values), 1.0)

    def test_data_copying(
        self,
        sample_parameter_grid,
        sample_chi_squared_values,
        sample_log_likelihood_values,
    ):
        """Test that input arrays are deep copied."""
        original_chi2 = sample_chi_squared_values.copy()
        result = LikelihoodResult(
            parameter_grid=sample_parameter_grid,
            chi_squared_values=sample_chi_squared_values,
            log_likelihood_values=sample_log_likelihood_values,
        )

        # Modify original
        sample_chi_squared_values[0] = -999

        # Result should not be affected
        assert result.chi_squared_values[0] == original_chi2[0]


class TestBestFit:
    """Test suite for best-fit parameter extraction."""

    def test_get_best_fit_returns_dict(self, sample_likelihood_result):
        """Test that get_best_fit returns a dictionary."""
        best_fit = sample_likelihood_result.get_best_fit()
        assert isinstance(best_fit, dict)

    def test_best_fit_contains_all_parameters(self, sample_likelihood_result):
        """Test that best-fit contains all parameter names."""
        best_fit = sample_likelihood_result.get_best_fit()
        expected_params = ["omega_b", "omega_c"]

        for param in expected_params:
            assert param in best_fit

    def test_best_fit_at_minimum_chi_squared(
        self,
        sample_parameter_grid,
        sample_chi_squared_values,
        sample_log_likelihood_values,
    ):
        """Test that best-fit corresponds to minimum chi-squared."""
        result = LikelihoodResult(
            parameter_grid=sample_parameter_grid,
            chi_squared_values=sample_chi_squared_values,
            log_likelihood_values=sample_log_likelihood_values,
        )

        best_fit = result.get_best_fit()

        # For our test data, best-fit should be at fiducial (0.022, 0.12)
        assert np.isclose(best_fit["omega_b"], 0.022, rtol=1e-10)
        assert np.isclose(best_fit["omega_c"], 0.12, rtol=1e-10)

    def test_get_chi_squared_minimum(self, sample_likelihood_result):
        """Test chi-squared minimum extraction."""
        chi2_min = sample_likelihood_result.get_chi_squared_minimum()

        assert isinstance(chi2_min, float)
        assert chi2_min == np.min(sample_likelihood_result.chi_squared_values)

    def test_get_maximum_likelihood(self, sample_likelihood_result):
        """Test maximum likelihood extraction."""
        max_like = sample_likelihood_result.get_maximum_likelihood()

        assert isinstance(max_like, float)
        assert max_like == np.max(sample_likelihood_result.likelihood_values)


class TestMarginalizedLikelihood:
    """Test suite for marginalized likelihood computation."""

    def test_marginalized_likelihood_valid_parameter(self, sample_likelihood_result):
        """Test marginalized likelihood for valid parameter."""
        marg = sample_likelihood_result.get_marginalized_likelihood("omega_b")

        assert isinstance(marg, np.ndarray)
        # Should be normalized to sum to 1
        assert np.isclose(np.sum(marg), 1.0)

    def test_marginalized_likelihood_invalid_parameter(self, sample_likelihood_result):
        """Test error handling for invalid parameter name."""
        with pytest.raises(ValueError, match="not found in grid"):
            sample_likelihood_result.get_marginalized_likelihood("invalid_param")

    def test_marginalized_likelihood_shape(self, sample_likelihood_result):
        """Test that marginalized likelihood has correct shape."""
        marg_omega_b = sample_likelihood_result.get_marginalized_likelihood("omega_b")
        marg_omega_c = sample_likelihood_result.get_marginalized_likelihood("omega_c")

        # Should have same length as parameter values
        assert len(marg_omega_b) == 3  # 3 omega_b values
        assert len(marg_omega_c) == 3  # 3 omega_c values

    def test_marginalized_likelihood_positive(self, sample_likelihood_result):
        """Test that marginalized likelihood is non-negative."""
        marg = sample_likelihood_result.get_marginalized_likelihood("omega_b")
        assert np.all(marg >= 0)


class TestConfidenceIntervals:
    """Test suite for confidence interval computation."""

    def test_confidence_intervals_returns_dict(self, sample_likelihood_result):
        """Test that get_confidence_intervals returns a dictionary."""
        intervals = sample_likelihood_result.get_confidence_intervals()
        assert isinstance(intervals, dict)

    def test_confidence_intervals_contain_all_parameters(self, sample_likelihood_result):
        """Test that intervals are computed for all parameters."""
        intervals = sample_likelihood_result.get_confidence_intervals()
        expected_params = ["omega_b", "omega_c"]

        for param in expected_params:
            assert param in intervals

    def test_confidence_intervals_are_tuples(self, sample_likelihood_result):
        """Test that each interval is a tuple of (lower, upper)."""
        intervals = sample_likelihood_result.get_confidence_intervals()

        for param, interval in intervals.items():
            assert isinstance(interval, tuple)
            assert len(interval) == 2
            lower, upper = interval
            assert lower <= upper

    def test_confidence_intervals_default_68(self, sample_likelihood_result):
        """Test default 68% confidence level."""
        intervals = sample_likelihood_result.get_confidence_intervals()

        # Just verify it runs without specifying confidence level
        assert intervals is not None

    def test_confidence_intervals_95(self, sample_likelihood_result):
        """Test 95% confidence intervals."""
        intervals_68 = sample_likelihood_result.get_confidence_intervals(0.68)
        intervals_95 = sample_likelihood_result.get_confidence_intervals(0.95)

        # 95% intervals should be wider than 68%
        for param in intervals_68:
            lower_68, upper_68 = intervals_68[param]
            lower_95, upper_95 = intervals_95[param]

            width_68 = upper_68 - lower_68
            width_95 = upper_95 - lower_95

            assert width_95 >= width_68


class TestSaveLoad:
    """Test suite for saving and loading results."""

    def test_save_creates_file(self, sample_likelihood_result, temp_output_dir):
        """Test that save creates a file."""
        output_path = temp_output_dir / "test_results.pkl"
        sample_likelihood_result.save(output_path)

        assert output_path.exists()

    def test_load_restores_result(self, sample_likelihood_result, temp_output_dir):
        """Test that load restores the result correctly."""
        output_path = temp_output_dir / "test_results.pkl"
        sample_likelihood_result.save(output_path)

        loaded = LikelihoodResult.load(output_path)

        np.testing.assert_array_equal(
            loaded.chi_squared_values,
            sample_likelihood_result.chi_squared_values,
        )
        np.testing.assert_array_equal(
            loaded.log_likelihood_values,
            sample_likelihood_result.log_likelihood_values,
        )

    def test_load_nonexistent_file(self, temp_output_dir):
        """Test error handling for loading non-existent file."""
        nonexistent_path = temp_output_dir / "nonexistent.pkl"

        with pytest.raises(FileNotFoundError):
            LikelihoodResult.load(nonexistent_path)

    def test_save_creates_parent_directories(
        self, sample_likelihood_result, temp_output_dir
    ):
        """Test that save creates parent directories if needed."""
        output_path = temp_output_dir / "subdir" / "nested" / "results.pkl"
        sample_likelihood_result.save(output_path)

        assert output_path.exists()

    def test_roundtrip_preserves_best_fit(
        self, sample_likelihood_result, temp_output_dir
    ):
        """Test that save/load preserves best-fit computation."""
        output_path = temp_output_dir / "test_results.pkl"
        original_best_fit = sample_likelihood_result.get_best_fit()

        sample_likelihood_result.save(output_path)
        loaded = LikelihoodResult.load(output_path)
        loaded_best_fit = loaded.get_best_fit()

        assert original_best_fit == loaded_best_fit


class TestSummaryStatistics:
    """Test suite for summary statistics."""

    def test_get_summary_statistics_returns_dict(self, sample_likelihood_result):
        """Test that get_summary_statistics returns a dictionary."""
        summary = sample_likelihood_result.get_summary_statistics()
        assert isinstance(summary, dict)

    def test_summary_contains_required_keys(self, sample_likelihood_result):
        """Test that summary contains all required keys."""
        summary = sample_likelihood_result.get_summary_statistics()

        required_keys = [
            "best_fit_parameters",
            "chi_squared_minimum",
            "maximum_likelihood",
            "confidence_intervals_68",
            "confidence_intervals_95",
            "total_parameter_points",
            "parameter_names",
        ]

        for key in required_keys:
            assert key in summary

    def test_summary_parameter_count(self, sample_likelihood_result):
        """Test that summary has correct parameter count."""
        summary = sample_likelihood_result.get_summary_statistics()

        assert summary["total_parameter_points"] == 9  # 3x3 grid


class TestRepresentation:
    """Test suite for string representation."""

    def test_repr_contains_info(self, sample_likelihood_result):
        """Test that __repr__ contains useful information."""
        repr_str = repr(sample_likelihood_result)

        assert "LikelihoodResult" in repr_str
        assert "parameters" in repr_str
        assert "grid_points" in repr_str
        assert "χ²_min" in repr_str

    def test_repr_is_string(self, sample_likelihood_result):
        """Test that __repr__ returns a string."""
        repr_str = repr(sample_likelihood_result)
        assert isinstance(repr_str, str)


class TestSingleParameter:
    """Test suite for single parameter cases."""

    def test_single_parameter_grid(self, single_param_grid):
        """Test LikelihoodResult with single parameter."""
        _ = single_param_grid.get_total_points()
        chi2 = np.array([10.0, 5.0, 10.0])  # Minimum at center
        log_like = -0.5 * chi2

        result = LikelihoodResult(
            parameter_grid=single_param_grid,
            chi_squared_values=chi2,
            log_likelihood_values=log_like,
        )

        best_fit = result.get_best_fit()
        assert "amplitude" in best_fit
        assert np.isclose(best_fit["amplitude"], 1.0)

    def test_single_parameter_marginalized(self, single_param_grid):
        """Test marginalized likelihood for single parameter."""
        chi2 = np.array([10.0, 5.0, 10.0])
        log_like = -0.5 * chi2

        result = LikelihoodResult(
            parameter_grid=single_param_grid,
            chi_squared_values=chi2,
            log_likelihood_values=log_like,
        )

        marg = result.get_marginalized_likelihood("amplitude")

        # For single parameter, marginalized = normalized likelihood
        assert np.isclose(np.sum(marg), 1.0)
        assert len(marg) == 3


class TestEdgeCases:
    """Test suite for edge cases."""

    def test_uniform_chi_squared(
        self,
        sample_parameter_grid,
    ):
        """Test handling of uniform chi-squared (flat likelihood)."""
        n_points = sample_parameter_grid.get_total_points()
        chi2 = np.ones(n_points) * 100  # Uniform chi-squared
        log_like = -0.5 * chi2

        result = LikelihoodResult(
            parameter_grid=sample_parameter_grid,
            chi_squared_values=chi2,
            log_likelihood_values=log_like,
        )

        # Should still work - best fit is first point
        best_fit = result.get_best_fit()
        assert best_fit is not None

    def test_zero_likelihood_confidence_interval(
        self,
        sample_parameter_grid,
    ):
        """Test confidence interval when likelihood sums to zero."""
        n_points = sample_parameter_grid.get_total_points()
        # Create extremely negative log-likelihood that results in zero likelihood
        chi2 = np.ones(n_points) * 1e10  # Very large chi-squared
        log_like = -0.5 * chi2  # Very negative log-likelihood

        result = LikelihoodResult(
            parameter_grid=sample_parameter_grid,
            chi_squared_values=chi2,
            log_likelihood_values=log_like,
        )

        # The _compute_parameter_interval handles zero likelihood case
        # by returning full parameter range
        intervals = result.get_confidence_intervals()
        assert intervals is not None
        for param, interval in intervals.items():
            assert interval[0] <= interval[1]

    def test_very_large_chi_squared(
        self,
        sample_parameter_grid,
    ):
        """Test handling of very large chi-squared values."""
        n_points = sample_parameter_grid.get_total_points()
        chi2 = np.arange(n_points) * 1000 + 1e6  # Very large values
        log_like = -0.5 * chi2

        result = LikelihoodResult(
            parameter_grid=sample_parameter_grid,
            chi_squared_values=chi2,
            log_likelihood_values=log_like,
        )

        # Should handle numerical stability
        assert np.all(np.isfinite(result.likelihood_values))
        assert np.max(result.likelihood_values) == 1.0  # Normalized

    def test_negative_chi_squared_values(
        self,
        sample_parameter_grid,
    ):
        """Test that negative chi-squared values are accepted (not validated)."""
        # Note: Negative chi2 is unphysical but shouldn't cause crash
        n_points = sample_parameter_grid.get_total_points()
        chi2 = np.arange(n_points) - n_points / 2  # Some negative values
        log_like = -0.5 * chi2

        result = LikelihoodResult(
            parameter_grid=sample_parameter_grid,
            chi_squared_values=chi2,
            log_likelihood_values=log_like,
        )

        # Should work without error
        best_fit = result.get_best_fit()
        assert best_fit is not None
