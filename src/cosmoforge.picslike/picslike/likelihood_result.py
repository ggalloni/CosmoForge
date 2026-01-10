"""
Results container for pixel-based likelihood analysis.

This module provides the LikelihoodResult class for storing, managing, and
analyzing results from pixel-based likelihood computations. It includes
functionality for result storage, best-fit parameter extraction, confidence
interval computation, and result visualization.

Classes
-------
LikelihoodResult
    Container for likelihood computation results and analysis tools.

Notes
-----
The LikelihoodResult class provides a comprehensive interface for working
with likelihood analysis results, including statistical analysis tools
and visualization capabilities for understanding parameter constraints.
"""

from __future__ import annotations

import pickle
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from .parameter_grid import ParameterGrid


class LikelihoodResult:
    """
    Container for pixel-based likelihood computation results.

    This class stores and manages results from likelihood analysis, providing
    methods for result analysis, best-fit parameter extraction, confidence
    interval computation, and result persistence.

    Parameters
    ----------
    parameter_grid : ParameterGrid
        Parameter grid used in the likelihood computation.
    chi_squared_values : numpy.ndarray
        Chi-squared values for each parameter grid point.
    log_likelihood_values : numpy.ndarray
        Log-likelihood values for each parameter grid point.

    Attributes
    ----------
    parameter_grid : ParameterGrid
        Parameter grid used in the computation.
    chi_squared_values : numpy.ndarray
        Array of chi-squared values.
    log_likelihood_values : numpy.ndarray
        Array of log-likelihood values.
    likelihood_values : numpy.ndarray
        Array of likelihood values (exponential of log-likelihood).

    Examples
    --------
    Basic result analysis:

    >>> result = LikelihoodResult(grid, chi2_values, log_like_values)
    >>> best_fit = result.get_best_fit()
    >>> confidence_intervals = result.get_confidence_intervals()
    >>> result.save("analysis_results.pkl")

    Statistical analysis:

    >>> # Get 68% confidence intervals
    >>> intervals_68 = result.get_confidence_intervals(confidence_level=0.68)
    >>> # Get marginalized likelihood for specific parameter
    >>> marg_like = result.get_marginalized_likelihood('omega_b')

    Notes
    -----
    The class assumes that likelihood values correspond to parameter grid
    points in the same order. It provides both chi-squared and likelihood
    perspectives on the same underlying computation.
    """

    def __init__(
        self,
        parameter_grid: ParameterGrid,
        chi_squared_values: np.ndarray,
        log_likelihood_values: np.ndarray,
    ) -> None:
        """
        Initialize likelihood result container.

        Parameters
        ----------
        parameter_grid : ParameterGrid
            Parameter grid used in the likelihood computation.
        chi_squared_values : numpy.ndarray
            Chi-squared values for each parameter grid point.
        log_likelihood_values : numpy.ndarray
            Log-likelihood values for each parameter grid point.

        Raises
        ------
        ValueError
            If array dimensions are inconsistent with parameter grid.
        """
        # Validate input dimensions
        n_points = parameter_grid.get_total_points()
        if len(chi_squared_values) != n_points:
            msg = (
                f"Chi-squared array length {len(chi_squared_values)} "
                f"doesn't match grid size {n_points}"
            )
            raise ValueError(msg)

        if len(log_likelihood_values) != n_points:
            msg = (
                f"Log-likelihood array length {len(log_likelihood_values)} "
                f"doesn't match grid size {n_points}"
            )
            raise ValueError(msg)

        self.parameter_grid = parameter_grid
        self.chi_squared_values = deepcopy(chi_squared_values)
        self.log_likelihood_values = deepcopy(log_likelihood_values)

        # Compute likelihood values (with numerical stability)
        self.likelihood_values = self._compute_likelihood_values()

    def _compute_likelihood_values(self) -> np.ndarray:
        """
        Compute likelihood values from log-likelihood with numerical stability.

        Returns
        -------
        likelihood_values : numpy.ndarray
            Likelihood values computed from log-likelihood.

        Notes
        -----
        Uses numerical tricks to avoid overflow/underflow in exponential
        computation by subtracting the maximum log-likelihood before
        exponentiating.
        """
        # Subtract maximum for numerical stability
        max_log_like = np.max(self.log_likelihood_values)
        log_like_normalized = self.log_likelihood_values - max_log_like
        return np.exp(log_like_normalized)

    def get_best_fit(self) -> dict[str, Any]:
        """
        Get best-fit parameter values.

        Returns
        -------
        best_fit_params : dict[str, Any]
            Dictionary mapping parameter names to their best-fit values.

        Notes
        -----
        Returns the parameter combination with the highest likelihood
        (lowest chi-squared value). In case of ties, returns the first
        occurrence.
        """
        # Find index of maximum likelihood (minimum chi-squared)
        best_fit_index = np.argmax(self.likelihood_values)

        # Get corresponding parameter point
        best_fit_point = self.parameter_grid.grid_points[best_fit_index]

        # Convert to named dictionary
        return self.parameter_grid.get_parameter_dict(best_fit_point)

    def get_chi_squared_minimum(self) -> float:
        """
        Get minimum chi-squared value.

        Returns
        -------
        chi_squared_min : float
            Minimum chi-squared value in the analysis.
        """
        return float(np.min(self.chi_squared_values))

    def get_maximum_likelihood(self) -> float:
        """
        Get maximum likelihood value.

        Returns
        -------
        likelihood_max : float
            Maximum likelihood value in the analysis.
        """
        return float(np.max(self.likelihood_values))

    def get_confidence_intervals(
        self,
        confidence_level: float = 0.68,
    ) -> dict[str, tuple[float, float]]:
        """
        Compute confidence intervals for each parameter.

        Parameters
        ----------
        confidence_level : float, optional
            Confidence level for interval computation (default: 0.68 for 1σ).

        Returns
        -------
        confidence_intervals : dict[str, tuple[float, float]]
            Dictionary mapping parameter names to (lower, upper) confidence bounds.

        Notes
        -----
        Computes confidence intervals by finding parameter ranges that contain
        the specified fraction of the total likelihood. Uses marginalized
        likelihood distributions for each parameter.
        """
        intervals = {}

        for param_name in self.parameter_grid.parameter_names:
            # Get marginalized likelihood for this parameter
            marg_likelihood = self.get_marginalized_likelihood(param_name)
            param_values = self.parameter_grid.parameter_ranges[param_name]

            # Find confidence interval
            lower, upper = self._compute_parameter_interval(
                param_values, marg_likelihood, confidence_level
            )
            intervals[param_name] = (lower, upper)

        return intervals

    def get_marginalized_likelihood(self, parameter_name: str) -> np.ndarray:
        """
        Get marginalized likelihood for a specific parameter.

        Parameters
        ----------
        parameter_name : str
            Name of the parameter to marginalize over.

        Returns
        -------
        marginalized_likelihood : numpy.ndarray
            Marginalized likelihood distribution for the parameter.

        Raises
        ------
        ValueError
            If parameter name is not found in the grid.

        Notes
        -----
        Computes the marginalized likelihood by integrating (summing) over
        all other parameters in the grid. The result is normalized to
        unit area under the curve.
        """
        if parameter_name not in self.parameter_grid.parameter_names:
            msg = f"Parameter '{parameter_name}' not found in grid"
            raise ValueError(msg)

        # Get parameter index
        param_index = self.parameter_grid.parameter_names.index(parameter_name)
        param_values = self.parameter_grid.parameter_ranges[parameter_name]

        # Initialize marginalized likelihood array
        marginalized = np.zeros(len(param_values))

        # Sum likelihood over all other parameters
        for i, param_value in enumerate(param_values):
            # Find all grid points with this parameter value
            matching_indices = []
            for j, point in enumerate(self.parameter_grid.grid_points):
                if point[param_index] == param_value:
                    matching_indices.append(j)

            # Sum likelihood over matching points
            if matching_indices:
                marginalized[i] = np.sum(self.likelihood_values[matching_indices])

        # Normalize to unit area
        if np.sum(marginalized) > 0:
            marginalized /= np.sum(marginalized)

        return marginalized

    def _compute_parameter_interval(
        self,
        param_values: np.ndarray,
        likelihood: np.ndarray,
        confidence_level: float,
    ) -> tuple[float, float]:
        """
        Compute confidence interval for a parameter from its likelihood.

        Parameters
        ----------
        param_values : numpy.ndarray
            Parameter values.
        likelihood : numpy.ndarray
            Likelihood values for each parameter value.
        confidence_level : float
            Desired confidence level (e.g., 0.68 for 1σ).

        Returns
        -------
        lower_bound : float
            Lower confidence bound.
        upper_bound : float
            Upper confidence bound.

        Notes
        -----
        Finds the smallest interval containing the specified fraction of
        the likelihood. Uses a simple threshold-based approach.
        """
        # Normalize likelihood
        if np.sum(likelihood) == 0:
            return float(param_values[0]), float(param_values[-1])

        likelihood_norm = likelihood / np.sum(likelihood)

        # Find maximum likelihood point
        max_index = np.argmax(likelihood_norm)

        # Expand outward from maximum until we reach desired confidence level
        indices = [max_index]
        total_prob = likelihood_norm[max_index]

        while total_prob < confidence_level and len(indices) < len(likelihood_norm):
            # Find next highest likelihood points on either side
            left_candidate = max_index - len(indices) // 2 - 1
            right_candidate = max_index + len(indices) // 2 + 1

            # Add valid candidates
            if left_candidate >= 0 and left_candidate not in indices:
                indices.append(left_candidate)
                total_prob += likelihood_norm[left_candidate]

            if (
                right_candidate < len(likelihood_norm)
                and right_candidate not in indices
                and total_prob < confidence_level
            ):
                indices.append(right_candidate)
                total_prob += likelihood_norm[right_candidate]

        # Get bounds from included indices
        indices.sort()
        lower_bound = float(param_values[indices[0]])
        upper_bound = float(param_values[indices[-1]])

        return lower_bound, upper_bound

    def save(self, output_path: str | Path) -> None:
        """
        Save likelihood results to file.

        Parameters
        ----------
        output_path : str or Path
            Path where results should be saved.

        Notes
        -----
        Saves the complete LikelihoodResult object using pickle format.
        The saved file can be loaded later for further analysis or
        visualization.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, input_path: str | Path) -> LikelihoodResult:
        """
        Load likelihood results from file.

        Parameters
        ----------
        input_path : str or Path
            Path to saved results file.

        Returns
        -------
        result : LikelihoodResult
            Loaded likelihood result object.

        Raises
        ------
        FileNotFoundError
            If the input file does not exist.
        """
        input_path = Path(input_path)

        if not input_path.exists():
            msg = f"Results file not found: {input_path}"
            raise FileNotFoundError(msg)

        with open(input_path, "rb") as f:
            return pickle.load(f)

    def get_summary_statistics(self) -> dict[str, Any]:
        """
        Get summary statistics for the likelihood analysis.

        Returns
        -------
        summary : dict[str, Any]
            Dictionary containing summary statistics including best-fit values,
            chi-squared minimum, and confidence intervals.

        Notes
        -----
        Provides a comprehensive summary of the analysis results suitable
        for reporting and quick interpretation of the results.
        """
        summary = {
            "best_fit_parameters": self.get_best_fit(),
            "chi_squared_minimum": self.get_chi_squared_minimum(),
            "maximum_likelihood": self.get_maximum_likelihood(),
            "confidence_intervals_68": self.get_confidence_intervals(0.68),
            "confidence_intervals_95": self.get_confidence_intervals(0.95),
            "total_parameter_points": len(self.parameter_grid),
            "parameter_names": deepcopy(self.parameter_grid.parameter_names),
        }

        return summary

    def __repr__(self) -> str:
        """Return string representation of the result object."""
        n_points = len(self.parameter_grid)
        n_params = len(self.parameter_grid.parameter_names)
        chi2_min = self.get_chi_squared_minimum()

        return (
            f"LikelihoodResult(parameters={n_params}, "
            f"grid_points={n_points}, χ²_min={chi2_min:.3f})"
        )
