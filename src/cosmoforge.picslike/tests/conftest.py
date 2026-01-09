"""
Test configuration for PICSLike package.

This module provides shared test fixtures and configuration for the
PICSLike test suite.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from cosmocore import InputParams

# Import from submodules to avoid heavy PICSLike import during collection
from picslike.likelihood_result import LikelihoodResult
from picslike.parameter_grid import ParameterGrid

# Disable JIT compilation for testing
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")


@pytest.fixture
def minimal_params():
    """Create minimal InputParams for testing."""
    params = InputParams()
    params.update(
        {
            "nside": 4,
            "lmax": 16,
            "spins": [0, 2],
            "labels": ["T", "E", "B"],
            "physical_labels": ["T", "Q", "U"],
        }
    )
    return params


@pytest.fixture
def sample_parameter_ranges():
    """Sample parameter ranges for testing."""
    return {
        "omega_b": np.array([0.020, 0.022, 0.024]),
        "omega_c": np.array([0.10, 0.12, 0.14]),
    }


@pytest.fixture
def sample_theoretical_spectra():
    """Sample theoretical spectra for testing (as dict expected by ParameterGrid)."""
    spectra = {}

    # Generate spectra dict for all parameter combinations
    omega_b_values = [0.020, 0.022, 0.024]
    omega_c_values = [0.10, 0.12, 0.14]

    for omega_b in omega_b_values:
        for omega_c in omega_c_values:
            # Create dict with spectrum keys expected by cosmocore
            # The spectrum dict should have keys like "TT", "EE", "BB", "TE", "TB", "EB"
            scale = (omega_b / 0.022) * (omega_c / 0.12)
            ell = np.arange(2, 101)
            cl_theory = scale * 1000 * ell ** (-1.1)
            spectra[(omega_b, omega_c)] = {
                "TT": cl_theory,
                "EE": cl_theory * 0.5,
                "BB": cl_theory * 0.1,
                "TE": cl_theory * 0.3,
                "TB": np.zeros_like(cl_theory),
                "EB": np.zeros_like(cl_theory),
            }

    return spectra


@pytest.fixture
def sample_parameter_grid(
    minimal_params, sample_parameter_ranges, sample_theoretical_spectra
):
    """Sample parameter grid for testing."""
    return ParameterGrid(
        core_params=minimal_params,
        parameter_ranges=sample_parameter_ranges,
        theoretical_spectra=sample_theoretical_spectra,
    )


@pytest.fixture
def sample_data_vector():
    """Sample data vector for testing."""
    return np.random.normal(0, 1, 100)


@pytest.fixture
def sample_noise_covariance():
    """Sample noise covariance matrix for testing."""
    n_pixels = 100
    return np.eye(n_pixels)


@pytest.fixture
def sample_chi_squared_values(sample_parameter_grid):
    """Sample chi-squared values for testing LikelihoodResult."""
    n_points = sample_parameter_grid.get_total_points()
    # Create chi-squared values with minimum at center of grid (fiducial point)
    chi2 = np.zeros(n_points)
    fiducial_point = (0.022, 0.12)

    for i, point in enumerate(sample_parameter_grid.grid_points):
        # Chi-squared increases quadratically from fiducial
        d_omega_b = (point[0] - fiducial_point[0]) / 0.002
        d_omega_c = (point[1] - fiducial_point[1]) / 0.02
        chi2[i] = 100 + d_omega_b**2 + d_omega_c**2

    return chi2


@pytest.fixture
def sample_log_likelihood_values(sample_chi_squared_values):
    """Sample log-likelihood values derived from chi-squared."""
    return -0.5 * sample_chi_squared_values


@pytest.fixture
def sample_likelihood_result(
    sample_parameter_grid,
    sample_chi_squared_values,
    sample_log_likelihood_values,
):
    """Sample LikelihoodResult for testing."""
    return LikelihoodResult(
        parameter_grid=sample_parameter_grid,
        chi_squared_values=sample_chi_squared_values,
        log_likelihood_values=sample_log_likelihood_values,
    )


@pytest.fixture
def single_param_ranges():
    """Single parameter range for testing edge cases."""
    return {"amplitude": np.array([0.95, 1.00, 1.05])}


@pytest.fixture
def single_param_spectra():
    """Single parameter spectra for testing."""
    spectra = {}
    amplitude_values = [0.95, 1.00, 1.05]
    ell = np.arange(2, 101)

    for amp in amplitude_values:
        spectra[(amp,)] = {
            "TT": amp * 1000 * ell ** (-1.1),
            "EE": amp * 500 * ell ** (-1.1),
            "BB": amp * 100 * ell ** (-1.1),
            "TE": amp * 300 * ell ** (-1.1),
            "TB": np.zeros_like(ell, dtype=float),
            "EB": np.zeros_like(ell, dtype=float),
        }

    return spectra


@pytest.fixture
def single_param_grid(minimal_params, single_param_ranges, single_param_spectra):
    """Single parameter grid for testing."""
    return ParameterGrid(
        core_params=minimal_params,
        parameter_ranges=single_param_ranges,
        theoretical_spectra=single_param_spectra,
    )


@pytest.fixture
def local_path():
    """Get the path to the picslike package directory."""
    test_dir = os.path.dirname(__file__)
    return os.path.dirname(test_dir)


@pytest.fixture
def data_resolver(local_path):
    """
    Fixture that provides a function to resolve test data file paths.

    This function resolves data file paths to work from both
    the project root and the package directory.
    """

    def resolve_data_path(data_path):
        """
        Resolve a test data file path to work from current location.

        Parameters
        ----------
        data_path : str
            Path to the data file relative to local_path
            (e.g., "tests/data/nside4/TQU/config.yaml")

        Returns
        -------
        str
            Absolute path to the data file
        """
        current_dir = os.getcwd()
        path_parts = current_dir.split(os.sep)

        package_prefix = ""
        try:
            path_parts.index("src")
            if not current_dir.endswith("cosmoforge.picslike"):
                package_prefix = "src/cosmoforge.picslike/"
        except ValueError:
            if not current_dir.endswith("cosmoforge.picslike"):
                package_prefix = "src/cosmoforge.picslike/"

        if data_path.startswith("tests/"):
            resolved_path = package_prefix + data_path
        else:
            resolved_path = data_path

        if not os.path.isabs(resolved_path):
            if resolved_path.startswith(package_prefix):
                relative_path = resolved_path[len(package_prefix) :]
                resolved_path = os.path.join(local_path, relative_path)
            else:
                resolved_path = os.path.join(local_path, resolved_path)

        return os.path.abspath(resolved_path)

    return resolve_data_path


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test output files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def config_path(data_resolver):
    """Path to the test configuration file."""
    return data_resolver("tests/data/nside4/TQU/config.yaml")
