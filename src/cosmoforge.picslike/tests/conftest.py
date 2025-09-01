"""
Test configuration for PICSLike package.

This module provides shared test fixtures and configuration for the
PICSLike test suite.
"""

import numpy as np
import pytest
from picslike import ParameterGrid


@pytest.fixture
def sample_parameter_ranges():
    """Sample parameter ranges for testing."""
    return {
        "omega_b": np.array([0.020, 0.022, 0.024]),
        "omega_c": np.array([0.10, 0.12, 0.14]),
    }


@pytest.fixture
def sample_theoretical_spectra():
    """Sample theoretical spectra for testing."""
    spectra = {}

    # Generate dummy spectra for all parameter combinations
    omega_b_values = [0.020, 0.022, 0.024]
    omega_c_values = [0.10, 0.12, 0.14]

    ell = np.arange(2, 101)  # Small range for testing

    for omega_b in omega_b_values:
        for omega_c in omega_c_values:
            # Simple scaling model
            scale = (omega_b / 0.022) * (omega_c / 0.12)
            cl_theory = scale * 1000 * ell ** (-1.1)
            spectra[(omega_b, omega_c)] = cl_theory

    return spectra


@pytest.fixture
def sample_parameter_grid(sample_parameter_ranges, sample_theoretical_spectra):
    """Sample parameter grid for testing."""
    return ParameterGrid(sample_parameter_ranges, sample_theoretical_spectra)


@pytest.fixture
def sample_data_vector():
    """Sample data vector for testing."""
    return np.random.normal(0, 1, 100)


@pytest.fixture
def sample_noise_covariance():
    """Sample noise covariance matrix for testing."""
    n_pixels = 100
    return np.eye(n_pixels)
