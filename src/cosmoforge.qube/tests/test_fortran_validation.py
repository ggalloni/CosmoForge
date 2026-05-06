"""
Validation tests comparing Python QML implementation against Fortran reference.

These tests ensure that the Python qube package produces results consistent
with the original Fortran implementation to machine precision. The reference
data was regenerated from a high-precision Fortran build (format e25.17 in
all ASCII outputs and full-precision pixel-coordinate exchange) to remove
the ASCII-serialization floor that previously masked sub-1e-3 agreement.

Tolerances (empirically determined against the HP reference):
- NCov (noise covariance): exact (same file)
- invCov (inverse covariance): ~1e-13 (LAPACK Cholesky roundoff)
- Signal covariance: ~1e-11 (Legendre-sum accumulation)
- Fisher diagonal: ~1e-7 (Numba-vs-Fortran summation order)
- Spectra: ~1e-12 (matches Python production)
- Noise bias: ~1e-7 (Numba-vs-Fortran summation order)
"""

import os

import numpy as np
import pytest

from qube import Spectra

FORTRAN_REF_DIR = "tests/data/nside8/B/fortran_reference"

TOLERANCES = {
    "ncov": 1e-12,
    "invcov": 1e-12,
    "signal": 1e-10,
    "fisher_diag": 1e-6,
    "spectra": 1e-10,
    "noise_bias": 1e-6,
}


@pytest.fixture
def fortran_ref_path(local_path):
    """Return path to Fortran reference data directory."""
    return os.path.join(local_path, FORTRAN_REF_DIR)


@pytest.fixture
def config_path(local_path, config_resolver):
    """Return resolved config file path."""
    return config_resolver(os.path.join(FORTRAN_REF_DIR, "config.yaml"))


@pytest.fixture
def python_spectra(config_path):
    """Run Python Spectra computation and return instance."""
    spectra = Spectra(config_path)
    spectra.run()
    return spectra


@pytest.fixture
def fortran_data(fortran_ref_path):
    """Load all Fortran reference data."""
    data = {}

    # Fisher matrix
    data["fisher"] = np.loadtxt(os.path.join(fortran_ref_path, "fisher.dat"))

    # Covariance matrices (determine size from file)
    invcov_path = os.path.join(fortran_ref_path, "invcov.bin")
    n = int(np.sqrt(os.path.getsize(invcov_path) / 8))

    data["invcov"] = np.fromfile(invcov_path).reshape((n, n))
    data["ncov"] = np.fromfile(os.path.join(fortran_ref_path, "ncov.bin")).reshape((n, n))
    data["signal"] = np.fromfile(os.path.join(fortran_ref_path, "signal.bin")).reshape(
        (n, n)
    )

    # Spectra and noise bias
    spectra_data = np.loadtxt(os.path.join(fortran_ref_path, "spectra.dat"))
    data["spectra_ell"] = spectra_data[:, 0].astype(int)
    data["spectra"] = spectra_data[:, 1]

    nb_data = np.loadtxt(os.path.join(fortran_ref_path, "noise_bias.dat"))
    data["noise_bias_ell"] = nb_data[:, 0].astype(int)
    data["noise_bias"] = nb_data[:, 1]

    return data


def test_fortran_validation(python_spectra, fortran_data):
    """
    Validate all quantities against Fortran reference in a single test.

    This test runs the full QML pipeline once and checks:
    1. NCov (noise covariance matrix)
    2. invCov (inverse covariance matrix)
    3. Signal covariance matrix
    4. Fisher matrix diagonal
    5. Power spectra
    6. Noise bias
    """
    results = {}

    # 1. NCov (noise covariance)
    diag_python = np.diag(python_spectra.noise_cov1)
    diag_fortran = np.diag(fortran_data["ncov"])
    results["ncov"] = np.max(np.abs((diag_python - diag_fortran) / diag_fortran))

    # 2. invCov (inverse covariance)
    diag_python = np.diag(python_spectra.inv_cov1)
    diag_fortran = np.diag(fortran_data["invcov"])
    results["invcov"] = np.max(np.abs((diag_python - diag_fortran) / diag_fortran))

    # 3. Signal covariance
    python_signal = python_spectra.fisher_instance.signal_matrix
    fortran_signal = fortran_data["signal"]
    rel_diff = np.abs((python_signal - fortran_signal) / fortran_signal)
    rel_diff = np.where(np.abs(fortran_signal) > 1e-20, rel_diff, 0)
    results["signal"] = np.max(rel_diff)

    # 4. Fisher diagonal
    # Fisher is beam-smoothed. Divide out beam smoothing for raw comparison.
    fisher_python = python_spectra.fisher_instance.fisher
    beam_smoothing = python_spectra.fisher_instance.beam_smoothing
    fisher_raw = fisher_python / np.outer(beam_smoothing, beam_smoothing)
    diag_python = np.diag(fisher_raw)
    diag_fortran = np.diag(fortran_data["fisher"])
    results["fisher_diag"] = np.max(np.abs((diag_python - diag_fortran) / diag_fortran))

    # 5. Power spectra
    python_clhat = python_spectra.get_power_spectra()[0]
    fortran_clhat = fortran_data["spectra"]
    n = min(len(python_clhat), len(fortran_clhat))
    results["spectra"] = np.max(
        np.abs((python_clhat[:n] - fortran_clhat[:n]) / fortran_clhat[:n])
    )

    # 6. Noise bias
    python_nlhat = python_spectra.get_noise_bias()
    fortran_nlhat = fortran_data["noise_bias"]
    n = min(len(python_nlhat), len(fortran_nlhat))
    results["noise_bias"] = np.max(
        np.abs((python_nlhat[:n] - fortran_nlhat[:n]) / fortran_nlhat[:n])
    )

    # Print summary
    print("\n" + "=" * 70)
    print("FORTRAN VALIDATION SUMMARY")
    print("=" * 70)
    print(f"{'Quantity':<20} {'Max Rel Diff':<15} {'Tolerance':<15} {'Status'}")
    print("-" * 70)

    for key, val in results.items():
        tol = TOLERANCES[key]
        status = "PASS" if val < tol else "FAIL"
        print(f"{key:<20} {val:<15.2e} {tol:<15.2e} {status}")

    print("=" * 70)

    # Assert all checks pass
    for key, val in results.items():
        assert val < TOLERANCES[key], (
            f"{key} max relative diff {val:.2e} exceeds tolerance {TOLERANCES[key]:.2e}"
        )
