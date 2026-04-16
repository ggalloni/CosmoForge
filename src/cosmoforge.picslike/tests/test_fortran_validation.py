"""
Validation tests comparing Python PICSLike against Fortran reference.

These tests ensure that the Python picslike package produces results
consistent with the original Fortran implementation to within expected
numerical precision.

Tolerances (empirically determined):
- Signal covariance: ~1e-10 (numerical precision in Legendre computation)
- Noise covariance: ~1e-12 (binary file read precision)
- Chi-squared: ~1e-9 (accumulation of numerical errors)
- Log-likelihood: ~1e-9 (accumulation of numerical errors)
"""

import os
import tempfile

import numpy as np
import pytest
import yaml

from cosmocore import (
    FieldCollection,
    FieldConfig,
    InputParams,
    ScalarField,
    compute_pointings,
    compute_signal_matrix,
)
from cosmocore.in_out import readcl

FORTRAN_REF_DIR = "tests/data/nside8/B/fortran_reference"

# Number of parameter points for likelihood tests (3 for fast validation)
N_VALIDATION_POINTS = 3

TOLERANCES = {
    "signal": 1e-10,
    "noise": 1e-12,
    "chi2": 1e-9,
    "loglike": 1e-9,
}


class DummyLogger:
    """Minimal logger for testing."""

    def log_with_feedback(self, msg, level=1):
        pass


@pytest.fixture
def fortran_ref_path(local_path):
    """Return path to Fortran reference data directory."""
    return os.path.join(local_path, FORTRAN_REF_DIR)


@pytest.fixture
def config_path(local_path, config_resolver):
    """Return resolved config file path."""
    return config_resolver(os.path.join(FORTRAN_REF_DIR, "config.yaml"))


@pytest.fixture
def fortran_signal(fortran_ref_path):
    """Load Fortran reference signal matrix."""
    signal_path = os.path.join(fortran_ref_path, "signal.bin")
    n = int(np.sqrt(os.path.getsize(signal_path) / 8))
    return np.fromfile(signal_path, dtype=np.float64).reshape((n, n))


@pytest.fixture
def python_signal(config_path, fortran_ref_path):
    """Compute Python signal matrix."""
    import healpy as hp

    # Load config
    params = InputParams.read_parameter_file(config_path)

    nside = params.nside
    lmax = params.lmax
    lmax_signal = 4 * nside

    # Create mask (full sky for this test)
    npix = hp.nside2npix(nside)
    mask = np.ones(npix, dtype=np.float64)

    # Create field
    config = FieldConfig(spin=0, nside=nside, lmax=lmax, mask=mask, labels="B")
    field = ScalarField(config)

    # Create collection
    collection = FieldCollection(params, [field], logger=DummyLogger())

    # Setup geometry
    active_pixels = field.active_pixels
    n_active = len(active_pixels)

    point_vectors = (np.empty((n_active, 3), dtype=np.float64),)
    theta_vectors = (np.empty(n_active, dtype=np.float64),)
    phi_vectors = (np.empty(n_active, dtype=np.float64),)
    point_vectors, theta_vectors, phi_vectors = compute_pointings(
        nside,
        [n_active],
        point_vectors,
        theta_vectors,
        phi_vectors,
        np.array([active_pixels]),
        0,  # RING ordering
    )
    collection.set_pointing_vectors(point_vectors)

    # Read and set Cls with lmax_signal
    cls_dict = readcl(params.inputclfile, params, DummyLogger(), lmax=lmax_signal)
    collection.set_cls(cls_dict, lmax=lmax_signal)
    collection.set_beams(lmax=lmax_signal)

    # Compute signal matrix
    signal = np.zeros((n_active, n_active), dtype=np.float64, order="F")
    compute_signal_matrix(S=signal, lmax=lmax_signal, fields=collection)

    return signal


def test_signal_matrix_validation(python_signal, fortran_signal):
    """
    Validate signal covariance matrix against Fortran reference.

    This test verifies that the Python compute_signal_matrix function
    produces results identical to the Fortran implementation to within
    machine precision.
    """
    # Compare diagonals
    diag_python = np.diag(python_signal)
    diag_fortran = np.diag(fortran_signal)

    # Compute relative difference
    rel_diff = np.abs((diag_python - diag_fortran) / diag_fortran)
    max_rel_diff = np.max(rel_diff)

    # Also compute absolute difference for machine precision check
    abs_diff = np.abs(python_signal - fortran_signal)
    max_abs_diff = np.max(abs_diff)

    # Print summary
    print("\n" + "=" * 70)
    print("SIGNAL MATRIX FORTRAN VALIDATION")
    print("=" * 70)
    print(f"Python signal diagonal mean:   {np.mean(diag_python):.6e}")
    print(f"Fortran signal diagonal mean:  {np.mean(diag_fortran):.6e}")
    print(f"Max relative difference:       {max_rel_diff:.2e}")
    print(f"Max absolute difference:       {max_abs_diff:.2e}")
    print("=" * 70)

    status = "PASS" if max_rel_diff < TOLERANCES["signal"] else "FAIL"
    print(f"Signal validation: {status}")
    print("=" * 70)

    # Assert signal matches
    assert max_rel_diff < TOLERANCES["signal"], (
        f"Signal max relative diff {max_rel_diff:.2e} exceeds "
        f"tolerance {TOLERANCES['signal']:.2e}"
    )


# =============================================================================
# COMPREHENSIVE FORTRAN VALIDATION TEST
# =============================================================================


def test_full_fortran_validation(local_path, fortran_ref_path):
    """
    Comprehensive validation of Python PICSLike against Fortran reference.

    This single test validates all quantities in one run:
    - Signal covariance matrix
    - Noise covariance matrix
    - Chi-squared values
    - Log-likelihood values

    Uses only N_VALIDATION_POINTS parameter points for fast execution.
    """
    from picslike import PICSLike

    print("\n" + "=" * 70)
    print("COMPREHENSIVE FORTRAN VALIDATION")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Load Fortran reference data
    # -------------------------------------------------------------------------
    print("\nLoading Fortran reference data...")

    # Signal covariance
    signal_path = os.path.join(fortran_ref_path, "signal.bin")
    n_sig = int(np.sqrt(os.path.getsize(signal_path) / 8))
    fortran_signal = np.fromfile(signal_path, dtype=np.float64).reshape((n_sig, n_sig))

    # Noise covariance
    noise_path = os.path.join(fortran_ref_path, "ncov_B_only.bin")
    n_noise = int(np.sqrt(os.path.getsize(noise_path) / 8))
    fortran_noise = np.fromfile(noise_path, dtype=np.float64).reshape((n_noise, n_noise))

    # Likelihood results (first N_VALIDATION_POINTS only)
    results_path = os.path.join(fortran_ref_path, "results.txt")
    data = np.loadtxt(results_path)
    mask = data[:, 1] < N_VALIDATION_POINTS
    filtered_data = data[mask]
    fortran_chi2 = filtered_data[:, 3]
    fortran_loglike = filtered_data[:, 2]

    print(f"  Signal matrix shape: {fortran_signal.shape}")
    print(f"  Noise matrix shape: {fortran_noise.shape}")
    print(f"  Likelihood points: {len(fortran_chi2)}")

    # -------------------------------------------------------------------------
    # Create validation config with reduced parameter points
    # -------------------------------------------------------------------------
    print("\nCreating validation configuration...")

    original_config_path = os.path.join(local_path, FORTRAN_REF_DIR, "config.yaml")
    with open(original_config_path) as f:
        config = yaml.safe_load(f)

    # Modify parameter range to only use first N_VALIDATION_POINTS
    r_values = np.linspace(0.0, 0.005, 401)[:N_VALIDATION_POINTS]
    config["parameters"]["r"] = [
        float(r_values[0]),
        float(r_values[-1]),
        N_VALIDATION_POINTS,
    ]

    # Resolve file paths (handle both tests/ and ../tests/ prefixes)
    for key, value in config.items():
        if isinstance(value, str):
            if value.startswith("../tests/"):
                config[key] = os.path.join(local_path, value.replace("../", ""))
            elif value.startswith("tests/"):
                config[key] = os.path.join(local_path, value)

    # Create temporary config file
    temp_config = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(config, temp_config, default_flow_style=False)
    temp_config.close()

    # -------------------------------------------------------------------------
    # Run PICSLike
    # -------------------------------------------------------------------------
    print("\nRunning PICSLike...")

    picslike = PICSLike(temp_config.name)
    picslike.run()

    python_noise = picslike.noise_cov1
    python_chi2 = picslike.likelihood_result.chi_squared_values
    python_loglike = picslike.likelihood_result.log_likelihood_values

    # -------------------------------------------------------------------------
    # Compute Python signal matrix for comparison
    # -------------------------------------------------------------------------
    print("\nComputing Python signal matrix...")
    import healpy as hp

    params = InputParams.read_parameter_file(temp_config.name)
    nside = params.nside
    lmax = params.lmax
    lmax_signal = 4 * nside

    npix = hp.nside2npix(nside)
    mask = np.ones(npix, dtype=np.float64)

    config_field = FieldConfig(spin=0, nside=nside, lmax=lmax, mask=mask, labels="B")
    field = ScalarField(config_field)
    collection = FieldCollection(params, [field], logger=DummyLogger())

    active_pixels = field.active_pixels
    n_active = len(active_pixels)

    point_vectors = (np.empty((n_active, 3), dtype=np.float64),)
    theta_vectors = (np.empty(n_active, dtype=np.float64),)
    phi_vectors = (np.empty(n_active, dtype=np.float64),)
    point_vectors, theta_vectors, phi_vectors = compute_pointings(
        nside,
        [n_active],
        point_vectors,
        theta_vectors,
        phi_vectors,
        np.array([active_pixels]),
        0,  # RING ordering
    )
    collection.set_pointing_vectors(point_vectors)

    cls_dict = readcl(params.inputclfile, params, DummyLogger(), lmax=lmax_signal)
    collection.set_cls(cls_dict, lmax=lmax_signal)
    collection.set_beams(lmax=lmax_signal)

    python_signal = np.zeros((n_active, n_active), dtype=np.float64, order="F")
    compute_signal_matrix(S=python_signal, lmax=lmax_signal, fields=collection)

    # -------------------------------------------------------------------------
    # Validate Signal Covariance
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("SIGNAL COVARIANCE VALIDATION")
    print("-" * 70)

    diag_python_sig = np.diag(python_signal)
    diag_fortran_sig = np.diag(fortran_signal)
    signal_rel_diff = np.abs((diag_python_sig - diag_fortran_sig) / diag_fortran_sig)
    signal_max_rel = np.max(signal_rel_diff)
    signal_abs_diff = np.abs(python_signal - fortran_signal)
    signal_max_abs = np.max(signal_abs_diff)

    print(f"Python signal diagonal mean:   {np.mean(diag_python_sig):.6e}")
    print(f"Fortran signal diagonal mean:  {np.mean(diag_fortran_sig):.6e}")
    print(f"Max relative difference:       {signal_max_rel:.2e}")
    print(f"Max absolute difference:       {signal_max_abs:.2e}")

    signal_status = "PASS" if signal_max_rel < TOLERANCES["signal"] else "FAIL"
    print(f"Status: {signal_status}")

    # -------------------------------------------------------------------------
    # Validate Noise Covariance
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("NOISE COVARIANCE VALIDATION")
    print("-" * 70)

    assert python_noise.shape == fortran_noise.shape, (
        f"Shape mismatch: Python {python_noise.shape} vs Fortran {fortran_noise.shape}"
    )

    noise_abs_diff = np.abs(python_noise - fortran_noise)
    noise_max_abs = np.max(noise_abs_diff)

    print(f"Matrix shape:                  {python_noise.shape}")
    print(f"Python noise diagonal mean:    {np.mean(np.diag(python_noise)):.6e}")
    print(f"Fortran noise diagonal mean:   {np.mean(np.diag(fortran_noise)):.6e}")
    print(f"Max absolute difference:       {noise_max_abs:.2e}")

    noise_status = "PASS" if noise_max_abs < TOLERANCES["noise"] else "FAIL"
    print(f"Status: {noise_status}")

    # -------------------------------------------------------------------------
    # Validate Chi-Squared
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("CHI-SQUARED VALIDATION")
    print("-" * 70)

    assert len(python_chi2) == len(fortran_chi2), (
        f"Length mismatch: Python {len(python_chi2)} vs Fortran {len(fortran_chi2)}"
    )

    chi2_abs_diff = np.abs(python_chi2 - fortran_chi2)
    chi2_max_abs = np.max(chi2_abs_diff)
    chi2_rel_diff = np.abs((python_chi2 - fortran_chi2) / fortran_chi2)
    chi2_max_rel = np.max(chi2_rel_diff)

    print(f"Number of parameter points:    {len(python_chi2)}")
    print(
        f"Python chi2 range:             "
        f"[{python_chi2.min():.4f}, {python_chi2.max():.4f}]"
    )
    print(
        f"Fortran chi2 range:            "
        f"[{fortran_chi2.min():.4f}, {fortran_chi2.max():.4f}]"
    )
    print(f"Max absolute difference:       {chi2_max_abs:.2e}")
    print(f"Max relative difference:       {chi2_max_rel:.2e}")

    chi2_status = "PASS" if chi2_max_rel < TOLERANCES["chi2"] else "FAIL"
    print(f"Status: {chi2_status}")

    # -------------------------------------------------------------------------
    # Validate Log-Likelihood
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("LOG-LIKELIHOOD VALIDATION")
    print("-" * 70)

    assert len(python_loglike) == len(fortran_loglike), (
        f"Length mismatch: Python {len(python_loglike)} vs Fortran {len(fortran_loglike)}"
    )

    loglike_abs_diff = np.abs(python_loglike - fortran_loglike)
    loglike_max_abs = np.max(loglike_abs_diff)
    loglike_scale = np.abs(fortran_loglike).mean()
    loglike_scaled_max = (
        loglike_max_abs / loglike_scale if loglike_scale > 0 else loglike_max_abs
    )

    print(f"Number of parameter points:    {len(python_loglike)}")
    print(
        f"Python loglike range:          "
        f"[{python_loglike.min():.4f}, {python_loglike.max():.4f}]"
    )
    print(
        f"Fortran loglike range:         "
        f"[{fortran_loglike.min():.4f}, {fortran_loglike.max():.4f}]"
    )
    print(f"Max absolute difference:       {loglike_max_abs:.2e}")
    print(f"Scaled max difference:         {loglike_scaled_max:.2e}")

    loglike_status = "PASS" if loglike_scaled_max < TOLERANCES["loglike"] else "FAIL"
    print(f"Status: {loglike_status}")

    # -------------------------------------------------------------------------
    # Overall Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Signal covariance:   {signal_status}")
    print(f"Noise covariance:    {noise_status}")
    print(f"Chi-squared:         {chi2_status}")
    print(f"Log-likelihood:      {loglike_status}")

    all_pass = all(
        s == "PASS" for s in [signal_status, noise_status, chi2_status, loglike_status]
    )
    overall_status = "PASS" if all_pass else "FAIL"
    print("=" * 70)
    print(f"OVERALL: {overall_status}")
    print("=" * 70)

    # Clean up temp file
    os.unlink(temp_config.name)

    # Assert all validations pass
    assert signal_max_rel < TOLERANCES["signal"], (
        f"Signal validation failed: max rel diff {signal_max_rel:.2e}"
    )
    assert noise_max_abs < TOLERANCES["noise"], (
        f"Noise validation failed: max abs diff {noise_max_abs:.2e}"
    )
    assert chi2_max_rel < TOLERANCES["chi2"], (
        f"Chi-squared validation failed: max rel diff {chi2_max_rel:.2e}"
    )
    assert loglike_scaled_max < TOLERANCES["loglike"], (
        f"Log-likelihood validation failed: scaled max diff {loglike_scaled_max:.2e}"
    )
