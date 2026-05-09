import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest


def _configure_plt():
    plt.rc("axes", labelsize=20, linewidth=1.5)
    plt.rc("xtick", direction="in", labelsize=15, top=True)
    plt.rc("ytick", direction="in", labelsize=15, right=True)

    plt.rc("xtick.major", width=1.1, size=5)
    plt.rc("ytick.major", width=1.1, size=5)

    plt.rc("xtick.minor", width=1.1, size=3)
    plt.rc("ytick.minor", width=1.1, size=3)

    plt.rc("lines", linewidth=2)
    plt.rc("legend", frameon=False, fontsize=15)
    plt.rc("figure", dpi=100, autolayout=True, figsize=[10, 7])
    plt.rc("savefig", dpi=300, bbox="tight")

    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
        color=[
            "red",
            "dodgerblue",
            "forestgreen",
            "goldenrod",
            "maroon",
            "cyan",
            "limegreen",
            "darkorange",
            "darkmagenta",
        ]
    )


@pytest.fixture
def configure_plt():
    return _configure_plt


@pytest.fixture
def local_path():
    """Get the path to the cosmocore package directory."""
    # Get the directory where this conftest.py file is located
    test_dir = os.path.dirname(__file__)
    # Return the parent directory (the package root)
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
            (e.g., "tests/data/ref_TQU_signal.dat")

        Returns
        -------
        str
            Absolute path to the data file
        """
        # Find the project root by looking for src directory
        current_dir = os.getcwd()
        path_parts = current_dir.split(os.sep)

        # Determine if we need to add package prefix for relative paths
        package_prefix = ""
        try:
            # If we find 'src' in the path, we might be running from project root
            path_parts.index("src")
            # If current dir doesn't end with package name, add prefix
            if not current_dir.endswith("cosmoforge.cosmocore"):
                package_prefix = "src/cosmoforge.cosmocore/"
        except ValueError:
            # No 'src' in path - check if we're in the package directory
            if not current_dir.endswith("cosmoforge.cosmocore"):
                # Assume we need the full path from wherever we are
                package_prefix = "src/cosmoforge.cosmocore/"

        # Add package prefix if needed
        if data_path.startswith("tests/"):
            resolved_path = package_prefix + data_path
        else:
            resolved_path = data_path

        # If it's not absolute, make it relative to local_path
        if not os.path.isabs(resolved_path):
            if resolved_path.startswith(package_prefix):
                # Remove the package prefix since we're going from local_path
                relative_path = resolved_path[len(package_prefix) :]
                resolved_path = os.path.join(local_path, relative_path)
            else:
                resolved_path = os.path.join(local_path, resolved_path)

        return os.path.abspath(resolved_path)

    return resolve_data_path


@pytest.fixture
def simple_compression_setup():
    """
    Create a simple test setup for compression classes.

    Returns a dictionary with:
    - N_inv: Simple diagonal noise inverse
    - theta, phi: Pixel positions on the sphere
    - lmax: Maximum multipole
    """
    np.random.seed(42)

    # Small number of pixels for fast tests
    n_pix = 100

    # Simple diagonal noise covariance (inverse)
    noise_variance = np.ones(n_pix) * 0.01
    N = np.diag(noise_variance)
    N_inv = np.diag(1.0 / noise_variance)

    # Random positions on the sphere
    theta = np.random.uniform(0, np.pi, n_pix)
    phi = np.random.uniform(0, 2 * np.pi, n_pix)

    # Small lmax for fast tests
    lmax = 10

    return {
        "N": N,
        "N_inv": N_inv,
        "theta": theta,
        "phi": phi,
        "lmax": lmax,
        "n_pix": n_pix,
    }


@pytest.fixture
def uniform_sky_setup():
    """
    Create a uniform sky setup with known geometry.

    Uses HEALPix-like uniform distribution for testing.
    """
    np.random.seed(123)

    # Moderate number of pixels
    n_pix = 50

    # Uniform distribution on sphere using golden spiral
    golden_ratio = (1 + np.sqrt(5)) / 2
    indices = np.arange(n_pix)
    theta = np.arccos(1 - 2 * (indices + 0.5) / n_pix)
    phi = (2 * np.pi * indices / golden_ratio) % (2 * np.pi)

    # Diagonal noise
    noise_variance = np.ones(n_pix) * 0.1
    N = np.diag(noise_variance)
    N_inv = np.diag(1.0 / noise_variance)

    lmax = 8

    return {
        "N": N,
        "N_inv": N_inv,
        "theta": theta,
        "phi": phi,
        "lmax": lmax,
        "n_pix": n_pix,
    }


@pytest.fixture
def two_scalar_field_setup():
    """
    Create a test setup with two scalar fields with different sky coverage.

    This mimics the multi-field setup used in signal matrix tests, with
    independent theta/phi coordinates per field (component).
    """
    np.random.seed(42)

    # Two fields with different pixel counts (different sky coverage)
    n_pix_1 = 60
    n_pix_2 = 40

    # Block-diagonal noise covariance (inverse)
    n_pix_total = n_pix_1 + n_pix_2
    noise_variance_1 = np.ones(n_pix_1) * 0.01
    noise_variance_2 = np.ones(n_pix_2) * 0.02  # Different noise level

    N = np.zeros((n_pix_total, n_pix_total))
    N[:n_pix_1, :n_pix_1] = np.diag(noise_variance_1)
    N[n_pix_1:, n_pix_1:] = np.diag(noise_variance_2)

    N_inv = np.zeros((n_pix_total, n_pix_total))
    N_inv[:n_pix_1, :n_pix_1] = np.diag(1.0 / noise_variance_1)
    N_inv[n_pix_1:, n_pix_1:] = np.diag(1.0 / noise_variance_2)

    # Random positions on the sphere - different for each field
    theta_1 = np.random.uniform(0, np.pi, n_pix_1)
    phi_1 = np.random.uniform(0, 2 * np.pi, n_pix_1)

    theta_2 = np.random.uniform(0, np.pi, n_pix_2)
    phi_2 = np.random.uniform(0, 2 * np.pi, n_pix_2)

    # Pack as tuples (multi-field format)
    theta_tuple = (theta_1, theta_2)
    phi_tuple = (phi_1, phi_2)

    # Small lmax for fast tests
    lmax = 8

    return {
        "N": N,
        "N_inv": N_inv,
        "theta": theta_tuple,
        "phi": phi_tuple,
        "lmax": lmax,
        "n_pix_1": n_pix_1,
        "n_pix_2": n_pix_2,
        "n_pix_total": n_pix_total,
    }


@pytest.fixture
def three_scalar_field_realistic_setup():
    """
    Create a realistic test setup with three scalar fields (T1, T2, T3).

    This mimics a real-world scenario where we have multiple independent
    temperature measurements (e.g., different frequency channels).
    """
    np.random.seed(42)

    # Three fields with different pixel counts (different sky coverage)
    n_pix_1 = 80  # Field 1: full coverage
    n_pix_2 = 60  # Field 2: partial coverage
    n_pix_3 = 50  # Field 3: smaller patch

    n_pix_total = n_pix_1 + n_pix_2 + n_pix_3

    # Generate uniform positions on sphere using golden spiral
    def golden_spiral_positions(n_pix, seed_offset=0):
        np.random.seed(42 + seed_offset)
        golden_ratio = (1 + np.sqrt(5)) / 2
        indices = np.arange(n_pix)
        theta = np.arccos(1 - 2 * (indices + 0.5) / n_pix)
        phi = (2 * np.pi * indices / golden_ratio) % (2 * np.pi)
        # Add small perturbation to avoid numerical issues
        theta += np.random.uniform(-0.01, 0.01, n_pix)
        phi += np.random.uniform(-0.01, 0.01, n_pix)
        return theta, phi

    theta_1, phi_1 = golden_spiral_positions(n_pix_1, seed_offset=0)
    theta_2, phi_2 = golden_spiral_positions(n_pix_2, seed_offset=100)
    theta_3, phi_3 = golden_spiral_positions(n_pix_3, seed_offset=200)

    # Pack as tuples (multi-field format)
    theta_tuple = (theta_1, theta_2, theta_3)
    phi_tuple = (phi_1, phi_2, phi_3)

    # Block-diagonal noise covariance with different noise levels
    noise_var_1 = np.ones(n_pix_1) * 0.01  # Low noise
    noise_var_2 = np.ones(n_pix_2) * 0.02  # Medium noise
    noise_var_3 = np.ones(n_pix_3) * 0.03  # Higher noise

    N = np.zeros((n_pix_total, n_pix_total))
    N[:n_pix_1, :n_pix_1] = np.diag(noise_var_1)
    N[n_pix_1 : n_pix_1 + n_pix_2, n_pix_1 : n_pix_1 + n_pix_2] = np.diag(noise_var_2)
    N[n_pix_1 + n_pix_2 :, n_pix_1 + n_pix_2 :] = np.diag(noise_var_3)

    N_inv = np.zeros((n_pix_total, n_pix_total))
    N_inv[:n_pix_1, :n_pix_1] = np.diag(1.0 / noise_var_1)
    N_inv[n_pix_1 : n_pix_1 + n_pix_2, n_pix_1 : n_pix_1 + n_pix_2] = np.diag(
        1.0 / noise_var_2
    )
    N_inv[n_pix_1 + n_pix_2 :, n_pix_1 + n_pix_2 :] = np.diag(1.0 / noise_var_3)

    lmax = 10

    # n_ell counts inference multipoles ℓ=2..lmax — drives Fisher dimension.
    n_ell = lmax - 1
    # Realistic power spectra (physical C_ell values, ℓ-indexed length lmax+1).
    ells_full = np.arange(lmax + 1, dtype=np.float64)
    with np.errstate(divide="ignore"):
        inv_ell2 = np.where(ells_full >= 2, 1.0 / ells_full**2, 0.0)

    # Auto-spectra: different amplitudes for different fields
    C_ell_11 = 1e-4 * inv_ell2  # Field 1 auto
    C_ell_22 = 0.8e-4 * inv_ell2  # Field 2 auto
    C_ell_33 = 0.6e-4 * inv_ell2  # Field 3 auto

    # Cross-spectra: correlated but not perfectly
    C_ell_12 = 0.5e-4 * inv_ell2  # 1-2 cross
    C_ell_13 = 0.3e-4 * inv_ell2  # 1-3 cross
    C_ell_23 = 0.4e-4 * inv_ell2  # 2-3 cross

    C_ell_dict = {
        (0, 0): C_ell_11,
        (1, 1): C_ell_22,
        (2, 2): C_ell_33,
        (0, 1): C_ell_12,
        (0, 2): C_ell_13,
        (1, 2): C_ell_23,
    }

    return {
        "N": N,
        "N_inv": N_inv,
        "theta": theta_tuple,
        "phi": phi_tuple,
        "lmax": lmax,
        "n_pix_1": n_pix_1,
        "n_pix_2": n_pix_2,
        "n_pix_3": n_pix_3,
        "n_pix_total": n_pix_total,
        "C_ell_dict": C_ell_dict,
        "n_ell": n_ell,
    }
