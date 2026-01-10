#!/usr/bin/env python3
"""
Simple example demonstrating pixel-based likelihood computation.

This script creates synthetic CMB-like data and performs pixel-based likelihood
analysis on a small parameter grid. It's designed to demonstrate the core
functionality without requiring large external datasets.
"""

import sys

import healpy as hp
import numpy as np
from tqdm import tqdm

# Set up the path to include our package
sys.path.insert(0, "/home/ggalloni/Projects/GitHub/CosmoForge/src/cosmoforge.picslike")


def generate_synthetic_cmb_map(nside=32, lmax=64, seed=42):
    """
    Generate a synthetic CMB temperature map.

    Parameters
    ----------
    nside : int
        HEALPix nside parameter (low resolution for testing).
    lmax : int
        Maximum multipole for power spectrum.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    cmb_map : numpy.ndarray
        Synthetic CMB temperature map.
    cl_theory : numpy.ndarray
        Theoretical power spectrum used to generate the map.
    """
    np.random.seed(seed)

    # Create a simple CMB-like power spectrum
    ell = np.arange(lmax + 1)
    # Simple model: C_l ∝ l^(-1.1) with exponential cutoff
    cl_theory = np.zeros(lmax + 1)
    cl_theory[2:] = 1000 * ell[2:] ** (-1.1) * np.exp(-ell[2:] / 40)

    # Generate random alm coefficients
    alm = hp.synalm(cl_theory, lmax=lmax)

    # Convert to map
    cmb_map = hp.alm2map(alm, nside)

    return cmb_map, cl_theory


def create_noise_map(nside=32, noise_level=0.1, seed=43):
    """
    Create a simple white noise map.

    Parameters
    ----------
    nside : int
        HEALPix nside parameter.
    noise_level : float
        Standard deviation of noise per pixel.
    seed : int
        Random seed.

    Returns
    -------
    noise_map : numpy.ndarray
        Gaussian white noise map.
    """
    np.random.seed(seed)
    npix = hp.nside2npix(nside)
    return np.random.normal(0, noise_level, npix)


def create_simple_mask(nside=32, mask_fraction=0.1):
    """
    Create a simple mask removing some pixels.

    Parameters
    ----------
    nside : int
        HEALPix nside parameter.
    mask_fraction : float
        Fraction of pixels to mask.

    Returns
    -------
    mask : numpy.ndarray
        Binary mask (1=good, 0=masked).
    """
    npix = hp.nside2npix(nside)
    mask = np.ones(npix)

    # Mask pixels near the galactic plane (simple approximation)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    galactic_lat = np.pi / 2 - theta

    # Mask pixels within some latitude range
    mask_condition = np.abs(galactic_lat) < mask_fraction * np.pi / 2
    mask[mask_condition] = 0

    return mask


def generate_parameter_grid_with_spectra(n_points=5, lmax=64):
    """
    Generate a small parameter grid with corresponding theoretical spectra.

    Parameters
    ----------
    n_points : int
        Number of points per parameter.
    lmax : int
        Maximum multipole.

    Returns
    -------
    param_ranges : dict
        Parameter ranges.
    theoretical_spectra : dict
        Theoretical spectra for each parameter combination.
    """
    # Define parameter ranges (small for testing)
    omega_b_values = np.linspace(0.020, 0.025, n_points)
    omega_c_values = np.linspace(0.10, 0.14, n_points)

    param_ranges = {"omega_b": omega_b_values, "omega_c": omega_c_values}

    # Generate theoretical spectra for each parameter combination
    theoretical_spectra = {}
    ell = np.arange(lmax + 1)

    for omega_b in omega_b_values:
        for omega_c in omega_c_values:
            # Simple scaling model
            scale_factor = (omega_b / 0.0225) * (omega_c / 0.12)

            # Base spectrum (similar to generate_synthetic_cmb_map)
            cl_theory = np.zeros(lmax + 1)
            cl_theory[2:] = (
                scale_factor * 1000 * ell[2:] ** (-1.1) * np.exp(-ell[2:] / 40)
            )

            theoretical_spectra[(omega_b, omega_c)] = cl_theory

    return param_ranges, theoretical_spectra


class SimplePICSLike:
    """
    Simplified version of PICSLike for demonstration purposes.

    This class implements the core pixel-based likelihood computation
    without all the complex infrastructure of the full PICSLike class.
    """

    def __init__(self, data_map, noise_variance, mask=None, nside=None):
        """
        Initialize with data and noise properties.

        Parameters
        ----------
        data_map : numpy.ndarray
            Observed data map.
        noise_variance : float
            Noise variance per pixel.
        mask : numpy.ndarray, optional
            Binary mask (1=good, 0=masked).
        nside : int, optional
            HEALPix nside parameter for the map.
        """
        self.data_map = data_map
        self.noise_variance = noise_variance
        self.nside = nside or hp.get_nside(data_map)

        if mask is None:
            self.mask = np.ones_like(data_map)
        else:
            self.mask = mask

        # Create masked data vector
        self.good_pixels = self.mask > 0.5
        self.data_vector = self.data_map[self.good_pixels]
        self.n_pixels = len(self.data_vector)

        # Set up field collection for efficient covariance computation
        self._setup_field_collection()

        print(f"Using {self.n_pixels} unmasked pixels")

    def _setup_field_collection(self):
        """Set up FieldCollection for efficient signal covariance computation."""

        # Create a simple BaseField-like object for temperature data
        class SimpleField:
            def __init__(self, data, mask, nside):
                self.data = data[mask > 0.5]  # Only unmasked pixels
                self.mask = mask
                self.nside = nside
                self.spin = 0  # Temperature is spin-0
                self.maps_label = "T"

                # Compute pointing vectors for unmasked pixels
                pixel_indices = np.where(mask > 0.5)[0]
                # Convert pixel indices to 3D pointing vectors
                import healpy as hp

                n_pixels = len(pixel_indices)
                self.point_vectors = np.empty((n_pixels, 3), dtype=np.float64)

                for i, pix in enumerate(pixel_indices):
                    theta, phi = hp.pix2ang(nside, pix, nest=False)
                    x = np.sin(theta) * np.cos(phi)
                    y = np.sin(theta) * np.sin(phi)
                    z = np.cos(theta)
                    norm = np.sqrt(x**2 + y**2 + z**2)
                    self.point_vectors[i, 0] = x / norm
                    self.point_vectors[i, 1] = y / norm
                    self.point_vectors[i, 2] = z / norm

                self.n_active = len(self.data)

                # Store power spectrum (will be updated for each likelihood calculation)
                self.cls = {}

        # Create field collection with our temperature field
        self.field = SimpleField(self.data_map, self.mask, self.nside)

        # Create minimal FieldCollection-like structure
        class SimpleFieldCollection:
            def __init__(self, field):
                self.fields = [field]
                self.spin = [0]  # Temperature only
                self.n_active = [field.n_active]

            def get_cls(self, i, j, spec_type):
                """Get power spectrum for field pair (i,j) of type spec_type."""
                if i == 0 and j == 0 and spec_type == 0:
                    return self.fields[0].cls.get("TT", np.zeros(100))
                return np.zeros(100)

            def update_cls(self, cl_theory):
                """Update power spectrum for current parameter point."""
                self.fields[0].cls["TT"] = cl_theory

        self.field_collection = SimpleFieldCollection(self.field)

    def compute_signal_covariance(self, cl_theory, nside):
        """
        Compute signal covariance matrix from power spectrum using cosmocore.

        Parameters
        ----------
        cl_theory : numpy.ndarray
            Theoretical power spectrum.
        nside : int
            HEALPix nside parameter.

        Returns
        -------
        signal_cov : numpy.ndarray
            Signal covariance matrix for unmasked pixels.
        """
        # Get active pixel indices
        pixel_indices = np.where(self.mask > 0.5)[0]
        n_pixels = self.n_pixels
        signal_cov = np.zeros((n_pixels, n_pixels), dtype=np.float64)

        # Use HealPy to compute pixel correlations
        import healpy as hp
        from scipy.special import eval_legendre

        lmax = len(cl_theory) - 1

        # For each pixel pair, compute the correlation
        for i in range(n_pixels):
            for j in range(i, n_pixels):
                # Get angles for both pixels
                theta_i, phi_i = hp.pix2ang(nside, pixel_indices[i])
                theta_j, phi_j = hp.pix2ang(nside, pixel_indices[j])

                # Compute angular separation
                cos_angle = np.cos(theta_i) * np.cos(theta_j) + np.sin(theta_i) * np.sin(
                    theta_j
                ) * np.cos(phi_i - phi_j)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)

                # Compute correlation using Legendre polynomials
                correlation = 0.0
                for ell in range(1, lmax + 1):  # Skip monopole for CMB
                    if ell < len(cl_theory):
                        # Legendre polynomial evaluation
                        p_l = eval_legendre(ell, cos_angle)
                        correlation += (2 * ell + 1) / (4 * np.pi) * cl_theory[ell] * p_l

                signal_cov[i, j] = correlation
                signal_cov[j, i] = correlation  # Symmetric matrix

        return signal_cov

    def compute_likelihood(self, cl_theory, nside):
        """
        Compute likelihood for given theoretical spectrum.

        Parameters
        ----------
        cl_theory : numpy.ndarray
            Theoretical power spectrum.
        nside : int
            HEALPix nside parameter.

        Returns
        -------
        chi_squared : float
            Chi-squared value.
        log_likelihood : float
            Log-likelihood value.
        """
        # Compute signal covariance
        signal_cov = self.compute_signal_covariance(cl_theory, nside)

        # Add noise (diagonal)
        noise_cov = self.noise_variance * np.eye(self.n_pixels)
        total_cov = signal_cov + noise_cov

        # Add small regularization for numerical stability
        total_cov += 1e-10 * np.eye(self.n_pixels)

        try:
            # Invert covariance matrix
            cov_inv = np.linalg.inv(total_cov)

            # Compute chi-squared
            chi_squared = float(self.data_vector.T @ cov_inv @ self.data_vector)

            # Log-likelihood (ignoring normalization constants)
            log_likelihood = -0.5 * chi_squared

            return chi_squared, log_likelihood

        except np.linalg.LinAlgError:
            print(
                "Warning: Covariance matrix inversion failed, returning large chi-squared"
            )
            return 1e10, -1e10


def main():
    """Main demonstration of pixel-based likelihood analysis."""
    print("=== Pixel-Based Likelihood Demonstration ===")
    print()

    # Parameters
    nside = 16  # Even lower resolution for speed (1536 pixels total)
    lmax = 32  # Reduced lmax for faster computation
    n_param_points = 3  # Small grid for demonstration

    print(f"Using nside={nside} (npix={hp.nside2npix(nside)})")
    print(
        f"Parameter grid: {n_param_points}x{n_param_points} = {n_param_points**2} points"
    )
    print()

    # Generate synthetic data
    print("1. Generating synthetic CMB data...")
    true_map, true_cl = generate_synthetic_cmb_map(nside=nside, lmax=lmax)
    noise_map = create_noise_map(nside=nside, noise_level=0.1)
    mask = create_simple_mask(nside=nside, mask_fraction=0.2)

    # Create observed data (signal + noise)
    observed_map = true_map + noise_map

    print(f"  True signal RMS: {np.std(true_map):.3f}")
    print(f"  Noise RMS: {np.std(noise_map):.3f}")
    print(f"  Mask: {np.sum(mask) / len(mask) * 100:.1f}% of sky observed")
    print()

    # Set up parameter grid
    print("2. Setting up parameter grid...")
    param_ranges, theoretical_spectra = generate_parameter_grid_with_spectra(
        n_points=n_param_points, lmax=lmax
    )

    print("  Parameter ranges:")
    for param_name, values in param_ranges.items():
        print(f"    {param_name}: {values[0]:.4f} - {values[-1]:.4f}")
    print()

    # Initialize likelihood computer
    print("3. Computing likelihoods...")
    likelihood_computer = SimplePICSLike(
        data_map=observed_map,
        noise_variance=0.1**2,  # noise_level^2
        mask=mask,
        nside=nside,
    )

    # Compute likelihood for each parameter combination
    chi_squared_values = []
    log_likelihood_values = []
    parameter_points = []

    total_points = n_param_points**2

    print("  Computing likelihoods with progress tracking...")

    for i, (param_point, cl_theory) in enumerate(
        tqdm(theoretical_spectra.items(), desc="Parameter points", total=total_points)
    ):
        omega_b, omega_c = param_point

        chi2, log_like = likelihood_computer.compute_likelihood(cl_theory, nside)

        chi_squared_values.append(chi2)
        log_likelihood_values.append(log_like)
        parameter_points.append(param_point)

        print(
            f"    Point {i + 1}/{total_points}: "
            f"ω_b={omega_b:.4f}, ω_c={omega_c:.3f} → χ²={chi2:.1f}"
        )

    print()

    # Analyze results
    print("4. Results Analysis:")
    chi_squared_values = np.array(chi_squared_values)
    log_likelihood_values = np.array(log_likelihood_values)

    # Find best-fit
    best_idx = np.argmin(chi_squared_values)
    best_chi2 = chi_squared_values[best_idx]
    best_params = parameter_points[best_idx]

    print(f"  Best-fit parameters: ω_b={best_params[0]:.4f}, ω_c={best_params[1]:.3f}")
    print(f"  Minimum χ²: {best_chi2:.1f}")
    print(
        "  Range of χ² values: "
        f"{np.min(chi_squared_values):.1f} - {np.max(chi_squared_values):.1f}"
    )
    print()

    # True parameters (used to generate the data)
    true_omega_b = 0.0225  # Middle of our range
    true_omega_c = 0.12  # Middle of our range
    print(f"  True parameters: ω_b={true_omega_b:.4f}, ω_c={true_omega_c:.3f}")

    # Find closest grid point to true parameters
    distances = [
        (abs(p[0] - true_omega_b) + abs(p[1] - true_omega_c)) for p in parameter_points
    ]
    true_idx = np.argmin(distances)
    true_chi2 = chi_squared_values[true_idx]

    print(f"  χ² at closest grid point to truth: {true_chi2:.1f}")
    print(f"  Δχ² from best-fit: {true_chi2 - best_chi2:.1f}")
    print()

    # Summary
    print("5. Summary:")
    print(
        "  Successfully computed pixel-based likelihood on "
        f"{total_points} parameter points"
    )
    print(
        f"  Used {likelihood_computer.n_pixels} pixels out of "
        f"{hp.nside2npix(nside)} total"
    )
    print(f"  Best-fit found at: ω_b={best_params[0]:.4f}, ω_c={best_params[1]:.3f}")

    if true_chi2 - best_chi2 < 5:  # Within reasonable range
        print("  ✓ Results look reasonable (true parameters have low χ²)")
    else:
        print(
            "  ⚠ True parameters have high χ² - may indicate issues or limited resolution"
        )

    print()
    print("=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
