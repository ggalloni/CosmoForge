#!/usr/bin/env python3
"""
Main script for pixel-based likelihood analysis.

This script demonstrates the complete workflow for performing pixel-based
likelihood analysis using the PICSLike package. It includes parameter
grid setup, theoretical spectrum management, and likelihood computation.

Usage
-----
Single process:
    python main_picslike.py config/pixel_analysis.yaml

MPI parallel:
    mpirun -n 4 python main_picslike.py config/pixel_analysis.yaml

Example configuration file format:
    analysis:
        lmax: 1000
        nside: 512
        output_dir: "outputs/"

    fields:
        - name: "temperature"
          file: "data/planck_temperature_map.fits"
          noise_file: "data/planck_noise_map.fits"

    parameters:
        omega_b:
            min: 0.020
            max: 0.025
            n_points: 10
        omega_c:
            min: 0.10
            max: 0.14
            n_points: 10

    theoretical_spectra:
        file: "theory/theoretical_spectra_grid.pkl"
"""

import sys
from pathlib import Path

import numpy as np

from picslike import PICSLike


def load_theoretical_spectra(spectra_file: str) -> dict:
    """
    Load theoretical spectra from file.

    Parameters
    ----------
    spectra_file : str
        Path to file containing theoretical spectra grid.

    Returns
    -------
    spectra_dict : dict
        Dictionary mapping parameter tuples to power spectra.

    Notes
    -----
    This is a placeholder implementation. In practice, this would load
    pre-computed theoretical spectra from CAMB, CLASS, or similar codes.
    """
    import pickle

    try:
        with open(spectra_file, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        # Generate dummy spectra for demonstration
        print("Warning: Theoretical spectra file not found. Generating dummy data.")
        return generate_dummy_spectra()


def generate_dummy_spectra() -> dict:
    """
    Generate dummy theoretical spectra for demonstration.

    Returns
    -------
    spectra_dict : dict
        Dictionary with dummy theoretical spectra.

    Notes
    -----
    This generates placeholder spectra for testing purposes.
    Real applications should use proper cosmological codes.
    """
    spectra = {}

    # Example parameter ranges
    omega_b_values = np.linspace(0.020, 0.025, 5)
    omega_c_values = np.linspace(0.10, 0.14, 5)

    # Generate dummy power spectra (TT only for simplicity)
    ell = np.arange(2, 1001)

    for omega_b in omega_b_values:
        for omega_c in omega_c_values:
            # Simple model: scale base spectrum by parameters
            scale_factor = (omega_b / 0.0225) * (omega_c / 0.12)

            # Base CMB-like spectrum (very simplified)
            base_spectrum = 1000 * ell ** (-1.1) * np.exp(-ell / 800)
            cl_theory = scale_factor * base_spectrum

            # Store as dictionary entry
            param_tuple = (omega_b, omega_c)
            spectra[param_tuple] = cl_theory

    return spectra


def setup_parameter_ranges(config: dict) -> dict:
    """
    Setup parameter ranges from configuration.

    Parameters
    ----------
    config : dict
        Configuration dictionary with parameter definitions.

    Returns
    -------
    param_ranges : dict
        Dictionary mapping parameter names to value arrays.
    """
    param_ranges = {}

    for param_name, param_config in config["parameters"].items():
        param_ranges[param_name] = np.linspace(
            param_config["min"], param_config["max"], param_config["n_points"]
        )

    return param_ranges


def main():
    """Main analysis workflow."""
    if len(sys.argv) != 2:
        print("Usage: python main_picslike.py <config_file>")
        sys.exit(1)

    config_file = sys.argv[1]

    # Initialize PICSLike analysis
    print("Initializing PICSLike analysis...")
    picslike = PICSLike(config_file)

    # Load theoretical spectra
    print("Loading theoretical spectra...")
    spectra_file = picslike.params.get("theoretical_spectra", {}).get("file", "")
    theoretical_spectra = load_theoretical_spectra(spectra_file)

    # Setup parameter grid
    print("Setting up parameter grid...")
    param_ranges = setup_parameter_ranges(picslike.params)
    picslike.setup_parameter_grid(param_ranges, theoretical_spectra)

    # Load observational data
    print("Loading observational data...")
    picslike.load_maps()

    # Compute likelihood across parameter grid
    print("Computing likelihood grid...")
    picslike.compute_likelihood_grid()

    # Extract and display results
    if picslike.rank == 0:
        print("\nAnalysis Results:")
        print("=" * 50)

        # Get best-fit parameters
        best_fit = picslike.get_best_fit()
        print(f"Best-fit parameters: {best_fit}")

        # Get chi-squared values
        chi2_values = picslike.get_chi_squared()
        print(f"Minimum χ²: {np.min(chi2_values):.3f}")

        # Save results
        output_dir = Path(
            picslike.params.get("analysis", {}).get("output_dir", "outputs/")
        )
        output_file = output_dir / "picslike_results.pkl"
        picslike.save_results(output_file)

        print(f"Results saved to: {output_file}")

        # Generate summary report
        result = picslike.likelihood_result
        summary = result.get_summary_statistics()

        print("\nSummary Statistics:")
        print("-" * 30)
        for key, value in summary.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for subkey, subvalue in value.items():
                    print(f"  {subkey}: {subvalue}")
            else:
                print(f"{key}: {value}")


if __name__ == "__main__":
    main()
