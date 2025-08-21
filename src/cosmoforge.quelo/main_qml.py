#!/usr/bin/env python3
"""
Main script for QML power spectrum estimation using the Spectra class.

This script demonstrates how to use the Spectra class for Quadratic Maximum
Likelihood power spectrum estimation.
"""

from quelo.spectra import Spectra


def main():
    """Main function to run QML power spectrum analysis."""
    # Create Spectra instance with parameter file
    qml_analyzer = Spectra("src/cosmoforge.quelo/quelo/TEB_defaults.yaml")

    # Run the complete QML analysis pipeline
    qml_analyzer.run()

    # Optionally, get results (only available on rank 0)
    power_spectra = qml_analyzer.get_power_spectra()
    if power_spectra is not None:
        print(f"Power spectra computed successfully with shape: {power_spectra.shape}")

        # Get noise bias if auto-correlation
        if not qml_analyzer.params.do_cross:
            noise_bias = qml_analyzer.get_noise_bias()
            if noise_bias is not None:
                print(f"Noise bias computed with shape: {noise_bias.shape}")


if __name__ == "__main__":
    main()
