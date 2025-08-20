#!/usr/bin/env python3
"""
Main script for Fisher matrix computation using the Fisher class.

This script replaces the original dofisher.py with a cleaner class-based approach.
"""

from quelo.fisher import Fisher


def main():
    """Main function to run Fisher matrix analysis."""
    # Create Fisher instance with parameter file
    fisher_analyzer = Fisher("src/cosmoforge.quelo/quelo/TEB_defaults.yaml")

    # Run the complete analysis pipeline
    fisher_analyzer.run()

    # Optionally, get results (only available on rank 0)
    fisher_matrix = fisher_analyzer.get_fisher_matrix()
    if fisher_matrix is not None:
        print(f"Fisher matrix computed successfully with shape: {fisher_matrix.shape}")

        # Get parameter errors
        errors = fisher_analyzer.get_parameter_errors()
        if errors is not None:
            print(f"Parameter errors: {errors}")


if __name__ == "__main__":
    main()
