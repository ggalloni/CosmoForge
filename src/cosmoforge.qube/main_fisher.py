#!/usr/bin/env python3
"""
Fisher Matrix Computation Pipeline for Cosmological Parameter Forecasting.

This script provides the main entry point for Fisher information matrix computation
using the CosmoForge framework. The Fisher matrix quantifies the information content
of CMB observations about cosmological parameters and enables parameter constraint
forecasting for future or simulated surveys.

The Fisher matrix F_ij represents the curvature of the likelihood function:
F_ij = -E[∂²ln L/∂θ_i∂θ_j] = (1/2) * Tr[C^(-1) * ∂C/∂θ_i * C^(-1) * ∂C/∂θ_j]

where C is the total covariance matrix (signal + noise) and θ_i are the parameters
of interest. The inverse Fisher matrix provides the parameter covariance matrix
under the assumption of Gaussian posteriors.

Features:
- MPI-parallelized computation for scalability
- Support for multiple field combinations (T, QU, TQU, TEB)
- Integration with HEALPix pixelization schemes
- Beam convolution and noise modeling
- Output of Fisher matrices and parameter error forecasts

Usage:
    # Serial execution
    python main_fisher.py

    # MPI parallel execution
    mpirun -n 8 python main_fisher.py

Configuration:
    Analysis parameters are read from the YAML configuration file specified
    in the Fisher class initialization. Key parameters include:
    - HEALPix resolution (nside)
    - Multipole range (lmin, lmax)
    - Field combinations and cross-correlations
    - Beam and noise specifications
    - Output file paths

Output Files:
    - Fisher matrix: Binary matrix file for further analysis
    - Parameter errors: Text file with marginal 1σ uncertainties
    - Covariance matrices: Inverted covariance for QML analysis
    - Log files: Computation progress and timing information

Notes:
    The Fisher matrix computation is memory and compute intensive, scaling as
    O(N_pix^3) for matrix operations and O(N_ell^2) for Fisher elements.
    For high-resolution analyses, ensure adequate computational resources.

See Also:
    main_qml.py : QML power spectrum estimation using computed Fisher matrix
    produce_mock_inputs.py : Generate synthetic data for testing

References:
    .. [1] Tegmark, M. et al. "How to measure CMB power spectra without losing
       information" Phys. Rev. D 55, 5895 (1997)
    .. [2] Seljak, U. & Zaldarriaga, M. "A Line-of-Sight Integration Approach to
       Cosmic Microwave Background Anisotropies" Astrophys. J. 469, 437 (1996)
"""

from qube.fisher import Fisher


def main():
    """
    Execute Fisher matrix computation pipeline.

    This function orchestrates the complete Fisher matrix analysis workflow
    including parameter loading, field setup, covariance matrix computation,
    and parallel Fisher matrix calculation. The function handles MPI
    coordination automatically and produces output files for subsequent analysis.

    The analysis pipeline consists of:
    1. Parameter file parsing and validation
    2. HEALPix field collection setup
    3. Geometry computation and pixel pointing
    4. Noise and signal covariance matrix assembly
    5. Parallel Fisher matrix element computation
    6. Results gathering and output file generation

    Results are available only on the master process (rank 0) and include:
    - Full Fisher information matrix
    - Marginal parameter error forecasts
    - Inverted covariance matrices for QML analysis

    Raises:
        FileNotFoundError: If configuration file is not found
        ValueError: If parameters are inconsistent or invalid
        MPIError: If MPI communication fails during parallel computation

    Notes:
        The function uses a hardcoded configuration file path. For production
        use, consider making this configurable via command line arguments.

        Memory usage scales with HEALPix resolution and number of fields.
        Monitor available memory for high-resolution analyses.    Examples:
        Serial execution:
        >>> main()  # Uses default configuration file

        MPI execution (from command line):
        $ mpirun -n 4 python main_fisher.py
    """
    # Initialize Fisher matrix computation with configuration file
    # TODO: Consider making config file path configurable via command line
    fisher_analyzer = Fisher("src/cosmoforge.qube/qube/TEB_defaults.yaml")
    logger = fisher_analyzer.logger

    logger.info("Initializing Fisher matrix computation...")

    # Execute the complete Fisher analysis pipeline
    # This includes all setup phases and parallel computation
    logger.info("Starting Fisher matrix analysis pipeline...")
    fisher_analyzer.run()

    # Retrieve and display results (only available on master process)
    fisher_matrix = fisher_analyzer.get_fisher_matrix()
    if fisher_matrix is not None:
        logger.info("=" * 60)
        logger.info("FISHER MATRIX COMPUTATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.log_matrix_info("Fisher matrix", fisher_matrix, level=1)
        condition_number = fisher_matrix.max() / fisher_matrix.min()
        logger.log_with_feedback(f"Matrix condition number: {condition_number:.2e}", 3)

        # Compute and display parameter error forecasts
        errors = fisher_analyzer.get_error_bars()
        if errors is not None:
            logger.log_with_feedback("\nParameter error forecasts (1σ marginal):", 3)
            logger.log_with_feedback(f"Number of parameters: {len(errors)}", 1)
            logger.log_with_feedback(
                f"Error range: [{errors.min():.3e}, {errors.max():.3e}]", 3
            )
            logger.log_with_feedback(f"Mean fractional error: {errors.mean():.3e}", 3)

            # Display first few errors as examples
            logger.log_with_feedback("\nFirst 5 parameter errors:", 3)
            for i, err in enumerate(errors[:5]):
                logger.log_with_feedback(f"  Parameter {i + 1}: {err:.3e}", 3)
        else:
            logger.warning(
                "Warning: Could not compute parameter errors (singular Fisher matrix)"
            )
    else:
        logger.info("Fisher matrix computation completed (worker process)")

    logger.info("\nAnalysis complete. Check output files for detailed results.")


if __name__ == "__main__":
    main()
