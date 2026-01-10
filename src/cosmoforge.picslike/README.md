# PICSLike: Pixel-based Likelihood for Cosmological Parameter Inference

PICSLike is a Python package for pixel-based likelihood analysis in cosmological parameter estimation. It provides tools for computing the likelihood of observational data given theoretical predictions in pixel space, offering an alternative to harmonic-space methods for cases where pixel-space analysis is more appropriate or computationally efficient.

## Features

- **Pixel-based likelihood computation**: Direct likelihood evaluation in map pixel space
- **Parameter grid evaluation**: Support for parameter matrices/ranges with theoretical spectra
- **Signal covariance computation**: Automatic computation of signal covariance matrices for each parameter point
- **Chi-squared statistics**: Efficient computation and storage of chi-squared values across parameter space
- **MPI parallelization**: Scalable computation across parameter grids
- **Integration with CosmoForge**: Seamless integration with the broader CosmoForge ecosystem

## Key Classes

- **PICSLike**: Main class for pixel-based likelihood analysis, inheriting from `cosmocore.Core`
- **ParameterGrid**: Helper class for managing parameter ranges and theoretical spectra
- **LikelihoodResult**: Container for storing and managing likelihood computation results

## Scientific Context

Pixel-based likelihood methods provide several advantages in certain scenarios:

1. **Incomplete sky coverage**: Natural handling of masked regions without harmonic-space complications
2. **Non-Gaussian features**: Direct treatment of non-Gaussian signals and systematics
3. **Cross-correlation analysis**: Efficient computation of cross-correlations between different maps
4. **Computational efficiency**: Potentially faster for certain analysis configurations

The likelihood function computed is:

```
ln L(θ) = -1/2 * (d - s(θ))^T * C^(-1) * (d - s(θ))
```

where:
- `d` is the observed data vector
- `s(θ)` is the theoretical signal for parameters θ
- `C` is the total covariance matrix (signal + noise)

## Usage

```python
from picslike import PICSLike

# Initialize with parameter file
picslike = PICSLike(params_file="config/pixel_analysis.yaml")

# Run full pipeline (setup + compute)
picslike.run()

# Or step-by-step:
picslike.setup_parameter_grid()
picslike.setup_fields()
picslike.setup_geometry()
picslike.setup_covariance_matrices()
picslike.setup_cls()
picslike.setup_beams()
picslike.setup_maps()
picslike.compute()

# Extract results
chi_squared_values = picslike.get_chi_squared()
best_fit_params = picslike.get_best_fit()
```

## Requirements

- Python ≥ 3.11
- cosmocore (CosmoForge core package)
- NumPy
- SciPy
- tqdm (for progress bars)
- mpi4py (for parallel computation)

## Installation

PICSLike is part of the CosmoForge ecosystem. Install from the monorepo root:

```bash
uv pip install -e src/cosmoforge.picslike/
```

## License

This project is licensed under the same terms as the CosmoForge project.
