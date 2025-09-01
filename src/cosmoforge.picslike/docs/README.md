# PICSLike Documentation

## Overview

PICSLike is a comprehensive Python package for pixel-based likelihood analysis in cosmological parameter estimation. It provides tools for computing the likelihood of observational cosmic microwave background (CMB) data given theoretical predictions in pixel space, offering an alternative approach to traditional harmonic-space methods.

## Key Features

### Pixel-Space Analysis
- Direct likelihood computation in map pixel space
- Natural handling of masked regions and incomplete sky coverage
- Efficient treatment of non-Gaussian signals and systematic effects

### Parameter Estimation
- Support for parameter grids and matrices with theoretical spectra
- Automatic signal covariance matrix computation for each parameter point
- Chi-squared statistics computation and storage across parameter space

### High Performance Computing
- MPI parallelization for scalable computation
- Memory-optimized algorithms for large-scale analyses
- Integration with the broader CosmoForge ecosystem

## Scientific Background

The pixel-based likelihood method computes the likelihood function directly in map space:

```
ln L(θ) = -1/2 * (d - s(θ))^T * C^(-1) * (d - s(θ))
```

Where:
- `d` is the observed data vector
- `s(θ)` is the theoretical signal prediction for parameters θ  
- `C` is the total covariance matrix (signal + noise)

This approach offers several advantages:

1. **Incomplete Sky Coverage**: No complications from harmonic transforms of masked data
2. **Non-Gaussian Features**: Direct treatment of non-Gaussian signals and systematics
3. **Cross-Correlations**: Efficient computation of cross-correlations between different maps
4. **Computational Efficiency**: Potentially faster for certain analysis configurations

## Package Structure

```
picslike/
├── __init__.py           # Package initialization and exports
├── picslike.py           # Main PICSLike class
├── parameter_grid.py     # Parameter space management
└── likelihood_result.py  # Results storage and analysis
```

## Main Classes

### PICSLike
The main analysis class that inherits from `cosmocore.Core`. Handles:
- Observational data loading and preparation
- Parameter grid setup and management
- Likelihood computation across parameter space
- Results storage and retrieval

### ParameterGrid
Helper class for managing parameter spaces:
- Grid generation from parameter ranges
- Theoretical spectrum management
- MPI process distribution
- Parameter indexing and retrieval

### LikelihoodResult
Container for analysis results:
- Chi-squared and likelihood value storage
- Best-fit parameter extraction
- Confidence interval computation
- Result persistence and loading

## Usage Examples

### Basic Analysis
```python
from cosmoforge.picslike import PICSLike

# Initialize analysis
picslike = PICSLike("config/pixel_analysis.yaml")

# Set up parameter grid
param_ranges = {
    'omega_b': np.linspace(0.020, 0.025, 10),
    'omega_c': np.linspace(0.10, 0.14, 10),
}
picslike.setup_parameter_grid(param_ranges, theoretical_spectra)

# Compute likelihood
picslike.compute_likelihood_grid()

# Extract results
best_fit = picslike.get_best_fit()
chi2_values = picslike.get_chi_squared()
```

### MPI Parallel Execution
```bash
mpirun -n 4 python pixel_analysis.py config.yaml
```

### Result Analysis
```python
# Load saved results
result = LikelihoodResult.load("results.pkl")

# Get confidence intervals
intervals_68 = result.get_confidence_intervals(0.68)
intervals_95 = result.get_confidence_intervals(0.95)

# Marginalize over parameters
marg_omega_b = result.get_marginalized_likelihood('omega_b')
```

## Configuration

PICSLike uses YAML configuration files for analysis setup:

```yaml
analysis:
  lmax: 1000
  nside: 512
  output_dir: "outputs/"

fields:
  - name: "temperature"
    file: "data/planck_temperature_map.fits"
    noise_file: "data/planck_noise_map.fits"
    mask_file: "data/planck_mask.fits"

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
```

## Performance Considerations

### Memory Usage
- Covariance matrices scale as O(N_pix²) where N_pix is the number of pixels
- Consider using appropriate masking to reduce effective pixel count
- Monitor memory usage for high-resolution analyses

### Computational Scaling
- Likelihood computation scales as O(N_param × N_pix³)
- Matrix inversion is the dominant computational cost
- MPI parallelization distributes parameter points across processes

### Optimization Tips
- Use appropriate HEALPix resolution (nside) for your analysis needs
- Pre-compute theoretical spectra grids for efficiency
- Consider using reduced covariance matrices for large-scale analyses

## Integration with CosmoForge

PICSLike seamlessly integrates with other CosmoForge packages:

- **cosmocore**: Provides base functionality and field management
- **quelo**: Can be used for complementary QML power spectrum analysis
- **meta**: Provides workflow management and configuration utilities

## Testing

The package includes comprehensive unit tests:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=picslike tests/

# Run specific test modules
pytest tests/test_parameter_grid.py
```

## Future Enhancements

Planned features for future releases:

1. **Advanced Covariance Models**: Support for more sophisticated noise models
2. **Bayesian Sampling**: Integration with MCMC samplers for full posterior exploration
3. **Visualization Tools**: Built-in plotting capabilities for results analysis
4. **Cross-Survey Analysis**: Enhanced support for multi-survey joint analyses

## References

1. Wandelt, B.D., Larson, D.L. & Lakshminarayanan, A. "Global, exact cosmic microwave background data analysis using Gibbs sampling" Phys. Rev. D 70, 083511 (2004)

2. Jewell, J., Levin, S. & Anderson, C.H. "Application of MCMC methods to multi-frequency CMB data sets" Astrophys. J. 609, 1-6 (2004)

3. Chu, M. et al. "Cosmic microwave background likelihood approximation by a Gaussianized Blackwell-Rao estimator" Phys. Rev. D 71, 103002 (2005)
