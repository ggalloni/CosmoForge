<p align="center">
  <img src="https://raw.githubusercontent.com/ggalloni/CosmoForge/master/src/cosmoforge.picslike/logos/logo_picslike_light.png#gh-light-mode-only" alt="PICSLike logo (light)" width="50%"/>
  <img src="https://raw.githubusercontent.com/ggalloni/CosmoForge/master/src/cosmoforge.picslike/logos/logo_picslike_dark.png#gh-dark-mode-only" alt="PICSLike logo (dark)" width="50%"/>
</p>

[![PyPI](https://img.shields.io/pypi/v/picslike?include_prereleases)](https://pypi.org/project/picslike/)
[![Python](https://img.shields.io/pypi/pyversions/picslike)](https://pypi.org/project/picslike/)
[![Documentation](https://img.shields.io/badge/docs-picslike-blue.svg)](https://ggalloni.github.io/CosmoForge/api/picslike.html)
[![Parallelization](https://img.shields.io/badge/MPI-parallel-orange.svg)](https://www.mpi-forum.org/)
[![Method](https://img.shields.io/badge/method-Pixel--based%20Likelihood-red.svg)](https://en.wikipedia.org/wiki/Likelihood_function)

> **[PICSLike Documentation](https://ggalloni.github.io/CosmoForge/api/picslike.html) | [Main Documentation](https://ggalloni.github.io/CosmoForge/)**

PICSLike (Pixel-based Inference with Correlated-Skies Likelihood) is the pixel-based likelihood analysis engine of CosmoForge, implementing direct likelihood evaluation in map pixel space for cosmological parameter inference from CMB data.

## Overview

PICSLike provides pixel-based likelihood computation as an alternative to harmonic-space methods:

- **Pixel-based Likelihood**: Direct likelihood evaluation in map pixel space
- **Parameter Grid Analysis**: Efficient likelihood computation across parameter grids

The approach is particularly useful for handling incomplete sky coverage, non-Gaussian features, and scenarios where pixel-space analysis offers computational or methodological advantages.

## Key Features

### Pixel-based Likelihood Analysis

- Direct likelihood evaluation in pixel space
- Natural handling of masked regions
- Support for temperature and polarization data
- Cross-correlation analysis support
- Chi-squared and log-likelihood computation

### Parameter Grid Management

- Flexible parameter range specification
- Theoretical spectra grid management
- Efficient MPI distribution across parameter points
- Best-fit parameter extraction
- Confidence interval computation

### Technical Features

- **MPI Parallelization**: Distributed computation across parameter grids
- **Memory Optimization**: Efficient covariance matrix handling
- **Instrumental Effects**: Beam convolution and pixel window functions
- **Flexible Configuration**: YAML-based parameter specification

## Installation

PICSLike is part of CosmoForge and installs automatically:

```bash
pip install -e /path/to/CosmoForge
```

For MPI support:

```bash
pip install mpi4py
```

## Documentation

For detailed API documentation and examples:

- **[PICSLike API](https://ggalloni.github.io/CosmoForge/api/picslike.html)** - Pixel-based likelihood computation
- **[Parameter Grid](https://ggalloni.github.io/CosmoForge/api/picslike/parameter_grid.html)** - Parameter space management
- **[Likelihood Results](https://ggalloni.github.io/CosmoForge/api/picslike/likelihood_result.html)** - Result storage and analysis

## Quick Start

### Basic Pixel-based Likelihood Analysis

```python
from picslike import PICSLike

# Initialize likelihood analysis
picslike = PICSLike("config/pixel_analysis.yaml")

# Run full pipeline
picslike.run()

# Get results
chi_squared = picslike.get_chi_squared()
best_fit = picslike.get_best_fit()
log_likelihood = picslike.get_log_likelihood()
```

### Step-by-Step Execution

```python
from picslike import PICSLike

# Initialize analysis
picslike = PICSLike("config/pixel_config.yaml")

# Setup pipeline components
picslike.setup_parameter_grid()
picslike.setup_fields()
picslike.setup_geometry()
picslike.setup_covariance_matrices()
picslike.setup_cls()
picslike.setup_beams()
picslike.setup_maps()

# Compute likelihood across parameter grid
picslike.compute()

# Save results
picslike.save_results("output/likelihood_results.pkl")
```

### Accessing Simulation Results

```python
from picslike import PICSLike

# Run analysis
picslike = PICSLike("config.yaml")
picslike.run()

# Get individual simulation results
simulation_results = picslike.get_simulation_results()

# Get mean likelihood result
mean_result = picslike.get_mean_likelihood_result()

# Extract summary statistics
summary = mean_result.get_summary_statistics()
```

## Configuration

PICSLike uses YAML configuration files:

### Basic Configuration

```yaml
# Data specification
nside: 32
lmax: 64
fields: "TEB"  # or "TQU"

# Input files
maskfile: "data/mask.fits"
inputclfile: "data/fiducial_cls.txt"
covmatfile1: "data/noise_cov1.bin"
covmatfile2: "data/noise_cov2.bin"  # for cross-correlation

# Analysis parameters
do_cross: false
calibration: 1.0
nsims: 100
```

### Parameter Grid Configuration

```yaml
# Parameter ranges for likelihood grid
parameters:
  amplitude:
    - 0.5    # min
    - 1.5    # max
    - 11     # n_points
  tilt:
    - -0.5   # min
    - 0.5    # max
    - 11     # n_points

# Theoretical spectra location
root_dir: "data/templates"
root_filename: "dls"
```

### Beam Configuration

```yaml
# Beam parameters
smoothing_type: gaussian  # none, gaussian, cosine_legacy, cosine_npipe, file
fwhmarcmin: 5.0
beam_file: "data/beam.fits"
apply_pixwin: true
smooth_pol: true
```

## MPI Usage

Run analyses in parallel:

```bash
# Pixel-based likelihood analysis
mpirun -n 4 python -c "
from picslike import PICSLike
picslike = PICSLike('config.yaml')
picslike.run()
"

# Full analysis with result saving
mpirun -n 8 python main_picslike.py config/pixel_analysis.yaml
```

## Analysis Pipeline

### Likelihood Computation Pipeline

1. **Setup**: Read parameters and initialize fields
2. **Parameter Grid**: Configure parameter ranges and load theoretical spectra
3. **Geometry**: Compute pixel pointing vectors
4. **Covariance**: Load noise covariance matrices
5. **Maps**: Read input observation maps
6. **Broadcast**: Distribute data across MPI processes
7. **Compute**: Evaluate likelihood at each parameter point
8. **Results**: Gather and store likelihood values

### Likelihood Function

The pixel-based likelihood is computed as:

```
ln L(theta) = -1/2 * (d - s(theta))^T * C^(-1) * (d - s(theta))
```

where:
- `d` is the observed data vector
- `s(theta)` is the theoretical signal for parameters theta
- `C` is the total covariance matrix (signal + noise)

## API Reference

### PICSLike Class

```python
class PICSLike(Core):
    """Pixel-based likelihood analysis implementation."""

    def run(self):
        """Execute complete likelihood analysis."""

    def compute(self):
        """Compute likelihood across parameter grid."""

    def get_chi_squared(self):
        """Get chi-squared values."""

    def get_log_likelihood(self):
        """Get log-likelihood values."""

    def get_best_fit(self):
        """Get best-fit parameter values."""

    def save_results(self, output_path):
        """Save results to file."""
```

### ParameterGrid Class

```python
class ParameterGrid:
    """Manager for parameter grids and theoretical spectra."""

    def get_total_points(self):
        """Get total number of grid points."""

    def get_points_for_process(self, rank, size):
        """Get parameter points for an MPI process."""

    def get_spectrum(self, param_point):
        """Get theoretical spectrum for a parameter point."""
```

### LikelihoodResult Class

```python
class LikelihoodResult:
    """Container for likelihood computation results."""

    def get_best_fit(self):
        """Get best-fit parameter values."""

    def get_confidence_intervals(self, confidence_level=0.68):
        """Compute confidence intervals."""

    def get_marginalized_likelihood(self, parameter_name):
        """Get marginalized likelihood for a parameter."""

    def get_summary_statistics(self):
        """Get comprehensive summary statistics."""

    def save(self, output_path):
        """Save results to file."""

    @classmethod
    def load(cls, input_path):
        """Load results from file."""
```

## Output Files

### Likelihood Analysis Outputs

- `likelihood_results.pkl`: Complete likelihood results
- `likelihood_results_sim_XX.pkl`: Individual simulation results

### Result Content

- Chi-squared values across parameter grid
- Log-likelihood values
- Best-fit parameters
- Confidence intervals (68%, 95%)
- Marginalized likelihoods

## Performance Optimization

### Memory Management

- Efficient covariance matrix handling
- In-place operations where possible
- Memory-mapped file I/O for large datasets

### Computational Efficiency

- Signal matrix computation optimized with Numba
- Optimized BLAS/LAPACK operations
- MPI load balancing across parameter points

### Scaling

Typical performance scaling:

```text
Processes    Speed-up
1           1.0x
2           1.9x
4           3.7x
8           7.3x
16          14.2x
```

## Testing

Run the test suite (from the repository root):

```bash
uv run pytest src/cosmoforge.picslike/tests/ -s
```

Specific tests:

```bash
uv run pytest src/cosmoforge.picslike/tests/test_likelihood_result.py -s
uv run pytest src/cosmoforge.picslike/tests/test_parameter_grid.py -s
uv run pytest src/cosmoforge.picslike/tests/test_picslike.py -s
```

## Examples

### Mock Data Generation

```bash
# Generate mock inputs for temperature-only analysis
python scripts/produce_mock_inputs.py T /data/mocks config.yaml

# Generate with Galactic mask
python scripts/produce_mock_inputs.py TQU /data/mocks config.yaml --mask
```

### Parameter Constraint Analysis

```python
from picslike import PICSLike, LikelihoodResult

# Run likelihood analysis
picslike = PICSLike("config.yaml")
picslike.run()

# Get likelihood result
result = picslike.get_mean_likelihood_result()

# Get confidence intervals
intervals_68 = result.get_confidence_intervals(0.68)
intervals_95 = result.get_confidence_intervals(0.95)

# Get marginalized likelihood for specific parameter
marg_like = result.get_marginalized_likelihood('amplitude')

print(f"Best-fit parameters: {result.get_best_fit()}")
print(f"68% confidence intervals: {intervals_68}")
```

## Scientific Context

Pixel-based likelihood methods provide several advantages:

1. **Incomplete sky coverage**: Natural handling of masked regions without harmonic-space complications
2. **Non-Gaussian features**: Direct treatment of non-Gaussian signals and systematics
3. **Cross-correlation analysis**: Efficient computation of cross-correlations between different maps
4. **Computational efficiency**: Faster for certain analysis configurations

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce nside or use more MPI processes
2. **Missing spectra errors**: Ensure theoretical spectra exist for all grid points
3. **MPI hanging**: Ensure consistent configuration across processes

### Debug Mode

Enable detailed logging:

```yaml
feedback: 4  # Maximum verbosity
```

### Performance Profiling

```python
import cProfile

def run_analysis():
    picslike = PICSLike("config.yaml")
    picslike.run()

cProfile.run('run_analysis()', 'profile_output.prof')
```

## Contributing

See the main CosmoForge README for contribution guidelines.

## Citation

If you use PICSLike (as part of CosmoForge) in your research, please cite:

> Galloni, G. & Pagano, L., *CosmoForge I: A unified framework for QML power spectrum estimation and pixel-based likelihood analysis*, in preparation (2026).

```bibtex
@article{GalloniPagano_CosmoForgeI,
    author = {Galloni, G. and Pagano, L.},
    title  = {{CosmoForge I}: A unified framework for {QML} power spectrum estimation and pixel-based likelihood analysis},
    year   = {2026},
    note   = {in preparation}
}
```

This entry will be updated with the arXiv identifier and journal reference once available.

## References

- Wandelt, B.D., Larson, D.L. & Lakshminarayanan, A. "Global, exact cosmic microwave background data analysis using Gibbs sampling" Phys. Rev. D 70, 083511 (2004)
- Jewell, J., Levin, S. & Anderson, C.H. "Application of Monte Carlo algorithms to the Bayesian analysis of the cosmic microwave background" Astrophys. J. 609, 1-14 (2004)
- Eriksen, H.K. et al. "Power Spectrum Estimation from High-Resolution Maps by Gibbs Sampling" Astrophys. J. Suppl. 155, 227-241 (2004)
- Planck Collaboration "Planck 2018 results. V. CMB power spectra and likelihoods" Astron. Astrophys. 641, A5 (2020)
- Tegmark, M. "How to measure CMB power spectra without losing information" Phys. Rev. D 55, 5895 (1997)
