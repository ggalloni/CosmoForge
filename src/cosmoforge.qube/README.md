<p align="center">
  <img src="https://raw.githubusercontent.com/ggalloni/CosmoForge/master/src/cosmoforge.qube/logos/QUBE_logo.png" alt="QUBE logo" width="40%"/>
</p>

[![PyPI](https://img.shields.io/pypi/v/qube-qml?include_prereleases)](https://pypi.org/project/qube-qml/)
[![Python](https://img.shields.io/pypi/pyversions/qube-qml)](https://pypi.org/project/qube-qml/)
[![Documentation](https://img.shields.io/badge/docs-qube-blue.svg)](https://cosmoforge.readthedocs.io/en/latest/api/qube.html)
[![Parallelization](https://img.shields.io/badge/MPI-parallel-orange.svg)](https://www.mpi-forum.org/)
[![QML Method](https://img.shields.io/badge/method-QML-red.svg)](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation)

> **📚 [QUBE Documentation](https://cosmoforge.readthedocs.io/en/latest/api/qube.html) | [Fisher Analysis API](https://cosmoforge.readthedocs.io/en/latest/api/qube/fisher.html) | [Spectra API](https://cosmoforge.readthedocs.io/en/latest/api/qube/spectra.html) | [Main Documentation](https://cosmoforge.readthedocs.io/en/latest/)**

QUBE: Quadratic maximum likelihood UnBiased Estimator is the analysis engine of CosmoForge, implementing Fisher matrix analysis and Quadratic Maximum Likelihood (QML) power spectrum estimation for any spin-0 or spin-2 field on the sphere. Applications include CMB temperature and polarization, galaxy surveys, 21 cm intensity mapping, and any other signal that can be described by angular power spectra.

## Overview

QUBE provides two main analysis methods:

- **Fisher Matrix Analysis**: Fast parameter forecasting and covariance estimation
- **QML Power Spectrum Estimation**: Optimal power spectrum recovery from noisy data

Both methods support MPI parallelization for large-scale analyses and can handle spin-0 and spin-2 data (e.g. temperature and polarization) with realistic instrumental effects.

## Key Features

### Fisher Matrix Analysis

- Fast parameter forecasting
- Covariance matrix computation
- Support for multiple field types (T, E, B)
- Cross-correlation analysis
- Instrumental noise modeling

### QML Power Spectrum Estimation

- Optimal power spectrum estimation
- Noise bias correction
- Cross-correlation support
- Three output modes: deconvolved, decorrelated, convolved
- Multipole binning for bandpower estimation

### Multipole Binning

- Native binned QML: Fisher and q-vector computed directly in bin space
- Configurable via Python API (`Bins` class) or YAML (`delta_ell`, `bin_lmins`/`bin_lmaxs`)
- Beam smoothing absorbed into binned derivatives for exact normalization
- Default `delta_ell=1` recovers standard per-multipole estimation
- Custom non-uniform bins supported

### Technical Features

- **MPI Parallelization**: Distributed computation across multiple processes
- **Computation Basis**: Harmonic and pixel basis with optional m-block compression and automatic field block-diagonal optimization
- **Instrumental Effects**: Beam convolution and pixel window functions
- **Flexible Configuration**: YAML-based parameter specification

## Installation

### Install from PyPI

```bash
pip install qube-qml
```

`qube-qml` is the distribution name on PyPI; the package is imported as ``qube``.

To enable MPI parallelism (requires a system MPI implementation):

```bash
pip install qube-qml[mpi]
```

Or install the full CosmoForge umbrella (cosmocore + qube-qml + picslike):

```bash
pip install cosmoforge
```

### Install from source

```bash
git clone https://github.com/ggalloni/CosmoForge.git
cd CosmoForge
uv sync --all-packages --all-extras --dev
```

## Documentation

For detailed API documentation and examples:

- **[Fisher Analysis](https://cosmoforge.readthedocs.io/en/latest/api/qube/fisher.html)** - Fisher matrix computation and parameter forecasting
- **[Spectra Analysis](https://cosmoforge.readthedocs.io/en/latest/api/qube/spectra.html)** - QML power spectrum estimation
- **[Main Scripts](https://cosmoforge.readthedocs.io/en/latest/api/qube/main_fisher.html)** - Command-line interfaces for analysis
- **[Mock Data Generation](https://cosmoforge.readthedocs.io/en/latest/api/qube/produce_mock_inputs.html)** - Mock data generation utilities

## Quick Start

### Fisher Matrix Analysis

```python
from qube import Fisher

# Initialize Fisher analysis
fisher = Fisher("config/fisher_params.yaml")

# Run Fisher computation
fisher.run()

# Get results
fisher_matrix = fisher.get_fisher_matrix()
covariance = fisher.get_covariance_matrix()
```

### QML Power Spectrum Estimation

```python
from qube import Spectra

# Initialize QML analysis
qml = Spectra("config/qml_params.yaml")

# Run QML computation
qml.run()

# Get power spectra
power_spectra = qml.get_power_spectra()
noise_bias = qml.get_noise_bias()
```

### Reusing Fisher for QML

```python
from qube import Fisher, Spectra

# Compute Fisher matrix once
fisher = Fisher("config/params.yaml")
fisher.run()

# Reuse for QML (more efficient)
qml = Spectra("config/params.yaml", fisher=fisher)
qml.run()
```

### Binned Power Spectrum Estimation

```python
from cosmocore import Bins
from qube import Fisher, Spectra

# Uniform bins of width 5
bins = Bins.fromdeltal(2, lmax, delta_ell=5)

# Or custom non-uniform bins (both bounds inclusive)
bins = Bins(lmins=[2, 10, 30], lmaxs=[9, 29, 64])

fisher = Fisher("config/params.yaml")
fisher.set_binning(bins)
fisher.run()

# Spectra inherits bins from Fisher
qml = Spectra("config/params.yaml", fisher=fisher)
qml.run()

cl_binned = qml.get_power_spectra()      # (nsims, nbins)
ell_eff = qml.get_effective_ells()        # bin midpoints
errors = qml.get_error_bars()             # (nbins,)
```

### In-Memory Inputs (no files)

Every input can be handed over as an array instead of a file path — for callers that
already hold their maps, mask and noise covariance and should not have to round-trip
them through disk. The config still supplies the scalars; an injected object wins over
the corresponding path.

```python
from qube import Fisher, Spectra

fisher = Fisher(
    params,
    mask=mask,             # (npix, nfields)
    noise_cov1=noise_cov,  # reduced (n_active, n_active)
    cls_data=cls,          # {label: C_ell}, physical C_ell
    fiducial_cls=cls,
    beam=beam,             # (>=3, lmax+1) T/E/B window functions
)
fisher.run()

# maps are a Spectra seam — Fisher never reads them
qml = Spectra(params, fisher=fisher, maps1=maps)
qml.run()
power_spectra = qml.get_power_spectra(mode="deconvolved")
```

Leave the `out*` paths unset and the run touches no disk at all.

Kwargs: `mask`, `noise_cov1`/`noise_cov2`, `maps1`/`maps2`, `cls_data`, `fiducial_cls`,
`beam`. Each injected object is *exactly what the corresponding reader would have
returned*. Two contracts are asymmetric — deliberately, because each mirrors what its
reader does:

| kwarg | contract |
|---|---|
| `noise_cov1` | reduced to active pixels |
| `maps1` | reduced to active pixels |

`noise_cov1`/`maps1` are *reduced* to the active (unmasked) pixels, concatenated across
fields — so the active-pixel ordering has to be known before they can be built.

Worked example: [`notebooks/in_memory_inputs.ipynb`](notebooks/in_memory_inputs.ipynb),
which drives the file adapter and the injection adapter over the same data and asserts
they produce identical spectra. Rationale: ADR-0017.

## Configuration

QUBE uses YAML configuration files:

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
remove_noise_bias: true
```

### QML-Specific Parameters

```yaml
# QML simulation parameters
nsims: 100
ssim: 0
zerofill: false

# Input maps
inputmapfile1: "data/maps1_{:04d}.fits"
inputmapfile2: "data/maps2_{:04d}.fits"
endname1: ".fits"
endname2: ".fits"

# Output files
outfisherfile: "output/fisher.dat"
outcovmatfile: "output/covariance.dat"
outerrfile: "output/errors.dat"
```

### Binning Configuration

```yaml
# Uniform bins (simplest)
delta_ell: 5

# Or custom bins (both bounds inclusive)
bin_lmins: [2, 10, 30]
bin_lmaxs: [9, 29, 64]
```

### Beam Configuration

```yaml
# Beam parameters
smoothing_type: gaussian  # none, gaussian, cosine_legacy, cosine_npipe, file
fwhmarcmin: 5.0
beam_file: "data/beam.fits"
apply_pixwin: true
```

## MPI Usage

Run analyses in parallel:

```bash
# Fisher analysis
mpirun -n 4 python -c "
from qube import Fisher
fisher = Fisher('config.yaml')
fisher.run()
"

# QML analysis
mpirun -n 8 python -c "
from qube import Spectra
qml = Spectra('config.yaml')
qml.run()
"
```

## Analysis Pipeline

### Fisher Matrix Pipeline

1. **Setup**: Read parameters and initialize fields
2. **Geometry**: Compute pixel pointing vectors
3. **Signal Matrix**: Compute theoretical signal covariance
4. **Derivatives**: Compute signal matrix derivatives
5. **Fisher Matrix**: Assemble Fisher information matrix
6. **Inversion**: Compute parameter covariance matrix

### QML Pipeline

1. **Setup**: Initialize fields and read Fisher matrix
2. **Maps**: Read input simulation maps
3. **Renormalization**: Apply Fisher matrix normalization
4. **E-operator**: Compute quadratic estimator operators
5. **Estimation**: Apply operators to data maps
6. **Reduction**: Combine results across MPI processes
7. **Final Spectra**: Apply Fisher matrix to get power spectra

## API Reference

### Fisher Class

```python
class Fisher(Core):
    """Fisher matrix analysis implementation."""
    
    def run(self):
        """Execute complete Fisher analysis."""
        
    def get_fisher_matrix(self):
        """Get computed Fisher matrix."""
        
    def get_covariance_matrix(self):
        """Get parameter covariance matrix."""
        
    def write_outputs(self):
        """Write results to files."""
```

### Spectra Class

```python
class Spectra(Core):
    """QML power spectrum estimation implementation."""
    
    def run(self):
        """Execute complete QML analysis."""
        
    def get_power_spectra(self):
        """Get estimated power spectra."""
        
    def get_noise_bias(self):
        """Get noise bias estimates."""
        
    def compute_qml_spectra(self):
        """Main QML computation loop."""
```

## Output Files

Persistence is opt-in (ADR-0015): each file below is written only when its
corresponding `out*` path is set in the config; with the path unset (the
default) nothing is written to disk.

### Fisher Analysis Outputs

- `fisher.dat`: Fisher information matrix
- `covariance.dat`: Parameter covariance matrix
- `errors.dat`: Parameter uncertainties
- `geometry.dat`: Pixel geometry information

### QML Analysis Outputs

- `spectra.dat`: Estimated power spectra
- `noise_bias.dat`: Noise bias estimates
- `covariance.dat`: Spectrum covariance matrix

## Performance Optimization

### Memory Management

- Efficient covariance matrix handling
- In-place operations where possible
- Memory-mapped file I/O for large datasets

### Computational Efficiency

- Numba JIT compilation for critical loops
- Optimized BLAS/LAPACK operations
- MPI load balancing

### Scaling

Typical performance scaling:

```text
Processes    Speed-up (Fisher)    Speed-up (QML)
1           1.0x                 1.0x
2           1.8x                 1.9x
4           3.5x                 3.7x
8           6.8x                 7.2x
16          12.5x                14.1x
```

## Testing

Run the test suite (from the repository root):

```bash
uv run pytest src/cosmoforge.qube/tests/ -s
```

Specific tests:

```bash
uv run pytest src/cosmoforge.qube/tests/test_fisher.py -s
uv run pytest src/cosmoforge.qube/tests/test_spectra.py -s
uv run pytest src/cosmoforge.qube/tests/test_signal_covmat.py -s
```

## Examples

### Mock Data Generation

```python
from qube.scripts.generate_mock_data import generate_cmb_maps

# Generate mock CMB maps
generate_cmb_maps(
    nside=32,
    lmax=64,
    nsims=100,
    output_dir="mock_data/",
    fiducial_cls="fiducial_cls.txt"
)
```

### Parameter Forecasting

```python
from qube import Fisher
import numpy as np

# Run Fisher analysis
fisher = Fisher("forecast_config.yaml")
fisher.run()

# Get parameter uncertainties
cov_matrix = fisher.get_covariance_matrix()
param_errors = np.sqrt(np.diag(cov_matrix))

print(f"Parameter uncertainties: {param_errors}")
```

## Advanced Usage

### Custom Analysis

```python
from qube import Spectra
from cosmocore import FieldCollection

# Custom field setup
fields = create_custom_fields()
collection = FieldCollection(params, fields)

# Custom analysis
qml = Spectra(params_file, fisher=None)
qml.collection = collection
qml.run()
```

### Cross-Validation

```python
# Split data for cross-validation
qml1 = Spectra("config_half1.yaml")
qml2 = Spectra("config_half2.yaml")

qml1.run()
qml2.run()

# Compare results
spectra1 = qml1.get_power_spectra()
spectra2 = qml2.get_power_spectra()
```

## Troubleshooting

### Common Issues

1. **Memory errors**: Reduce nside or use more MPI processes
2. **Convergence issues**: Check mask coverage and noise levels
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
    qml = Spectra("config.yaml")
    qml.run()

cProfile.run('run_analysis()', 'profile_output.prof')
```

## Contributing

See the main CosmoForge README for contribution guidelines.

## Citation

If you use QUBE (as part of CosmoForge) in your research, please cite:

> Galloni, G. & Pagano, L., *CosmoForge I: A unified framework for QML power spectrum estimation and pixel-based likelihood analysis*, [arXiv:2605.21149](https://arxiv.org/abs/2605.21149) (2026).

```bibtex
@article{GalloniPagano_CosmoForgeI,
    author        = {Galloni, G. and Pagano, L.},
    title         = {{CosmoForge I}: A unified framework for {QML} power spectrum estimation and pixel-based likelihood analysis},
    year          = {2026},
    eprint        = {2605.21149},
    archivePrefix = {arXiv},
    primaryClass  = {astro-ph.CO},
}
```

## References

- Quadratic Maximum Likelihood: Tegmark (1997), Phys. Rev. D 55, 5895
- Bandpower estimation: Bond, Jaffe & Knox (1998), Phys. Rev. D 57, 2117
- Binned QML: Bilbao-Ahedo et al. (2021), arXiv:2104.08528
