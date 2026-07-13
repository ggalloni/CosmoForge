<p align="center">
  <img src="https://raw.githubusercontent.com/ggalloni/CosmoForge/master/logos/logo_cosmoforge_light.png#gh-light-mode-only" alt="CosmoForge logo (light)" width="60%"/>
  <img src="https://raw.githubusercontent.com/ggalloni/CosmoForge/master/logos/logo_cosmoforge_dark.png#gh-dark-mode-only" alt="CosmoForge logo (dark)" width="60%"/>
</p>

[![Build Status](https://github.com/ggalloni/CosmoForge/actions/workflows/test.yml/badge.svg?branch=master)](https://github.com/ggalloni/CosmoForge/actions/workflows/test.yml)
[![Documentation Status](https://readthedocs.org/projects/cosmoforge/badge/?version=latest)](https://cosmoforge.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/ggalloni/CosmoForge/branch/master/graph/badge.svg?token=UOm3LdvL7J)](https://codecov.io/gh/ggalloni/CosmoForge)
[![PyPI](https://img.shields.io/pypi/v/cosmoforge?include_prereleases)](https://pypi.org/project/cosmoforge/)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/charliermarsh/ruff)
[![Python](https://img.shields.io/pypi/pyversions/cosmoforge)](https://pypi.org/project/cosmoforge/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **📚 [Complete Documentation](https://cosmoforge.readthedocs.io/en/latest/) | [Installation Guide](https://cosmoforge.readthedocs.io/en/latest/installation.html) | [Quick Start](https://cosmoforge.readthedocs.io/en/latest/quickstart.html) | [API Reference](https://cosmoforge.readthedocs.io/en/latest/api/cosmocore.html)**

CosmoForge is a comprehensive Python framework for power spectrum estimation and likelihood analysis of spin-0 and spin-2 fields on the sphere, using Fisher matrix, Quadratic Maximum Likelihood (QML), and pixel-based likelihood methods. While widely applicable to any sky signal (e.g. CMB, galaxy surveys, 21 cm), it is particularly optimized for the analysis of partial-sky, noisy observations with complex noise covariance.

## Overview

CosmoForge consists of several interconnected packages designed for efficient and accurate cosmological parameter estimation:

- **cosmocore**: Core functionality for cosmological analysis, including field management, matrix operations, and I/O utilities
- **qube-qml**: QML and Fisher matrix implementations for power spectrum estimation (imported as `qube`)
- **picslike**: Pixel-based Inference with Correlated-Skies Likelihood — pixel-space likelihood analysis for parameter estimation

## Features

- **Fisher Matrix Analysis**: Fast Fisher matrix computation for cosmological parameter forecasting
- **QML Power Spectrum Estimation**: Quadratic Maximum Likelihood estimation for optimal power spectrum recovery
- **Pixel-Based Likelihood**: Direct likelihood evaluation in map pixel space for parameter estimation
- **MPI Parallelization**: Efficient parallel computation support for large-scale analyses
- **HEALPix Integration**: Full support for HEALPix pixelization scheme
- **Flexible Field Management**: Support for scalar (spin-0) and tensor (spin-2) fields
- **Beam and Noise Modeling**: Comprehensive instrumental effects modeling

## Installation

> **📖 For detailed installation instructions, see the [Installation Guide](https://cosmoforge.readthedocs.io/en/latest/installation.html)**

### Requirements

- Python 3.11–3.13
- NumPy, SciPy
- healpy
- mpi4py (for parallel computation)

### Install from PyPI

The umbrella distribution installs all three subpackages:

```bash
pip install cosmoforge
```

Or pick subpackages individually:

```bash
pip install cosmocore       # core utilities only
pip install qube-qml        # adds QML / Fisher (imported as `qube`)
pip install picslike        # adds pixel-space likelihood
```

### Install from source

```bash
git clone https://github.com/ggalloni/CosmoForge.git
cd CosmoForge
uv sync --all-packages --all-extras --dev
```

## Quick Start

> **🚀 For comprehensive tutorials and examples, visit the [Quick Start Guide](https://cosmoforge.readthedocs.io/en/latest/quickstart.html) and [Tutorials](https://cosmoforge.readthedocs.io/en/latest/tutorials/index.html)**

### Fisher Matrix Analysis

```python
from qube import Fisher

# Initialize Fisher analysis
fisher = Fisher("config/fisher_config.yaml")
fisher.run()

# Get Fisher matrix
fisher_matrix = fisher.get_fisher_matrix()
```

### QML Power Spectrum Estimation

```python
from qube import Spectra

# Initialize QML analysis
qml = Spectra("config/qml_config.yaml")
qml.run()

# Get power spectra
power_spectra = qml.get_power_spectra()
noise_bias = qml.get_noise_bias()
```

### Pixel-Based Likelihood Analysis

```python
from picslike import PICSLike

# Initialize pixel-based likelihood
picslike = PICSLike(params_file="config/picslike_config.yaml")
picslike.run()

# Get results
chi_squared = picslike.get_chi_squared()
best_fit = picslike.get_best_fit()
```

### Using with Precomputed Fisher

```python
from qube import Fisher, Spectra

# Compute Fisher matrix first
fisher = Fisher("config/fisher_config.yaml")
fisher.run()

# Reuse Fisher computation for QML
qml = Spectra("config/qml_config.yaml", fisher=fisher)
qml.run()
```

### In-Memory Inputs

Every pipeline input can be handed over as an **in-memory array** instead of a file path.
This is for callers that already hold their maps, mask and noise covariance — an upstream
component pipeline, or a notebook — and should not have to round-trip them through disk.

The config still supplies the scalars (`nside`, `lmax`, …); an injected object simply wins
over the corresponding path.

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

# maps are a Spectra / PICSLike seam — Fisher never reads them
spectra = Spectra(params, fisher=fisher, maps1=maps)
spectra.run()
power_spectra = spectra.get_power_spectra(mode="deconvolved")
```

Combine this with opt-in persistence (leave the `out*` paths unset) and the whole run
touches no disk at all. The same kwargs exist on `PICSLike`.

The full vocabulary: `mask`, `noise_cov1`/`noise_cov2`, `maps1`/`maps2`, `cls_data`,
`fiducial_cls`, `beam`. Each injected object is defined as *exactly what the corresponding
reader would have returned* — two contracts are worth calling out, because they mirror what
each reader does rather than being symmetric:

| kwarg | contract |
|---|---|
| `noise_cov1` | reduced to active pixels |
| `maps1` | reduced to active pixels |

"Reduced" means restricted to the active (unmasked) pixels and ordered by the
**active-pixel index** — a pure function of the mask, so no `Fisher` is needed to learn it:

```python
from cosmocore import active_pixel_index

index = active_pixel_index(mask)
maps = maps_full.reshape(nfields * npix, nsims)[index]
cov = cov_full[np.ix_(index, index)]
```

Worked example: [`in_memory_inputs.ipynb`](src/cosmoforge.qube/notebooks/in_memory_inputs.ipynb),
which drives both adapters over the same data and asserts they agree bit-for-bit. Design
rationale: [ADR-0017](docs/adr/0017-file-or-array-loading-seams.md).

## Package Structure

```text
CosmoForge/
├── src/
│   ├── cosmoforge.cosmocore/    # Core functionality (published as `cosmocore`)
│   ├── cosmoforge.qube/         # QML and Fisher analysis (published as `qube-qml`, imported as `qube`)
│   ├── cosmoforge.picslike/     # Pixel-space likelihood (published as `picslike`)
│   └── cosmoforge.meta/         # Umbrella metapackage (published as `cosmoforge`)
├── tests/                       # Workspace-level fixtures (per-package suites under src/cosmoforge.*/tests/)
└── docs/                        # Sphinx documentation
```

## Configuration

CosmoForge uses YAML configuration files to specify analysis parameters:

```yaml
# Example configuration
nside: 4
lmax: 16
fields: "TEB"
maskfile: "data/mask.fits"
inputclfile: "data/fiducial_cls.txt"
# ... additional parameters
```

## Testing

Run the test suite:

```bash
# Run all tests
uv run pytest

# Run specific package tests
uv run --package cosmocore pytest src/cosmoforge.cosmocore/tests/
uv run --package qube pytest src/cosmoforge.qube/tests/
uv run --package picslike pytest src/cosmoforge.picslike/tests/
```

## Performance

CosmoForge is designed for high-performance cosmological analysis:

- **Numba JIT compilation** for critical mathematical operations
- **MPI parallelization** for distributed computing
- **Optimized matrix operations** using LAPACK/BLAS
- **Memory-efficient algorithms** for large datasets

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

[Add license information]

## Citation

If you use CosmoForge in your research, please cite:

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

## Support

> **📖 Complete documentation is available at: [https://cosmoforge.readthedocs.io/en/latest/](https://cosmoforge.readthedocs.io/en/latest/)**

For questions and support:

- Open an issue on GitHub
- Contact: [contact information]

## Acknowledgments

CosmoForge builds upon established cosmological analysis methods and libraries:

- HEALPix for pixelization
- NumPy/SciPy for numerical computations
- MPI for parallelization
