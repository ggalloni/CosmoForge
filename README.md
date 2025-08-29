# CosmoForge

[![Build Status](https://github.com/ggalloni/CosmoForge/workflows/Test%20CosmoForge%20Packages/badge.svg)](https://github.com/ggalloni/CosmoForge/actions)
[![Documentation](https://github.com/ggalloni/CosmoForge/workflows/Build%20and%20Deploy%20Documentation/badge.svg)](https://ggalloni.github.io/CosmoForge/)
[![codecov](https://codecov.io/gh/ggalloni/CosmoForge/graph/badge.svg?token=UOm3LdvL7J)](https://codecov.io/gh/ggalloni/CosmoForge)
[![PyPI](https://img.shields.io/pypi/v/cosmoforge.svg)](https://pypi.org/project/cosmoforge/)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/charliermarsh/ruff)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

<p align="center">
  <img src="logos/logo_cosmoforge_light.png#gh-light-mode-only" alt="CosmoForge logo (light)" style="max-width:40%; height:auto;"/>
  <img src="logos/logo_cosmoforge_dark.png#gh-dark-mode-only" alt="CosmoForge logo (dark)" style="max-width:40%; height:auto;"/>
</p>

> **📚 [Complete Documentation](https://ggalloni.github.io/CosmoForge/) | [Installation Guide](https://ggalloni.github.io/CosmoForge/installation.html) | [Quick Start](https://ggalloni.github.io/CosmoForge/quickstart.html) | [API Reference](https://ggalloni.github.io/CosmoForge/api/cosmocore.html)**

CosmoForge is a comprehensive Python framework for cosmological analysis, focusing on Cosmic Microwave Background (CMB) data analysis using Fisher matrix and Quadratic Maximum Likelihood (QML) power spectrum estimation methods.

## Overview

CosmoForge consists of several interconnected packages designed for efficient and accurate cosmological parameter estimation:

- **cosmocore**: Core functionality for cosmological analysis including field management, matrix operations, and I/O utilities
- **quelo**: QML and Fisher matrix implementations for power spectrum estimation
- **meta**: Metadata and utilities package

## Features

- **Fisher Matrix Analysis**: Fast Fisher matrix computation for cosmological parameter forecasting
- **QML Power Spectrum Estimation**: Quadratic Maximum Likelihood estimation for optimal power spectrum recovery
- **MPI Parallelization**: Efficient parallel computation support for large-scale analyses
- **HEALPix Integration**: Full support for HEALPix pixelization scheme
- **Flexible Field Management**: Support for scalar (temperature) and tensor (polarization) fields
- **Beam and Noise Modeling**: Comprehensive instrumental effects modeling

## Installation

> **📖 For detailed installation instructions, see the [Installation Guide](https://ggalloni.github.io/CosmoForge/installation.html)**

### Requirements

- Python 3.8+
- NumPy
- SciPy
- healpy
- mpi4py (for parallel computation)
- matplotlib (for plotting)
- pytest (for testing)

### Install from source

```bash
git clone https://github.com/ggalloni/CosmoForge.git
cd CosmoForge
pip install -e .
```

## Quick Start

> **🚀 For comprehensive tutorials and examples, visit the [Quick Start Guide](https://ggalloni.github.io/CosmoForge/quickstart.html) and [Tutorials](https://ggalloni.github.io/CosmoForge/tutorials/index.html)**

### Fisher Matrix Analysis

```python
from quelo import Fisher

# Initialize Fisher analysis
fisher = Fisher("config/fisher_config.yaml")
fisher.run()

# Get Fisher matrix
fisher_matrix = fisher.get_fisher_matrix()
```

### QML Power Spectrum Estimation

```python
from quelo import Spectra

# Initialize QML analysis
qml = Spectra("config/qml_config.yaml")
qml.run()

# Get power spectra
power_spectra = qml.get_power_spectra()
noise_bias = qml.get_noise_bias()
```

### Using with Precomputed Fisher

```python
from quelo import Fisher, Spectra

# Compute Fisher matrix first
fisher = Fisher("config/fisher_config.yaml")
fisher.run()

# Reuse Fisher computation for QML
qml = Spectra("config/qml_config.yaml", fisher=fisher)
qml.run()
```

## Package Structure

```text
CosmoForge/
├── src/
│   ├── cosmoforge.cosmocore/    # Core functionality
│   ├── cosmoforge.quelo/        # QML and Fisher analysis
│   └── cosmoforge.meta/         # Metadata package
├── tests/                       # Test suite
├── docs/                        # Documentation
└── examples/                    # Example configurations
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
cd src/cosmoforge.quelo
python -m pytest tests/
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

[Add citation information]

## Support

> **📖 Complete documentation is available at: [https://ggalloni.github.io/CosmoForge/](https://ggalloni.github.io/CosmoForge/)**

For questions and support:

- Open an issue on GitHub
- Contact: [contact information]

## Acknowledgments

CosmoForge builds upon established cosmological analysis methods and libraries:

- HEALPix for pixelization
- NumPy/SciPy for numerical computations
- MPI for parallelization
