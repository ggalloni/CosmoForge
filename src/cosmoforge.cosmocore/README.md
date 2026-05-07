# CosmoCore

[![Python](https://img.shields.io/badge/python-3.11%7C3.12%7C3.13-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-cosmocore-blue.svg)](https://ggalloni.github.io/CosmoForge/api/cosmocore.html)
[![Performance](https://img.shields.io/badge/performance-numba-green.svg)](https://numba.pydata.org/)

> [CosmoCore Documentation](https://ggalloni.github.io/CosmoForge/api/cosmocore.html) | [API Reference](https://ggalloni.github.io/CosmoForge/api/cosmocore/basics.html) | [Main Documentation](https://ggalloni.github.io/CosmoForge/)

CosmoCore is the foundational package of CosmoForge, providing core functionality for the analysis of spin-0 and spin-2 fields on the sphere, including field management, computation basis methods, matrix operations, I/O utilities, and spherical harmonic operations.

## Overview

CosmoCore serves as the base layer for all cosmological computations in CosmoForge. It provides:

- **Field Management**: Scalar (spin-0) and tensor (spin-2) field handling with HEALPix integration
- **Basis**: Harmonic and pixel computation basis for Fisher matrix computation, with multi-field and spin-2 support
- **Matrix Operations**: Optimized LAPACK-based linear algebra with Numba acceleration
- **Harmonic Analysis**: Power spectrum management, beam handling, and spherical harmonic transforms
- **Pixel Operations**: Signal matrix computation and HEALPix pixel-based operations
- **I/O Utilities**: Reading and writing of cosmological data formats

## Key Components

### Computation Basis

Two computation basis methods for efficient Fisher matrix and QML estimation:

- **`HarmonicBasis`** (Tegmark-like): Direct transformation to harmonic space (n_pix -> n_modes). Fast when n_modes << n_pix. Supports optional m-block compression (`compress=True`) that approximates K as block-diagonal in azimuthal number m, giving ~lmax^2 speedup.
- **`PixelBasis`** (Gjerlow-like): Pixel-space projector with eigenvalue truncation (n_pix -> n_kept). Supports multiple basis choices and per-field threshold tuning.
- **`create_computation_basis`**: Factory function for creating basis instances by name.

Both methods support:

- **Multi-field**: Multiple components with independent sky coverage and noise
- **Spin-2 polarization**: E/B mode decomposition with spin-weighted spherical harmonics
- **Per-field eigenspectrum inspection**: `compute_eigenspectrum_per_field()` for choosing eigenmode thresholds, with E/B breakdown for spin-2 fields
- **Visualization**: `plot_eigenvalue_spectrum()` and `plot_eigenvalue_comparison()` with per-component subplots

### Fields

- **`ScalarField`**: Spin-0 field implementation (e.g. CMB temperature, convergence)
- **`PolarizationField`**: Spin-2 field implementation (e.g. CMB polarization, cosmic shear)
- **`FieldCollection`**: Container for managing multiple fields
- **`create_field`**: Factory function for field creation

### Managers

- **`SpectraManager`**: Power spectrum handling and normalization
- **`BeamManager`**: Instrumental beam function management

### Binning

- **`Bins`**: Multipole binning specification for bandpower estimation. Supports uniform bins (`Bins.fromdeltal`) and custom non-uniform bins. Both bounds are inclusive. Used by QUBE for binned QML estimation.

### Mathematical Operations

- **Legendre Polynomials**: `legendre_00`, `legendre_02`, `legendre_22`, `legendre_plm`
- **Matrix Operations**: `matrix_mult`, `matrix_inverse_symm`, `matrix_trace`, `matrix_slogdet_symm`
- **Harmonic Transforms**: `cl_to_vec`, `vec_to_cl`
- **Wigner d-matrices**: Spin-weighted harmonic basis functions
- **SMW Formula**: Sherman-Morrison-Woodbury compressed inverse and log-determinant

### Core

- **`Core`**: Base class for cosmological analysis pipelines
- **`InputParams`**: Parameter file management
- **`CosmoLogger`** / **`Timer`**: Logging and profiling utilities

## Installation

CosmoCore is automatically installed as part of CosmoForge:

```bash
pip install -e /path/to/CosmoForge
```

## Usage

### Computation Basis Workflow

```python
from cosmocore.basis import PixelBasis
import numpy as np

# Set up pixel basis
ppc = PixelBasis(N, N_inv, theta, phi, lmax=100)
ppc.setup()

# Inspect per-field eigenspectra to choose thresholds
fig, axes = ppc.plot_eigenvalue_spectrum(basis="noise_weighted")

# Apply eigenmode truncation with chosen threshold
ppc.apply_compression(epsilon=1e-4, basis="noise_weighted")

# Compute Fisher matrix
fisher = ppc.compute_fisher_matrix(C_ell)
```

### Multi-Field with Spin-2 Polarization

```python
from cosmocore.basis import PixelBasis

# T (spin-0) + QU (spin-2) setup
ppc = PixelBasis(
    N, N_inv,
    theta=(theta_t, theta_p),
    phi=(phi_t, phi_p),
    lmax=100,
    spins=[0, 2],
)
ppc.setup()

# Per-field eigenspectrum with E/B breakdown
spectra = ppc.compute_eigenspectrum_per_field(basis="noise_weighted")
for entry in spectra:
    print(f"{entry['label']}: {len(entry['eigenvalues'])} modes")

# Per-field thresholds (scalar for T, E/B tuple for polarization)
ppc.apply_compression(epsilon=[1e-4, (1e-4, 1e-3)])

# Multi-field Fisher matrix
fisher = ppc.compute_fisher_matrix(C_ell_dict, spectra_list)
```

### Field Creation

```python
from cosmocore import create_field, FieldCollection

# Create temperature field
temp_field = create_field(
    spin=0, nside=32, lmax=64,
    mask=mask_array, labels="T"
)

# Create polarization field
pol_field = create_field(
    spin=2, nside=32, lmax=64,
    mask=mask_array, labels=["E", "B"]
)

# Create field collection
collection = FieldCollection(params, [temp_field, pol_field])
```

### Power Spectrum Management

```python
from cosmocore import SpectraManager, BeamManager

spectra_mgr = SpectraManager(fields)
beam_mgr = BeamManager(fields)

spectra_mgr.set_cls_from_file("fiducial_cls.txt", params)
beam_mgr.apply_smoothing(spectra_mgr)
```

### Matrix Operations

```python
from cosmocore import matrix_mult, matrix_inverse_symm, matrix_trace

C = matrix_mult(A, B)
inv_A = matrix_inverse_symm(A)
tr_AB = matrix_trace(A, B)
```

## Configuration

CosmoCore uses parameter files for configuration:

```yaml
# Field configuration
nside: 32
lmax: 64
spins: [0, 2]  # Temperature and polarization
labels: ["T", "E", "B"]

# I/O configuration
maskfile: "data/mask.fits"
inputclfile: "data/fiducial_cls.txt"
input_convention: Dl  # "Cl" (default) or "Dl" for input files
output_convention: Cl  # "Cl" (default) or "Dl" for QML output
covmatfile1: "data/noise_cov.bin"

# Beam configuration
smoothing_type: gaussian
fwhmarcmin: 5.0
beam_file: "data/beam.fits"
```

## Architecture

```text
cosmocore/
├── __init__.py              # Public API
├── core.py                  # Core base class
├── fields.py                # Field implementations (Scalar, Polarization)
├── settings.py              # Parameter management
├── harmonic.py              # Power spectrum and beam managers
├── pixel.py                 # Pixel-based signal matrix operations
├── in_out.py                # I/O utilities
├── logger.py                # Logging and timing
├── basics/                  # Mathematical primitives
│   ├── linalg.py            #   Matrix operations (mult, inverse, trace)
│   ├── legendre.py          #   Legendre polynomial evaluation
│   ├── wigner.py            #   Wigner d-matrices for spin-2
│   ├── geometry.py          #   Rotation angles and coordinate transforms
│   ├── indexing.py          #   Spectrum index utilities
│   └── smw.py               #   Sherman-Morrison-Woodbury formula
└── basis/             # Computation basis for Fisher/QML
    ├── base.py              #   Abstract base class and SMW types
    ├── harmonic_basis.py    #   Harmonic basis builder (V operator, Lambda)
    ├── harmonic.py          #   HarmonicBasis (Tegmark-like)
    └── pixel.py             #   PixelBasis (Gjerlow-like)
```

## Performance Features

### Numba Acceleration

Critical functions use Numba JIT compilation for near-C performance:

```python
@njit(cache=True)
def legendre_unified(cos_theta, lmax, pl_00, pl_02, pl_22):
    ...
```

### Memory Optimization

- In-place operations and pre-allocated buffers for hot paths
- Efficient memory layouts (Fortran-order for LAPACK calls)
- SMW formula avoids forming full n_pix x n_pix inverses

### Vectorized Operations

- NumPy/BLAS vectorization for matrix operations
- Precomputed derivative diagonals for O(l^2) Fisher computation
- Block-diagonal structure exploited for multi-field computation
- M-block compression: block-diagonal K in azimuthal number m for ~lmax^2 speedup
- Field block-diagonal K: automatic detection and exploitation when fields are independent

## Testing

Run CosmoCore tests:

```bash
uv run pytest src/cosmoforge.cosmocore/tests/
```

## Dependencies

- NumPy: Numerical computations
- SciPy: Scientific computing (LAPACK wrappers)
- Numba: JIT compilation
- Matplotlib: Eigenspectrum visualization
- HEALPix: Pixelization (via healpy)

## Documentation

- [Core Module](https://ggalloni.github.io/CosmoForge/api/cosmocore/core.html) - Main Core class and pipeline
- [Fields Module](https://ggalloni.github.io/CosmoForge/api/cosmocore/fields.html) - Field management and HEALPix
- [Basics Module](https://ggalloni.github.io/CosmoForge/api/cosmocore/basics.html) - Mathematical utilities
- [Harmonic Module](https://ggalloni.github.io/CosmoForge/api/cosmocore/harmonic.html) - Power spectrum and beams
- [I/O Module](https://ggalloni.github.io/CosmoForge/api/cosmocore/in_out.html) - Data input/output

## References

- Tegmark, M. "How to measure CMB power spectra without losing information" Phys. Rev. D 55, 5895 (1997)
- Gjerlow, E. et al. "Component separation for the CMB with a low-resolution analysis" A&A 629, A51 (2019)
- Gorski, K.M. et al. "HEALPix: A Framework for High-Resolution Discretization and Fast Analysis of Data Distributed on the Sphere" Astrophys. J. 622, 759-771 (2005)
