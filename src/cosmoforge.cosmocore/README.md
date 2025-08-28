# CosmoForge.CosmoCore

CosmoCore is the foundational package of CosmoForge, providing core functionality for cosmological analysis including field management, matrix operations, I/O utilities, and fundamental mathematical operations.

## Overview

CosmoCore serves as the base layer for all cosmological computations in CosmoForge. It provides:

- **Field Management**: Scalar and polarization field handling with HEALPix integration
- **Matrix Operations**: Optimized linear algebra operations with Numba acceleration
- **I/O Utilities**: Reading and writing of cosmological data formats
- **Harmonic Analysis**: Power spectrum management and beam handling
- **Pixel Operations**: HEALPix pixel-based computations

## Key Components

### Core Classes

- **`Core`**: Base class for all cosmological analysis pipelines
- **`BaseField`**: Abstract base for cosmological fields
- **`ScalarField`**: Temperature field implementation
- **`PolarizationField`**: Polarization (E/B) field implementation
- **`FieldCollection`**: Container for managing multiple fields

### Managers

- **`SpectraManager`**: Power spectrum handling and normalization
- **`BeamManager`**: Instrumental beam function management

### Mathematical Operations

- **Legendre Polynomials**: `legendre_00`, `legendre_02`, `legendre_22`
- **Matrix Operations**: `matrix_mult`, `matrix_inverse_symm`, `matrix_trace`
- **Harmonic Transforms**: `cl_to_vec`, `vec_to_cl`

## Installation

CosmoCore is automatically installed as part of CosmoForge:

```bash
pip install -e /path/to/CosmoForge
```

## Usage

### Basic Field Creation

```python
from cosmocore import create_field, FieldCollection
import numpy as np

# Create temperature field
temp_field = create_field(
    spin=0,
    nside=32,
    lmax=64,
    mask=mask_array,
    labels="T"
)

# Create polarization field
pol_field = create_field(
    spin=2,
    nside=32,
    lmax=64,
    mask=mask_array,
    labels=["E", "B"]
)

# Create field collection
fields = [temp_field, pol_field]
collection = FieldCollection(params, fields)
```

### Power Spectrum Management

```python
from cosmocore import SpectraManager, BeamManager

# Create managers
spectra_mgr = SpectraManager(fields)
beam_mgr = BeamManager(fields)

# Set power spectra from file
spectra_mgr.set_cls_from_file("fiducial_cls.txt", params)

# Apply normalization
spectra_mgr.apply_normalization()

# Apply beam smoothing
beam_mgr.apply_smoothing(spectra_mgr)
```

### Matrix Operations

```python
from cosmocore import matrix_mult, matrix_inverse_symm, matrix_trace

# Matrix multiplication
C = matrix_mult(A, B)

# Symmetric matrix inversion
inv_A = matrix_inverse_symm(A)

# Matrix trace
tr_AB = matrix_trace(A, B)
```

### I/O Operations

```python
from cosmocore import read_mask, read_covmat, readcl

# Read mask
mask = read_mask(maskfile, mask_array)

# Read covariance matrix
covmat = read_covmat(covmatfile, npix, active_pixels, C)

# Read power spectra
cls = readcl(clfile, cl_array)
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
covmatfile1: "data/noise_cov.bin"

# Beam configuration
smoothing_type: 1  # Gaussian
fwhmarcmin: 5.0
beam_file: "data/beam.fits"
```

## Performance Features

### Numba Acceleration

Critical functions use Numba JIT compilation:

```python
@njit(cache=True)
def legendre_unified(cos_theta, lmax, pl_00, pl_02, pl_22):
    # High-performance Legendre polynomial computation
    ...
```

### Memory Optimization

- In-place operations where possible
- Efficient memory layouts for matrix operations
- Optimized data structures for large datasets

### Vectorized Operations

- NumPy vectorization for array operations
- Efficient broadcasting for field operations
- Optimized inner loops for critical computations

## API Reference

### Field Creation

```python
def create_field(spin, nside, lmax, mask=None, labels=None):
    """Create field based on spin value."""
    
def compute_pointings(nside, active_pixels, ordering=1):
    """Compute pointing vectors for active pixels."""
```

### Matrix Operations

```python
def matrix_mult(A, B):
    """Multiply two matrices using BLAS."""

def matrix_inverse_symm(M):
    """Invert symmetric matrix using Cholesky decomposition."""

def matrix_trace(A, B):
    """Compute trace of matrix product."""
```

### Harmonic Operations

```python
def cl_to_vec(cl, vec):
    """Convert Cl array to vector format."""

def vec_to_cl(vec, cl):
    """Convert vector to Cl array format."""
```

## Testing

Run CosmoCore tests:

```bash
cd src/cosmoforge.cosmocore
python -m pytest tests/
```

## Dependencies

- NumPy: Numerical computations
- SciPy: Scientific computing
- Numba: JIT compilation
- HEALPix: Pixelization (via healpy)

## Architecture

CosmoCore follows a modular architecture:

```text
cosmocore/
├── __init__.py          # Public API
├── core.py              # Core base class
├── fields.py            # Field implementations
├── settings.py          # Parameter management
├── basics.py            # Basic mathematical operations
├── harmonic.py          # Harmonic analysis tools
├── pixel.py             # Pixel-based operations
└── in_out.py           # I/O utilities
```

## Extension Points

CosmoCore is designed to be extensible:

- **Custom Fields**: Inherit from `BaseField`
- **Custom Operations**: Add to existing modules
- **New Algorithms**: Implement using existing infrastructure

## Performance Benchmarks

Typical performance on modern hardware:

- Field creation: ~1ms for nside=32
- Matrix operations: ~10ms for 1000x1000 matrices
- Power spectrum operations: ~1ms for lmax=100
- I/O operations: ~100ms for typical data files
