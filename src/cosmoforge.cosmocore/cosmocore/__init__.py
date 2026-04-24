"""
CosmoCore: Core infrastructure for analysis of spin-0 and spin-2 fields on the sphere.

This package provides foundational utilities for cosmological analysis of angular
fields on the sphere, including pixel-space operations, spherical harmonic
transformations, and matrix operations. It serves as the common computational
backbone for the CosmoForge framework.

The package implements:

- **Pixel operations**: Signal matrix computation, pixel-space correlations,
  and HEALPix-based pixelization handling
- **Harmonic operations**: Power spectrum management, beam window functions,
  and spherical harmonic transforms
- **Field management**: Scalar (spin-0) and tensor (spin-2) field handling
  with proper coordinate transformations
- **Matrix operations**: Optimized LAPACK-based routines for covariance matrix
  inversion and determinant computation

Key References
--------------
Infrastructure:
.. [1] Gorski, K.M. et al. "HEALPix: A Framework for High-Resolution Discretization
   and Fast Analysis of Data Distributed on the Sphere"
   Astrophys. J. 622, 759-771 (2005)
.. [2] Reinecke, M. & Seljak, U. "Libsharp - spherical harmonic transforms revisited"
   Astron. Astrophys. 554, A112 (2013)

Spin-2 Field Formalism:
.. [3] Kamionkowski, M., Kosowsky, A. & Stebbins, A. "Statistics of cosmic microwave
   background polarization" Phys. Rev. D 55, 7368 (1997)
.. [4] Zaldarriaga, M. & Seljak, U. "All-sky analysis of polarization in the microwave
   background" Phys. Rev. D 55, 1830 (1997)

Theoretical Power Spectra:
.. [5] Lewis, A., Challinor, A. & Lasenby, A. "Efficient Computation of Cosmic
   Microwave Background Anisotropies in Closed Friedmann-Robertson-Walker Models"
   Astrophys. J. 538, 473-476 (2000) - CAMB
.. [6] Lesgourgues, J. "The Cosmic Linear Anisotropy Solving System (CLASS) I: Overview"
   arXiv:1104.2932 (2011) - CLASS

Numerical Methods:
.. [7] Anderson, E. et al. "LAPACK Users' Guide" SIAM, Philadelphia (1999)
   - For matrix operations (Cholesky decomposition, LU factorization)
"""

from .basics import (
    get_rotation_angle,
    idx2spec,
    legendre_00,
    legendre_02,
    legendre_22,
    legendre_plm,
    matrix_inverse_symm,
    matrix_mult,
    matrix_slogdet_symm,
    matrix_trace,
    spec2idx,
)
from .basis import (
    ComputationBasis,
    HarmonicBasis,
    PixelBasis,
    create_computation_basis,
)
from .bins import Bins
from .core import Core
from .fields import (
    BaseField,
    FieldCollection,
    FieldConfig,
    PolarizationField,
    ScalarField,
    create_field,
)
from .harmonic import (
    BeamManager,
    SpectraManager,
    cl_to_vec,
    vec_to_cl,
)
from .in_out import (
    convert_spectra_normalization,
    output_geometry,
    read_covmat,
    read_maps,
    read_mask,
    readcl,
    write_covmat_reduced,
    write_out_matrix,
    writecl,
)
from .logger import CosmoLogger, Timer, get_logger, get_logger_from_params
from .mpi_utils import MPISharedMemoryMixin
from .pixel import (
    compute_00_contribution,
    compute_02_contribution,
    compute_22_contribution,
    compute_pointings,
    compute_signal_matrix,
    count_nonzero_mask,
    derivative_step_00,
    derivative_step_02,
    derivative_step_22,
    do_derivative_step,
)
from .settings import InputParams

__all__ = [
    # settings
    "InputParams",
    # logger
    "CosmoLogger",
    "Timer",
    "get_logger",
    "get_logger_from_params",
    # pixel
    "compute_00_contribution",
    "compute_02_contribution",
    "compute_22_contribution",
    "compute_pointings",
    "compute_signal_matrix",
    "count_nonzero_mask",
    "derivative_step_00",
    "derivative_step_02",
    "derivative_step_22",
    "do_derivative_step",
    # in_out
    "convert_spectra_normalization",
    "output_geometry",
    "read_covmat",
    "read_maps",
    "read_mask",
    "readcl",
    "write_covmat_reduced",
    "write_out_matrix",
    "writecl",
    # harmonic
    "BeamManager",
    "SpectraManager",
    "cl_to_vec",
    "vec_to_cl",
    # bins
    "Bins",
    # core
    "Core",
    # basis
    "ComputationBasis",
    "HarmonicBasis",
    "PixelBasis",
    "create_computation_basis",
    # fields
    "BaseField",
    "FieldCollection",
    "FieldConfig",
    "PolarizationField",
    "ScalarField",
    "create_field",
    # mpi_utils
    "MPISharedMemoryMixin",
    # basics
    "get_rotation_angle",
    "idx2spec",
    "legendre_00",
    "legendre_02",
    "legendre_22",
    "legendre_plm",
    "matrix_inverse_symm",
    "matrix_mult",
    "matrix_slogdet_symm",
    "matrix_trace",
    "spec2idx",
]
