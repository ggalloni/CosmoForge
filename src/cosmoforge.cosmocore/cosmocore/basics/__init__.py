"""
Basic mathematical utilities for cosmological computations.

This package provides fundamental mathematical functions optimized for
cosmological analysis, including Legendre polynomials, Wigner d-matrices,
rotation calculations, and matrix operations.
"""

from .geometry import _project_and_norm, get_rotation_angle
from .indexing import idx2spec, spec2idx
from .legendre import legendre_00, legendre_02, legendre_22, legendre_plm
from .linalg import (
    _copy_lower_to_upper,
    add_diagonal,
    cholesky_decomposition,
    cholesky_factor,
    cholesky_solve,
    matrix_inverse_symm,
    matrix_mult,
    matrix_slogdet,
    matrix_slogdet_symm,
    matrix_trace,
    symmetrize_inplace,
)
from .smw import smw_inverse, smw_kernel, smw_logdet, smw_quadratic_form
from .wigner import _wigner_d_start, wigner_d_matrix, wigner_d_small

__all__ = [
    # legendre
    "legendre_00",
    "legendre_22",
    "legendre_02",
    "legendre_plm",
    # wigner
    "wigner_d_small",
    "_wigner_d_start",
    "wigner_d_matrix",
    # indexing
    "spec2idx",
    "idx2spec",
    # geometry
    "_project_and_norm",
    "get_rotation_angle",
    # linalg
    "matrix_mult",
    "add_diagonal",
    "matrix_trace",
    "_copy_lower_to_upper",
    "matrix_inverse_symm",
    "matrix_slogdet",
    "cholesky_decomposition",
    "cholesky_factor",
    "cholesky_solve",
    "matrix_slogdet_symm",
    "symmetrize_inplace",
    # smw
    "smw_inverse",
    "smw_logdet",
    "smw_kernel",
    "smw_quadratic_form",
]
