"""Matrix operations: multiplication, trace, inversion, determinants."""

from __future__ import annotations

import numpy as np
from numba import njit
from scipy.linalg import cho_factor, cho_solve, lapack


def matrix_mult(A, B):
    """
    Matrix multiplication wrapper.

    Parameters
    ----------
    A, B : numpy.ndarray
        2D matrices to multiply.

    Returns
    -------
    numpy.ndarray
        Result of matrix multiplication A @ B.
    """
    return np.matmul(A, B)


@njit(cache=True)
def add_diagonal(M, d, out=None):
    """
    Add diagonal vector to matrix: result = M + diag(d).

    Parameters
    ----------
    M : numpy.ndarray
        2D square matrix.
    d : numpy.ndarray
        1D vector to add to diagonal.
    out : numpy.ndarray, optional
        Output array. If None, a new array is created.
        If provided, must have same shape as M.

    Returns
    -------
    numpy.ndarray
        M + diag(d), computed without creating full diagonal matrix.
    """
    if out is None:
        out = M.copy()
    elif out is not M:
        out[:] = M
    n = M.shape[0]
    # Explicit loop works with both C and F ordered arrays in numba
    for i in range(n):
        out[i, i] += d[i]
    return out


@njit(cache=True)
def matrix_trace(A, B):
    """
    Compute trace of matrix product A @ B.

    Parameters
    ----------
    A, B : numpy.ndarray
        2D square matrices of the same size.

    Returns
    -------
    float
        Trace of the matrix product: tr(A @ B) = sum_i sum_j A_ij B_ji.
    """
    n = A.shape[0]
    s = 0.0
    for i in range(n):
        for j in range(n):
            s += A[i, j] * B[j, i]
    return s


@njit
def _copy_lower_to_upper(M):
    """
    Copy lower triangular part to upper triangular part of matrix.

    Parameters
    ----------
    M : numpy.ndarray
        2D square matrix to symmetrize.

    Returns
    -------
    numpy.ndarray
        Symmetric matrix with upper part copied from lower part.
    """
    n = M.shape[0]
    for i in range(n):
        for j in range(i):
            M[j, i] = M[i, j]
    return M


def matrix_inverse_symm(M, overwrite=False):
    """
    Compute inverse of symmetric positive definite matrix.

    Parameters
    ----------
    M : numpy.ndarray
        2D square symmetric positive definite matrix to invert.
    overwrite : bool, optional
        If True, the input matrix M may be overwritten with intermediate
        results for better performance. Default is False.

    Returns
    -------
    numpy.ndarray
        Inverse of the input matrix.

    Raises
    ------
    ValueError
        If matrix is not square or if Cholesky decomposition fails even
        after symmetric-scaling preconditioning.

    Notes
    -----
    Uses LAPACK's dpotrf (Cholesky) and dpotri (inverse) for efficient
    and numerically stable inversion of symmetric positive definite matrices.

    The matrix is always symmetrically preconditioned before Cholesky as
    ``D M D`` with ``D = diag(1/sqrt(|M_ii|))``, then unscaled back after
    inversion via ``F^{-1} = D (D M D)^{-1} D``. This recovers PD-ness
    for matrices that are mathematically positive definite but numerically
    ill-conditioned because of large dynamic range across the diagonal —
    the typical case of multi-spectrum QML Fisher matrices that mix
    cosmic-variance-limited (huge signal, tiny Fisher entry) and noise-
    limited (large Fisher entry) bandpowers. The preconditioning is cheap
    (two diagonal scalings) and well-conditioned matrices are unaffected
    in finite precision.
    """
    if M.shape[0] != M.shape[1]:
        raise ValueError("Matrix must be square")

    if not overwrite:
        M = np.asfortranarray(M.copy())

    # Symmetric-scale in place: D M D with D = diag(1/sqrt(|M_ii|)).
    # Brings the diagonal close to unity and shrinks the condition number
    # for matrices with large dynamic range (e.g., multi-spectrum QML
    # Fisher matrices that mix cosmic-variance-limited and noise-limited
    # bandpowers). No extra full-matrix allocation.
    diag = np.diag(M).copy()
    abs_diag = np.abs(diag)
    abs_diag[abs_diag == 0.0] = 1.0
    d_inv = 1.0 / np.sqrt(abs_diag)
    np.multiply(M, d_inv[None, :], out=M)
    np.multiply(M, d_inv[:, None], out=M)

    L, info = lapack.dpotrf(M, lower=True, overwrite_a=True, clean=True)
    if info != 0:
        raise ValueError(f"dpotrf failed with info={info}")
    inv_L, info_i = lapack.dpotri(L, lower=True, overwrite_c=True)
    if info_i != 0:
        raise ValueError(f"dpotri failed with info={info_i}")
    inv = _copy_lower_to_upper(inv_L)
    # Unscale: F^{-1} = D (D M D)^{-1} D
    np.multiply(inv, d_inv[None, :], out=inv)
    np.multiply(inv, d_inv[:, None], out=inv)
    return inv


def matrix_slogdet(M):
    """
    Compute sign and logarithm of the determinant of a matrix.

    Parameters
    ----------
    M : numpy.ndarray
        2D square matrix for which to compute the signed log determinant.

    Returns
    -------
    tuple of float
        (sign, logdet) where sign is +/-1 and logdet is log(|det(M)|).
        If det(M) = 0, returns (0.0, -inf).

    Raises
    ------
    ValueError
        If matrix is not square or if LU decomposition fails.

    Notes
    -----
    Uses LAPACK's dgetrf (LU decomposition with partial pivoting).
    """
    if M.shape[0] != M.shape[1]:
        raise ValueError("Matrix must be square")

    # Use LU decomposition for general matrices
    lu, piv, info = lapack.dgetrf(M, overwrite_a=False)
    if info != 0:
        raise ValueError(f"dgetrf failed with info={info}")

    # Compute log determinant from diagonal of U
    logdet = 0.0
    sign = 1.0

    n = M.shape[0]
    for i in range(n):
        diag_val = lu[i, i]
        if abs(diag_val) < 1e-15:  # Singular matrix
            return 0.0, -np.inf
        if diag_val < 0:
            sign *= -1.0
        logdet += np.log(abs(diag_val))

    # Account for permutations in pivoting
    for i in range(n):
        if piv[i] != i + 1:  # LAPACK uses 1-based indexing
            sign *= -1.0

    return sign, logdet


def cholesky_decomposition(M):
    """
    Perform Cholesky decomposition of a symmetric positive definite matrix.

    Parameters
    ----------
    M : numpy.ndarray
        2D square symmetric positive definite matrix to decompose.

    Returns
    -------
    numpy.ndarray
        Lower triangular matrix L such that M = L @ L.T.

    Raises
    ------
    ValueError
        If matrix is not square or if Cholesky decomposition fails.
    """
    if M.shape[0] != M.shape[1]:
        raise ValueError("Matrix must be square")

    L, info = lapack.dpotrf(M, lower=True, overwrite_a=False, clean=True)
    if info != 0:
        raise ValueError(f"dpotrf failed with info={info}")

    return L


def cholesky_factor(M, overwrite_a=False):
    """
    Cholesky factor (L, lower=True) of a symmetric positive definite matrix.

    Pairs with :func:`cholesky_solve`. The lower triangle of the returned
    matrix holds L; the upper triangle is left untouched (LAPACK convention).

    Parameters
    ----------
    M : numpy.ndarray
        2D square symmetric positive definite matrix.
    overwrite_a : bool, optional
        If True, M is overwritten with the factor in place (no extra
        n×n allocation). Default is False.

    Returns
    -------
    tuple
        ``(c, True)`` where c is the n×n array holding L in its lower
        triangle. Suitable as the first argument to :func:`cholesky_solve`.

    Raises
    ------
    numpy.linalg.LinAlgError
        If M is not positive definite.
    """
    return cho_factor(M, lower=True, overwrite_a=overwrite_a)


def cholesky_solve(c_and_lower, b, overwrite_b=False):
    """
    Solve A x = b given the Cholesky factor of A from :func:`cholesky_factor`.

    Parameters
    ----------
    c_and_lower : tuple
        ``(c, lower)`` tuple as returned by :func:`cholesky_factor`.
    b : numpy.ndarray
        Right-hand side, shape ``(n,)`` or ``(n, k)``.
    overwrite_b : bool, optional
        If True, b may be overwritten with the solution. Default is False.

    Returns
    -------
    numpy.ndarray
        Solution x with the same shape as b.
    """
    return cho_solve(c_and_lower, b, overwrite_b=overwrite_b)


def matrix_slogdet_symm(M):
    """
    Compute sign and logarithm of determinant for symmetric positive
    definite matrix.

    Parameters
    ----------
    M : numpy.ndarray
        2D square symmetric positive definite matrix.

    Returns
    -------
    tuple of float
        (sign, logdet) where sign is +1 and logdet is log(det(M)).
        For positive definite matrices, sign is always +1.

    Raises
    ------
    ValueError
        If matrix is not square or if Cholesky decomposition fails.

    Notes
    -----
    Uses Cholesky decomposition: det(M) = det(L)^2 = (prod L_ii)^2.
    """
    if M.shape[0] != M.shape[1]:
        raise ValueError("Matrix must be square")

    # Cholesky decomposition
    L = cholesky_decomposition(M)

    # log(det(M)) = 2 * sum log(L_ii)
    logdet = 0.0
    n = M.shape[0]
    for i in range(n):
        diag_val = L[i, i]
        if diag_val <= 0:  # Should not happen for positive definite
            raise ValueError("Non-positive diagonal element in Cholesky factor")
        logdet += np.log(diag_val)

    # Factor of 2 because det(M) = det(L)^2
    logdet *= 2.0

    # Sign is always +1 for positive definite matrices
    return 1.0, logdet
