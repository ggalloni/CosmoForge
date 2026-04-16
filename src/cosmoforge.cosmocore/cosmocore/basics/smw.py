"""Sherman-Morrison-Woodbury formula implementations."""

from __future__ import annotations

import numpy as np
from scipy.linalg import lapack

from .linalg import (
    cholesky_decomposition,
    matrix_inverse_symm,
    matrix_mult,
    matrix_slogdet_symm,
)


def smw_inverse(N_inv, V_N_inv, V_Ninv_VT, Lambda_diag, threshold=1e-30):
    """
    Sherman-Morrison-Woodbury inverse: (N + V^T Lambda V)^{-1}.

    Parameters
    ----------
    N_inv : numpy.ndarray
        Precomputed inverse of N, shape (n, n).
    V_N_inv : numpy.ndarray
        Precomputed V @ N^{-1}, shape (k, n).
    V_Ninv_VT : numpy.ndarray
        Precomputed V @ N^{-1} @ V^T, shape (k, k).
    Lambda_diag : numpy.ndarray
        Diagonal of Lambda, shape (k,).
    threshold : float, optional
        Minimum value for Lambda diagonal elements. Default is 1e-30.

    Returns
    -------
    numpy.ndarray
        The inverse (N + V^T Lambda V)^{-1}, shape (n, n).
    """
    Lambda_inv_diag = np.where(
        Lambda_diag > threshold, 1.0 / Lambda_diag, 1.0 / threshold
    )
    K = np.asfortranarray(V_Ninv_VT.copy())
    n = K.shape[0]
    K.flat[:: n + 1] += Lambda_inv_diag
    kernel_inv = matrix_inverse_symm(K, overwrite=True)
    return N_inv - V_N_inv.T @ kernel_inv @ V_N_inv


def smw_logdet(log_det_N, V_Ninv_VT, Lambda_diag, threshold=1e-30):
    """
    Sherman-Morrison-Woodbury log determinant: log|N + V^T Lambda V|.

    Parameters
    ----------
    log_det_N : float
        Precomputed log|N|.
    V_Ninv_VT : numpy.ndarray
        Precomputed V @ N^{-1} @ V^T, shape (k, k).
    Lambda_diag : numpy.ndarray
        Diagonal of Lambda, shape (k,).
    threshold : float, optional
        Minimum value for Lambda diagonal elements. Default is 1e-30.

    Returns
    -------
    float
        The log determinant log|N + V^T Lambda V|.
    """
    log_det_Lambda = np.sum(np.log(np.maximum(Lambda_diag, threshold)))
    Lambda_inv_diag = np.where(
        Lambda_diag > threshold, 1.0 / Lambda_diag, 1.0 / threshold
    )
    K = np.asfortranarray(V_Ninv_VT.copy())
    n = K.shape[0]
    K.flat[:: n + 1] += Lambda_inv_diag
    _, log_det_K = matrix_slogdet_symm(K)
    return log_det_N + log_det_Lambda + log_det_K


def smw_kernel(V_Ninv_VT, Lambda_diag, threshold=1e-30):
    """
    Build the SMW kernel matrix K = Lambda^{-1} + V N^{-1} V^T.

    Parameters
    ----------
    V_Ninv_VT : numpy.ndarray
        Precomputed V @ N^{-1} @ V^T, shape (k, k).
    Lambda_diag : numpy.ndarray
        Diagonal of Lambda, shape (k,).
    threshold : float, optional
        Minimum value for Lambda diagonal elements. Default is 1e-30.

    Returns
    -------
    numpy.ndarray
        The kernel matrix K, shape (k, k), in Fortran order.
    """
    Lambda_inv_diag = np.where(
        Lambda_diag > threshold, 1.0 / Lambda_diag, 1.0 / threshold
    )
    K = np.asfortranarray(V_Ninv_VT.copy())
    n = K.shape[0]
    K.flat[:: n + 1] += Lambda_inv_diag
    return K


def smw_quadratic_form(data, N_inv, V_N_inv, V_Ninv_VT, Lambda_diag, threshold=1e-30):
    """
    SMW quadratic form: d^T (N + V^T Lambda V)^{-1} d.

    Parameters
    ----------
    data : numpy.ndarray
        Data vector d, shape (n,).
    N_inv : numpy.ndarray
        Precomputed inverse of N, shape (n, n).
    V_N_inv : numpy.ndarray
        Precomputed V @ N^{-1}, shape (k, n).
    V_Ninv_VT : numpy.ndarray
        Precomputed V @ N^{-1} @ V^T, shape (k, k).
    Lambda_diag : numpy.ndarray
        Diagonal of Lambda, shape (k,).
    threshold : float, optional
        Minimum value for Lambda diagonal elements. Default is 1e-30.

    Returns
    -------
    float
        The quadratic form d^T C^{-1} d.
    """
    # Term 1: d^T @ N^{-1} @ d
    term1 = float(matrix_mult(data.T, matrix_mult(N_inv, data)))

    # y = V @ N^{-1} @ d
    y = matrix_mult(V_N_inv, data)

    # Build and solve with K
    K = smw_kernel(V_Ninv_VT, Lambda_diag, threshold)
    L = cholesky_decomposition(K)
    kernel_inv_y = lapack.dpotrs(L, y, lower=True)[0]

    # Term 2: y^T @ K^{-1} @ y
    term2 = float(matrix_mult(y.T, kernel_inv_y))

    return term1 - term2
