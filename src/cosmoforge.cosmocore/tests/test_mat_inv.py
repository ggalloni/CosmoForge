from timeit import timeit

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from scipy.linalg import inv as scipy_inv
from scipy.linalg import lapack


# 1. Numba-based Cholesky + inversion
@njit(cache=True)
def cholesky_lower(A):
    n = A.shape[0]
    L = np.zeros_like(A)
    for i in range(n):
        for j in range(i + 1):
            s = A[i, j]
            for k in range(j):
                s -= L[i, k] * L[j, k]
            if i == j:
                if s <= 0.0:
                    raise ValueError("Matrix is not positive definite")
                L[i, j] = np.sqrt(s)
            else:
                L[i, j] = s / L[j, j]
    return L


@njit(cache=True)
def invert_lower_triangular(L):
    n = L.shape[0]
    Linv = np.zeros_like(L)
    for i in range(n):
        Linv[i, i] = 1.0 / L[i, i]
        for j in range(i):
            s = 0.0
            for k in range(j, i):
                s -= L[i, k] * Linv[k, j]
            Linv[i, j] = s / L[i, i]
    return Linv


@njit(cache=True)
def matrix_inverse_symm(M):
    n = M.shape[0]
    if M.shape[1] != n:
        raise ValueError("Non‑square matrix")
    L = cholesky_lower(M)
    Linv = invert_lower_triangular(L)
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += Linv[k, i] * Linv[k, j]
            M[i, j] = s


@njit
def copy_lower_to_upper(M):
    n = M.shape[0]
    for i in range(n):
        for j in range(i):
            M[j, i] = M[i, j]


# 2. SciPy + LAPACK in-place
def matrix_inverse_symm_inplace(M):
    M = np.asfortranarray(M, dtype=np.float64)
    _, info = lapack.dpotrf(M, lower=True, overwrite_a=True, clean=True)
    if info != 0:
        raise ValueError(f"dpotrf failed: info={info}")
    _, info = lapack.dpotri(M, lower=True, overwrite_c=True)
    if info != 0:
        raise ValueError(f"dpotri failed: info={info}")
    n = M.shape[0]
    for i in range(n):
        for j in range(i):
            M[j, i] = M[i, j]
    return M


# 2. SciPy + LAPACK in-place + Numba sym. copy
def matrix_inverse_symm_inplace_numba(M):
    M = np.asfortranarray(M, dtype=np.float64)
    _, _ = lapack.dpotrf(M, lower=True, overwrite_a=True, clean=True)
    _, _ = lapack.dpotri(M, lower=True, overwrite_c=True)
    copy_lower_to_upper(M)
    return M


def matrix_inverse_inplace(M):
    lu, piv, info = lapack.dgetrf(M)
    M, info = lapack.dgetri(lu, piv)
    return M


def test_matrix_inverse(configure_plt, show_fig=False):
    configure_plt()

    sizes = [200, 500, 800]  # , 1000, 2000, 3072, 5000]
    times = {
        name: []
        for name in [
            "NumPy inv",
            "SciPy inv",
            "LAPACK in-place",
            "LAPACK dgetri",
            "LAPACK + Numba",
            # "Numba manual", # Uncomment if you want to test the manual Numba version
        ]
    }

    n_runs = 3

    # Benchmark functions
    def lapack_inplace():
        return matrix_inverse_symm_inplace(SPD.copy())

    def lapack_inplace_numba():
        return matrix_inverse_symm_inplace_numba(SPD.copy())

    def lapack_dgetri():
        return matrix_inverse_inplace(SPD.copy())

    def numpy_inv():
        return np.linalg.inv(SPD.copy())

    def inv_scipy():
        return scipy_inv(SPD.copy())

    def numba_manual():
        M = SPD.copy()
        matrix_inverse_symm(M)
        return M

    globals_dict = {
        "NumPy inv": numpy_inv,
        "SciPy inv": inv_scipy,
        "LAPACK in-place": lapack_inplace,
        "LAPACK dgetri": lapack_dgetri,
        "LAPACK + Numba": lapack_inplace_numba,
        # "Numba manual": test_numba_manual,  # Uncomment for manual Numba version
    }

    for n in sizes:
        print(f"Matrix size: {n}x{n}")
        np.random.seed(42)
        A = np.random.randn(n, n)
        A = np.asfortranarray(A)
        SPD = A @ A.T + n * np.eye(n)
        SPD = np.asfortranarray(SPD)

        # Warm-up
        _ = matrix_inverse_symm_inplace(SPD.copy())
        _ = matrix_inverse_symm_inplace_numba(SPD.copy())
        # _ = matrix_inverse_symm(SPD.copy()) # Uncomment for manual Numba version

        for i, name in enumerate(times.keys()):
            print(f"Running {name}...")
            funcname = globals_dict[name]
            # Run the function and time it
            t = timeit(funcname, number=n_runs)
            times[name].append(t / n_runs)

    for name, tlist in times.items():
        plt.plot(sizes, tlist, marker="o", label=name)
    plt.xlabel("Matrix size (n)")
    plt.ylabel("Average time per run (s)")
    plt.title("Benchmark: Matrix Inversion Performance")
    plt.legend()
    if show_fig:
        plt.show()


if __name__ == "__main__":
    from conftest import _configure_plt

    test_matrix_inverse(_configure_plt, show_fig=True)
