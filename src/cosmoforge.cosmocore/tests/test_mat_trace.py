from timeit import timeit

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from scipy.linalg.blas import dgemm


# ---- Method 1: Numba double loop ----
@njit(cache=True)
def matrix_trace(A, B):
    n = A.shape[0]
    s = 0.0
    for i in range(n):
        for j in range(n):
            s += A[i, j] * B[j, i]
    return s


def test_mat_trace(configure_plt):

    configure_plt()

    # ---- Setup ----
    n = 3072  # Size of matrices (adjust as needed)
    np.random.seed(42)
    A = np.random.randn(n, n)
    B = np.random.randn(n, n)

    # --- Setup ---
    sizes = [200, 500, 800, 1000, 2000, 3072, 5000]
    # sizes = [200, 500, 800, 1000, 2000]
    times = {
        name: []
        for name in [
            "Numba double loop",  # Uncomment if you want to test the Numba version
            "NumPy sum",
            "NumPy trace dot",
            "BLAS dgemm",
        ]
    }
    n_runs = 10

    # Warm up JIT
    _ = matrix_trace(A, B)

    # ---- Benchmark functions ----

    def numba_impl():
        return matrix_trace(A, B)

    def numpy_sum():
        return np.sum(A * B.T)

    def numpy_trace_dot():
        return np.trace(A @ B.T)

    def blas_dgemm():
        # A_f = np.asfortranarray(A)
        # B_f = np.asfortranarray(B)
        # C = dgemm(alpha=1.0, a=A_f, b=B_f, trans_b=True)
        C = dgemm(alpha=1.0, a=A, b=B, trans_b=True)
        return np.trace(C)

    globals_dict = {
        # Uncomment if you want to test the Numba version
        "Numba double loop": numba_impl,
        "NumPy sum": numpy_sum,
        "NumPy trace dot": numpy_trace_dot,
        "BLAS dgemm": blas_dgemm,
    }

    for n in sizes:
        print(f"Matrix size: {n}x{n}")
        np.random.seed(42)
        A = np.random.randn(n, n)
        B = np.random.randn(n, n)
        A = np.asfortranarray(A)
        B = np.asfortranarray(B)

        # Warm up JIT
        _ = matrix_trace(A, B)

        # Run the function and time it
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
    plt.title("Benchmark: Matrix Trace Performance")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    from conftest import _configure_plt

    test_mat_trace(_configure_plt)
