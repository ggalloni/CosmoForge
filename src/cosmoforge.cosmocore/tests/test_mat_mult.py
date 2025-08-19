from timeit import timeit

import matplotlib.pyplot as plt
import numpy as np
from numba import njit
from scipy.linalg.blas import dgemm


# --- Matrix Multiplication: Numba version ---
@njit(cache=True)
def matrix_mult(A, B):
    n = A.shape[0]
    C = np.empty((n, n), dtype=A.dtype)
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i, k] * B[k, j]
            C[i, j] = s
    return C


def test_mat_multiplication(configure_plt):
    configure_plt()

    # --- Setup ---
    sizes = [200, 500, 800, 1000, 2000, 3072, 5000]
    times = {
        name: []
        for name in [
            # "Numba matmul", # Uncomment if you want to test the Numba version
            "NumPy np.dot",
            "NumPy @ operator",
            # "NumPy einsum", # Uncomment if you want to test einsum
            "NumPy matmul",
            "BLAS dgemm",
        ]
    }
    n_runs = 10

    # --- Multiplication Benchmarks ---
    def numba_mult():
        return matrix_mult(A, B)

    def numpy_dot():
        return np.dot(A, B)

    def numpy_at():
        return A @ B

    def blas_dgemm():
        # A_f = np.asfortranarray(A)
        # B_f = np.asfortranarray(B)
        # return dgemm(alpha=1.0, a=A_f, b=B_f)
        return dgemm(alpha=1.0, a=A, b=B)

    def einsum():
        return np.einsum("ij,jk->ik", A, B)

    def matmul():
        return np.matmul(A, B)

    globals_dict = {
        # Uncomment if you want to test the Numba version
        # "Numba matmul": numba_mult,
        "NumPy np.dot": numpy_dot,
        "NumPy @ operator": numpy_at,
        # Uncomment if you want to test einsum
        # "NumPy einsum": einsum,
        "NumPy matmul": matmul,
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
        # _ = matrix_mult(A, B) # Uncomment if you want to test the Numba version

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
    plt.title("Benchmark: Matrix Multiplication Performance")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    from conftest import _configure_plt

    test_mat_multiplication(_configure_plt)
