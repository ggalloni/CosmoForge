import numpy as np
import pytest

from cosmocore import (
    idx2spec,
    spec2idx,
)
from cosmocore.basics import (
    _project_and_norm,
    cholesky_factor,
    cholesky_solve,
    matrix_inverse_symm,
    matrix_slogdet,
    matrix_slogdet_symm,
    smw_inverse,
    smw_kernel,
    smw_logdet,
    smw_quadratic_form,
)


def test_idx2spec():
    """Test inverse spectrum index conversion."""
    nfields = 3

    # Test auto-spectra indices
    assert idx2spec(0, nfields) == (0, 0)
    assert idx2spec(1, nfields) == (1, 1)
    assert idx2spec(2, nfields) == (2, 2)

    # Test cross-spectra indices (this will cover lines 228-233)
    # For nfields=3, cross indices start at 3
    assert idx2spec(3, nfields) == (0, 1)  # First cross spectrum
    assert idx2spec(4, nfields) == (0, 2)  # Second cross spectrum
    assert idx2spec(5, nfields) == (1, 2)  # Third cross spectrum

    # Test with more fields to exercise the while loop
    nfields = 4
    assert idx2spec(4, nfields) == (0, 1)  # First cross
    assert idx2spec(5, nfields) == (0, 2)  # Second cross
    assert idx2spec(6, nfields) == (0, 3)  # Third cross
    assert idx2spec(7, nfields) == (1, 2)  # Fourth cross
    assert idx2spec(8, nfields) == (1, 3)  # Fifth cross
    assert idx2spec(9, nfields) == (2, 3)  # Sixth cross

    # Test error case - out of bounds (covers line 192)
    with pytest.raises(ValueError, match="Index .* out of bounds"):
        idx2spec(100, nfields)


def test_spec2idx_idx2spec_consistency():
    """Test that spec2idx and idx2spec are inverse operations."""
    nfields = 3

    # Test auto-spectra
    for i in range(nfields):
        idx = spec2idx(i, i, nfields)
        assert idx2spec(idx, nfields) == (i, i)

    # Test cross-spectra
    for i in range(nfields):
        for j in range(i + 1, nfields):
            idx = spec2idx(i, j, nfields)
            assert idx2spec(idx, nfields) == (i, j)
            idx = spec2idx(j, i, nfields)
            assert idx2spec(idx, nfields) == (i, j)


def test_project_and_norm():
    """Test the _project_and_norm function to cover epsilon bump logic."""
    # Test normal case
    vx, vy, vz = 1.0, 1.0, 1.0
    px, py, pz = _project_and_norm(vx, vy, vz)

    # Result should be normalized
    norm = np.sqrt(px * px + py * py + pz * pz)
    assert abs(norm - 1.0) < 1e-10

    # Test edge case where projection gives near-zero norm (covers lines 271-274)
    # When v is parallel to z-axis, cross product with z gives near-zero
    vx, vy, vz = 0.0, 0.0, 1.0  # Parallel to z-axis
    px, py, pz = _project_and_norm(vx, vy, vz)

    # Should still be normalized after epsilon bump
    norm = np.sqrt(px * px + py * py + pz * pz)
    assert abs(norm - 1.0) < 1e-10
    assert pz == 0.0  # z-component should be zero (projection onto xy-plane)


def test_matrix_slogdet():
    """Test matrix_slogdet for general matrices."""
    # Test with positive definite matrix
    M = np.array([[2.0, 1.0], [1.0, 2.0]])
    sign, logdet = matrix_slogdet(M)
    np_sign, np_logdet = np.linalg.slogdet(M)
    assert sign == np_sign
    np.testing.assert_almost_equal(logdet, np_logdet)

    # Test with negative determinant matrix
    M_neg = np.array([[1.0, 2.0], [3.0, 4.0]])  # det = -2
    sign, logdet = matrix_slogdet(M_neg)
    np_sign, np_logdet = np.linalg.slogdet(M_neg)
    assert sign == np_sign
    np.testing.assert_almost_equal(logdet, np_logdet)

    # Test non-square matrix error
    with pytest.raises(ValueError, match="Matrix must be square"):
        matrix_slogdet(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))


def test_matrix_slogdet_symm():
    """Test matrix_slogdet_symm for symmetric positive definite matrices."""
    # Test with positive definite matrix
    M = np.array([[4.0, 2.0], [2.0, 3.0]])
    sign, logdet = matrix_slogdet_symm(M)
    assert sign == 1.0
    np_sign, np_logdet = np.linalg.slogdet(M)
    np.testing.assert_almost_equal(logdet, np_logdet)

    # Test with larger positive definite matrix
    M_large = np.array([[5.0, 1.0, 2.0], [1.0, 4.0, 1.0], [2.0, 1.0, 6.0]])
    sign, logdet = matrix_slogdet_symm(M_large)
    assert sign == 1.0
    np_sign, np_logdet = np.linalg.slogdet(M_large)
    np.testing.assert_almost_equal(logdet, np_logdet)

    # Test non-square matrix error
    with pytest.raises(ValueError, match="Matrix must be square"):
        matrix_slogdet_symm(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    # Test non-positive definite matrix (Cholesky fails)
    M_not_pd = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues: 3, -1
    with pytest.raises(ValueError, match="dpotrf failed"):
        matrix_slogdet_symm(M_not_pd)


class TestSMWInverse:
    """Tests for Sherman-Morrison-Woodbury inverse."""

    def test_smw_inverse_matches_direct(self):
        """SMW inverse should match direct inversion."""
        np.random.seed(42)
        n, k = 100, 20

        # Create positive definite N
        A = np.random.randn(n, n)
        N = A @ A.T + np.eye(n)
        N_inv = matrix_inverse_symm(np.asfortranarray(N.copy()))

        # Create projector V and diagonal Lambda
        V = np.random.randn(k, n)
        Lambda_diag = np.abs(np.random.randn(k)) + 0.1

        # Precompute SMW quantities
        V_N_inv = V @ N_inv
        V_Ninv_VT = V_N_inv @ V.T

        # SMW inverse
        smw_result = smw_inverse(N_inv, V_N_inv, V_Ninv_VT, Lambda_diag)

        # Direct inverse: (N + V^T Λ V)^{-1}
        Lambda = np.diag(Lambda_diag)
        M = N + V.T @ Lambda @ V
        direct_result = matrix_inverse_symm(np.asfortranarray(M.copy()))

        np.testing.assert_allclose(smw_result, direct_result, rtol=1e-9)

    def test_smw_inverse_symmetric(self):
        """SMW inverse should produce symmetric result."""
        np.random.seed(123)
        n, k = 50, 10

        A = np.random.randn(n, n)
        N = A @ A.T + np.eye(n)
        N_inv = matrix_inverse_symm(np.asfortranarray(N.copy()))

        V = np.random.randn(k, n)
        Lambda_diag = np.abs(np.random.randn(k)) + 0.1

        V_N_inv = V @ N_inv
        V_Ninv_VT = V_N_inv @ V.T

        result = smw_inverse(N_inv, V_N_inv, V_Ninv_VT, Lambda_diag)

        np.testing.assert_allclose(result, result.T, rtol=1e-12)

    @pytest.mark.parametrize("n,k", [(100, 10), (200, 20), (500, 50)])
    def test_smw_inverse_various_sizes(self, n, k):
        """SMW inverse should work for various matrix sizes."""
        np.random.seed(42)

        A = np.random.randn(n, n)
        N = A @ A.T + np.eye(n)
        N_inv = matrix_inverse_symm(np.asfortranarray(N.copy()))

        V = np.random.randn(k, n)
        Lambda_diag = np.abs(np.random.randn(k)) + 0.1

        V_N_inv = V @ N_inv
        V_Ninv_VT = V_N_inv @ V.T

        smw_result = smw_inverse(N_inv, V_N_inv, V_Ninv_VT, Lambda_diag)

        # Verify against direct
        Lambda = np.diag(Lambda_diag)
        M = N + V.T @ Lambda @ V
        direct_result = matrix_inverse_symm(np.asfortranarray(M.copy()))

        np.testing.assert_allclose(smw_result, direct_result, rtol=1e-8)


def test_smw_performance_comparison(capsys):
    """Compare SMW vs direct inversion performance."""
    import time

    np.random.seed(42)
    n, k = 500, 50
    n_iterations = 10

    # Setup
    A = np.random.randn(n, n)
    N = A @ A.T + np.eye(n)
    N_inv = matrix_inverse_symm(np.asfortranarray(N.copy()))

    V = np.random.randn(k, n)

    # Precompute SMW quantities (one-time cost)
    t0 = time.perf_counter()
    V_N_inv = V @ N_inv
    V_Ninv_VT = V_N_inv @ V.T
    precompute_time = time.perf_counter() - t0

    # Generate different Lambda_diag values
    Lambda_diags = [np.abs(np.random.randn(k)) + 0.1 for _ in range(n_iterations)]

    # Time SMW inverse (multiple calls)
    t0 = time.perf_counter()
    for Lambda_diag in Lambda_diags:
        _ = smw_inverse(N_inv, V_N_inv, V_Ninv_VT, Lambda_diag)
    smw_time = time.perf_counter() - t0

    # Time direct inverse (multiple calls)
    t0 = time.perf_counter()
    for Lambda_diag in Lambda_diags:
        Lambda = np.diag(Lambda_diag)
        M = N + V.T @ Lambda @ V
        _ = matrix_inverse_symm(np.asfortranarray(M.copy()))
    direct_time = time.perf_counter() - t0

    # Print results
    print(f"\n{'=' * 60}")
    print(f"SMW Performance (n={n}, k={k}, {n_iterations} iterations)")
    print(f"{'=' * 60}")
    print(f"Precompute time:     {precompute_time * 1000:8.2f} ms (one-time)")
    print(
        f"SMW total time:      "
        f"{smw_time * 1000:8.2f} ms ({smw_time / n_iterations * 1000:.2f} ms/iter)"
    )
    print(
        f"Direct total time:   "
        f"{direct_time * 1000:8.2f} ms ({direct_time / n_iterations * 1000:.2f} ms/iter)"
    )
    print(f"Speedup:             {direct_time / smw_time:.1f}x")
    print(f"{'=' * 60}")

    # No timing assertion: at (n=500, k=50) the SMW precompute amortises across
    # the n_iterations inner loop, so the per-iter delta against direct inversion
    # is comparable to CI runner noise (~0.3 ms on 14 ms baseline). This test
    # exists for reporting; correctness is covered by other SMW tests.


class TestSMWLogdet:
    """Tests for Sherman-Morrison-Woodbury log determinant."""

    def test_smw_logdet_matches_direct(self):
        """SMW logdet should match direct computation."""
        np.random.seed(42)
        n, k = 100, 20

        # Create positive definite N
        A = np.random.randn(n, n)
        N = A @ A.T + np.eye(n)
        N_inv = matrix_inverse_symm(N.copy())
        _, log_det_N = matrix_slogdet_symm(N.copy())

        # Create projector V and diagonal Lambda
        V = np.random.randn(k, n)
        Lambda_diag = np.abs(np.random.randn(k)) + 0.1

        # Precompute SMW quantities
        V_Ninv_VT = V @ N_inv @ V.T

        # SMW logdet
        smw_result = smw_logdet(log_det_N, V_Ninv_VT, Lambda_diag)

        # Direct logdet: log|N + V^T Λ V|
        Lambda = np.diag(Lambda_diag)
        M = N + V.T @ Lambda @ V
        _, direct_result = matrix_slogdet_symm(M.copy())

        np.testing.assert_allclose(smw_result, direct_result, rtol=1e-9)

    @pytest.mark.parametrize("n,k", [(100, 10), (200, 20), (500, 50)])
    def test_smw_logdet_various_sizes(self, n, k):
        """SMW logdet should work for various matrix sizes."""
        np.random.seed(42)

        A = np.random.randn(n, n)
        N = A @ A.T + np.eye(n)
        N_inv = matrix_inverse_symm(N.copy())
        _, log_det_N = matrix_slogdet_symm(N.copy())

        V = np.random.randn(k, n)
        Lambda_diag = np.abs(np.random.randn(k)) + 0.1

        V_Ninv_VT = V @ N_inv @ V.T

        smw_result = smw_logdet(log_det_N, V_Ninv_VT, Lambda_diag)

        # Verify against direct
        Lambda = np.diag(Lambda_diag)
        M = N + V.T @ Lambda @ V
        _, direct_result = matrix_slogdet_symm(M.copy())

        np.testing.assert_allclose(smw_result, direct_result, rtol=1e-8)


class TestSMWKernel:
    """Tests for SMW kernel matrix construction."""

    def test_smw_kernel_shape(self):
        """SMW kernel should have correct shape."""
        np.random.seed(42)
        k = 20
        V_Ninv_VT = np.random.randn(k, k)
        V_Ninv_VT = V_Ninv_VT @ V_Ninv_VT.T  # Make symmetric
        Lambda_diag = np.abs(np.random.randn(k)) + 0.1

        K = smw_kernel(V_Ninv_VT, Lambda_diag)

        assert K.shape == (k, k)

    def test_smw_kernel_values(self):
        """SMW kernel should equal Λ^{-1} + V N^{-1} V^T."""
        np.random.seed(42)
        k = 10
        V_Ninv_VT = np.random.randn(k, k)
        V_Ninv_VT = V_Ninv_VT @ V_Ninv_VT.T
        Lambda_diag = np.abs(np.random.randn(k)) + 0.1

        K = smw_kernel(V_Ninv_VT, Lambda_diag)

        # Direct computation
        expected = V_Ninv_VT + np.diag(1.0 / Lambda_diag)

        np.testing.assert_allclose(K, expected, rtol=1e-12)


class TestSMWQuadraticForm:
    """Tests for SMW quadratic form."""

    def test_smw_quadratic_form_matches_direct(self):
        """SMW quadratic form should match direct computation."""
        np.random.seed(42)
        n, k = 100, 20

        # Create positive definite N
        A = np.random.randn(n, n)
        N = A @ A.T + np.eye(n)
        N_inv = matrix_inverse_symm(N.copy())
        N_chol = cholesky_factor(N.copy())

        # Create projector V and diagonal Lambda
        V = np.random.randn(k, n)
        Lambda_diag = np.abs(np.random.randn(k)) + 0.1

        # Precompute SMW quantities
        V_N_inv = V @ N_inv
        V_Ninv_VT = V_N_inv @ V.T

        # Data vector
        data = np.random.randn(n)

        # SMW quadratic form
        smw_result = smw_quadratic_form(data, N_chol, V_N_inv, V_Ninv_VT, Lambda_diag)

        # Direct: d^T C^{-1} d where C = N + V^T Λ V
        Lambda = np.diag(Lambda_diag)
        C = N + V.T @ Lambda @ V
        C_inv = matrix_inverse_symm(C.copy())
        direct_result = float(data.T @ C_inv @ data)

        np.testing.assert_allclose(smw_result, direct_result, rtol=1e-9)

    @pytest.mark.parametrize("n,k", [(100, 10), (200, 20), (500, 50)])
    def test_smw_quadratic_form_various_sizes(self, n, k):
        """SMW quadratic form should work for various sizes."""
        np.random.seed(42)

        A = np.random.randn(n, n)
        N = A @ A.T + np.eye(n)
        N_inv = matrix_inverse_symm(N.copy())
        N_chol = cholesky_factor(N.copy())

        V = np.random.randn(k, n)
        Lambda_diag = np.abs(np.random.randn(k)) + 0.1

        V_N_inv = V @ N_inv
        V_Ninv_VT = V_N_inv @ V.T

        data = np.random.randn(n)

        smw_result = smw_quadratic_form(data, N_chol, V_N_inv, V_Ninv_VT, Lambda_diag)

        # Direct computation
        Lambda = np.diag(Lambda_diag)
        C = N + V.T @ Lambda @ V
        C_inv = matrix_inverse_symm(C.copy())
        direct_result = float(data.T @ C_inv @ data)

        np.testing.assert_allclose(smw_result, direct_result, rtol=1e-8)


def test_smw_quadratic_form_performance(capsys):
    """Compare SMW vs direct quadratic form performance."""
    import time

    np.random.seed(42)
    n, k = 500, 50
    n_iterations = 10

    # Setup
    A = np.random.randn(n, n)
    N = A @ A.T + np.eye(n)
    N_inv = matrix_inverse_symm(np.asfortranarray(N.copy()))
    N_chol = cholesky_factor(np.asfortranarray(N.copy()))

    V = np.random.randn(k, n)

    # Precompute SMW quantities (one-time cost)
    t0 = time.perf_counter()
    V_N_inv = V @ N_inv
    V_Ninv_VT = V_N_inv @ V.T
    precompute_time = time.perf_counter() - t0

    # Generate different Lambda_diag values and data vectors
    Lambda_diags = [np.abs(np.random.randn(k)) + 0.1 for _ in range(n_iterations)]
    data_vectors = [np.random.randn(n) for _ in range(n_iterations)]

    # Time SMW quadratic form (multiple calls)
    t0 = time.perf_counter()
    for Lambda_diag, data in zip(Lambda_diags, data_vectors):
        _ = smw_quadratic_form(data, N_chol, V_N_inv, V_Ninv_VT, Lambda_diag)
    smw_time = time.perf_counter() - t0

    # Time direct quadratic form (multiple calls)
    t0 = time.perf_counter()
    for Lambda_diag, data in zip(Lambda_diags, data_vectors):
        Lambda = np.diag(Lambda_diag)
        C = N + V.T @ Lambda @ V
        C_inv = matrix_inverse_symm(np.asfortranarray(C.copy()))
        _ = float(data.T @ C_inv @ data)
    direct_time = time.perf_counter() - t0

    # Print results
    print(f"\n{'=' * 60}")
    print(f"SMW Quadratic Form Performance (n={n}, k={k}, {n_iterations} iters)")
    print(f"{'=' * 60}")
    print(f"Precompute time:     {precompute_time * 1000:8.2f} ms (one-time)")
    print(
        f"SMW total time:      "
        f"{smw_time * 1000:8.2f} ms ({smw_time / n_iterations * 1000:.2f} ms/iter)"
    )
    print(
        f"Direct total time:   "
        f"{direct_time * 1000:8.2f} ms ({direct_time / n_iterations * 1000:.2f} ms/iter)"
    )
    print(f"Speedup:             {direct_time / smw_time:.1f}x")
    print(f"{'=' * 60}")

    # SMW should be faster for k << n
    assert smw_time < direct_time, "SMW should be faster than direct"


class TestCholeskyFactorSolve:
    """Tests for cholesky_factor / cholesky_solve wrappers."""

    def test_factor_solve_matches_direct_inverse(self):
        rng = np.random.default_rng(0)
        n = 32
        A = rng.standard_normal((n, n))
        M = A @ A.T + np.eye(n)
        b = rng.standard_normal(n)

        N_chol = cholesky_factor(M)
        x = cholesky_solve(N_chol, b)
        x_ref = np.linalg.solve(M, b)
        np.testing.assert_allclose(x, x_ref, rtol=1e-10, atol=1e-12)

    def test_solve_matrix_rhs(self):
        rng = np.random.default_rng(1)
        n, k = 24, 7
        A = rng.standard_normal((n, n))
        M = A @ A.T + np.eye(n)
        B = rng.standard_normal((n, k))

        N_chol = cholesky_factor(M)
        X = cholesky_solve(N_chol, B)
        np.testing.assert_allclose(M @ X, B, rtol=1e-10, atol=1e-12)

    def test_overwrite_a_aliases_input(self):
        rng = np.random.default_rng(2)
        n = 16
        A = rng.standard_normal((n, n))
        M = np.asfortranarray(A @ A.T + np.eye(n))
        M_id = id(M)

        c, lower = cholesky_factor(M, overwrite_a=True)
        assert lower is True
        assert id(c) == M_id  # in-place: factor shares storage with M

    def test_non_positive_definite_raises(self):
        M = np.array([[1.0, 2.0], [2.0, 1.0]])
        with pytest.raises(np.linalg.LinAlgError):
            cholesky_factor(M)
