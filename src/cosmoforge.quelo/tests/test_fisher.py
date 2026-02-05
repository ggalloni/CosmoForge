import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from mpi4py import MPI

from quelo import Fisher


def get_fisher_matrix(
    fields: str = "TEB",
    config_resolver=None,
    config_type: str = "config",
) -> np.ndarray:
    # Create Fisher instance with parameter file
    config_file = config_resolver(f"tests/data/nside4/{fields}/{config_type}.yaml")
    fisher_analyzer = Fisher(config_file)

    # Run the complete analysis pipeline
    fisher_analyzer.run()

    error_bars = fisher_analyzer.get_error_bars()

    # Optionally, get results (only available on rank 0)
    fisher_matrix = fisher_analyzer.get_fisher_matrix()

    # Clean up temporary config file
    os.unlink(config_file)

    return fisher_matrix, error_bars


@pytest.mark.parametrize("fields", ["T", "QU", "TQU", "TEB"])
def test_fisher_computation(fields, local_path, config_resolver):
    # Test the Fisher matrix computation for the specified fields
    fisher_matrix, error_bars = get_fisher_matrix(fields, config_resolver=config_resolver)
    assert fisher_matrix is not None, "Fisher matrix should not be None"
    assert error_bars is not None, "Error bars should not be None"

    file = local_path + f"/tests/data/nside4/{fields}/ref_fisher.dat"
    ref = np.loadtxt(file, dtype=np.float64)

    assert fisher_matrix.shape == ref.shape, (
        f"Fisher matrix shape should match reference: {ref.shape}"
    )

    np.testing.assert_allclose(
        fisher_matrix,
        ref,
        atol=1e-3,
        rtol=1e-5,
        err_msg=f"Fisher matrix for {fields} does not match reference.",
    )


@pytest.mark.parametrize("fields", ["QU"])
def test_cross_fisher_computation(fields, local_path, config_resolver):
    fisher_matrix, error_bars = get_fisher_matrix(
        fields, config_resolver=config_resolver, config_type="cross_config"
    )
    assert fisher_matrix is not None, "Fisher matrix should not be None"
    assert error_bars is not None, "Error bars should not be None"

    file = local_path + f"/tests/data/nside4/{fields}/ref_fisher.dat"
    ref = np.loadtxt(file, dtype=np.float64)

    assert fisher_matrix.shape == ref.shape, (
        f"Fisher matrix shape should match reference: {ref.shape}"
    )

    np.testing.assert_allclose(
        fisher_matrix,
        ref,
        atol=1e-3,
        rtol=1e-5,
        err_msg=f"Fisher matrix for {fields} does not match reference.",
    )


@patch("quelo.fisher.MPI")
def test_fisher_worker_rank_behavior(mock_mpi, config_resolver):
    """
    Test Fisher matrix computation behavior for worker ranks (rank != 0).
    Worker ranks should return None for Fisher matrix and error bars.
    """
    from cosmocore import InputParams

    # Mock MPI to simulate rank=1 (worker process)
    mock_comm = MagicMock()
    mock_comm.Get_rank.return_value = 1  # Worker rank
    mock_comm.Get_size.return_value = 2  # Total 2 processes
    mock_comm.Barrier.return_value = None
    mock_mpi.COMM_WORLD = mock_comm
    mock_mpi.LAND = MPI.LAND  # Use real MPI constant

    fields = "TEB"
    config_file = config_resolver(f"tests/data/nside4/{fields}/config.yaml")

    try:
        # Create Fisher instance with mocked MPI (this will think it's rank 1)
        fisher_analyzer = Fisher(config_file)

        # Verify that the mock worked - fisher should think it's rank 1
        assert fisher_analyzer.rank == 1, "Fisher instance should have rank=1"
        assert fisher_analyzer.size == 2, "Fisher instance should have size=2"

        # Load real params for broadcast simulation
        real_params = InputParams.read_parameter_file(config_file)

        # Mock the broadcast to return the params when needed
        broadcast_call_count = 0

        def mock_bcast(data, root=0):
            nonlocal broadcast_call_count
            broadcast_call_count += 1

            # Return the appropriate broadcast data based on call order
            if broadcast_call_count == 1:  # params
                return real_params
            elif broadcast_call_count == 2:  # collection
                return None  # Worker can handle None collection
            else:  # Other broadcasts (npixs, pixact, etc.)
                return None

        mock_comm.bcast.side_effect = mock_bcast
        # Workers don't contribute to final result
        mock_comm.allreduce.return_value = None

        # For rank=1, we just test that get_fisher_matrix() and get_error_bars()
        # return None without running the full pipeline
        fisher_matrix = fisher_analyzer.get_fisher_matrix()
        error_bars = fisher_analyzer.get_error_bars()

        # Worker ranks should get None for both Fisher matrix and error bars
        assert fisher_matrix is None, "Worker rank should get None for Fisher matrix"
        assert error_bars is None, "Worker rank should get None for error bars"

    finally:
        # Clean up temporary config file
        os.unlink(config_file)


def get_fisher_matrix_compressed(
    fields: str = "T",
    config_resolver=None,
    compression_method: str = "harmonic",
    epsilon: float = None,
) -> np.ndarray:
    """Run Fisher computation with compression enabled."""
    config_file = config_resolver(f"tests/data/nside4/{fields}/config.yaml")

    # Compression configuration
    compression_config = {
        "method": compression_method,
        "epsilon": epsilon,
    }

    fisher_analyzer = Fisher(config_file, compression=compression_config)
    fisher_analyzer.run()

    fisher_matrix = fisher_analyzer.get_fisher_matrix()
    error_bars = fisher_analyzer.get_error_bars()

    # Clean up temporary config file
    os.unlink(config_file)

    return fisher_matrix, error_bars


@pytest.mark.parametrize("compression_method", ["harmonic"])
def test_compressed_fisher_runs(compression_method, local_path, config_resolver):
    """
    Test that compressed Fisher computation runs without errors.

    The compression algorithm is an approximation that works best when
    n_pix >> n_modes. For marginal cases (like the test data with
    n_pix/n_modes ≈ 2.5), exact agreement with pixel-space Fisher is not expected.

    This test verifies:
    1. The compression pipeline executes without errors
    2. The Fisher matrix has the correct shape
    3. The Fisher matrix is positive definite (physically meaningful)
    """
    fields = "T"  # Compression only supports single-spectrum

    fisher_matrix, error_bars = get_fisher_matrix_compressed(
        fields,
        config_resolver=config_resolver,
        compression_method=compression_method,
        epsilon=1e-10,
    )

    assert fisher_matrix is not None, "Fisher matrix should not be None"
    assert error_bars is not None, "Error bars should not be None"

    # Load Fortran reference for shape comparison
    file = local_path + f"/tests/data/nside4/{fields}/ref_fisher.dat"
    ref = np.loadtxt(file, dtype=np.float64)

    assert fisher_matrix.shape == ref.shape, (
        f"Fisher matrix shape should match reference: {ref.shape}"
    )

    # Check positive definiteness (all eigenvalues > 0)
    eigenvalues = np.linalg.eigvalsh(fisher_matrix)
    assert np.all(eigenvalues > -1e-10), "Fisher matrix should be positive semi-definite"

    # Diagonal should be positive (variance terms)
    assert np.all(fisher_matrix.diagonal() > 0), "Fisher diagonal should be positive"


def test_compressed_fisher_algorithm_consistency(local_path, config_resolver):
    """
    Test that the compression algorithm is internally consistent.

    The compressed Fisher F_ij = (1/2) Tr[(V C^{-1} V^T) E_i (V C^{-1} V^T) E_j]
    uses the correct SMW formula for V C^{-1} V^T, which should match the
    pixel-space Fisher to high precision.

    This test verifies that the compressed Fisher matches the pixel-space
    (Fortran reference) Fisher matrix within numerical tolerance.
    """
    fields = "T"

    # Get both Fisher matrices
    fisher_uncompressed, _ = get_fisher_matrix(fields, config_resolver=config_resolver)
    fisher_compressed, _ = get_fisher_matrix_compressed(
        fields,
        config_resolver=config_resolver,
        compression_method="harmonic",
        epsilon=1e-10,
    )

    assert fisher_uncompressed is not None
    assert fisher_compressed is not None

    # Check symmetry
    np.testing.assert_allclose(
        fisher_compressed,
        fisher_compressed.T,
        rtol=1e-10,
        err_msg="Compressed Fisher should be symmetric",
    )

    # Check that both are positive definite
    eig_uncomp = np.linalg.eigvalsh(fisher_uncompressed)
    eig_comp = np.linalg.eigvalsh(fisher_compressed)

    assert np.all(eig_uncomp > -1e-10), "Uncompressed Fisher should be positive semi-def"
    assert np.all(eig_comp > -1e-10), "Compressed Fisher should be positive semi-def"

    # The compressed Fisher should match pixel-space Fisher exactly
    # (up to numerical precision) when using SMW formula for V C^{-1} V^T
    np.testing.assert_allclose(
        fisher_compressed,
        fisher_uncompressed,
        atol=1e-3,
        rtol=1e-5,
        err_msg="Compressed Fisher should match pixel-space Fisher",
    )


def test_compressed_fisher_matches_reference(local_path, config_resolver):
    """
    Test that compressed Fisher matches the Fortran reference.

    This is the ultimate validation: the compressed algorithm should
    reproduce the same results as the original Fortran implementation.
    """
    fields = "T"

    # Get compressed Fisher matrix
    fisher_compressed, _ = get_fisher_matrix_compressed(
        fields,
        config_resolver=config_resolver,
        compression_method="harmonic",
        epsilon=1e-10,
    )

    assert fisher_compressed is not None, "Compressed Fisher should not be None"

    # Load Fortran reference
    file = local_path + f"/tests/data/nside4/{fields}/ref_fisher.dat"
    ref = np.loadtxt(file, dtype=np.float64)

    assert fisher_compressed.shape == ref.shape, (
        f"Compressed Fisher shape should match reference: {ref.shape}"
    )

    # Compressed Fisher should match Fortran reference
    np.testing.assert_allclose(
        fisher_compressed,
        ref,
        atol=1e-3,
        rtol=1e-5,
        err_msg="Compressed Fisher does not match Fortran reference",
    )


def test_pixel_projected_fisher_degradation(local_path, config_resolver):
    """
    Test that PixelProjected Fisher stays within acceptable degradation.

    Based on Gjerløw et al., PixelProjected compression is an approximation
    that should give <10% error bar degradation when keeping sufficient modes.

    We compare against Harmonic compression (exact via SMW) as the reference.
    """
    fields = "T"

    # Get Harmonic Fisher (reference - exact via SMW)
    fisher_harmonic, _ = get_fisher_matrix_compressed(
        fields,
        config_resolver=config_resolver,
        compression_method="harmonic",
        epsilon=1e-10,
    )

    # Get PixelProjected Fisher with tight epsilon (should be close to exact)
    fisher_pixel_projected, _ = get_fisher_matrix_compressed(
        fields,
        config_resolver=config_resolver,
        compression_method="pixel_projected",
        epsilon=1e-6,  # Tight threshold to keep most modes
    )

    assert fisher_harmonic is not None
    assert fisher_pixel_projected is not None

    # Check shapes match
    assert fisher_pixel_projected.shape == fisher_harmonic.shape

    # Check symmetry
    np.testing.assert_allclose(
        fisher_pixel_projected,
        fisher_pixel_projected.T,
        rtol=1e-10,
        err_msg="PixelProjected Fisher should be symmetric",
    )

    # Check positive semi-definiteness
    eigenvalues = np.linalg.eigvalsh(fisher_pixel_projected)
    assert np.all(eigenvalues > -1e-10), "Fisher should be positive semi-definite"

    # Compute error bar degradation
    # Error bars = sqrt(diag(F^{-1}))
    fisher_harm_inv = np.linalg.pinv(fisher_harmonic)
    fisher_pp_inv = np.linalg.pinv(fisher_pixel_projected)

    sigma_harm = np.sqrt(np.maximum(np.diag(fisher_harm_inv), 0))
    sigma_pp = np.sqrt(np.maximum(np.diag(fisher_pp_inv), 0))

    # Compute degradation where sigma_harm > 0
    valid = sigma_harm > 1e-15
    if np.any(valid):
        degradation = np.abs(sigma_pp[valid] - sigma_harm[valid]) / sigma_harm[valid]
        max_degradation = np.max(degradation) * 100  # percent

        # Based on Gjerløw et al., with sufficient modes we should be <10%
        assert max_degradation < 15, (
            f"PixelProjected error bar degradation ({max_degradation:.1f}%) "
            f"exceeds 15% tolerance"
        )

    # Also check Fisher diagonal values are close
    diag_harm = np.diag(fisher_harmonic)
    diag_pp = np.diag(fisher_pixel_projected)

    valid_diag = np.abs(diag_harm) > 1e-15
    if np.any(valid_diag):
        diag_diff = (
            np.abs(diag_pp[valid_diag] - diag_harm[valid_diag]) / diag_harm[valid_diag]
        )
        max_diag_diff = np.max(diag_diff) * 100

        assert max_diag_diff < 20, (
            f"PixelProjected Fisher diagonal difference ({max_diag_diff:.1f}%) "
            f"exceeds 20% tolerance"
        )


if __name__ == "__main__":
    fields_list = ["T", "QU", "TQU", "TEB"]

    path = os.path.abspath(__file__.split("/tests/test_fisher.py")[0])

    print(f"Running tests in directory: {path}")

    for fields in fields_list:
        print(f"Testing Fisher matrix computation for fields: {fields}")
        test_fisher_computation(fields, local_path=path)
