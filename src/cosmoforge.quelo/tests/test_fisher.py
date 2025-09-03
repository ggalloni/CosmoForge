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


if __name__ == "__main__":
    fields_list = ["T", "QU", "TQU", "TEB"]

    path = os.path.abspath(__file__.split("/tests/test_fisher.py")[0])

    print(f"Running tests in directory: {path}")

    for fields in fields_list:
        print(f"Testing Fisher matrix computation for fields: {fields}")
        test_fisher_computation(fields, local_path=path)
