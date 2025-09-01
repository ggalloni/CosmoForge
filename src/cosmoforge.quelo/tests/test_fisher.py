import os

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

    errors = fisher_analyzer.get_error_bars()

    assert errors is not None, "Error bars should not be None"

    # Optionally, get results (only available on rank 0)
    fisher_matrix = fisher_analyzer.get_fisher_matrix()

    # Clean up temporary config file
    os.unlink(config_file)

    return fisher_matrix


@pytest.mark.skipif(
    "OMPI_COMM_WORLD_RANK" not in os.environ and "PMI_RANK" not in os.environ,
    reason="MPI test: run with mpirun/mpiexec",
)
def test_fisher_mpi_structure(local_path, config_resolver):
    """
    Test that only rank 0 gets the Fisher matrix, others get None,
    and all ranks reach the end of run without error.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    fields = "TEB"
    fisher_matrix = get_fisher_matrix(fields, config_resolver=config_resolver)

    if rank == 0:
        assert fisher_matrix is not None, "Rank 0 should get Fisher matrix"
    else:
        assert fisher_matrix is None, f"Rank {rank} should get None for Fisher matrix"

    # Optionally, broadcast a success flag to ensure all ranks finished
    success = True
    all_success = comm.allreduce(success, op=MPI.LAND)
    assert all_success, "All ranks should reach the end of the test"

    if rank == 0:
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


@pytest.mark.parametrize("fields", ["T", "QU", "TQU", "TEB"])
def test_fisher_computation(fields, local_path, config_resolver):
    # Test the Fisher matrix computation for the specified fields
    fisher_matrix = get_fisher_matrix(fields, config_resolver=config_resolver)
    assert fisher_matrix is not None, "Fisher matrix should not be None"

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
    fisher_matrix = get_fisher_matrix(
        fields, config_resolver=config_resolver, config_type="cross_config"
    )
    assert fisher_matrix is not None, "Fisher matrix should not be None"

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


if __name__ == "__main__":
    fields_list = ["T", "QU", "TQU", "TEB"]

    path = os.path.abspath(__file__.split("/tests/test_fisher.py")[0])

    print(f"Running tests in directory: {path}")

    for fields in fields_list:
        print(f"Testing Fisher matrix computation for fields: {fields}")
        test_fisher_computation(fields, local_path=path)
