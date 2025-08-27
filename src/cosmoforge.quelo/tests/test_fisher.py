import os

import numpy as np
import pytest

from quelo import Fisher


def get_fisher_matrix(fields: str = "TEB", local_path: str = None) -> np.ndarray:
    # Create Fisher instance with parameter file
    fisher_analyzer = Fisher(local_path + f"/tests/data/nside4/{fields}/config.yaml")

    # Run the complete analysis pipeline
    fisher_analyzer.run()

    # Optionally, get results (only available on rank 0)
    fisher_matrix = fisher_analyzer.get_fisher_matrix()
    return fisher_matrix


@pytest.mark.parametrize("fields", ["T", "QU", "TQU", "TEB"])
def test_fisher_computation(fields, local_path):
    # Test the Fisher matrix computation for the specified fields
    fisher_matrix = get_fisher_matrix(fields, local_path=local_path)
    assert fisher_matrix is not None, "Fisher matrix should not be None"

    file = local_path + f"/tests/data/nside4/{fields}/ref_fisher.dat"
    ref = np.loadtxt(file, dtype=np.float64)

    assert fisher_matrix.shape == ref.shape, (
        f"Fisher matrix shape should match reference: {ref.shape}"
    )

    diff = fisher_matrix - ref

    np.testing.assert_allclose(
        diff,
        0.0,
        rtol=1e-5,
        atol=1e-8,
        err_msg=f"Fisher matrix for {fields} does not match reference.",
    )


if __name__ == "__main__":
    fields_list = ["TQU", "TEB"]

    path = os.path.abspath(__file__.split("/tests/test_fisher.py")[0])

    print(f"Running tests in directory: {path}")

    for fields in fields_list:
        print(f"Testing Fisher matrix computation for fields: {fields}")
        test_fisher_computation(fields, local_path=path)
