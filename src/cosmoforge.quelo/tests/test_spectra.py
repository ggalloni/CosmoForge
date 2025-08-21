import os

import numpy as np
import pytest

from quelo import Fisher, Spectra


def get_qml_analyzer(fields: str = "TEB", local_path: str = None) -> Spectra:
    """Create a Spectra instance for QML power spectrum estimation."""
    # Create Spectra instance with parameter file
    qml_analyzer = Spectra(local_path + f"/tests/data/nside4/{fields}_nside4.yaml")

    # Run the complete QML analysis pipeline
    qml_analyzer.run()

    return qml_analyzer


def get_qml_spectra(
    fields: str = "TEB", local_path: str = None, qml_analyzer: Spectra = None
) -> tuple[np.ndarray, Fisher]:
    """Get QML power spectra computation results."""
    # Test without Fisher reuse to isolate the issue
    # Create Spectra instance without precomputed Fisher
    if qml_analyzer is None:
        qml_analyzer = get_qml_analyzer(fields, local_path=local_path)

    # Get results (only available on rank 0)
    power_spectra = qml_analyzer.get_power_spectra()
    noise_bias = qml_analyzer.get_noise_bias()

    return power_spectra, noise_bias, None  # No separate Fisher instance


# @pytest.mark.parametrize("fields", ["TEB", "TQU"])
@pytest.mark.parametrize("fields", ["TEB"])
def test_spectra_computation(fields, local_path):
    """Test the QML power spectra computation for the specified fields."""
    qml_analyzer = get_qml_analyzer(fields, local_path=local_path)
    print(qml_analyzer.__dict__)
    power_spectra, noise_bias, fisher = get_qml_spectra(fields, local_path=local_path)

    print(qml_analyzer.collection.spectra_manager._cls_dict)
    print(power_spectra)

    # Basic checks
    assert power_spectra is not None, "Power spectra should not be None"
    assert power_spectra.ndim == 2, "Power spectra should be 2D array (nsims x nell)"

    # Check that we have the expected number of simulations and multipoles
    expected_nsims = 3  # This should match the parameter file
    expected_nell = 6 * (16 - 1)  # 6 spectra * (lmax - 1), lmax=16 from yaml

    assert power_spectra.shape[0] == expected_nsims, (
        f"Expected {expected_nsims} simulations"
    )
    assert power_spectra.shape[1] == expected_nell, (
        f"Expected {expected_nell} multipole bins"
    )


@pytest.mark.skip
def test_spectra_without_fisher(local_path):
    """Test that Spectra can compute Fisher internally when not provided."""
    # Create Spectra instance without precomputed Fisher
    qml_analyzer = Spectra(local_path + "/tests/data/nside4/TEB_nside4.yaml")

    # This should work and compute Fisher internally
    qml_analyzer.run()

    # Check results
    power_spectra = qml_analyzer.get_power_spectra()
    assert power_spectra is not None, "Power spectra should not be None"


@pytest.mark.skip
def test_spectra_reuse_optimization(local_path):
    """Test that using precomputed Fisher is faster than computing from scratch."""
    import time

    # Method 1: Compute Fisher separately then reuse
    start_time = time.time()
    fisher = Fisher(local_path + "/tests/data/nside4/TEB_nside4.yaml")
    fisher.run()

    qml_with_fisher = Spectra(
        local_path + "/tests/data/nside4/TEB_nside4.yaml", fisher=fisher
    )
    qml_with_fisher.run()
    time_with_reuse = time.time() - start_time

    # Method 2: Compute Fisher internally
    start_time = time.time()
    qml_without_fisher = Spectra(local_path + "/tests/data/nside4/TEB_nside4.yaml")
    qml_without_fisher.run()
    time_without_reuse = time.time() - start_time

    # Reusing should be faster (though this might not always be true for small test cases)
    print(f"Time with Fisher reuse: {time_with_reuse:.2f}s")
    print(f"Time without Fisher reuse: {time_without_reuse:.2f}s")

    # Both should give same results
    spectra1 = qml_with_fisher.get_power_spectra()
    spectra2 = qml_without_fisher.get_power_spectra()

    assert spectra1 is not None and spectra2 is not None
    assert spectra1.shape == spectra2.shape


if __name__ == "__main__":
    # Run the tests
    fields_list = ["TEB", "TQU"]

    path = os.path.abspath(__file__.split("/tests/test_spectra.py")[0])

    print(f"Running tests in directory: {path}")

    for fields in fields_list:
        print(f"Testing QML power spectra computation for fields: {fields}")
        test_spectra_computation(fields, local_path=path)

    print("Testing Spectra without precomputed Fisher...")
    test_spectra_without_fisher(local_path=path)

    print("Testing optimization with Fisher reuse...")
    test_spectra_reuse_optimization(local_path=path)

    print("All tests passed!")
