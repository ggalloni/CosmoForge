import os
import time

import numpy as np
import pytest

from quelo import Fisher, Spectra


def get_fisher_instance(
    fields: str = "TEB", config_resolver=None, local_path: str = None
) -> Fisher:
    """Create a Fisher instance for QML power spectrum estimation."""

    print("Creating Fisher instance...")
    config_file = config_resolver(f"tests/data/nside4/{fields}/config.yaml")
    fisher = Fisher(config_file)
    fisher.run()

    # Clean up temporary config file
    os.unlink(config_file)

    return fisher


def get_qml_analyzer(
    fields: str = "TEB", config_resolver=None, local_path: str = None
) -> Spectra:
    """Create a Spectra instance for QML power spectrum estimation."""

    print("Creating QML analyzer...")

    config_file = config_resolver(f"tests/data/nside4/{fields}/config.yaml")
    qml_analyzer = Spectra(config_file)

    qml_analyzer.run()

    # Clean up temporary config file
    os.unlink(config_file)

    return qml_analyzer


def get_qml_spectra(
    fields: str = "TEB",
    config_resolver=None,
    local_path: str = None,
    qml_analyzer: Spectra = None,
) -> tuple[np.ndarray, Fisher]:
    """Get QML power spectra computation results."""
    if qml_analyzer is None:
        qml_analyzer = get_qml_analyzer(fields, config_resolver=config_resolver)

    power_spectra = qml_analyzer.get_power_spectra()
    noise_bias = qml_analyzer.get_noise_bias()

    return power_spectra, noise_bias, qml_analyzer.fisher_instance


@pytest.mark.parametrize("fields", ["T", "QU", "TQU", "TEB"])
def test_spectra_computation(fields, local_path, config_resolver):
    """Test the QML power spectra computation for the specified fields."""
    qml_analyzer = get_qml_analyzer(fields, config_resolver=config_resolver)
    power_spectra, noise_bias, fisher = get_qml_spectra(
        fields, config_resolver=config_resolver, qml_analyzer=qml_analyzer
    )

    file = local_path + f"/tests/data/nside4/{fields}/ref_spectra.txt"
    ref = np.loadtxt(file)

    diff = power_spectra - ref

    np.testing.assert_allclose(
        diff,
        0,
        atol=1e-5,
        rtol=1e-8,
    )


def test_spectra_reuse_optimization(local_path, config_resolver):
    fisher = get_fisher_instance(fields="TQU", config_resolver=config_resolver)

    start_time = time.time()
    config_file1 = config_resolver("tests/data/nside4/TQU/config.yaml")
    qml_with_fisher = Spectra(config_file1, fisher=fisher)
    qml_with_fisher.run()
    time_with_reuse = time.time() - start_time
    os.unlink(config_file1)

    start_time = time.time()
    config_file2 = config_resolver("tests/data/nside4/TQU/config.yaml")
    qml_without_fisher = Spectra(config_file2)
    qml_without_fisher.run()
    time_without_reuse = time.time() - start_time
    os.unlink(config_file2)

    print(f"Time with Fisher reuse: {time_with_reuse:.2f}s")
    print(f"Time without Fisher reuse: {time_without_reuse:.2f}s")

    # Both should give same results
    spectra1 = qml_with_fisher.get_power_spectra()
    spectra2 = qml_without_fisher.get_power_spectra()

    assert spectra1 is not None and spectra2 is not None
    assert spectra1.shape == spectra2.shape


if __name__ == "__main__":
    # Run the tests
    fields_list = ["T", "QU", "TQU", "TEB"]

    path = os.path.abspath(__file__.split("/tests/test_spectra.py")[0])

    print(f"Running tests in directory: {path}")

    for fields in fields_list:
        print(f"Testing QML power spectra computation for fields: {fields}")
        test_spectra_computation(fields, local_path=path)

    print("Testing optimization with Fisher reuse...")
    test_spectra_reuse_optimization(local_path=path)

    print("All tests passed!")
