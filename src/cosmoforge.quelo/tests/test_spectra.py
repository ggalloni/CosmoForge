import os
import time

import numpy as np
import pytest
from mpi4py import MPI

from quelo import Fisher, Spectra


class TestSpectra:
    """
    Test class for QML power spectra computation with Fisher matrix reuse.

    This class computes Fisher instances once and reuses them across multiple tests
    to improve performance while maintaining comprehensive test coverage.
    """

    def __init__(self):
        self._fisher_cache = {}
        self._config_resolver = None
        self._local_path = None

    def setup_method(self, method):
        """Setup method called before each test method."""
        pass

    def teardown_method(self, method):
        """Teardown method called after each test method."""
        pass

    def get_fisher_instance(self, fields: str) -> Fisher:
        """
        Get a Fisher instance for the specified fields, using cache for efficiency.

        Parameters
        ----------
        fields : str
            Field specification (e.g., "T", "QU", "TQU", "TEB")

        Returns
        -------
        Fisher
            Fisher instance for the specified fields
        """
        if fields not in self._fisher_cache:
            print(f"Creating Fisher instance for fields: {fields}")
            config_file = self._config_resolver(f"tests/data/nside4/{fields}/config.yaml")
            fisher = Fisher(config_file)
            fisher.run()
            os.unlink(config_file)
            self._fisher_cache[fields] = fisher
        else:
            print(f"Reusing cached Fisher instance for fields: {fields}")

        return self._fisher_cache[fields]

    def get_qml_analyzer(
        self, fields: str, config_type: str = "config", fisher: Fisher = None
    ) -> Spectra:
        """
        Create a Spectra instance for QML power spectrum estimation.

        Parameters
        ----------
        fields : str
            Field specification (e.g., "T", "QU", "TQU", "TEB")
        config_type : str, optional
            Configuration type ("config" or "cross_config"), by default "config"
        fisher : Fisher, optional
            Pre-computed Fisher instance to reuse, by default None

        Returns
        -------
        Spectra
            Configured Spectra instance
        """
        print(f"Creating QML analyzer for fields: {fields}, config: {config_type}")

        config_file = self._config_resolver(
            f"tests/data/nside4/{fields}/{config_type}.yaml"
        )
        qml_analyzer = Spectra(config_file, fisher=fisher)
        qml_analyzer.run()
        os.unlink(config_file)

        return qml_analyzer

    def get_qml_spectra(
        self, qml_analyzer: Spectra
    ) -> tuple[np.ndarray, np.ndarray, Fisher]:
        """
        Get QML power spectra computation results.

        Parameters
        ----------
        qml_analyzer : Spectra
            Configured Spectra instance

        Returns
        -------
        tuple[np.ndarray, np.ndarray, Fisher]
            Power spectra, noise bias, and Fisher instance
        """
        power_spectra = qml_analyzer.get_power_spectra()
        noise_bias = qml_analyzer.get_noise_bias()
        return power_spectra, noise_bias, qml_analyzer.fisher_instance

    def assert_spectra_match_reference(self, power_spectra: np.ndarray, fields: str):
        """
        Assert that computed spectra match reference values.

        Parameters
        ----------
        power_spectra : np.ndarray
            Computed power spectra
        fields : str
            Field specification for reference file lookup
        """
        ref_file = self._local_path + f"/tests/data/nside4/{fields}/ref_spectra.txt"
        ref = np.loadtxt(ref_file)

        np.testing.assert_allclose(
            power_spectra,
            ref,
            atol=1e-3,
            rtol=1e-5,
            err_msg=f"Power spectra for fields {fields} do not match reference",
        )


# Create global test instance
test_spectra = TestSpectra()


@pytest.fixture(autouse=True)
def setup_test_spectra(local_path, config_resolver):
    """Setup fixture to initialize test_spectra with required parameters."""
    test_spectra._local_path = local_path
    test_spectra._config_resolver = config_resolver


@pytest.mark.parametrize("fields", ["T", "QU", "TQU", "TEB"])
def test_spectra_computation(fields, local_path, config_resolver):
    """Test the QML power spectra computation for the specified fields."""
    # Get cached Fisher instance
    fisher = test_spectra.get_fisher_instance(fields)

    # Create QML analyzer using the cached Fisher
    qml_analyzer = test_spectra.get_qml_analyzer(fields, fisher=fisher)

    # Get spectra results
    power_spectra, noise_bias, fisher_instance = test_spectra.get_qml_spectra(
        qml_analyzer
    )

    # Assert results match reference
    test_spectra.assert_spectra_match_reference(power_spectra, fields)


@pytest.mark.skipif(
    "OMPI_COMM_WORLD_RANK" not in os.environ and "PMI_RANK" not in os.environ,
    reason="MPI test: run with mpirun/mpiexec",
)
def test_spectra_mpi_structure(local_path, config_resolver):
    """Test MPI structure and behavior for power spectra computation."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    fields = "TQU"

    # Get cached Fisher instance
    fisher = test_spectra.get_fisher_instance(fields)

    # Create QML analyzer using the cached Fisher
    qml_analyzer = test_spectra.get_qml_analyzer(fields, fisher=fisher)

    # Get spectra results
    power_spectra, noise_bias, fisher_instance = test_spectra.get_qml_spectra(
        qml_analyzer
    )

    if rank == 0:
        assert power_spectra is not None, "Rank 0 should get power spectra"
        assert noise_bias is not None, "Rank 0 should get noise bias"
        # Check against reference for rank 0 only
        test_spectra.assert_spectra_match_reference(power_spectra, fields)
    else:
        assert power_spectra is None, f"Rank {rank} should get None for power spectra"
        assert noise_bias is None, f"Rank {rank} should get None for noise bias"

    # Ensure all ranks finished successfully
    success = True
    all_success = comm.allreduce(success, op=MPI.LAND)
    assert all_success, "All ranks should reach the end of the test"


@pytest.mark.parametrize("fields", ["QU"])
def test_cross_spectra_computation(fields, local_path, config_resolver):
    """Test the QML cross power spectra computation for the specified fields."""
    # Get cached Fisher instance (same Fisher can be used for cross-spectra)
    fisher = test_spectra.get_fisher_instance(fields)

    # Create QML analyzer for cross-spectra using the cached Fisher
    qml_analyzer = test_spectra.get_qml_analyzer(
        fields, config_type="cross_config", fisher=fisher
    )

    # Get spectra results
    power_spectra, noise_bias, fisher_instance = test_spectra.get_qml_spectra(
        qml_analyzer
    )

    # Assert results match reference (same reference as regular spectra)
    test_spectra.assert_spectra_match_reference(power_spectra, fields)


def test_spectra_reuse_optimization(local_path, config_resolver):
    """Test that Fisher reuse optimization works correctly and gives same results."""
    fields = "TQU"

    # Get cached Fisher instance
    fisher = test_spectra.get_fisher_instance(fields)

    # Test with Fisher reuse
    start_time = time.time()
    qml_with_fisher = test_spectra.get_qml_analyzer(fields, fisher=fisher)
    power_spectra_with_reuse, _, _ = test_spectra.get_qml_spectra(qml_with_fisher)
    time_with_reuse = time.time() - start_time

    # Test without Fisher reuse (create fresh instance)
    start_time = time.time()
    qml_without_fisher = test_spectra.get_qml_analyzer(fields)  # No fisher parameter
    power_spectra_without_reuse, _, _ = test_spectra.get_qml_spectra(qml_without_fisher)
    time_without_reuse = time.time() - start_time

    print(f"Time with Fisher reuse: {time_with_reuse:.2f}s")
    print(f"Time without Fisher reuse: {time_without_reuse:.2f}s")

    # Both should give same results
    assert (
        power_spectra_with_reuse is not None and power_spectra_without_reuse is not None
    )
    assert power_spectra_with_reuse.shape == power_spectra_without_reuse.shape
    np.testing.assert_allclose(
        power_spectra_with_reuse,
        power_spectra_without_reuse,
        atol=1e-3,
        rtol=1e-5,
        err_msg="Spectra with and without Fisher reuse do not match.",
    )


def test_fisher_cache_efficiency():
    """Test that Fisher instances are properly cached and reused."""
    fields = "TEB"

    # First call should create Fisher
    fisher1 = test_spectra.get_fisher_instance(fields)

    # Second call should return cached Fisher
    fisher2 = test_spectra.get_fisher_instance(fields)

    # Should be the same object
    assert fisher1 is fisher2, "Fisher instances should be cached and reused"

    # Cache should contain the fields
    assert fields in test_spectra._fisher_cache, f"Fisher cache should contain {fields}"


# Backward compatibility functions for direct usage
def get_fisher_instance(
    fields: str = "TEB", config_resolver=None, local_path: str = None
) -> Fisher:
    """Create a Fisher instance for QML power spectrum estimation."""
    if test_spectra._config_resolver is None:
        test_spectra._config_resolver = config_resolver
        test_spectra._local_path = local_path
    return test_spectra.get_fisher_instance(fields)


def get_qml_analyzer(
    fields: str = "TEB",
    config_resolver=None,
    local_path: str = None,
    config_type="config",
) -> Spectra:
    """Create a Spectra instance for QML power spectrum estimation."""
    if test_spectra._config_resolver is None:
        test_spectra._config_resolver = config_resolver
        test_spectra._local_path = local_path
    return test_spectra.get_qml_analyzer(fields, config_type)


def get_qml_spectra(
    fields: str = "TEB",
    config_resolver=None,
    local_path: str = None,
    qml_analyzer: Spectra = None,
) -> tuple[np.ndarray, np.ndarray, Fisher]:
    """Get QML power spectra computation results."""
    if qml_analyzer is None:
        if test_spectra._config_resolver is None:
            test_spectra._config_resolver = config_resolver
            test_spectra._local_path = local_path
        qml_analyzer = test_spectra.get_qml_analyzer(fields)
    return test_spectra.get_qml_spectra(qml_analyzer)


if __name__ == "__main__":
    # Run the tests
    import tempfile

    # Mock config resolver for standalone execution
    def mock_config_resolver(config_path):
        # Create temporary file with the config path
        temp_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        # For standalone testing, you'd need to copy the actual config content
        # This is a simplified version for demonstration
        temp_file.write("# Mock config file\n")
        temp_file.close()
        return temp_file.name

    path = os.path.abspath(__file__.split("/tests/test_spectra.py")[0])

    # Initialize test instance
    test_spectra._local_path = path
    test_spectra._config_resolver = mock_config_resolver

    print(f"Running tests in directory: {path}")

    # Test all fields
    fields_list = ["T", "QU", "TQU", "TEB"]
    for fields in fields_list:
        print(f"Testing QML power spectra computation for fields: {fields}")
        try:
            # This would need actual config files to work
            test_spectra_computation(fields, path, mock_config_resolver)
            print(f"✓ Tests passed for {fields}")
        except Exception as e:
            print(f"✗ Tests failed for {fields}: {e}")

    print("Testing optimization with Fisher reuse...")
    try:
        test_spectra_reuse_optimization(path, mock_config_resolver)
        print("✓ Fisher reuse optimization test passed")
    except Exception as e:
        print(f"✗ Fisher reuse optimization test failed: {e}")

    print("Testing Fisher cache efficiency...")
    try:
        test_fisher_cache_efficiency()
        print("✓ Fisher cache efficiency test passed")
    except Exception as e:
        print(f"✗ Fisher cache efficiency test failed: {e}")

    print("All tests completed!")
