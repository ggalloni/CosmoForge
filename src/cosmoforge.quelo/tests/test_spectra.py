import os
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from mpi4py import MPI

from quelo import Fisher, Spectra


class HelperSpectra:
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

    def get_fisher_instance(self, fields: str, config_type: str = "config") -> Fisher:
        """
        Get a Fisher instance for the specified fields and config type, using cache for
        efficiency.

        Parameters
        ----------
        fields : str
            Field specification (e.g., "T", "QU", "TQU", "TEB")
        config_type : str, optional
            Configuration type ("config" or "cross_config"), by default "config"

        Returns
        -------
        Fisher
            Fisher instance for the specified fields and configuration
        """
        cache_key = f"{fields}_{config_type}"
        if cache_key not in self._fisher_cache:
            print(f"Creating Fisher instance for fields: {fields}, config: {config_type}")
            config_file = self._config_resolver(
                f"tests/data/nside4/{fields}/{config_type}.yaml"
            )
            fisher = Fisher(config_file)
            fisher.run()
            os.unlink(config_file)
            self._fisher_cache[cache_key] = fisher
        else:
            print(
                f"Reusing cached Fisher instance for fields: {fields}, "
                f"config: {config_type}"
            )

        return self._fisher_cache[cache_key]

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
test_spectra = HelperSpectra()


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


@patch("quelo.spectra.MPI")
def test_spectra_worker_rank_behavior(mock_mpi, config_resolver):
    """
    Test QML power spectra computation behavior for worker ranks (rank != 0).
    Worker ranks should return None for power spectra and noise bias.
    """
    # Mock MPI to simulate rank=1 (worker process)
    mock_comm = MagicMock()
    mock_comm.Get_rank.return_value = 1  # Worker rank
    mock_comm.Get_size.return_value = 2  # Total 2 processes
    mock_comm.Barrier.return_value = None
    mock_mpi.COMM_WORLD = mock_comm
    mock_mpi.LAND = MPI.LAND  # Use real MPI constant

    fields = "T"  # Use simple field for faster test
    config_file = config_resolver(f"tests/data/nside4/{fields}/config.yaml")

    try:
        # Create Spectra instance with mocked MPI (this will think it's rank 1)
        spectra_analyzer = Spectra(config_file)

        # Verify that the mock worked - spectra should think it's rank 1
        assert spectra_analyzer.rank == 1, "Spectra instance should have rank=1"
        assert spectra_analyzer.size == 2, "Spectra instance should have size=2"

        # For rank=1, we just test that get_power_spectra() and get_noise_bias()
        # return None without running the full pipeline
        power_spectra = spectra_analyzer.get_power_spectra()
        noise_bias = spectra_analyzer.get_noise_bias()

        # Worker ranks should get None for both power spectra and noise bias
        assert power_spectra is None, "Worker rank should get None for power spectra"
        assert noise_bias is None, "Worker rank should get None for noise bias"

    finally:
        # Clean up temporary config file
        os.unlink(config_file)


@pytest.mark.parametrize("fields", ["QU"])
def test_cross_spectra_computation(fields, local_path, config_resolver):
    """Test the QML cross power spectra computation for the specified fields."""
    # Get cached Fisher instance for cross-configuration (different from regular Fisher)
    fisher = test_spectra.get_fisher_instance(fields, config_type="cross_config")

    # Create QML analyzer for cross-spectra using the cross-configuration Fisher
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
    fields = "QU"

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

    # First call should create Fisher for regular config
    fisher1 = test_spectra.get_fisher_instance(fields)

    # Second call should return cached Fisher for regular config
    fisher2 = test_spectra.get_fisher_instance(fields)

    # Should be the same object
    assert fisher1 is fisher2, "Fisher instances should be cached and reused"

    # Cache should contain the fields with config type
    cache_key = f"{fields}_config"
    assert cache_key in test_spectra._fisher_cache, (
        f"Fisher cache should contain {cache_key}"
    )

    # Test that cross-config creates a different Fisher instance
    if fields == "QU":  # Only QU has cross_config available
        fisher_cross = test_spectra.get_fisher_instance(
            fields, config_type="cross_config"
        )
        assert fisher_cross is not fisher1, (
            "Cross-config should create different Fisher instance"
        )

        cache_key_cross = f"{fields}_cross_config"
        assert cache_key_cross in test_spectra._fisher_cache, (
            f"Fisher cache should contain {cache_key_cross}"
        )


@pytest.mark.parametrize("fields", ["T", "QU"])
def test_normalization_modes(fields, local_path, config_resolver):
    """Test the three normalization modes for QML power spectra."""
    # Get cached Fisher instance
    fisher = test_spectra.get_fisher_instance(fields)

    # Create QML analyzer
    qml_analyzer = test_spectra.get_qml_analyzer(fields, fisher=fisher)

    # Test deconvolved mode (default, backwards compatible)
    cl_deconv = qml_analyzer.get_power_spectra(mode="deconvolved")
    cl_default = qml_analyzer.get_power_spectra()  # Should be same as deconvolved

    assert cl_deconv is not None, "Deconvolved mode should return results"
    assert cl_default is not None, "Default mode should return results"
    np.testing.assert_array_equal(
        cl_deconv, cl_default, err_msg="Default mode should equal deconvolved mode"
    )

    # Test decorrelated mode
    cl_decorr = qml_analyzer.get_power_spectra(mode="decorrelated")
    assert cl_decorr is not None, "Decorrelated mode should return results"
    assert cl_decorr.shape == cl_deconv.shape, "Decorrelated shape should match"

    # Test convolved mode
    result = qml_analyzer.get_power_spectra(mode="convolved")
    assert result is not None, "Convolved mode should return results"
    assert isinstance(result, tuple), "Convolved mode should return tuple"
    assert len(result) == 3, "Convolved tuple should have 3 elements"

    y, W, convolve_func = result
    assert y is not None, "Convolved y should not be None"
    assert W is not None, "Window matrix should not be None"
    assert callable(convolve_func), "convolve_func should be callable"

    # Test that window matrix is square
    nell = cl_deconv.shape[1]
    assert W.shape == (nell, nell), f"Window matrix should be ({nell}, {nell})"

    # Test invalid mode
    with pytest.raises(ValueError, match="mode must be one of"):
        qml_analyzer.get_power_spectra(mode="invalid_mode")


@pytest.mark.parametrize("fields", ["T", "QU"])
def test_covariance_methods(fields, local_path, config_resolver):
    """Test get_covariance and get_error_bars methods."""
    # Get cached Fisher instance
    fisher = test_spectra.get_fisher_instance(fields)

    # Create QML analyzer
    qml_analyzer = test_spectra.get_qml_analyzer(fields, fisher=fisher)

    # Get nell from power spectra shape
    cl = qml_analyzer.get_power_spectra()
    nell = cl.shape[1]

    # Test deconvolved covariance (should be F^-1)
    cov_deconv = qml_analyzer.get_covariance(mode="deconvolved")
    assert cov_deconv is not None, "Deconvolved covariance should not be None"
    assert cov_deconv.shape == (nell, nell), "Covariance should be (nell, nell)"

    # Test decorrelated covariance (should be identity)
    cov_decorr = qml_analyzer.get_covariance(mode="decorrelated")
    assert cov_decorr is not None, "Decorrelated covariance should not be None"
    np.testing.assert_allclose(
        cov_decorr,
        np.eye(nell),
        atol=1e-10,
        err_msg="Decorrelated covariance should be identity",
    )

    # Test convolved covariance (should be Fisher)
    cov_conv = qml_analyzer.get_covariance(mode="convolved")
    assert cov_conv is not None, "Convolved covariance should not be None"
    assert cov_conv.shape == (nell, nell), "Convolved covariance should be (nell, nell)"

    # Test error bars
    errors_deconv = qml_analyzer.get_error_bars(mode="deconvolved")
    assert errors_deconv is not None, "Deconvolved errors should not be None"
    assert errors_deconv.shape == (nell,), "Errors should be 1D array"
    np.testing.assert_array_equal(
        errors_deconv,
        np.sqrt(np.diag(cov_deconv)),
        err_msg="Errors should be sqrt of diagonal",
    )

    # Decorrelated errors should all be 1.0
    errors_decorr = qml_analyzer.get_error_bars(mode="decorrelated")
    np.testing.assert_allclose(
        errors_decorr,
        np.ones(nell),
        atol=1e-10,
        err_msg="Decorrelated errors should all be 1.0",
    )

    # Test invalid mode
    with pytest.raises(ValueError, match="mode must be one of"):
        qml_analyzer.get_covariance(mode="invalid_mode")


def test_convolve_theory(local_path, config_resolver):
    """Test the convolve_theory helper method."""
    fields = "T"

    # Get cached Fisher instance
    fisher = test_spectra.get_fisher_instance(fields)

    # Create QML analyzer
    qml_analyzer = test_spectra.get_qml_analyzer(fields, fisher=fisher)

    # Create a mock theory spectrum
    cl = qml_analyzer.get_power_spectra()
    nell = cl.shape[1]
    theory = np.ones(nell)  # Simple test theory

    # Test convolve_theory
    convolved = qml_analyzer.convolve_theory(theory)
    assert convolved is not None, "convolve_theory should return result"
    assert convolved.shape == (nell,), "Convolved theory should have shape (nell,)"

    # Compare with convolved mode's helper
    _, W, convolve_func = qml_analyzer.get_power_spectra(mode="convolved")
    convolved_via_func = convolve_func(theory)

    np.testing.assert_allclose(
        convolved,
        convolved_via_func,
        atol=1e-10,
        err_msg="convolve_theory should match convolved mode's helper",
    )


def test_inv_fisher_sqrt_computation(local_path, config_resolver):
    """Test that F^(-1/2) is computed correctly."""
    fields = "T"

    # Get cached Fisher instance
    fisher = test_spectra.get_fisher_instance(fields)

    # Create QML analyzer
    qml_analyzer = test_spectra.get_qml_analyzer(fields, fisher=fisher)

    # Check that inv_fisher_sqrt exists
    assert qml_analyzer.inv_fisher_sqrt is not None, "inv_fisher_sqrt should be computed"

    # F^(-1/2) @ F^(-1/2) should equal F^(-1)
    # Note: This is only approximate due to eigenvalue truncation
    F_inv_sqrt = qml_analyzer.inv_fisher_sqrt
    F_inv_reconstructed = F_inv_sqrt @ F_inv_sqrt
    F_inv_actual = qml_analyzer.invfisher

    # Check that they're close (allowing for numerical precision)
    np.testing.assert_allclose(
        F_inv_reconstructed,
        F_inv_actual,
        rtol=1e-5,
        atol=1e-10,
        err_msg="F^(-1/2) @ F^(-1/2) should approximate F^(-1)",
    )


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
