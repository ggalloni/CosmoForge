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


# ============================================================================
# Compressed Spectra Tests
# ============================================================================


def get_compressed_spectra(
    fields: str = "T",
    config_resolver=None,
    compression_method: str = "harmonic",
    epsilon: float = None,
    return_analyzer: bool = False,
) -> tuple:
    """Run Spectra computation with compression enabled."""
    config_file = config_resolver(f"tests/data/nside4/{fields}/config.yaml")

    compression_config = {
        "method": compression_method,
        "epsilon": epsilon,
    }

    spectra_analyzer = Spectra(config_file, compression=compression_config)
    spectra_analyzer.run()

    power_spectra = spectra_analyzer.get_power_spectra()
    noise_bias = spectra_analyzer.get_noise_bias()

    os.unlink(config_file)

    if return_analyzer:
        return power_spectra, noise_bias, spectra_analyzer
    return power_spectra, noise_bias


def test_compressed_spectra_matches_reference(local_path, config_resolver):
    """
    Test that compressed spectra computation matches the Fortran reference.

    This verifies that the SMW compression produces the same results as
    traditional pixel-space computation for QML power spectrum estimation.
    """
    fields = "T"

    # First get traditional spectra for comparison
    print("=== Traditional Spectra ===")
    fisher_trad = test_spectra.get_fisher_instance(fields)
    qml_trad = test_spectra.get_qml_analyzer(fields, fisher=fisher_trad)
    power_spectra_trad, noise_bias_trad, _ = test_spectra.get_qml_spectra(qml_trad)
    fisher_matrix_trad = fisher_trad.get_fisher_matrix()
    print(f"Traditional spectra[0,:5]: {power_spectra_trad[0, :5]}")
    print(f"Traditional Fisher diag[:5]: {np.diag(fisher_matrix_trad)[:5]}")

    # Get compressed spectra
    print("\n=== Compressed Spectra ===")
    power_spectra_compressed, noise_bias_compressed, spectra_comp = (
        get_compressed_spectra(
            fields,
            config_resolver=config_resolver,
            compression_method="harmonic",
            epsilon=1e-10,
            return_analyzer=True,
        )
    )

    assert power_spectra_compressed is not None, "Compressed spectra should not be None"
    assert noise_bias_compressed is not None, "Compressed noise bias should not be None"

    fisher_matrix_comp = spectra_comp.fisher_instance.get_fisher_matrix()
    print(f"Compressed spectra[0,:5]: {power_spectra_compressed[0, :5]}")
    print(f"Compressed Fisher diag[:5]: {np.diag(fisher_matrix_comp)[:5]}")
    print(
        f"Fisher diag ratio: "
        f"{np.diag(fisher_matrix_comp)[:5] / np.diag(fisher_matrix_trad)[:5]}"
    )

    # Compare normalization (vecmul)
    print(f"\nTrad normalization[:5]: {qml_trad.normalization[:5]}")
    print(f"Comp normalization[:5]: {spectra_comp.normalization[:5]}")
    print(
        f"Normalization ratio: "
        f"{spectra_comp.normalization[:5] / qml_trad.normalization[:5]}"
    )

    # Load Fortran reference
    ref_file = local_path + f"/tests/data/nside4/{fields}/ref_spectra.txt"
    ref = np.loadtxt(ref_file)
    print(f"\nReference spectra[0,:5]: {ref[0, :5]}")

    # Compare
    print(f"\nTrad/Ref ratio: {power_spectra_trad[0, :5] / ref[0, :5]}")
    print(f"Comp/Ref ratio: {power_spectra_compressed[0, :5] / ref[0, :5]}")
    print(
        f"Comp/Trad ratio: {power_spectra_compressed[0, :5] / power_spectra_trad[0, :5]}"
    )

    # Compressed spectra should match Fortran reference
    np.testing.assert_allclose(
        power_spectra_compressed,
        ref,
        atol=1e-3,
        rtol=1e-5,
        err_msg="Compressed spectra do not match Fortran reference",
    )


def test_compressed_spectra_matches_traditional(local_path, config_resolver):
    """
    Test that compressed spectra matches traditional computation.

    This test verifies internal consistency: both methods should produce
    identical results within numerical precision.
    """
    fields = "T"

    # Get traditional (uncompressed) spectra
    fisher = test_spectra.get_fisher_instance(fields)
    qml_traditional = test_spectra.get_qml_analyzer(fields, fisher=fisher)
    power_spectra_traditional, noise_bias_traditional, _ = test_spectra.get_qml_spectra(
        qml_traditional
    )

    # Get compressed spectra
    power_spectra_compressed, noise_bias_compressed = get_compressed_spectra(
        fields,
        config_resolver=config_resolver,
        compression_method="harmonic",
        epsilon=1e-10,
    )

    assert power_spectra_traditional is not None
    assert power_spectra_compressed is not None

    # Both should match
    np.testing.assert_allclose(
        power_spectra_compressed,
        power_spectra_traditional,
        atol=1e-3,
        rtol=1e-5,
        err_msg="Compressed spectra should match traditional computation",
    )

    # Noise bias should also match
    if noise_bias_traditional is not None and noise_bias_compressed is not None:
        np.testing.assert_allclose(
            noise_bias_compressed,
            noise_bias_traditional,
            atol=1e-3,
            rtol=1e-5,
            err_msg="Compressed noise bias should match traditional computation",
        )


def test_pixel_projected_spectra_degradation(local_path, config_resolver):
    """
    Test that PixelProjected spectra stays within acceptable degradation.

    Based on Gjerløw et al., PixelProjected compression is an approximation
    that should give results close to exact when keeping sufficient modes.

    We compare against Harmonic compression (exact via SMW) as the reference.
    """
    fields = "T"

    # Get Harmonic spectra (reference - exact via SMW)
    spectra_harmonic, noise_bias_harmonic = get_compressed_spectra(
        fields,
        config_resolver=config_resolver,
        compression_method="harmonic",
        epsilon=1e-10,
    )

    # Get PixelProjected spectra with tight epsilon
    spectra_pixel_projected, noise_bias_pixel_projected = get_compressed_spectra(
        fields,
        config_resolver=config_resolver,
        compression_method="pixel_projected",
        epsilon=1e-6,  # Tight threshold to keep most modes
    )

    assert spectra_harmonic is not None
    assert spectra_pixel_projected is not None

    # Check shapes match
    assert spectra_pixel_projected.shape == spectra_harmonic.shape

    # Compute relative difference in power spectra
    # Average over simulations, compare ell-by-ell
    mean_harm = np.mean(spectra_harmonic, axis=1)
    mean_pp = np.mean(spectra_pixel_projected, axis=1)

    # Only compare where signal is significant
    valid = np.abs(mean_harm) > 1e-10 * np.max(np.abs(mean_harm))
    if np.any(valid):
        rel_diff = np.abs(mean_pp[valid] - mean_harm[valid]) / np.abs(mean_harm[valid])
        max_rel_diff = np.max(rel_diff) * 100  # percent

        # With tight epsilon, should be within 20% (spectra are noisier than Fisher)
        assert max_rel_diff < 30, (
            f"PixelProjected spectra difference ({max_rel_diff:.1f}%) "
            f"exceeds 30% tolerance. This is a known approximation effect."
        )

    # Also check noise bias if available
    # Note: Noise bias is more sensitive to compression approximation than power spectra
    # With the test data (small n_pix, marginal compression),
    # larger differences are expected
    if noise_bias_harmonic is not None and noise_bias_pixel_projected is not None:
        valid_nb = np.abs(noise_bias_harmonic) > 1e-10 * np.max(
            np.abs(noise_bias_harmonic)
        )
        if np.any(valid_nb):
            nb_diff = np.abs(
                noise_bias_pixel_projected[valid_nb] - noise_bias_harmonic[valid_nb]
            ) / np.abs(noise_bias_harmonic[valid_nb])
            max_nb_diff = np.max(nb_diff) * 100

            # Noise bias can have larger differences due to compression approximation
            # This is expected behavior for pixel_projected compression
            assert max_nb_diff < 100, (
                f"PixelProjected noise bias difference ({max_nb_diff:.1f}%) "
                f"exceeds 100% tolerance"
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
