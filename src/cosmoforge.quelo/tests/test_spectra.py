import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from mpi4py import MPI

from quelo import Fisher, Spectra

# =============================================================================
# Helpers & caching
# =============================================================================

# Module-level caches to avoid redundant pipeline runs.
_fisher_cache: dict[str, Fisher] = {}
_spectra_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_compressed_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
_analyzer_cache: dict[str, Spectra] = {}


def _get_fisher(fields: str, config_resolver, config_type: str = "config") -> Fisher:
    """Get or create a cached Fisher instance."""
    key = f"{fields}_{config_type}"
    if key not in _fisher_cache:
        config_file = config_resolver(f"tests/data/nside4/{fields}/{config_type}.yaml")
        fisher = Fisher(config_file)
        fisher.run()
        os.unlink(config_file)
        _fisher_cache[key] = fisher
    return _fisher_cache[key]


def _get_spectra(
    fields: str, config_resolver, config_type: str = "config"
) -> tuple[np.ndarray, np.ndarray]:
    """Get or create cached traditional spectra results."""
    key = f"{fields}_{config_type}"
    if key not in _spectra_cache:
        fisher = _get_fisher(fields, config_resolver, config_type)
        config_file = config_resolver(f"tests/data/nside4/{fields}/{config_type}.yaml")
        qml = Spectra(config_file, fisher=fisher)
        qml.run()
        os.unlink(config_file)
        _spectra_cache[key] = (qml.get_power_spectra(), qml.get_noise_bias())
    return _spectra_cache[key]


def _get_qml_analyzer(fields: str, config_resolver) -> Spectra:
    """Get or create a cached Spectra (QML analyzer) instance."""
    key = fields
    if key not in _analyzer_cache:
        fisher = _get_fisher(fields, config_resolver)
        config_file = config_resolver(f"tests/data/nside4/{fields}/config.yaml")
        qml = Spectra(config_file, fisher=fisher)
        qml.run()
        os.unlink(config_file)
        _analyzer_cache[key] = qml
    return _analyzer_cache[key]


def _get_compressed_spectra(
    fields: str,
    config_resolver,
    method: str = "harmonic",
    epsilon: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Get or create cached compressed spectra results."""
    key = f"{fields}_{method}_{epsilon}"
    if key not in _compressed_cache:
        config_file = config_resolver(f"tests/data/nside4/{fields}/config.yaml")
        qml = Spectra(
            config_file,
            compression={"method": method, "epsilon": epsilon},
        )
        qml.run()
        os.unlink(config_file)
        _compressed_cache[key] = (qml.get_power_spectra(), qml.get_noise_bias())
    return _compressed_cache[key]


# =============================================================================
# Traditional (pixel-space) Spectra tests
# =============================================================================


@pytest.mark.parametrize("fields", ["T", "QU", "TQU", "TEB"])
def test_spectra_computation(fields, local_path, config_resolver):
    """Traditional QML spectra vs Fortran reference for all field configurations."""
    power_spectra, noise_bias = _get_spectra(fields, config_resolver)

    ref_file = local_path + f"/tests/data/nside4/{fields}/ref_spectra.txt"
    ref = np.loadtxt(ref_file)

    np.testing.assert_allclose(
        power_spectra,
        ref,
        atol=1e-3,
        rtol=1e-5,
        err_msg=f"Power spectra for {fields} do not match reference",
    )


@pytest.mark.parametrize("fields", ["QU"])
def test_cross_spectra_computation(fields, local_path, config_resolver):
    """Cross-correlation QML spectra vs Fortran reference."""
    power_spectra, noise_bias = _get_spectra(
        fields, config_resolver, config_type="cross_config"
    )

    ref_file = local_path + f"/tests/data/nside4/{fields}/ref_spectra.txt"
    ref = np.loadtxt(ref_file)

    np.testing.assert_allclose(
        power_spectra,
        ref,
        atol=1e-3,
        rtol=1e-5,
        err_msg=f"Cross spectra for {fields} do not match reference",
    )


@patch("quelo.spectra.MPI")
def test_spectra_worker_rank_behavior(mock_mpi, config_resolver):
    """Worker ranks (rank != 0) should return None for spectra and noise bias."""
    mock_comm = MagicMock()
    mock_comm.Get_rank.return_value = 1
    mock_comm.Get_size.return_value = 2
    mock_comm.Barrier.return_value = None
    mock_mpi.COMM_WORLD = mock_comm
    mock_mpi.LAND = MPI.LAND

    config_file = config_resolver("tests/data/nside4/T/config.yaml")

    try:
        spectra_analyzer = Spectra(config_file)
        assert spectra_analyzer.rank == 1
        assert spectra_analyzer.size == 2

        assert spectra_analyzer.get_power_spectra() is None
        assert spectra_analyzer.get_noise_bias() is None
    finally:
        os.unlink(config_file)


# =============================================================================
# Compressed Spectra — single-field T
# (Merges: test_compressed_spectra_matches_reference +
#  test_compressed_spectra_matches_traditional)
# =============================================================================


def test_compressed_spectra_T(local_path, config_resolver):
    """Compressed harmonic T spectra matches both traditional and Fortran reference."""
    # Reuses cached traditional T from test_spectra_computation[T]
    power_trad, noise_trad = _get_spectra("T", config_resolver)
    power_comp, noise_comp = _get_compressed_spectra(
        "T",
        config_resolver,
        method="harmonic",
        epsilon=1e-10,
    )

    assert power_comp is not None

    # Match traditional
    np.testing.assert_allclose(
        power_comp,
        power_trad,
        atol=1e-3,
        rtol=1e-5,
        err_msg="Compressed T spectra should match traditional",
    )

    # Match noise bias
    if noise_trad is not None and noise_comp is not None:
        np.testing.assert_allclose(
            noise_comp,
            noise_trad,
            atol=1e-3,
            rtol=1e-5,
            err_msg="Compressed T noise bias should match traditional",
        )

    # Match Fortran reference
    ref_file = local_path + "/tests/data/nside4/T/ref_spectra.txt"
    ref = np.loadtxt(ref_file)
    np.testing.assert_allclose(
        power_comp,
        ref,
        atol=1e-3,
        rtol=1e-5,
        err_msg="Compressed T spectra do not match Fortran reference",
    )


# =============================================================================
# TEB multi-field compressed spectra
# =============================================================================


def test_teb_compressed_spectra_matches_traditional(local_path, config_resolver):
    """TEB (3 scalar fields) compressed spectra matches traditional."""
    # Reuses cached traditional TEB from test_spectra_computation[TEB]
    power_trad, _ = _get_spectra("TEB", config_resolver)
    power_comp, _ = _get_compressed_spectra(
        "TEB",
        config_resolver,
        method="harmonic",
        epsilon=1e-10,
    )

    assert power_comp is not None

    # Correlation check
    corr = np.corrcoef(power_trad.flatten(), power_comp.flatten())[0, 1]
    assert corr > 0.999, f"Correlation should be > 0.999, got {corr}"

    # Relative Frobenius norm
    rel_fro = np.linalg.norm(power_comp - power_trad, "fro") / np.linalg.norm(
        power_trad, "fro"
    )
    assert rel_fro < 0.05, f"Relative Frobenius should be < 5%, got {rel_fro:.2%}"


# =============================================================================
# Pixel-projected spectra degradation
# =============================================================================


def test_pixel_projected_spectra_degradation(local_path, config_resolver):
    """PixelProjected spectra stays within acceptable degradation vs Harmonic."""
    # Reuses cached harmonic T from test_compressed_spectra_T
    spectra_harmonic, noise_harmonic = _get_compressed_spectra(
        "T",
        config_resolver,
        method="harmonic",
        epsilon=1e-10,
    )
    spectra_pp, noise_pp = _get_compressed_spectra(
        "T",
        config_resolver,
        method="pixel_projected",
        epsilon=1e-6,
    )

    assert spectra_pp is not None
    assert spectra_pp.shape == spectra_harmonic.shape

    # Mean spectra comparison
    mean_harm = np.mean(spectra_harmonic, axis=1)
    mean_pp = np.mean(spectra_pp, axis=1)

    valid = np.abs(mean_harm) > 1e-10 * np.max(np.abs(mean_harm))
    if np.any(valid):
        rel_diff = np.abs(mean_pp[valid] - mean_harm[valid]) / np.abs(mean_harm[valid])
        max_rel_diff = np.max(rel_diff) * 100
        assert max_rel_diff < 30, (
            f"PixelProjected spectra difference ({max_rel_diff:.1f}%) exceeds 30%"
        )

    # Noise bias comparison (more lenient)
    if noise_harmonic is not None and noise_pp is not None:
        valid_nb = np.abs(noise_harmonic) > 1e-10 * np.max(np.abs(noise_harmonic))
        if np.any(valid_nb):
            nb_diff = np.abs(noise_pp[valid_nb] - noise_harmonic[valid_nb]) / np.abs(
                noise_harmonic[valid_nb]
            )
            assert np.max(nb_diff) * 100 < 100, (
                f"PixelProjected noise bias diff "
                f"({np.max(nb_diff) * 100:.1f}%) exceeds 100%"
            )


@pytest.mark.parametrize("fields", ["T", "QU"])
def test_normalization_modes(fields, local_path, config_resolver):
    """Test the three normalization modes for QML power spectra."""
    qml_analyzer = _get_qml_analyzer(fields, config_resolver)

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
    qml_analyzer = _get_qml_analyzer(fields, config_resolver)

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
    qml_analyzer = _get_qml_analyzer(fields, config_resolver)

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
    qml_analyzer = _get_qml_analyzer(fields, config_resolver)

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


@pytest.mark.parametrize("fields", ["T"])
def test_write_power_spectra_deconvolved(fields, local_path, config_resolver, tmp_path):
    """Test write_power_spectra for deconvolved mode."""
    qml_analyzer = _get_qml_analyzer(fields, config_resolver)

    # Test with custom filename
    output_file = tmp_path / "test_deconvolved.dat"
    qml_analyzer.write_power_spectra(mode="deconvolved", filename=str(output_file))

    assert output_file.exists(), "Deconvolved spectra file should be created"

    # Check error file was also created
    error_file = tmp_path / "test_deconvolved_errors.dat"
    assert error_file.exists(), "Error bars file should be created"

    # Verify file contents are valid
    spectra_data = np.loadtxt(output_file)
    assert spectra_data.size > 0, "Spectra file should contain data"

    error_data = np.loadtxt(error_file)
    assert error_data.size > 0, "Error file should contain data"


@pytest.mark.parametrize("fields", ["T"])
def test_write_power_spectra_no_errors(fields, local_path, config_resolver, tmp_path):
    """Test write_power_spectra with include_errors=False."""
    qml_analyzer = _get_qml_analyzer(fields, config_resolver)

    output_file = tmp_path / "test_no_errors.dat"
    qml_analyzer.write_power_spectra(
        mode="deconvolved", filename=str(output_file), include_errors=False
    )

    assert output_file.exists(), "Spectra file should be created"

    # Error file should NOT be created
    error_file = tmp_path / "test_no_errors_errors.dat"
    assert not error_file.exists(), "Error file should not be created when disabled"


@pytest.mark.parametrize("fields", ["T"])
def test_write_power_spectra_decorrelated(fields, local_path, config_resolver, tmp_path):
    """Test write_power_spectra for decorrelated mode."""
    qml_analyzer = _get_qml_analyzer(fields, config_resolver)

    output_file = tmp_path / "test_decorrelated.dat"
    qml_analyzer.write_power_spectra(mode="decorrelated", filename=str(output_file))

    assert output_file.exists(), "Decorrelated spectra file should be created"

    spectra_data = np.loadtxt(output_file)
    assert spectra_data.size > 0, "Spectra file should contain data"


@pytest.mark.parametrize("fields", ["T"])
def test_write_power_spectra_convolved(fields, local_path, config_resolver, tmp_path):
    """Test write_power_spectra for convolved mode."""
    qml_analyzer = _get_qml_analyzer(fields, config_resolver)

    output_file = tmp_path / "test_convolved.dat"
    qml_analyzer.write_power_spectra(mode="convolved", filename=str(output_file))

    # Check y vector file
    assert output_file.exists(), "Convolved y vector file should be created"
    y_data = np.loadtxt(output_file)
    assert y_data.size > 0, "Y vector file should contain data"

    # Check window matrix file
    window_file = tmp_path / "test_convolved_window.dat"
    assert window_file.exists(), "Window matrix file should be created"
    W_data = np.loadtxt(window_file)
    assert W_data.ndim == 2, "Window matrix should be 2D"
    assert W_data.shape[0] == W_data.shape[1], "Window matrix should be square"


@pytest.mark.parametrize("fields", ["T"])
def test_write_power_spectra_auto_filename(fields, local_path, config_resolver, tmp_path):
    """Test write_power_spectra auto-generates filename based on mode."""
    qml_analyzer = _get_qml_analyzer(fields, config_resolver)

    # Set outclfile to use tmp_path (not a default param, so we add it)
    qml_analyzer.params.outclfile = str(tmp_path / "output_spectra.dat")

    qml_analyzer.write_power_spectra(mode="deconvolved")
    expected_file = tmp_path / "output_spectra_deconvolved.dat"
    assert expected_file.exists(), "Auto-generated filename should work"


@pytest.mark.parametrize("fields", ["T"])
def test_write_power_spectra_fallback_filename(
    fields, local_path, config_resolver, tmp_path, monkeypatch
):
    """Test write_power_spectra falls back to default filename when outclfile not set."""
    qml_analyzer = _get_qml_analyzer(fields, config_resolver)

    # Ensure outclfile is not set (use delattr if it exists)
    if hasattr(qml_analyzer.params, "outclfile"):
        delattr(qml_analyzer.params, "outclfile")

    # Change to tmp_path so file is created there
    monkeypatch.chdir(tmp_path)

    qml_analyzer.write_power_spectra(mode="deconvolved")
    expected_file = tmp_path / "spectra_deconvolved.dat"
    assert expected_file.exists(), "Fallback filename should be spectra_{mode}.dat"


@pytest.mark.parametrize("fields", ["T"])
def test_write_power_spectra_invalid_mode(fields, local_path, config_resolver, tmp_path):
    """Test write_power_spectra raises error for invalid mode."""
    qml_analyzer = _get_qml_analyzer(fields, config_resolver)

    output_file = tmp_path / "test_invalid.dat"
    with pytest.raises(ValueError, match="mode must be one of"):
        qml_analyzer.write_power_spectra(mode="invalid_mode", filename=str(output_file))


# =============================================================================
# Spin-2 compressed spectra
# (Merges: test_compressed_spectra_spin2_matches_traditional +
#  test_compressed_spectra_spin2_matches_reference)
# =============================================================================


@pytest.mark.parametrize("fields", ["QU", "TQU"])
def test_compressed_spectra_spin2(fields, local_path, config_resolver):
    """Compressed harmonic spin-2 spectra matches traditional and Fortran reference."""
    # Reuses cached traditional from test_spectra_computation[QU/TQU]
    power_trad, _ = _get_spectra(fields, config_resolver)
    power_comp, _ = _get_compressed_spectra(
        fields,
        config_resolver,
        method="harmonic",
        epsilon=1e-10,
    )

    assert power_comp is not None
    assert power_comp.shape == power_trad.shape

    # Match traditional (correlation + Frobenius)
    corr = np.corrcoef(power_trad.flatten(), power_comp.flatten())[0, 1]
    assert corr > 0.999, f"Correlation for {fields} should be > 0.999, got {corr}"

    rel_fro = np.linalg.norm(power_comp - power_trad, "fro") / np.linalg.norm(
        power_trad, "fro"
    )
    assert rel_fro < 0.05, (
        f"Relative Frobenius for {fields} should be < 5%, got {rel_fro:.2%}"
    )

    # Match Fortran reference
    ref_file = local_path + f"/tests/data/nside4/{fields}/ref_spectra.txt"
    ref = np.loadtxt(ref_file)
    assert power_comp.shape == ref.shape

    corr_ref = np.corrcoef(power_comp.flatten(), ref.flatten())[0, 1]
    assert corr_ref > 0.999, (
        f"Correlation for {fields} vs reference should be > 0.999, got {corr_ref}"
    )

    rel_fro_ref = np.linalg.norm(power_comp - ref, "fro") / np.linalg.norm(ref, "fro")
    assert rel_fro_ref < 0.05, (
        f"Relative Frobenius for {fields} vs reference should be < 5%, "
        f"got {rel_fro_ref:.2%}"
    )
