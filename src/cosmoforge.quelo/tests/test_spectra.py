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
