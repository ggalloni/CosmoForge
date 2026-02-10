import os
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from mpi4py import MPI

from qube import Fisher

# =============================================================================
# Helpers & caching
# =============================================================================

# Module-level cache to avoid redundant Fisher pipeline runs.
# Tests execute top-to-bottom, so early tests populate the cache for later ones.
_fisher_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _run_fisher(
    fields: str,
    config_resolver,
    nside: int = 4,
    config_type: str = "config",
    compression_method: str | None = None,
    epsilon: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a Fisher pipeline and return (fisher_matrix, error_bars)."""
    config_file = config_resolver(f"tests/data/nside{nside}/{fields}/{config_type}.yaml")
    kwargs = {}
    if compression_method is not None:
        kwargs["compression"] = {"method": compression_method, "epsilon": epsilon}
    fisher_analyzer = Fisher(config_file, **kwargs)
    fisher_analyzer.run()
    fisher_matrix = fisher_analyzer.get_fisher_matrix()
    error_bars = fisher_analyzer.get_error_bars()
    os.unlink(config_file)
    return fisher_matrix, error_bars


def _cached_traditional(
    fields: str, config_resolver, nside: int = 4, config_type: str = "config"
) -> tuple[np.ndarray, np.ndarray]:
    """Get traditional Fisher, computing only on first call per key."""
    key = f"trad_{nside}_{fields}_{config_type}"
    if key not in _fisher_cache:
        _fisher_cache[key] = _run_fisher(
            fields, config_resolver, nside=nside, config_type=config_type
        )
    return _fisher_cache[key]


def _cached_compressed(
    fields: str,
    config_resolver,
    nside: int = 4,
    method: str = "harmonic",
    epsilon: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Get compressed Fisher, computing only on first call per key."""
    key = f"comp_{nside}_{method}_{fields}_{epsilon}"
    if key not in _fisher_cache:
        _fisher_cache[key] = _run_fisher(
            fields,
            config_resolver,
            nside=nside,
            compression_method=method,
            epsilon=epsilon,
        )
    return _fisher_cache[key]


# =============================================================================
# Traditional (pixel-space) Fisher tests
# =============================================================================


@pytest.mark.parametrize("fields", ["T", "QU", "TQU", "TEB"])
def test_fisher_computation(fields, local_path, config_resolver):
    """Traditional Fisher vs Fortran reference for all field configurations."""
    fisher_matrix, error_bars = _cached_traditional(fields, config_resolver)
    assert fisher_matrix is not None
    assert error_bars is not None

    file = local_path + f"/tests/data/nside4/{fields}/ref_fisher.dat"
    ref = np.loadtxt(file, dtype=np.float64)

    assert fisher_matrix.shape == ref.shape
    np.testing.assert_allclose(
        fisher_matrix,
        ref,
        atol=1e-3,
        rtol=1e-5,
        err_msg=f"Fisher matrix for {fields} does not match reference.",
    )


@pytest.mark.parametrize("fields", ["QU"])
def test_cross_fisher_computation(fields, local_path, config_resolver):
    """Cross-correlation Fisher vs Fortran reference."""
    fisher_matrix, error_bars = _cached_traditional(
        fields, config_resolver, config_type="cross_config"
    )
    assert fisher_matrix is not None
    assert error_bars is not None

    file = local_path + f"/tests/data/nside4/{fields}/ref_fisher.dat"
    ref = np.loadtxt(file, dtype=np.float64)

    assert fisher_matrix.shape == ref.shape
    np.testing.assert_allclose(
        fisher_matrix,
        ref,
        atol=1e-3,
        rtol=1e-5,
        err_msg=f"Cross Fisher for {fields} does not match reference.",
    )


@patch("qube.fisher.MPI")
def test_fisher_worker_rank_behavior(mock_mpi, config_resolver):
    """Worker ranks (rank != 0) should return None for Fisher matrix and error bars."""
    from cosmocore import InputParams

    mock_comm = MagicMock()
    mock_comm.Get_rank.return_value = 1
    mock_comm.Get_size.return_value = 2
    mock_comm.Barrier.return_value = None
    mock_mpi.COMM_WORLD = mock_comm
    mock_mpi.LAND = MPI.LAND

    fields = "TEB"
    config_file = config_resolver(f"tests/data/nside4/{fields}/config.yaml")

    try:
        fisher_analyzer = Fisher(config_file)
        assert fisher_analyzer.rank == 1
        assert fisher_analyzer.size == 2

        real_params = InputParams.read_parameter_file(config_file)
        broadcast_call_count = 0

        def mock_bcast(data, root=0):
            nonlocal broadcast_call_count
            broadcast_call_count += 1
            if broadcast_call_count == 1:
                return real_params
            elif broadcast_call_count == 2:
                return None
            else:
                return None

        mock_comm.bcast.side_effect = mock_bcast
        mock_comm.allreduce.return_value = None

        assert fisher_analyzer.get_fisher_matrix() is None
        assert fisher_analyzer.get_error_bars() is None
    finally:
        os.unlink(config_file)


# =============================================================================
# Compressed harmonic Fisher — single-field T
# =============================================================================


def test_compressed_harmonic_fisher_T(local_path, config_resolver):
    """Compressed harmonic Fisher for T: shape, PSD, symmetry, and Fortran reference."""
    fisher_compressed, error_bars = _cached_compressed(
        "T",
        config_resolver,
        method="harmonic",
        epsilon=1e-10,
    )

    assert fisher_compressed is not None
    assert error_bars is not None

    file = local_path + "/tests/data/nside4/T/ref_fisher.dat"
    ref = np.loadtxt(file, dtype=np.float64)
    assert fisher_compressed.shape == ref.shape

    # Symmetry
    np.testing.assert_allclose(
        fisher_compressed,
        fisher_compressed.T,
        rtol=1e-10,
        err_msg="Compressed Fisher should be symmetric",
    )

    # Positive semi-definite
    eigenvalues = np.linalg.eigvalsh(fisher_compressed)
    assert np.all(eigenvalues > -1e-10), "Should be positive semi-definite"
    assert np.all(fisher_compressed.diagonal() > 0), "Diagonal should be positive"

    # Match Fortran reference
    np.testing.assert_allclose(
        fisher_compressed,
        ref,
        atol=1e-3,
        rtol=1e-5,
        err_msg="Compressed T Fisher does not match Fortran reference",
    )


# =============================================================================
# Pixel-projected Fisher degradation
# =============================================================================


def test_pixel_projected_fisher_degradation(local_path, config_resolver):
    """PixelProjected Fisher stays within acceptable degradation vs Harmonic."""
    # Reuses cached harmonic T from test_compressed_harmonic_fisher_T
    fisher_harmonic, _ = _cached_compressed(
        "T",
        config_resolver,
        method="harmonic",
        epsilon=1e-10,
    )
    fisher_pixel_projected, _ = _cached_compressed(
        "T",
        config_resolver,
        method="pixel_projected",
        epsilon=1e-6,
    )

    assert fisher_pixel_projected.shape == fisher_harmonic.shape

    # Symmetry + PSD
    np.testing.assert_allclose(
        fisher_pixel_projected,
        fisher_pixel_projected.T,
        rtol=1e-10,
    )
    eigenvalues = np.linalg.eigvalsh(fisher_pixel_projected)
    assert np.all(eigenvalues > -1e-10)

    # Error bar degradation < 15%
    fisher_harm_inv = np.linalg.pinv(fisher_harmonic)
    fisher_pp_inv = np.linalg.pinv(fisher_pixel_projected)
    sigma_harm = np.sqrt(np.maximum(np.diag(fisher_harm_inv), 0))
    sigma_pp = np.sqrt(np.maximum(np.diag(fisher_pp_inv), 0))

    valid = sigma_harm > 1e-15
    if np.any(valid):
        degradation = np.abs(sigma_pp[valid] - sigma_harm[valid]) / sigma_harm[valid]
        max_degradation = np.max(degradation) * 100
        assert max_degradation < 15, (
            f"PixelProjected error bar degradation ({max_degradation:.1f}%) exceeds 15%"
        )

    # Fisher diagonal values < 20%
    diag_harm = np.diag(fisher_harmonic)
    diag_pp = np.diag(fisher_pixel_projected)
    valid_diag = np.abs(diag_harm) > 1e-15
    if np.any(valid_diag):
        diag_diff = (
            np.abs(diag_pp[valid_diag] - diag_harm[valid_diag]) / diag_harm[valid_diag]
        )
        max_diag_diff = np.max(diag_diff) * 100
        assert max_diag_diff < 20, (
            f"PixelProjected Fisher diagonal difference "
            f"({max_diag_diff:.1f}%) exceeds 20%"
        )


# =============================================================================
# TEB multi-field compression (3 scalar fields)
# =============================================================================


def test_teb_compression_matches_traditional(local_path, config_resolver):
    """TEB (3 scalar fields) compressed Fisher matches traditional."""
    # Reuses cached traditional TEB from test_fisher_computation[TEB]
    fisher_traditional, _ = _cached_traditional("TEB", config_resolver)
    fisher_compressed, _ = _cached_compressed(
        "TEB",
        config_resolver,
        method="harmonic",
        epsilon=1e-10,
    )

    assert fisher_compressed.shape == fisher_traditional.shape

    # Symmetry + PSD
    np.testing.assert_allclose(
        fisher_compressed,
        fisher_compressed.T,
        rtol=1e-10,
        err_msg="Compressed Fisher should be symmetric",
    )
    eig_comp = np.linalg.eigvalsh(fisher_compressed)
    assert np.all(eig_comp > -1e-10)

    # Match traditional
    np.testing.assert_allclose(
        fisher_compressed,
        fisher_traditional,
        atol=1e-3,
        rtol=0.01,
        err_msg="Compressed TEB Fisher should match traditional within 1%",
    )


# =============================================================================
# Spin-2 compressed Fisher
# =============================================================================


@pytest.mark.parametrize("fields", ["QU", "TQU"])
def test_compressed_fisher_spin2(fields, local_path, config_resolver):
    """Compressed spin-2 Fisher matches traditional and reference."""
    # Reuses cached traditional from test_fisher_computation[QU/TQU]
    fisher_traditional, _ = _cached_traditional(fields, config_resolver)
    fisher_compressed, _ = _cached_compressed(
        fields,
        config_resolver,
        method="harmonic",
        epsilon=1e-10,
    )

    assert fisher_compressed.shape == fisher_traditional.shape

    # Symmetry + PSD
    np.testing.assert_allclose(
        fisher_compressed,
        fisher_compressed.T,
        rtol=1e-10,
        err_msg=f"Compressed {fields} Fisher should be symmetric",
    )
    eig_comp = np.linalg.eigvalsh(fisher_compressed)
    assert np.all(eig_comp > -1e-10), (
        f"Compressed {fields} Fisher should be positive semi-definite"
    )

    # Match traditional (diagonal-normalized metric)
    diag_t = np.abs(np.diag(fisher_traditional))
    scale = np.sqrt(np.outer(diag_t, diag_t))
    norm_diff = np.abs(fisher_compressed - fisher_traditional) / (scale + 1e-30)
    assert norm_diff.max() < 1e-6, (
        f"Compressed {fields} Fisher diagonal-normalized diff {norm_diff.max():.2e} "
        f"exceeds 1e-6"
    )

    # Error bars vs traditional
    sigma_c = np.sqrt(np.diag(np.linalg.inv(fisher_compressed)))
    sigma_t = np.sqrt(np.diag(np.linalg.inv(fisher_traditional)))
    max_sigma_diff = np.max(np.abs(sigma_c - sigma_t) / sigma_t)
    assert max_sigma_diff < 1e-6, (
        f"Compressed {fields} error bars differ by {max_sigma_diff:.2e}"
    )

    # Match Fortran reference
    file = local_path + f"/tests/data/nside4/{fields}/ref_fisher.dat"
    ref = np.loadtxt(file, dtype=np.float64)
    assert fisher_compressed.shape == ref.shape

    diag_r = np.abs(np.diag(ref))
    scale_r = np.sqrt(np.outer(diag_r, diag_r))
    norm_diff_r = np.abs(fisher_compressed - ref) / (scale_r + 1e-30)
    assert norm_diff_r.max() < 1e-6, (
        f"Compressed {fields} Fisher vs reference diff "
        f"{norm_diff_r.max():.2e} exceeds 1e-6"
    )

    sigma_r = np.sqrt(np.diag(np.linalg.inv(ref)))
    max_sigma_diff_r = np.max(np.abs(sigma_c - sigma_r) / sigma_r)
    assert max_sigma_diff_r < 1e-6, (
        f"Compressed {fields} error bars vs reference differ by {max_sigma_diff_r:.2e}"
    )


# =============================================================================
# nside=8 tests — marked slow
# =============================================================================


@pytest.mark.slow
def test_teb_nside8_compression_vs_reference(local_path, config_resolver):
    """
    Compressed TEB nside=8 vs Fortran reference (skips traditional computation).

    Accuracy is validated at nside=4 via traditional comparisons above.
    This test validates that compression works at a larger problem size.
    """
    fields = "TEB"

    t0 = time.perf_counter()
    fisher_compressed, _ = _run_fisher(
        fields,
        config_resolver,
        nside=8,
        compression_method="harmonic",
        epsilon=1e-10,
    )
    elapsed = time.perf_counter() - t0
    print(f"\n=== TEB nside=8 Compressed: {elapsed:.1f} s ===")

    # Symmetry + PSD
    np.testing.assert_allclose(
        fisher_compressed,
        fisher_compressed.T,
        rtol=1e-10,
        err_msg="Compressed Fisher should be symmetric",
    )
    eig = np.linalg.eigvalsh(fisher_compressed)
    assert np.all(eig > -1e-10), "Should be positive semi-definite"

    # Match Fortran reference
    ref_file = local_path + f"/tests/data/nside8/{fields}/{fields}_ref_fisher.dat"
    ref = np.loadtxt(ref_file, dtype=np.float64)
    assert fisher_compressed.shape == ref.shape

    np.testing.assert_allclose(
        fisher_compressed,
        ref,
        atol=1e-2,
        rtol=0.01,
        err_msg="Compressed nside8 TEB Fisher should match Fortran reference",
    )


if __name__ == "__main__":
    fields_list = ["T", "QU", "TQU", "TEB"]

    path = os.path.abspath(__file__.split("/tests/test_fisher.py")[0])

    print(f"Running tests in directory: {path}")

    for fields in fields_list:
        print(f"Testing Fisher matrix computation for fields: {fields}")
        test_fisher_computation(fields, local_path=path)
