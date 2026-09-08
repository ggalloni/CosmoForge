"""Tests for the QUBE memory budget calculator.

Calibration target: eclipse-QU on commit d11ab0b, log
mem_eclipse_qu_20114986.out. Inputs: n_pix=59136, n_modes=33274,
lmax_signal=256, lmax=128 (switch active). Measured at basis_setup exit:
86,111 MiB total RSS minus 26,920 MiB baseline = 57.80 GiB of persistent
state. Measured at basis_setup peak: 142,816 MiB minus baseline = 113.2
GiB total transient + persistent.
"""

import pytest

from qube.memory_budget import (
    GIBIBYTE,
    BudgetConfig,
    PixelDirectBudgetConfig,
    StageBudget,
    predict_pixel_direct_budget,
    predict_qube_budget,
)


def test_persistent_state_uses_full_pixel_and_mode_squares():
    cfg = BudgetConfig(n_pix=1000, n_modes=500, lmax_signal=64)
    budget = predict_qube_budget(cfg)

    pix_sq = 1000 * 1000 * 8
    mode_sq = 500 * 500 * 8
    mode_pix = 500 * 1000 * 8

    cov = budget.stage("covariance_setup")
    assert cov.persistent["Cov_T"] == pix_sq

    basis = budget.stage("basis_setup")
    assert "Cov_T" not in basis.persistent, "Cov_T released to basis at handoff"
    assert basis.persistent["L (Cholesky factor of N, in-place)"] == pix_sq
    assert basis.persistent["V_N_inv"] == mode_pix
    assert basis.persistent["V_Ninv_VT (M kernel)"] == mode_sq


def test_release_pixel_projector_drops_v_from_persistent():
    cfg_release = BudgetConfig(
        n_pix=1000, n_modes=500, lmax_signal=64, release_pixel_projector=True
    )
    cfg_keep = BudgetConfig(
        n_pix=1000, n_modes=500, lmax_signal=64, release_pixel_projector=False
    )

    persistent_release = predict_qube_budget(cfg_release).stage("basis_setup").persistent
    persistent_keep = predict_qube_budget(cfg_keep).stage("basis_setup").persistent

    assert "V (pixel projector)" not in persistent_release
    assert persistent_keep["V (pixel projector)"] == 500 * 1000 * 8


def test_switch_adds_independent_noise_bias_kernel_t():
    no_switch = BudgetConfig(n_pix=1000, n_modes=500, lmax_signal=64, lmax=64)
    with_switch = BudgetConfig(n_pix=1000, n_modes=500, lmax_signal=64, lmax=32)

    no_t = predict_qube_budget(no_switch).stage("basis_setup").persistent
    with_t = predict_qube_budget(with_switch).stage("basis_setup").persistent

    assert "T (noise-bias kernel, switch path)" not in no_t
    assert with_t["T (noise-bias kernel, switch path)"] == 500 * 500 * 8


def test_switch_adds_s_fixed_and_corr_intermediate_to_basis_transient():
    no_switch = BudgetConfig(n_pix=1000, n_modes=500, lmax_signal=64, lmax=64)
    with_switch = BudgetConfig(n_pix=1000, n_modes=500, lmax_signal=64, lmax=32)

    no_t = predict_qube_budget(no_switch).stage("basis_setup").transient
    with_t = predict_qube_budget(with_switch).stage("basis_setup").transient

    assert "S_fixed (switch optimisation)" not in no_t
    assert "corr intermediate (V_N_inv @ S_fixed)" not in no_t
    assert with_t["S_fixed (switch optimisation)"] == 1000 * 1000 * 8
    assert with_t["corr intermediate (V_N_inv @ S_fixed)"] == 500 * 1000 * 8


def test_covariance_setup_transient_is_one_pix_square_for_post_revert():
    cfg = BudgetConfig(n_pix=1000, n_modes=500, lmax_signal=64)
    cov = predict_qube_budget(cfg).stage("covariance_setup")
    assert cov.transient["Cov_T (asfortranarray copy on read)"] == 1000 * 1000 * 8


def test_spectra_run_adds_noise_cov_t():
    cfg = BudgetConfig(n_pix=1000, n_modes=500, lmax_signal=64)
    budget = predict_qube_budget(cfg)

    basis_persistent = budget.stage("basis_setup").persistent
    spectra_persistent = budget.stage("spectra_run").persistent

    extra = set(spectra_persistent) - set(basis_persistent)
    assert extra == {"noise_cov_T (Spectra)"}
    assert spectra_persistent["noise_cov_T (Spectra)"] == 500 * 500 * 8


def test_eclipse_qu_basis_persistent_within_2_percent_of_d11ab0b_log():
    cfg = BudgetConfig(n_pix=59136, n_modes=33274, lmax_signal=256, lmax=128)
    budget = predict_qube_budget(cfg)

    measured_gib = (86_111 - 26_920) / 1024  # mem_eclipse_qu_20114986.out
    predicted_gib = budget.stage("basis_setup").persistent_bytes / GIBIBYTE
    relative_error = abs(predicted_gib - measured_gib) / measured_gib
    assert relative_error < 0.02, (
        f"prediction {predicted_gib:.2f} GiB drifted from measured "
        f"{measured_gib:.2f} GiB by {relative_error:.1%}"
    )


def test_eclipse_qu_basis_peak_within_2_percent_of_d11ab0b_log():
    cfg = BudgetConfig(n_pix=59136, n_modes=33274, lmax_signal=256, lmax=128)
    budget = predict_qube_budget(cfg)

    measured_gib = (142_816 - 26_920) / 1024  # peak above baseline
    predicted_gib = budget.stage("basis_setup").peak_bytes / GIBIBYTE
    relative_error = abs(predicted_gib - measured_gib) / measured_gib
    assert relative_error < 0.02, (
        f"prediction {predicted_gib:.2f} GiB drifted from measured "
        f"{measured_gib:.2f} GiB by {relative_error:.1%}"
    )


def test_lifetime_peak_is_max_across_stages():
    cfg = BudgetConfig(n_pix=1000, n_modes=500, lmax_signal=64, lmax=32)
    budget = predict_qube_budget(cfg)
    assert budget.lifetime_peak_bytes == max(s.peak_bytes for s in budget.stages)


def test_format_table_lists_every_stage_with_a_total():
    cfg = BudgetConfig(n_pix=1000, n_modes=500, lmax_signal=64, lmax=32)
    table = predict_qube_budget(cfg).format_table()
    for stage_name in ("covariance_setup", "basis_setup", "fisher_run", "spectra_run"):
        assert stage_name in table
    assert "lifetime peak" in table.lower()


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        BudgetConfig(n_pix=0, n_modes=500, lmax_signal=64)
    with pytest.raises(ValueError):
        BudgetConfig(n_pix=1000, n_modes=0, lmax_signal=64)
    with pytest.raises(ValueError):
        BudgetConfig(n_pix=1000, n_modes=500, lmax_signal=0)
    with pytest.raises(ValueError):
        BudgetConfig(n_pix=1000, n_modes=500, lmax_signal=64, lmax=128)


def test_stage_budget_peak_is_persistent_plus_sum_transient():
    sb = StageBudget(
        name="x",
        persistent={"a": 100, "b": 50},
        transient={"t1": 30, "t2": 20},
    )
    assert sb.persistent_bytes == 150
    assert sb.peak_bytes == 200


def test_qube_budget_stage_lookup_raises_on_unknown():
    cfg = BudgetConfig(n_pix=1000, n_modes=500, lmax_signal=64)
    budget = predict_qube_budget(cfg)
    with pytest.raises(KeyError):
        budget.stage("not_a_stage")


# -- Pixel-direct path -------------------------------------------------------


def test_pixel_direct_basis_setup_carries_two_pix_squares():
    """Two n_pix² buffers, never three: Core resolves the pixel-direct path
    before the S_fixed branch, so the fixed-multipole matrix is never built."""
    cfg = PixelDirectBudgetConfig(n_pix=1000, lmax_signal=64, n_bins=6, n_params=18)
    basis = predict_pixel_direct_budget(cfg).stage("basis_setup")

    pix_sq = 1000 * 1000 * 8
    assert basis.persistent["Cov_T (carried from covariance_setup)"] == pix_sq
    assert basis.persistent["basis._N (asfortranarray F-order copy)"] == pix_sq
    assert basis.persistent_bytes == 2 * pix_sq


def test_pixel_direct_fisher_run_scales_transient_with_n_params():
    cfg = PixelDirectBudgetConfig(n_pix=1000, lmax_signal=64, n_bins=6, n_params=18)
    fisher = predict_pixel_direct_budget(cfg).stage("fisher_run")
    pix_sq = 1000 * 1000 * 8
    key = "cinv_times_dcb (n_params dense pixel matrices)"
    assert fisher.transient[key] == 18 * pix_sq


def test_pixel_direct_qu_nside64_fsky010_basis_persistent_one_pix_sq():
    """basis_setup adds exactly one n_pix² buffer over covariance_setup.

    The old calibration run mem_nc_20103507.out (ccabffd, QU_nside64_lmax128_
    fsky0.1) measured 4241.1 - 2767.7 = 1473.4 MiB ≈ 2 × pix_sq, matching the
    code of the time: the F-order copy plus the S_fixed buffer Core built
    before resolving the path. S_fixed is no longer built here, so only the
    structural F-order copy remains. Confirmed locally on
    benchmark_memory_isolated_fsky0p100_pixel_pt14 (QU, nside=32, n_pix=2400,
    pix_sq = 43.9 MiB): basis_setup delta +44.0 MB, covariance_setup delta
    +43.9 MB — one pix_sq each. Cluster re-measurement at nside=64 pending
    for the paper's Table C.2 absolute numbers.
    """
    cfg = PixelDirectBudgetConfig(n_pix=9800, lmax_signal=128, n_bins=6, n_params=18)
    budget = predict_pixel_direct_budget(cfg)
    basis = budget.stage("basis_setup")
    cov = budget.stage("covariance_setup")
    pix_sq = 9800 * 9800 * 8
    assert basis.persistent_bytes - cov.persistent_bytes == pix_sq


def test_pixel_direct_defaults_to_uncached_matching_fisher_default():
    cfg = PixelDirectBudgetConfig(n_pix=1000, lmax_signal=64, n_bins=6, n_params=18)
    assert cfg.cache_derivatives is False
    fisher = predict_pixel_direct_budget(cfg).stage("fisher_run")
    assert not any("derivative cache" in term for term in fisher.persistent)


def test_cache_derivatives_adds_n_params_pix_sq_from_fisher_through_spectra():
    """The cache is built in fisher.compute.derivative_cache and retained for
    Spectra, so it lands in persistent state on both stages, not transient."""
    kwargs = dict(n_pix=1000, lmax_signal=64, n_bins=6, n_params=18)
    off = predict_pixel_direct_budget(PixelDirectBudgetConfig(**kwargs))
    on = predict_pixel_direct_budget(
        PixelDirectBudgetConfig(**kwargs, cache_derivatives=True)
    )

    cache_bytes = 18 * 1000 * 1000 * 8
    for stage in ("fisher_run", "spectra_run"):
        assert (
            on.stage(stage).persistent_bytes - off.stage(stage).persistent_bytes
            == cache_bytes
        )
        assert on.stage(stage).transient == off.stage(stage).transient
    for stage in ("covariance_setup", "basis_setup"):
        assert on.stage(stage).peak_bytes == off.stage(stage).peak_bytes


@pytest.mark.parametrize(
    "name, n_pix, n_params, measured_gib",
    [
        ("QU_nside64", 9800, 18, 28.69),
        ("T_nside64", 4900, 6, 3.00),
    ],
)
def test_cached_fisher_peak_matches_isolated_g100_rerun(
    name, n_pix, n_params, measured_gib
):
    """Isolated single-cell g100 runs, fsky=0.1, pixel-direct, post-pt-14
    (2026-08-26 re-run notes, "Cached vs uncached, measured"). Measured is the
    fisher-stage peak RSS above that cell's own baseline. Tolerance covers
    allocator/BLAS slack; cells below n_pix ~ 2400 are baseline-dominated and
    are deliberately not pinned.
    """
    cfg = PixelDirectBudgetConfig(
        n_pix=n_pix,
        lmax_signal=128,
        n_bins=6,
        n_params=n_params,
        cache_derivatives=True,
    )
    predicted_gib = (
        predict_pixel_direct_budget(cfg).stage("fisher_run").peak_bytes / GIBIBYTE
    )
    relative_error = abs(predicted_gib - measured_gib) / measured_gib
    assert relative_error < 0.10, (
        f"{name}: prediction {predicted_gib:.2f} GiB drifted from measured "
        f"{measured_gib:.2f} GiB by {relative_error:.1%}"
    )


def test_pixel_direct_invalid_config_rejected():
    with pytest.raises(ValueError):
        PixelDirectBudgetConfig(n_pix=0, lmax_signal=64, n_bins=6, n_params=18)
    with pytest.raises(ValueError):
        PixelDirectBudgetConfig(n_pix=1000, lmax_signal=0, n_bins=6, n_params=18)
    with pytest.raises(ValueError):
        PixelDirectBudgetConfig(n_pix=1000, lmax_signal=64, n_bins=0, n_params=18)
    with pytest.raises(ValueError):
        PixelDirectBudgetConfig(n_pix=1000, lmax_signal=64, n_bins=6, n_params=0)


def test_pixel_direct_format_table_lists_all_stages():
    cfg = PixelDirectBudgetConfig(n_pix=1000, lmax_signal=64, n_bins=6, n_params=18)
    table = predict_pixel_direct_budget(cfg).format_table()
    assert "[pixel_direct]" in table
    for stage in ("covariance_setup", "basis_setup", "fisher_run", "spectra_run"):
        assert stage in table
