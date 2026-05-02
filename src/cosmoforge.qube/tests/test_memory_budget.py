"""Tests for the QUBE memory budget calculator.

Calibration target: eclipse-QU on commit d11ab0b, log
mem_eclipse_qu_20114986.out. Inputs: n_pix=59136, n_modes=33274, basis_lmax
=256, lswitch_high=128 (switch active). Measured at basis_setup exit:
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
    cfg = BudgetConfig(n_pix=1000, n_modes=500, lmax=64)
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
        n_pix=1000, n_modes=500, lmax=64, release_pixel_projector=True
    )
    cfg_keep = BudgetConfig(
        n_pix=1000, n_modes=500, lmax=64, release_pixel_projector=False
    )

    persistent_release = predict_qube_budget(cfg_release).stage("basis_setup").persistent
    persistent_keep = predict_qube_budget(cfg_keep).stage("basis_setup").persistent

    assert "V (pixel projector)" not in persistent_release
    assert persistent_keep["V (pixel projector)"] == 500 * 1000 * 8


def test_switch_adds_independent_noise_bias_kernel_t():
    no_switch = BudgetConfig(n_pix=1000, n_modes=500, lmax=64, lswitch_high=64)
    with_switch = BudgetConfig(n_pix=1000, n_modes=500, lmax=64, lswitch_high=32)

    no_t = predict_qube_budget(no_switch).stage("basis_setup").persistent
    with_t = predict_qube_budget(with_switch).stage("basis_setup").persistent

    assert "T (noise-bias kernel, switch path)" not in no_t
    assert with_t["T (noise-bias kernel, switch path)"] == 500 * 500 * 8


def test_switch_adds_s_fixed_and_corr_intermediate_to_basis_transient():
    no_switch = BudgetConfig(n_pix=1000, n_modes=500, lmax=64, lswitch_high=64)
    with_switch = BudgetConfig(n_pix=1000, n_modes=500, lmax=64, lswitch_high=32)

    no_t = predict_qube_budget(no_switch).stage("basis_setup").transient
    with_t = predict_qube_budget(with_switch).stage("basis_setup").transient

    assert "S_fixed (switch optimisation)" not in no_t
    assert "corr intermediate (V_N_inv @ S_fixed)" not in no_t
    assert with_t["S_fixed (switch optimisation)"] == 1000 * 1000 * 8
    assert with_t["corr intermediate (V_N_inv @ S_fixed)"] == 500 * 1000 * 8


def test_covariance_setup_transient_is_one_pix_square_for_post_revert():
    cfg = BudgetConfig(n_pix=1000, n_modes=500, lmax=64)
    cov = predict_qube_budget(cfg).stage("covariance_setup")
    assert cov.transient["Cov_T (asfortranarray copy on read)"] == 1000 * 1000 * 8


def test_spectra_run_adds_noise_cov_t():
    cfg = BudgetConfig(n_pix=1000, n_modes=500, lmax=64)
    budget = predict_qube_budget(cfg)

    basis_persistent = budget.stage("basis_setup").persistent
    spectra_persistent = budget.stage("spectra_run").persistent

    extra = set(spectra_persistent) - set(basis_persistent)
    assert extra == {"noise_cov_T (Spectra)"}
    assert spectra_persistent["noise_cov_T (Spectra)"] == 500 * 500 * 8


def test_eclipse_qu_basis_persistent_within_2_percent_of_d11ab0b_log():
    cfg = BudgetConfig(n_pix=59136, n_modes=33274, lmax=256, lswitch_high=128)
    budget = predict_qube_budget(cfg)

    measured_gib = (86_111 - 26_920) / 1024  # mem_eclipse_qu_20114986.out
    predicted_gib = budget.stage("basis_setup").persistent_bytes / GIBIBYTE
    relative_error = abs(predicted_gib - measured_gib) / measured_gib
    assert relative_error < 0.02, (
        f"prediction {predicted_gib:.2f} GiB drifted from measured "
        f"{measured_gib:.2f} GiB by {relative_error:.1%}"
    )


def test_eclipse_qu_basis_peak_within_2_percent_of_d11ab0b_log():
    cfg = BudgetConfig(n_pix=59136, n_modes=33274, lmax=256, lswitch_high=128)
    budget = predict_qube_budget(cfg)

    measured_gib = (142_816 - 26_920) / 1024  # peak above baseline
    predicted_gib = budget.stage("basis_setup").peak_bytes / GIBIBYTE
    relative_error = abs(predicted_gib - measured_gib) / measured_gib
    assert relative_error < 0.02, (
        f"prediction {predicted_gib:.2f} GiB drifted from measured "
        f"{measured_gib:.2f} GiB by {relative_error:.1%}"
    )


def test_lifetime_peak_is_max_across_stages():
    cfg = BudgetConfig(n_pix=1000, n_modes=500, lmax=64, lswitch_high=32)
    budget = predict_qube_budget(cfg)
    assert budget.lifetime_peak_bytes == max(s.peak_bytes for s in budget.stages)


def test_format_table_lists_every_stage_with_a_total():
    cfg = BudgetConfig(n_pix=1000, n_modes=500, lmax=64, lswitch_high=32)
    table = predict_qube_budget(cfg).format_table()
    for stage_name in ("covariance_setup", "basis_setup", "fisher_run", "spectra_run"):
        assert stage_name in table
    assert "lifetime peak" in table.lower()


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        BudgetConfig(n_pix=0, n_modes=500, lmax=64)
    with pytest.raises(ValueError):
        BudgetConfig(n_pix=1000, n_modes=0, lmax=64)
    with pytest.raises(ValueError):
        BudgetConfig(n_pix=1000, n_modes=500, lmax=0)
    with pytest.raises(ValueError):
        BudgetConfig(n_pix=1000, n_modes=500, lmax=64, lswitch_high=128)


def test_stage_budget_peak_is_persistent_plus_sum_transient():
    sb = StageBudget(
        name="x",
        persistent={"a": 100, "b": 50},
        transient={"t1": 30, "t2": 20},
    )
    assert sb.persistent_bytes == 150
    assert sb.peak_bytes == 200


def test_qube_budget_stage_lookup_raises_on_unknown():
    cfg = BudgetConfig(n_pix=1000, n_modes=500, lmax=64)
    budget = predict_qube_budget(cfg)
    with pytest.raises(KeyError):
        budget.stage("not_a_stage")


# -- Pixel-direct path -------------------------------------------------------


def test_pixel_direct_basis_setup_carries_two_pix_squares_with_switch():
    cfg = PixelDirectBudgetConfig(n_pix=1000, lmax=64, n_bins=6, n_params=18)
    basis = predict_pixel_direct_budget(cfg).stage("basis_setup")

    pix_sq = 1000 * 1000 * 8
    assert basis.persistent["Cov_T (carried from covariance_setup)"] == pix_sq
    assert basis.persistent["basis._N (asfortranarray F-order copy)"] == pix_sq
    assert basis.persistent["S_fixed (allocator pool retained)"] == pix_sq


def test_pixel_direct_no_switch_drops_s_fixed_term():
    cfg = PixelDirectBudgetConfig(
        n_pix=1000, lmax=64, n_bins=6, n_params=18, has_switch=False
    )
    basis = predict_pixel_direct_budget(cfg).stage("basis_setup")
    assert "S_fixed (allocator pool retained)" not in basis.persistent


def test_pixel_direct_fisher_run_scales_transient_with_n_params():
    cfg = PixelDirectBudgetConfig(n_pix=1000, lmax=64, n_bins=6, n_params=18)
    fisher = predict_pixel_direct_budget(cfg).stage("fisher_run")
    pix_sq = 1000 * 1000 * 8
    key = "cinv_times_dcb (n_params dense pixel matrices)"
    assert fisher.transient[key] == 18 * pix_sq


def test_pixel_direct_qu_nside64_fsky010_basis_persistent_within_2_percent():
    """Calibration: mem_nc_20103507.out, ccabffd, QU_nside64_lmax128_fsky0.1.

    Measured basis_setup persistent above covariance_setup exit:
    4241.1 - 2767.7 = 1473.4 MiB. Predicted: 2 × pix_sq = 1465.6 MiB.
    """
    cfg = PixelDirectBudgetConfig(n_pix=9800, lmax=128, n_bins=6, n_params=18)
    budget = predict_pixel_direct_budget(cfg)
    basis = budget.stage("basis_setup")
    cov = budget.stage("covariance_setup")
    pix_sq = 9800 * 9800 * 8
    basis_above_cov_gib = (basis.persistent_bytes - cov.persistent_bytes) / GIBIBYTE
    measured_gib = (4241.1 - 2767.7) / 1024
    relative_error = abs(basis_above_cov_gib - measured_gib) / measured_gib
    assert relative_error < 0.02, (
        f"prediction {basis_above_cov_gib:.2f} GiB drifted from measured "
        f"{measured_gib:.2f} GiB by {relative_error:.1%}"
    )
    assert basis.persistent_bytes - cov.persistent_bytes == 2 * pix_sq


def test_pixel_direct_invalid_config_rejected():
    with pytest.raises(ValueError):
        PixelDirectBudgetConfig(n_pix=0, lmax=64, n_bins=6, n_params=18)
    with pytest.raises(ValueError):
        PixelDirectBudgetConfig(n_pix=1000, lmax=0, n_bins=6, n_params=18)
    with pytest.raises(ValueError):
        PixelDirectBudgetConfig(n_pix=1000, lmax=64, n_bins=0, n_params=18)
    with pytest.raises(ValueError):
        PixelDirectBudgetConfig(n_pix=1000, lmax=64, n_bins=6, n_params=0)


def test_pixel_direct_format_table_lists_all_stages():
    cfg = PixelDirectBudgetConfig(n_pix=1000, lmax=64, n_bins=6, n_params=18)
    table = predict_pixel_direct_budget(cfg).format_table()
    assert "[pixel_direct]" in table
    for stage in ("covariance_setup", "basis_setup", "fisher_run", "spectra_run"):
        assert stage in table
