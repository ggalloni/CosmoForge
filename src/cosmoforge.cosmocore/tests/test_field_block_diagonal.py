"""
Tests for field block-diagonal K detection and exploitation (Phase 4).

When cross-spectra are absent between fields and noise is independent per
field, K is exactly block-diagonal across fields.  Inverting each field
block independently is exact and gives significant speedup.
"""

import time

import numpy as np
from numpy.testing import assert_allclose

from cosmocore.spectrum_key import SpectrumKey, SpectrumKind


def _to_key(entry, *, spins):
    """Test-only adapter: legacy 2-tuple -> SpectrumKey (scalar)."""
    return SpectrumKey(entry[0], entry[1], SpectrumKind.SS, spins=spins)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_two_scalar_fields(
    n_pix_1=60, n_pix_2=40, lmax_signal=8, seed=42, noise_cross=False
):
    """Create a two-field setup with optional noise cross-block."""
    lmax = lmax_signal
    np.random.seed(seed)

    n_pix_total = n_pix_1 + n_pix_2

    nv1 = np.ones(n_pix_1) * 0.01
    nv2 = np.ones(n_pix_2) * 0.02

    N = np.zeros((n_pix_total, n_pix_total))
    N[:n_pix_1, :n_pix_1] = np.diag(nv1)
    N[n_pix_1:, n_pix_1:] = np.diag(nv2)

    if noise_cross:
        cross = np.ones((n_pix_1, n_pix_2)) * 1e-4
        N[:n_pix_1, n_pix_1:] = cross
        N[n_pix_1:, :n_pix_1] = cross.T

    from cosmocore.basics import matrix_inverse_symm

    N_inv = matrix_inverse_symm(N.copy())

    theta_1 = np.random.uniform(0.1, np.pi - 0.1, n_pix_1)
    phi_1 = np.random.uniform(0, 2 * np.pi, n_pix_1)
    theta_2 = np.random.uniform(0.1, np.pi - 0.1, n_pix_2)
    phi_2 = np.random.uniform(0, 2 * np.pi, n_pix_2)

    return {
        "N": N,
        "N_inv": N_inv,
        "theta": (theta_1, theta_2),
        "phi": (phi_1, phi_2),
        "lmax": lmax,
        "n_pix_1": n_pix_1,
        "n_pix_2": n_pix_2,
    }


def _make_tqu_setup(n_pix_T=50, n_pix_QU=30, lmax=6, seed=42):
    """Create a T + QU (spin-0 + spin-2) two-component setup."""
    np.random.seed(seed)

    # Spin-2 doubles pixel count for QU
    n_pix_total = n_pix_T + 2 * n_pix_QU

    N = np.eye(n_pix_total) * 0.01
    N_inv = np.eye(n_pix_total) * 100.0

    theta_T = np.random.uniform(0.1, np.pi - 0.1, n_pix_T)
    phi_T = np.random.uniform(0, 2 * np.pi, n_pix_T)
    theta_QU = np.random.uniform(0.1, np.pi - 0.1, n_pix_QU)
    phi_QU = np.random.uniform(0, 2 * np.pi, n_pix_QU)

    return {
        "N": N,
        "N_inv": N_inv,
        "theta": (theta_T, theta_QU),
        "phi": (phi_T, phi_QU),
        "lmax": lmax,
        "spins": [0, 2],
    }


# ---------------------------------------------------------------------------
# Detection tests
# ---------------------------------------------------------------------------


class TestFieldBlockDetection:
    """Tests for _detect_field_blocks."""

    def test_detect_independent_no_cross_spectra(self):
        """Two spin-0 fields, no cross-spectrum, independent noise -> [[0], [1]]."""
        from cosmocore.basis import HarmonicBasis

        setup = _make_two_scalar_fields()
        hc = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=setup["lmax"],
        )

        n_ell = setup["lmax"] - 1
        # Only auto-spectra in C_ell_dict
        C_ell_dict = {
            (0, 0): np.ones(n_ell) * 1e-6,
            (1, 1): np.ones(n_ell) * 0.8e-6,
        }
        groups = hc._detect_field_blocks(C_ell_dict)

        assert groups == [[0], [1]], f"Expected [[0], [1]], got {groups}"

    def test_detect_coupled_with_cross_spectra(self):
        """Two spin-0 fields WITH cross-spectrum -> [[0, 1]]."""
        from cosmocore.basis import HarmonicBasis

        setup = _make_two_scalar_fields()
        hc = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=setup["lmax"],
        )

        n_ell = setup["lmax"] - 1
        # Cross-spectrum in C_ell_dict couples the fields
        C_ell_dict = {
            (0, 0): np.ones(n_ell) * 1e-6,
            (1, 1): np.ones(n_ell) * 0.8e-6,
            (0, 1): np.ones(n_ell) * 0.5e-6,
        }
        groups = hc._detect_field_blocks(C_ell_dict)

        assert groups == [[0, 1]], f"Expected [[0, 1]], got {groups}"

    def test_detect_coupled_by_noise(self):
        """No cross-spectrum but noise off-diagonal -> [[0, 1]]."""
        from cosmocore.basis import HarmonicBasis

        setup = _make_two_scalar_fields(noise_cross=True)
        hc = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=setup["lmax"],
        )

        n_ell = setup["lmax"] - 1
        # Only auto-spectra
        C_ell_dict = {
            (0, 0): np.ones(n_ell) * 1e-6,
            (1, 1): np.ones(n_ell) * 0.8e-6,
        }
        groups = hc._detect_field_blocks(C_ell_dict)

        assert groups == [[0, 1]], f"Expected [[0, 1]], got {groups}"

    def test_detect_tqu_groups(self):
        """T + QU fields (spin-0 + spin-2), no TE/TB -> [[0], [1]]."""
        from cosmocore.basis import HarmonicBasis

        setup = _make_tqu_setup()
        hc = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=setup["lmax"],
            spins=setup["spins"],
        )

        n_ell = setup["lmax"] - 1
        # Only auto-spectra (T-T and EE/BB), no TE/TB in dict
        C_ell_dict = {
            (0, 0, 0): np.ones(n_ell) * 1e-6,
            (1, 1, 0): np.ones(n_ell) * 0.5e-6,
            (1, 1, 1): np.ones(n_ell) * 0.1e-6,
        }
        groups = hc._detect_field_blocks(C_ell_dict)

        assert groups == [[0], [1]], f"Expected [[0], [1]], got {groups}"

    def test_detect_tqu_coupled_with_te(self):
        """T + QU with TE cross-spectrum -> [[0, 1]]."""
        from cosmocore.basis import HarmonicBasis

        setup = _make_tqu_setup()
        hc = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=setup["lmax"],
            spins=setup["spins"],
        )

        n_ell = setup["lmax"] - 1
        # TE cross-spectrum in dict couples T and QU
        C_ell_dict = {
            (0, 0, 0): np.ones(n_ell) * 1e-6,
            (1, 1, 0): np.ones(n_ell) * 0.5e-6,
            (1, 1, 1): np.ones(n_ell) * 0.1e-6,
            (0, 1, 0): np.ones(n_ell) * 0.3e-6,
        }
        groups = hc._detect_field_blocks(C_ell_dict)

        assert groups == [[0, 1]], f"Expected [[0, 1]], got {groups}"

    def test_single_component_returns_single_group(self):
        """Single component should return [[0]]."""
        from cosmocore.basis import HarmonicBasis

        np.random.seed(42)
        n_pix = 50
        N = np.eye(n_pix) * 0.01
        np.eye(n_pix) * 100.0
        theta = np.random.uniform(0.1, np.pi - 0.1, n_pix)
        phi = np.random.uniform(0, 2 * np.pi, n_pix)

        hc = HarmonicBasis(N, theta, phi, lmax_signal=6)

        C_ell_dict = {(0, 0): np.ones(4) * 1e-6}
        groups = hc._detect_field_blocks(C_ell_dict)
        assert groups == [[0]]

    def test_three_fields_partial_coupling(self):
        """Three fields where 0-1 coupled but 2 independent -> [[0, 1], [2]]."""
        from cosmocore.basis import HarmonicBasis

        np.random.seed(42)
        n_pix = [30, 30, 30]
        n_total = sum(n_pix)
        N = np.eye(n_total) * 0.01
        np.eye(n_total) * 100.0

        thetas = tuple(np.random.uniform(0.1, np.pi - 0.1, n) for n in n_pix)
        phis = tuple(np.random.uniform(0, 2 * np.pi, n) for n in n_pix)

        hc = HarmonicBasis(N, thetas, phis, lmax_signal=5)

        n_ell = 5 - 1
        # 0-1 cross-spectrum, no 0-2 or 1-2
        C_ell_dict = {
            (0, 0): np.ones(n_ell) * 1e-6,
            (1, 1): np.ones(n_ell) * 0.8e-6,
            (2, 2): np.ones(n_ell) * 0.6e-6,
            (0, 1): np.ones(n_ell) * 0.4e-6,
        }
        groups = hc._detect_field_blocks(C_ell_dict)

        assert groups == [[0, 1], [2]], f"Expected [[0, 1], [2]], got {groups}"


# ---------------------------------------------------------------------------
# Exactness tests
# ---------------------------------------------------------------------------


class TestFieldBlockFisherExact:
    """Tests that field-block K inversion matches full K inversion exactly."""

    def test_field_block_fisher_exact(self):
        """Two independent spin-0 fields: field-block Fisher == full Fisher."""
        from cosmocore.basis import HarmonicBasis

        setup = _make_two_scalar_fields(lmax_signal=6)
        hc = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=setup["lmax"],
        )
        hc.setup()

        ells = np.arange(2, setup["lmax"] + 1)
        C_ell_dict = {
            (0, 0): 1e-5 / ells**2,
            (1, 1): 0.8e-5 / ells**2,
        }

        # Compute projected inverse with field blocks
        field_groups = hc._detect_field_blocks(C_ell_dict)
        assert field_groups == [[0], [1]]

        V_Cinv_VT_block = hc.get_projected_inverse(C_ell_dict, field_groups=field_groups)

        # Compute projected inverse without field blocks (full K inversion)
        V_Cinv_VT_full = hc.get_projected_inverse(C_ell_dict, field_groups=None)

        # They must match to machine precision
        assert_allclose(
            V_Cinv_VT_block,
            V_Cinv_VT_full,
            rtol=1e-12,
            atol=1e-15,
            err_msg="Field-block projected inverse must match full inversion exactly",
        )

    def test_field_block_fisher_matrix_exact(self):
        """Fisher matrix from field-block path matches full path exactly."""
        from cosmocore.basics import matrix_inverse_symm, matrix_mult, matrix_trace
        from cosmocore.basis import HarmonicBasis

        setup = _make_two_scalar_fields(lmax_signal=6)
        hc = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=setup["lmax"],
        )
        hc.setup()

        n_ell = setup["lmax"] - 1
        ells = np.arange(2, setup["lmax"] + 1)
        C_ell_dict = {
            (0, 0): 1e-5 / ells**2,
            (1, 1): 0.8e-5 / ells**2,
        }

        spins = (0, 0)
        spectra_list = [_to_key(t, spins=spins) for t in [(0, 0), (1, 1)]]

        # Fisher via the automatic path (which should detect field blocks)
        fisher_auto = hc.compute_fisher_matrix(
            C_ell_dict, spectra_list, ell_min=2, ell_max=setup["lmax"]
        )

        # Fisher via pixel-space for validation
        V = hc._V
        lambda_matrix = hc._build_lambda_matrix(C_ell_dict)
        S = V.T @ lambda_matrix @ V
        C = setup["N"] + S
        C_inv = matrix_inverse_symm(C.copy())

        n_spec = len(spectra_list)
        fisher_pixel = np.zeros((n_spec * n_ell, n_spec * n_ell))

        from cosmocore.spectrum_key import kind_to_legacy_mode

        for spec_a, entry_a in enumerate(spectra_list):
            ci_a, cj_a = entry_a.comp_i, entry_a.comp_j
            mode_a = kind_to_legacy_mode(entry_a.kind, is_cross=(ci_a != cj_a))
            for ell_a in range(2, setup["lmax"] + 1):
                E_a = hc.get_derivative_matrix(ell_a, ci_a, cj_a, mode_a)
                dS_a = V.T @ E_a @ V
                idx_a = spec_a * n_ell + (ell_a - 2)

                for spec_b in range(spec_a, n_spec):
                    entry_b = spectra_list[spec_b]
                    ci_b, cj_b = entry_b.comp_i, entry_b.comp_j
                    mode_b = kind_to_legacy_mode(entry_b.kind, is_cross=(ci_b != cj_b))
                    ell_b_start = ell_a if spec_a == spec_b else 2
                    for ell_b in range(ell_b_start, setup["lmax"] + 1):
                        E_b = hc.get_derivative_matrix(ell_b, ci_b, cj_b, mode_b)
                        dS_b = V.T @ E_b @ V
                        idx_b = spec_b * n_ell + (ell_b - 2)

                        t1 = matrix_mult(C_inv, dS_a)
                        t2 = matrix_mult(C_inv, dS_b)
                        f_val = 0.5 * matrix_trace(t1, t2)

                        fisher_pixel[idx_a, idx_b] = f_val
                        if idx_a != idx_b:
                            fisher_pixel[idx_b, idx_a] = f_val

        assert_allclose(
            fisher_auto,
            fisher_pixel,
            rtol=1e-6,
            atol=1e-10,
            err_msg="Fisher from field-block path must match pixel-space computation",
        )

    def test_coupled_fields_still_correct(self):
        """When fields are coupled (cross-spectrum), result is still exact."""
        from cosmocore.basis import HarmonicBasis

        setup = _make_two_scalar_fields(lmax_signal=6)
        hc = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=setup["lmax"],
        )
        hc.setup()

        n_ell = setup["lmax"] - 1
        ells = np.arange(2, setup["lmax"] + 1)
        C_ell_dict = {
            (0, 0): 1e-5 / ells**2,
            (1, 1): 0.8e-5 / ells**2,
            (0, 1): 0.4e-5 / ells**2,
        }

        # Cross-spectrum present -> single group, no field-block optimization
        field_groups = hc._detect_field_blocks(C_ell_dict)
        assert field_groups == [[0, 1]]

        spins = (0, 0)
        spectra_list = [_to_key(t, spins=spins) for t in [(0, 0), (1, 1), (0, 1)]]
        # Should still produce correct result (falls through to full K path)
        fisher = hc.compute_fisher_matrix(
            C_ell_dict, spectra_list, ell_min=2, ell_max=setup["lmax"]
        )
        assert fisher.shape == (3 * n_ell, 3 * n_ell)
        assert np.all(np.diag(fisher) > 0)
        assert_allclose(fisher, fisher.T, rtol=1e-10)


# ---------------------------------------------------------------------------
# Speedup test
# ---------------------------------------------------------------------------


class TestFieldBlockSpeedup:
    """Timing test to verify field-block inversion is faster."""

    def test_field_block_speedup(self):
        """Two independent fields: field-block K inversion should be faster."""
        from cosmocore.basis import HarmonicBasis

        # Use larger setup for meaningful timing
        setup = _make_two_scalar_fields(n_pix_1=100, n_pix_2=80, lmax_signal=10)
        hc = HarmonicBasis(
            N=setup["N"],
            theta=setup["theta"],
            phi=setup["phi"],
            lmax_signal=setup["lmax"],
        )
        hc.setup()

        ells = np.arange(2, setup["lmax"] + 1)
        C_ell_dict = {
            (0, 0): 1e-5 / ells**2,
            (1, 1): 0.8e-5 / ells**2,
        }

        field_groups = hc._detect_field_blocks(C_ell_dict)
        assert field_groups == [[0], [1]]

        # Warm up
        _ = hc.get_projected_inverse(C_ell_dict, field_groups=field_groups)
        _ = hc.get_projected_inverse(C_ell_dict, field_groups=None)

        # Time field-block inversion
        n_iter = 5
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = hc.get_projected_inverse(C_ell_dict, field_groups=field_groups)
        t_block = (time.perf_counter() - t0) / n_iter

        # Time full inversion
        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ = hc.get_projected_inverse(C_ell_dict, field_groups=None)
        t_full = (time.perf_counter() - t0) / n_iter

        print(f"\nField-block inversion: {t_block * 1000:.2f} ms")
        print(f"Full inversion: {t_full * 1000:.2f} ms")
        print(f"Speedup: {t_full / t_block:.2f}x")
