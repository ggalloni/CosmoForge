"""HarmonicBasis anchored at the fiducial covariance (``anchor_C_ell``).

The anchored basis factorises C0 = N_eff + V^T Λ0 V instead of N_eff and works
with Λ − Λ0 thereafter. Every quantity QUBE consumes must be unchanged: the
projected inverse V C^{-1} V^T, the weighted data V C^{-1} d and the
noise-bias covariance V C^{-1} N C^{-1} V^T, at the fiducial and away from it.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from cosmocore.basics import add_transposed_product
from cosmocore.basis import HarmonicBasis
from cosmocore.spectrum_key import SpectrumKey, SpectrumKind

LMAX = 6
N_PIX = 24


def _noise(n, seed):
    """Non-diagonal positive-definite noise: the anchor must not assume white N."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n))
    return 0.01 * np.eye(n) + 2e-4 * (X @ X.T) / n


def _geometry(spins, seed=0):
    rng = np.random.default_rng(seed)
    thetas = tuple(rng.uniform(0.1, np.pi - 0.1, N_PIX) for _ in spins)
    phis = tuple(rng.uniform(0, 2 * np.pi, N_PIX) for _ in spins)
    n = sum(2 * N_PIX if s == 2 else N_PIX for s in spins)
    if len(spins) == 1:
        return thetas[0], phis[0], n
    return thetas, phis, n


def _spectra(spins, scale=1.0):
    ells = np.arange(LMAX + 1, dtype=np.float64)
    with np.errstate(divide="ignore"):
        shape = np.where(ells >= 2, 1.0 / ells**2, 0.0)
    cl = 0.05 * scale * shape
    sp = tuple(spins)
    if spins == [0]:
        return cl
    if spins == [2]:
        return {
            SpectrumKey(0, 0, SpectrumKind.GG, spins=sp): cl,
            SpectrumKey(0, 0, SpectrumKind.CC, spins=sp): 0.3 * cl,
            SpectrumKey(0, 0, SpectrumKind.GC, spins=sp): 0.05 * cl,
        }
    return {
        SpectrumKey(0, 0, SpectrumKind.SS, spins=sp): cl,
        SpectrumKey(1, 1, SpectrumKind.GG, spins=sp): 0.5 * cl,
        SpectrumKey(1, 1, SpectrumKind.CC, spins=sp): 0.2 * cl,
        SpectrumKey(1, 1, SpectrumKind.GC, spins=sp): 0.02 * cl,
        SpectrumKey(0, 1, SpectrumKind.SG, spins=sp): 0.3 * cl,
        SpectrumKey(0, 1, SpectrumKind.SC, spins=sp): 0.01 * cl,
    }


def _perturb(C, factor):
    """Scale each multipole differently, so Λ − Λ0 is not a multiple of Λ0."""
    tilt = 1.0 + factor * np.linspace(-1.0, 1.0, LMAX + 1)
    if isinstance(C, dict):
        return {k: v * tilt for k, v in C.items()}
    return C * tilt


def _pair(spins, **kwargs):
    """(noise-anchored, fiducial-anchored, N, fiducial spectra) on one geometry."""
    theta, phi, n = _geometry(spins)
    N = _noise(n, seed=len(spins) + sum(spins))
    C0 = _spectra(spins)
    bases = []
    for anchor in (None, C0):
        hb = HarmonicBasis(
            N.copy(), theta, phi, LMAX, spins=spins, anchor_C_ell=anchor, **kwargs
        )
        hb.setup()
        bases.append(hb)
    return bases[0], bases[1], N, C0


SPINS = [[0], [2], [0, 2]]


@pytest.mark.parametrize("spins", SPINS, ids=["T", "QU", "TQU"])
@pytest.mark.parametrize("factor", [0.0, 0.3, 2.0], ids=["fiducial", "near", "far"])
def test_projected_inverse_matches_noise_anchor(spins, factor):
    ref, anchored, _, C0 = _pair(spins)
    C = _perturb(C0, factor)
    assert_allclose(
        anchored.get_projected_inverse(C),
        ref.get_projected_inverse(C),
        rtol=1e-9,
        atol=1e-9 * np.abs(ref.get_projected_inverse(C)).max(),
    )


@pytest.mark.parametrize("spins", SPINS, ids=["T", "QU", "TQU"])
@pytest.mark.parametrize("factor", [0.0, 0.3, 2.0], ids=["fiducial", "near", "far"])
def test_weighted_data_and_noise_bias_match_noise_anchor(spins, factor):
    ref, anchored, _, C0 = _pair(spins)
    C = _perturb(C0, factor)
    d = np.random.default_rng(7).standard_normal((ref.n_pix, 4))

    def weighted_and_noise_cov(hb):
        inner = hb.prepare_stable_inner_inv(C)
        A = inner.T
        return hb.get_weighted_data(d, C, stable_inner_inv=inner), (
            A @ hb.get_noise_for_bias() @ A.T
        )

    w_ref, cov_ref = weighted_and_noise_cov(ref)
    w, cov = weighted_and_noise_cov(anchored)
    assert_allclose(w, w_ref, rtol=1e-9, atol=1e-9 * np.abs(w_ref).max())
    assert_allclose(cov, cov_ref, rtol=1e-9, atol=1e-9 * np.abs(cov_ref).max())


@pytest.mark.parametrize("spins", SPINS, ids=["T", "QU", "TQU"])
def test_matches_brute_force_pixel_space(spins):
    """Against C = N + V^T Λ V inverted directly in pixel space."""
    _, anchored, N, C0 = _pair(spins)
    C = _perturb(C0, 0.3)
    V = anchored._V
    c_arr, c_dict, is_single = anchored._normalize_c_ell(C)
    Lam = (
        np.diag(anchored._build_lambda_diagonal(c_arr))
        if is_single
        else anchored._build_lambda_matrix(c_dict)
    )
    C_inv_Vt = np.linalg.solve(N + V.T @ Lam @ V, V.T)

    assert_allclose(
        anchored.get_projected_inverse(C), V @ C_inv_Vt, rtol=1e-8, atol=1e-12
    )
    inner = anchored.prepare_stable_inner_inv(C)
    noise_cov = inner.T @ anchored.get_noise_for_bias() @ inner
    assert_allclose(noise_cov, C_inv_Vt.T @ N @ C_inv_Vt, rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize("spins", SPINS, ids=["T", "QU", "TQU"])
def test_fisher_matches_noise_anchor(spins):
    ref, anchored, _, C0 = _pair(spins)
    if spins == [0]:
        args = (C0,)
    else:
        args = (C0, list(C0))
    assert_allclose(
        anchored.compute_fisher_matrix(*args),
        ref.compute_fisher_matrix(*args),
        rtol=1e-9,
    )


def test_at_fiducial_returns_m_without_a_solve():
    """At the anchor Λ − Λ0 = 0: the projected inverse is the stored M itself."""
    _, anchored, _, C0 = _pair([0, 2])
    A = anchored.get_projected_inverse(C0)
    assert np.shares_memory(A, anchored._V_Ninv_VT)
    assert not A.flags.writeable
    assert_allclose(anchored.prepare_stable_inner_inv(C0), np.eye(anchored.dim))


def test_switch_optimisation_matches_noise_anchor():
    """S_fixed and the anchor stack: T must drop both, M must keep both."""
    theta, phi, n = _geometry([0])
    N = _noise(n, seed=11)
    lmax_signal = 10
    fiducial = np.full(lmax_signal + 1, 0.05)
    window = fiducial.copy()
    window[LMAX + 1 :] = 0.0
    bases = []
    for anchor in (None, window):
        hb = HarmonicBasis(
            N.copy(),
            theta,
            phi,
            lmax_signal=lmax_signal,
            lmin=2,
            lmax=LMAX,
            fiducial_C_ell=fiducial,
            anchor_C_ell=anchor,
        )
        hb.setup()
        assert hb._use_switch_optimization
        bases.append(hb)
    ref, anchored = bases
    d = np.random.default_rng(3).standard_normal((n, 3))
    for C in (window, 1.7 * window):
        assert_allclose(
            anchored.get_projected_inverse(C), ref.get_projected_inverse(C), rtol=1e-9
        )
        inner_ref, inner = (hb.prepare_stable_inner_inv(C) for hb in bases)
        assert_allclose(
            inner.T @ anchored.get_noise_for_bias() @ inner,
            inner_ref.T @ ref.get_noise_for_bias() @ inner_ref,
            rtol=1e-9,
        )
        assert_allclose(
            anchored.get_weighted_data(d, C), ref.get_weighted_data(d, C), rtol=1e-9
        )


@pytest.mark.parametrize(
    "method",
    [
        "get_covariance",
        "get_full_inverse",
        "get_full_logdet",
        "quadratic_form",
        "prepare_for_basis",
    ],
)
def test_noise_factor_methods_refuse(method):
    _, anchored, _, C0 = _pair([0, 2])
    args = {
        "quadratic_form": (np.ones(anchored.n_pix), C0),
    }.get(method, (C0,))
    with pytest.raises(NotImplementedError, match="anchored"):
        getattr(anchored, method)(*args)


def test_anchor_with_mblock_compression_refuses():
    theta, phi, n = _geometry([0])
    with pytest.raises(NotImplementedError, match="m-block"):
        HarmonicBasis(
            np.eye(n), theta, phi, LMAX, compress=True, anchor_C_ell=_spectra([0])
        )


@pytest.mark.parametrize("a_order", ["C", "F"])
def test_add_transposed_product_updates_in_place(a_order):
    rng = np.random.default_rng(0)
    A = np.array(rng.standard_normal((5, 7)), order=a_order)
    B = rng.standard_normal((5, 3))
    C = np.asfortranarray(rng.standard_normal((7, 3)))
    expected = C + A.T @ B
    out = add_transposed_product(C, A, B)
    assert out is C
    assert_allclose(C, expected, rtol=1e-14)


def test_add_transposed_product_rejects_c_order_target():
    with pytest.raises(ValueError, match="F-contiguous"):
        add_transposed_product(np.zeros((3, 3)), np.ones((2, 3)), np.ones((2, 3)))


def test_anchor_is_a_copy_of_the_callers_spectra():
    """Mutating the caller's array must not move the anchor under Λ0."""
    theta, phi, n = _geometry([0])
    N = _noise(n, seed=5)
    C0 = _spectra([0])
    ref = HarmonicBasis(N.copy(), theta, phi, LMAX)
    anchored = HarmonicBasis(N.copy(), theta, phi, LMAX, anchor_C_ell=C0)
    ref.setup()
    anchored.setup()
    C0 *= 2.0
    assert_allclose(
        anchored.get_projected_inverse(C0), ref.get_projected_inverse(C0), rtol=1e-9
    )
