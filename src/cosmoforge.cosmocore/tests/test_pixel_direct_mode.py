"""
Tests for PixelBasis direct (V-free) mode.

Direct mode operates on full pixel-space matrices, reusing the existing
``cosmocore.pixel`` machinery. It is selected by ``setup_computation_basis(
method="auto")`` whenever ``n_pix <= n_modes``, or explicitly via
``PixelBasis(use_direct=True)``. These tests target the direct-mode-only
branches of ``PixelBasis``: ``_setup_direct``, ``_spectrum_idx_from_components``,
``_get_derivative_direct``, ``get_binned_derivative_direct``, and the
single/multi-spin combinations they dispatch to.
"""

import tempfile
from pathlib import Path

import healpy as hp
import numpy as np
import pytest

from cosmocore.basis import PixelBasis
from cosmocore.bins import Bins
from cosmocore.core import Core
from cosmocore.settings import InputParams


class _ConcreteCore(Core):
    """Minimal Core subclass for setting up fields + geometry without I/O."""

    def compute(self):  # pragma: no cover - required abstract stub
        return None

    def run(self):  # pragma: no cover - required abstract stub
        return None


def _make_core(spins, labels, nfields, nside=8, lmax=16, mask_half=True):
    """Build a Core with fields and geometry set up over a small fixture.

    Returns the Core instance with ``collection``, ``theta``, ``phi``,
    ``noise_cov1`` populated. The mask blanks half the sky when ``mask_half``,
    keeping the active pixel count low (~190 at nside=8).
    """
    params = InputParams()
    params.nside = nside
    params.lmax = lmax
    params.nfields = nfields
    params.spins = spins
    params.labels = labels
    params.ordering = "RING"

    npix = 12 * nside**2
    f = tempfile.NamedTemporaryFile(suffix=".fits", delete=False)
    f.close()
    if nfields == 1:
        mask = np.ones(npix, dtype=np.float64)
    else:
        mask = np.ones((nfields, npix), dtype=np.float64)
    if mask_half:
        if mask.ndim == 1:
            mask[: npix // 2] = 0.0
        else:
            mask[:, : npix // 2] = 0.0
    hp.write_map(f.name, mask, overwrite=True)
    params.maskfile = f.name

    core = _ConcreteCore(params)
    core.setup_fields()
    core.setup_geometry()

    n_total = core.collection.total_active_pixels
    rng = np.random.default_rng(0)
    diag = rng.uniform(0.5, 1.5, size=n_total)
    core.noise_cov1 = np.diag(diag)
    return core, params.maskfile


def _build_direct_basis(core, lmax):
    """Construct a PixelBasis with use_direct=True from a configured Core."""
    spins = [field.spin for field in core.collection.fields]
    n_pix = core.noise_cov1.shape[0]
    np.diag(1.0 / np.diag(core.noise_cov1))
    bm = PixelBasis(
        N=core.noise_cov1,
        theta=core.theta,
        phi=core.phi,
        lmax=lmax,
        spins=spins,
        fields=core.collection,
        use_direct=True,
    )
    bm.setup()
    assert bm.n_pix == n_pix
    return bm


def _per_ell_reference(bm, bins, bin_idx, comp_i, comp_j, mode):
    """Sum per-ℓ direct derivatives across a bin (uniform weights)."""
    lmin_b = bins.lmins[bin_idx]
    lmax_b = bins.lmaxs[bin_idx]
    ref = None
    for ell in range(lmin_b, lmax_b + 1):
        dC_ell = bm._get_derivative_direct(ell, comp_i, comp_j, mode).copy()
        ref = dC_ell if ref is None else ref + dC_ell
    return ref


# =========================================================================
# Setup paths
# =========================================================================


def test_setup_direct_requires_fields():
    """_setup_direct raises if fields=None — guards a documented contract."""
    n_pix = 20
    rng = np.random.default_rng(1)
    diag = rng.uniform(0.5, 1.5, size=n_pix)
    N = np.diag(diag)
    np.diag(1.0 / diag)
    theta = rng.uniform(0, np.pi, n_pix)
    phi = rng.uniform(0, 2 * np.pi, n_pix)

    bm = PixelBasis(
        N=N,
        theta=theta,
        phi=phi,
        lmax=8,
        use_direct=True,
    )
    with pytest.raises(ValueError, match="fields"):
        bm.setup()


# =========================================================================
# Single spin-0 field (TT)
# =========================================================================


def test_binned_derivative_direct_spin0():
    """Spin-0 binned direct derivative matches per-ℓ direct sum."""
    core, mask_file = _make_core(spins=[0], labels=["T"], nfields=1)
    try:
        bm = _build_direct_basis(core, lmax=core.params.lmax)
        bins = Bins.fromdeltal(2, core.params.lmax, 3)
        dC_b = bm.get_binned_derivative_direct(
            bin_idx=1,
            bins=bins,
            beam_smoothing=None,
            comp_i=0,
            comp_j=0,
            mode=0,
        )
        ref = _per_ell_reference(bm, bins, 1, 0, 0, 0)
        np.testing.assert_allclose(dC_b, ref, rtol=1e-10, atol=1e-12)
        # symmetry holds for auto-spectrum
        np.testing.assert_allclose(dC_b, dC_b.T, rtol=1e-12, atol=1e-12)
    finally:
        Path(mask_file).unlink()


# =========================================================================
# Single spin-2 field (EE / BB / EB)
# =========================================================================


@pytest.mark.parametrize("mode,name", [(0, "EE"), (1, "BB"), (2, "EB")])
def test_binned_derivative_direct_spin2(mode, name):
    """Spin-2 EE/BB/EB binned direct derivative matches per-ℓ direct sum."""
    core, mask_file = _make_core(spins=[2], labels=["E", "B"], nfields=2)
    try:
        bm = _build_direct_basis(core, lmax=core.params.lmax)
        bins = Bins.fromdeltal(2, core.params.lmax, 3)
        dC_b = bm.get_binned_derivative_direct(
            bin_idx=1,
            bins=bins,
            beam_smoothing=None,
            comp_i=0,
            comp_j=0,
            mode=mode,
        )
        ref = _per_ell_reference(bm, bins, 1, 0, 0, mode)
        np.testing.assert_allclose(
            dC_b,
            ref,
            rtol=1e-10,
            atol=1e-12,
            err_msg=f"spin-2 {name} direct binned derivative mismatch",
        )
    finally:
        Path(mask_file).unlink()


# =========================================================================
# Spin-0 × spin-2 (TQU: TE / TB)
# =========================================================================


@pytest.mark.parametrize("mode,name", [(0, "TE"), (1, "TB")])
def test_binned_derivative_direct_cross_spin(mode, name):
    """Cross-spin (T×P) binned direct derivative matches per-ℓ direct sum.

    Spectrum labels follow the spin-0-first convention (TE/TB), so the
    canonical caller passes comp_i=0 (T), comp_j=1 (P). Both off-diagonal
    blocks of dC must be filled (regression test for an earlier bug where
    only one was populated, breaking trace formulas downstream).
    """
    core, mask_file = _make_core(
        spins=[0, 2],
        labels=["T", "E", "B"],
        nfields=3,
    )
    try:
        bm = _build_direct_basis(core, lmax=core.params.lmax)
        bins = Bins.fromdeltal(2, core.params.lmax, 3)
        dC_te = bm.get_binned_derivative_direct(
            bin_idx=1,
            bins=bins,
            beam_smoothing=None,
            comp_i=0,
            comp_j=1,
            mode=mode,
        )
        ref_te = _per_ell_reference(bm, bins, 1, 0, 1, mode)
        np.testing.assert_allclose(
            dC_te,
            ref_te,
            rtol=1e-10,
            atol=1e-12,
            err_msg=f"{name} (spin-0×2) direct binned derivative mismatch",
        )
        # Sanity: both off-diagonal cross-blocks should be filled (regression).
        n_t = core.collection.fields[0].n_active[0]
        n_p = core.collection.fields[1].n_active[0]
        assert np.any(dC_te[:n_t, n_t : n_t + n_p] != 0)
        assert np.any(dC_te[n_t : n_t + n_p, :n_t] != 0)
    finally:
        Path(mask_file).unlink()


# =========================================================================
# Beam-smoothing weights
# =========================================================================


def test_binned_derivative_direct_with_beam_smoothing():
    """beam_smoothing argument scales each ℓ in the bin."""
    core, mask_file = _make_core(spins=[0], labels=["T"], nfields=1)
    try:
        bm = _build_direct_basis(core, lmax=core.params.lmax)
        bins = Bins.fromdeltal(2, core.params.lmax, 3)
        # Inference-range beam of length lmax-1 (ell=2..lmax, offset-from-2);
        # this matches Fisher's per-spectrum beam_smoothing layout.
        n_ell = core.params.lmax - 1
        beam = np.linspace(1.0, 0.5, n_ell) ** 2
        dC_b = bm.get_binned_derivative_direct(
            bin_idx=1,
            bins=bins,
            beam_smoothing=beam,
            comp_i=0,
            comp_j=0,
            mode=0,
        )

        # Reference: same bin, weighted per-ℓ sum.
        lmin_b, lmax_b = bins.lmins[1], bins.lmaxs[1]
        ref = None
        for ell in range(lmin_b, lmax_b + 1):
            d = bm._get_derivative_direct(ell, 0, 0, 0).copy()
            ref = beam[ell - 2] * d if ref is None else ref + beam[ell - 2] * d
        np.testing.assert_allclose(dC_b, ref, rtol=1e-10, atol=1e-12)
    finally:
        Path(mask_file).unlink()
