"""End-to-end: ``basis={"anchor": "fiducial"}`` leaves Fisher and Spectra unchanged.

The nside-4 configs run with lmax=8 < lmax_signal=16, so the switch
optimisation (S_fixed) is active underneath the anchor.
"""

import numpy as np
import pytest

from qube import Fisher, Spectra

HARMONIC = {"method": "harmonic"}
ANCHORED = {"method": "harmonic", "anchor": "fiducial"}
FIELDS = ["T", "QU", "TQU"]


def _cfg(config_resolver, fields):
    return config_resolver(f"tests/data/nside4/{fields}/config.yaml")


@pytest.mark.parametrize("fields", FIELDS)
def test_fisher_unchanged(config_resolver, fields):
    fishers = []
    for basis in (HARMONIC, ANCHORED):
        f = Fisher(_cfg(config_resolver, fields), basis=basis)
        f.run()
        assert f.basis_manager._use_switch_optimization
        fishers.append(f.get_fisher_matrix())
    # Scale-relative: entries that vanish by symmetry (~1e-37) carry only
    # round-off, and the noise-anchored path's own error is ~1e-10 of max|F|.
    ref = fishers[0]
    np.testing.assert_allclose(fishers[1], ref, rtol=1e-8, atol=1e-9 * np.abs(ref).max())


def test_fiducial_anchor_is_closer_to_the_pixel_space_reference(config_resolver):
    """T: anchored Fisher matches the direct pixel-space C^{-1} to ~1e-13.

    The noise-anchored path reaches only ~1e-8 on the same config (its
    n_modes³ solve sits on top of N_eff^{-1}); the anchor removes that step at
    the fiducial, so this pins the accuracy gain, not just equivalence.
    """

    def eig(basis):
        f = Fisher(_cfg(config_resolver, "T"), basis=basis)
        f.run()
        F = f.get_fisher_matrix()
        return np.sort(np.linalg.eigvalsh((F + F.T) / 2))

    reference = eig(False)
    np.testing.assert_allclose(eig(ANCHORED), reference, rtol=1e-11)


@pytest.mark.parametrize("fields", FIELDS)
def test_fisher_takes_the_fiducial_shortcut(config_resolver, fields):
    """The spectra Fisher passes must equal the anchor exactly.

    Otherwise Λ − Λ0 is nonzero, the n_modes³ solve runs anyway and the
    anchor buys nothing, with every number still correct.
    """
    f = Fisher(_cfg(config_resolver, fields), basis=ANCHORED)
    f.run()
    C_ell_dict, _ = f._build_multi_spectrum_inputs()
    bm = f.basis_manager
    assert np.shares_memory(bm.get_projected_inverse(C_ell_dict), bm._V_Ninv_VT)


@pytest.mark.parametrize("fields", FIELDS)
def test_spectra_unchanged(config_resolver, fields):
    results = []
    for basis in (HARMONIC, ANCHORED):
        s = Spectra(_cfg(config_resolver, fields), basis=basis)
        s.run()
        results.append((s.get_power_spectra(), s.get_noise_bias()))
    (spec_ref, nb_ref), (spec, nb) = results
    np.testing.assert_allclose(
        spec, spec_ref, rtol=1e-7, atol=1e-10 * np.abs(spec_ref).max()
    )
    np.testing.assert_allclose(nb, nb_ref, rtol=1e-7, atol=1e-10 * np.abs(nb_ref).max())


def test_unknown_anchor_raises(config_resolver):
    f = Fisher(_cfg(config_resolver, "T"), basis={"method": "harmonic", "anchor": "x"})
    with pytest.raises(ValueError, match="anchor"):
        f.run()
