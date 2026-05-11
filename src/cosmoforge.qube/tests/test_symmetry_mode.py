"""Tests for SymmetryMode wiring through Fisher and Spectra (ADR-0011)."""

import os

import pytest

from cosmocore.spectrum_key import SpectrumKind, SymmetryMode
from qube import Fisher, Spectra


@pytest.fixture
def fisher_t_qu(config_resolver):
    """Fisher instance over T + QU components — exercises cross-component
    spin-0 x spin-2 (TE, TB) and auto-pair spin-2 x spin-2 (EE, BB, EB)."""
    config_file = config_resolver("tests/data/nside4/TQU/config.yaml")
    fisher = Fisher(config_file)
    fisher.setup_fields()
    fisher.setup_cls(lmax=fisher.lmax_signal)
    os.unlink(config_file)
    return fisher


def test_fisher_default_symmetry_mode_is_symmetric(fisher_t_qu):
    """Fisher defaults to SYMMETRIC — pre-Slice-5 behaviour is the default."""
    assert fisher_t_qu.symmetry_mode is SymmetryMode.SYMMETRIC


def test_fisher_accepts_explicit_symmetry_mode(config_resolver):
    """Constructor accepts an explicit SymmetryMode kwarg and stores it."""
    config_file = config_resolver("tests/data/nside4/TQU/config.yaml")
    fisher = Fisher(config_file, symmetry_mode=SymmetryMode.DIRECTIONAL)
    os.unlink(config_file)
    assert fisher.symmetry_mode is SymmetryMode.DIRECTIONAL


def test_build_inputs_forwards_symmetry_mode(fisher_t_qu):
    """_build_multi_spectrum_inputs forwards the flag into SpectraManager.

    Verified by setting DIRECTIONAL on a T+QU collection: there are no
    spin-2 cross-component pairs here so the emitted keys are identical
    to SYMMETRIC, but the forwarding contract is exercised end-to-end
    (no crash, valid keys returned)."""
    fisher_t_qu.symmetry_mode = SymmetryMode.DIRECTIONAL
    cl_dict, keys = fisher_t_qu._build_multi_spectrum_inputs()
    assert len(keys) > 0
    # T + QU has no spin-2 cross-component pair, so no CG should appear.
    assert all(k.kind is not SpectrumKind.CG for k in keys)


def test_symmetric_t_qu_end_to_end_smoke(config_resolver):
    """End-to-end Fisher.run() smoke test on T+QU under default SYMMETRIC.

    Exercises the int-mode call sites in fisher.py (fixed in Slice 5 to pass
    is_cross to kind_to_legacy_mode) and the symmetry_mode threading through
    get_binned_derivative_matrix. T+QU has spin-0 x spin-2 cross pairs (TE,
    TB), which use a different encoding path than the spin-2 x spin-2
    cross-pair encoding affected by the is_cross fix — this test guards
    against a regression in the unaffected paths."""
    config_file = config_resolver("tests/data/nside4/TQU/config.yaml")
    fisher = Fisher(config_file)
    os.unlink(config_file)
    fisher.run()
    assert fisher.fisher is not None
    assert fisher.fisher.shape[0] == fisher.fisher.shape[1]


def test_spectra_inherits_symmetry_mode_from_fisher(config_resolver):
    """Spectra inherits symmetry_mode from the Fisher instance (ADR-0011) —
    the user sets the flag once on Fisher and Spectra can never diverge."""
    config_file = config_resolver("tests/data/nside4/TQU/config.yaml")
    fisher = Fisher(config_file)
    fisher.run()
    # Override on the Fisher instance after construction to verify the
    # propagation contract, then construct Spectra.
    fisher.symmetry_mode = SymmetryMode.DIRECTIONAL

    config_file2 = config_resolver("tests/data/nside4/TQU/config.yaml")
    spectra = Spectra(config_file2, fisher=fisher)
    os.unlink(config_file)
    os.unlink(config_file2)
    assert spectra.symmetry_mode is SymmetryMode.DIRECTIONAL
