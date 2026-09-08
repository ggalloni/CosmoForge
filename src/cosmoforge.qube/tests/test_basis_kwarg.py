"""Tests for the ``basis=`` constructor kwarg and its ``compression=`` shim (ADR-0018).

These exercise the public resolution of ``basis=`` into the internal
``_basis_config`` sentinel (``None`` = traditional path, dict = basis path).
Construction is cheap (no ``run()``); path selection is asserted separately in
the equivalence tests.
"""

import warnings

import numpy as np
import pytest

from qube import Fisher, Spectra


def _cfg(config_resolver):
    return config_resolver("tests/data/nside4/T/config.yaml")


def test_basis_dict_is_passed_through(config_resolver):
    f = Fisher(_cfg(config_resolver), basis={"method": "harmonic"})
    assert f._basis_config == {"method": "harmonic"}


@pytest.mark.parametrize("method", ["auto", "harmonic", "pixel"])
def test_basis_string_sugar(config_resolver, method):
    f = Fisher(_cfg(config_resolver), basis=method)
    assert f._basis_config == {"method": method}


def test_basis_false_is_traditional(config_resolver):
    f = Fisher(_cfg(config_resolver), basis=False)
    assert f._basis_config is None


def test_basis_unknown_string_raises(config_resolver):
    with pytest.raises(ValueError, match="unknown basis"):
        Fisher(_cfg(config_resolver), basis="bogus")


def test_compression_kwarg_deprecated_but_forwarded(config_resolver):
    with pytest.warns(DeprecationWarning, match="compression="):
        f = Fisher(_cfg(config_resolver), compression={"method": "harmonic"})
    assert f._basis_config == {"method": "harmonic"}


def test_basis_and_compression_together_is_error(config_resolver):
    with pytest.raises(TypeError, match="only basis="):
        Fisher(
            _cfg(config_resolver),
            basis="harmonic",
            compression={"method": "harmonic"},
        )


def test_default_resolves_to_auto(config_resolver):
    # The default (no basis arg) must select method="auto" — and emit no
    # deprecation warning (only DeprecationWarning is escalated so unrelated
    # dependency warnings don't make this flaky).
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        f = Fisher(_cfg(config_resolver))
    assert f._basis_config == {"method": "auto"}


def test_default_with_do_cross_stays_traditional(config_resolver):
    # auto only selects a basis for auto-spectra: the basis layer has no second
    # noise covariance, so do_cross=True keeps the default on the traditional
    # path (ADR-0018).
    cfg = config_resolver("tests/data/nside4/QU/cross_config.yaml")
    f = Fisher(cfg)
    assert f.params.do_cross is True
    assert f._basis_config is None


def test_default_run_builds_auto_basis(config_resolver):
    # End-to-end: a default Fisher run reaches the ADR-0003 selector and builds
    # a basis (harmonic for the full-sky test config).
    f = Fisher(_cfg(config_resolver))
    f.run()
    assert f.basis_manager is not None
    assert f.basis_manager.method == "harmonic"


def _fisher_matrix(config_resolver, fields, basis=Fisher._UNSET):
    cfg = config_resolver(f"tests/data/nside4/{fields}/config.yaml")
    f = Fisher(cfg) if basis is Fisher._UNSET else Fisher(cfg, basis=basis)
    f.run()
    return f.get_fisher_matrix()


@pytest.mark.parametrize("fields", ["T", "QU"])
def test_default_equivalent_to_traditional_and_explicit_harmonic(config_resolver, fields):
    """The auto default must not change results (ADR-0018 guardrail).

    Default (auto) resolves to harmonic on these full-sky configs, so it is
    numerically identical to explicit ``basis="harmonic"``; and it is the same
    Fisher *operator* as the traditional ``basis=False`` reference. Element-wise
    parameter ordering differs between the harmonic and pixel-space paths for
    spin-2, so the traditional comparison is permutation-invariant (eigenvalues).
    """
    F_default = _fisher_matrix(config_resolver, fields)
    F_harmonic = _fisher_matrix(config_resolver, fields, basis="harmonic")
    F_traditional = _fisher_matrix(config_resolver, fields, basis=False)

    # auto deterministically resolves to harmonic → same numbers.
    np.testing.assert_allclose(F_default, F_harmonic, rtol=1e-12, atol=0)

    # Same operator as the traditional reference (order-independent).
    eig = lambda M: np.sort(np.linalg.eigvalsh((M + M.T) / 2))  # noqa: E731
    np.testing.assert_allclose(
        eig(F_default),
        eig(F_traditional),
        rtol=1e-6,
        err_msg="auto default is not the same Fisher operator as basis=False",
    )


def _with_ceiling(cfg, cls_, via, ceiling=8):
    """Build ``cls_`` asking for ``lmax_signal=ceiling`` by one of two routes."""
    if via == "basis_dict":
        return cls_(cfg, basis={"method": "harmonic", "lmax_signal": ceiling})
    obj = cls_(cfg, basis={"method": "harmonic"})
    if via == "setter":
        obj.lmax_signal = ceiling
    return obj


@pytest.mark.parametrize("via", ["setter", "basis_dict"])
def test_fisher_lmax_signal_reaches_the_basis(config_resolver, via):
    """The resolved ``lmax_signal`` is the ceiling the basis is built at.

    Regression: the builder read ``params.lmax_signal`` directly, so a
    setter-set ceiling left the Cls/beams at one lmax and the basis at 4*nside,
    surfacing as a "beam too short" error deep in the spectra loader.
    """
    f = _with_ceiling(_cfg(config_resolver), Fisher, via)
    assert f.lmax_signal == 8  # not the 4*nside=16 default
    f.run()
    assert f.basis_manager.lmax_signal == 8


def test_spectra_lmax_signal_reaches_the_basis(config_resolver):
    """Same contract on ``Spectra``, where the dict is the only live route.

    The dict used to reach the internal Fisher and not ``Spectra`` itself,
    leaving one run with two ceilings: basis at 8, Cls and beams at 16.
    """
    s = _with_ceiling(_cfg(config_resolver), Spectra, "basis_dict")
    assert s.lmax_signal == 8
    s.run()
    assert s.basis_manager.lmax_signal == 8


def test_spectra_lmax_signal_setter_refuses_once_the_fisher_exists(config_resolver):
    """The setter cannot work after construction, so it says so.

    ``Spectra.__init__`` runs the internal Fisher, so a later assignment
    cannot change the ceiling anything was computed at. It was previously
    accepted and ignored: ``s.lmax_signal = 8`` moved the reported attribute
    and produced output bit-identical to setting nothing (measured max|diff|
    exactly 0.0).
    """
    s = Spectra(_cfg(config_resolver), basis={"method": "harmonic"})
    with pytest.raises(RuntimeError, match="cannot change what is computed"):
        s.lmax_signal = 8


def test_spectra_ceiling_changes_the_answer(config_resolver):
    """The ceiling must reach the numbers, not just the attribute.

    This is the assertion the attribute checks above cannot make: before the
    fix the dict route built the basis at 8 against Cls and beams at 16, which
    shifted the deconvolved Cl by ~0.6% at the lowest bandpower rather than
    producing the coherent lmax_signal=8 answer.
    """

    def deconvolved(via):
        s = _with_ceiling(_cfg(config_resolver), Spectra, via)
        s.run()
        return np.asarray(s.get_power_spectra(mode="deconvolved")).ravel()

    assert np.max(np.abs(deconvolved("basis_dict") - deconvolved("neither"))) > 1e-9, (
        "lmax_signal=8 produced the 4*nside=16 answer, so the ceiling is inert"
    )
