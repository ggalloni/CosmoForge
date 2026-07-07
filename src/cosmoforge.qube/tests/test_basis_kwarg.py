"""Tests for the ``basis=`` constructor kwarg and its ``compression=`` shim (ADR-0018).

These exercise the public resolution of ``basis=`` into the internal
``_basis_config`` sentinel (``None`` = traditional path, dict = basis path).
Construction is cheap (no ``run()``); path selection is asserted separately in
the equivalence tests.
"""

import warnings

import pytest

from qube import Fisher


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


def test_default_is_traditional_before_flip(config_resolver):
    # C1 only: default still resolves to the traditional path. The auto flip
    # lands in C2 and this expectation changes there.
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no spurious deprecation on the default
        f = Fisher(_cfg(config_resolver))
    assert f._basis_config is None
