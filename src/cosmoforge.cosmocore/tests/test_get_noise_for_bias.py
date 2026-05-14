"""Both ComputationBasis subclasses must implement get_noise_for_bias.

The returned value is basis-specific in form (pixel: U^T N U;
harmonic: V N_eff^{-1} N N_eff^{-1} V^T) but fills the same slot in the
QML noise-bias formula. The ABC declares the method; subclasses implement.
"""

import inspect

from cosmocore.basis.base import ComputationBasis


def test_abc_declares_get_noise_for_bias():
    assert hasattr(ComputationBasis, "get_noise_for_bias")
    method = ComputationBasis.get_noise_for_bias
    assert getattr(method, "__isabstractmethod__", False), (
        "get_noise_for_bias must be declared abstract on the ABC"
    )


def test_signature_takes_self_only():
    sig = inspect.signature(ComputationBasis.get_noise_for_bias)
    params = [p for p in sig.parameters.values() if p.name != "self"]
    assert params == [], "get_noise_for_bias takes no arguments"
