import os

import pytest

from qube import Fisher


@pytest.fixture
def fisher_instance_t_qu(config_resolver):
    """Fisher instance with T (spin-0) at index 0 and QU (spin-2) at index 1."""
    config_file = config_resolver("tests/data/nside4/TQU/config.yaml")
    fisher_analyzer = Fisher(config_file)
    fisher_analyzer.setup_fields()
    fisher_analyzer.setup_cls(lmax=fisher_analyzer.lmax_signal)
    os.unlink(config_file)
    return fisher_analyzer


def test_fisher_keyed_inputs_returns_spectrum_keys(fisher_instance_t_qu):
    cl_dict, keys = fisher_instance_t_qu._build_multi_spectrum_inputs()
    from cosmocore.spectrum_key import SpectrumKey

    assert all(isinstance(k, SpectrumKey) for k in keys)
