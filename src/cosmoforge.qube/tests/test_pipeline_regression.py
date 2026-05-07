"""End-to-end pipeline regression gate.

Locks the nside4/TEB Fisher + Spectra outputs against a frozen snapshot so any
future refactor that perturbs numerical results fails immediately rather than
slipping through unit tests. The snapshot lives next to the config at
``tests/data/nside4/TEB/regression_pipeline.npz``.

To intentionally regenerate the snapshot (e.g. when a numerical contract is
deliberately being changed), run this file directly:

    uv run python src/cosmoforge.qube/tests/test_pipeline_regression.py
"""

import os

import numpy as np

from qube import Fisher, Spectra

REF_PATH = os.path.join(
    os.path.dirname(__file__),
    "data",
    "nside4",
    "TEB",
    "regression_pipeline.npz",
)


def _run_pipeline(config_file: str) -> dict[str, np.ndarray]:
    """Run nside4/TEB Fisher + Spectra and return the four contract arrays."""
    fisher = Fisher(config_file)
    fisher.run()
    spectra = Spectra(config_file, fisher=fisher)
    spectra.run()
    return {
        "fisher_matrix": fisher.get_fisher_matrix(),
        "error_bars": fisher.get_error_bars(),
        "bandpowers": spectra.get_power_spectra(),
        "noise_bias": spectra.get_noise_bias(),
    }


def test_nside4_teb_pipeline_byte_identity(config_resolver):
    """nside4/TEB end-to-end pipeline must match the frozen snapshot exactly."""
    config_file = config_resolver("tests/data/nside4/TEB/config.yaml")
    try:
        outputs = _run_pipeline(config_file)
    finally:
        os.unlink(config_file)

    ref = np.load(REF_PATH)
    for name, value in outputs.items():
        np.testing.assert_array_equal(value, ref[name], err_msg=f"{name} changed")


if __name__ == "__main__":
    import sys
    import tempfile

    import yaml

    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rel_config = "tests/data/nside4/TEB/config.yaml"

    with open(os.path.join(package_root, rel_config)) as f:
        config = yaml.safe_load(f)
    for key, value in config.items():
        if not isinstance(value, str):
            continue
        clean = value[3:] if value.startswith("../") else value
        if clean.startswith(("tests/", "inputs/", "scripts/")):
            config[key] = os.path.join(package_root, clean)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(config, tmp, default_flow_style=False)
    tmp.close()

    try:
        outputs = _run_pipeline(tmp.name)
    finally:
        os.unlink(tmp.name)

    np.savez(REF_PATH, **outputs)
    print(f"Snapshot written: {REF_PATH}")
    for name, value in outputs.items():
        print(f"  {name:<14} shape={value.shape} dtype={value.dtype}")
    sys.exit(0)
