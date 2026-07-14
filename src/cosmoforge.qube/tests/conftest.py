import os
import tempfile

import pytest
import yaml

#: Config keys that name output artifacts. Strip them all and opt-in persistence
#: (ADR-0015) means the run writes nothing at all — which lets a test assert on
#: the filesystem rather than mock the writers.
OUTPUT_KEYS = {
    "output_geometry_file",
    "outnoisecovmat1",
    "outnoisecovmat2",
    "outinvcovmatfile1",
    "outinvcovmatfile2",
    "outfilefisher",
    "outcovmatfile",
    "outerrfile",
}


@pytest.fixture
def sandboxed_config(local_path):
    """Resolve a config to absolute inputs and strip every ``out*`` key.

    The shipped fixture configs point their ``out*`` keys back into
    ``tests/data/``, so a run driven by one writes artifacts into the fixture
    tree. A run driven by *this* config reads the same inputs and writes
    nothing, so the caller can assert on an empty output directory.
    """
    written = []

    def resolve(config_path):
        with open(os.path.join(local_path, config_path)) as f:
            config = yaml.safe_load(f)

        resolved = {}
        for key, value in config.items():
            if key in OUTPUT_KEYS:
                continue
            if isinstance(value, str) and value.startswith("../"):
                resolved[key] = os.path.join(local_path, value[3:])
            else:
                resolved[key] = value

        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(resolved, tmp, default_flow_style=False)
        tmp.close()
        written.append(tmp.name)
        return tmp.name

    yield resolve

    for path in written:
        os.unlink(path)


@pytest.fixture
def local_path():
    # Get the directory where this conftest.py file is located
    test_dir = os.path.dirname(__file__)
    # Return the parent directory (the package root)
    return os.path.dirname(test_dir)


@pytest.fixture
def config_resolver(local_path):
    """
    Fixture that provides a function to resolve config files with correct paths.

    This function reads a config file and resolves paths to work from both
    the project root and the package directory.
    """

    def resolve_config(config_path):
        """
        Read a config file and resolve all paths to work from current location.

        Parameters
        ----------
        config_path : str
            Path to the config file relative to local_path

        Returns
        -------
        str
            Path to a temporary config file with resolved paths
        """
        # Read the original config
        full_config_path = os.path.join(local_path, config_path)
        with open(full_config_path) as f:
            config = yaml.safe_load(f)

        # Find the project root by looking for src directory
        current_dir = os.getcwd()
        path_parts = current_dir.split(os.sep)

        # Determine if we need to add package prefix for relative paths
        package_prefix = ""
        try:
            # If we find 'src' in the path, we might be running from project root
            path_parts.index("src")
            # If current dir doesn't end with package name, add prefix
            if not current_dir.endswith("cosmoforge.qube"):
                package_prefix = "src/cosmoforge.qube/"
        except ValueError:
            # No 'src' in path - check if we're in the package directory
            if not current_dir.endswith("cosmoforge.qube"):
                # Assume we need the full path from wherever we are
                package_prefix = "src/cosmoforge.qube/"

        # Update relative paths in config
        for key, value in config.items():
            if isinstance(value, str):
                # Handle paths with ../ prefix (strip it first)
                clean_value = value
                if value.startswith("../"):
                    clean_value = value[3:]  # Remove "../" prefix

                if (
                    clean_value.startswith("tests/")
                    or clean_value.startswith("inputs/")
                    or clean_value.startswith("scripts/")
                ):
                    # Add package prefix if needed
                    config[key] = package_prefix + clean_value

        # Create a temporary config file with resolved paths
        temp_config = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
        yaml.dump(config, temp_config, default_flow_style=False)
        temp_config.close()

        return temp_config.name

    return resolve_config
