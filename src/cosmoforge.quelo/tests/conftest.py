import os
import tempfile

import pytest
import yaml


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
            if not current_dir.endswith("cosmoforge.quelo"):
                package_prefix = "src/cosmoforge.quelo/"
        except ValueError:
            # No 'src' in path - check if we're in the package directory
            if not current_dir.endswith("cosmoforge.quelo"):
                # Assume we need the full path from wherever we are
                package_prefix = "src/cosmoforge.quelo/"

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
