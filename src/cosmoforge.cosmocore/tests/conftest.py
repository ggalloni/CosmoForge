import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")


def _configure_plt():
    plt.rc("axes", labelsize=20, linewidth=1.5)
    plt.rc("xtick", direction="in", labelsize=15, top=True)
    plt.rc("ytick", direction="in", labelsize=15, right=True)

    plt.rc("xtick.major", width=1.1, size=5)
    plt.rc("ytick.major", width=1.1, size=5)

    plt.rc("xtick.minor", width=1.1, size=3)
    plt.rc("ytick.minor", width=1.1, size=3)

    plt.rc("lines", linewidth=2)
    plt.rc("legend", frameon=False, fontsize=15)
    plt.rc("figure", dpi=100, autolayout=True, figsize=[10, 7])
    plt.rc("savefig", dpi=300, bbox="tight")

    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
        color=[
            "red",
            "dodgerblue",
            "forestgreen",
            "goldenrod",
            "maroon",
            "cyan",
            "limegreen",
            "darkorange",
            "darkmagenta",
        ]
    )


@pytest.fixture
def configure_plt():
    return _configure_plt


@pytest.fixture
def local_path():
    """Get the path to the cosmocore package directory."""
    # Get the directory where this conftest.py file is located
    test_dir = os.path.dirname(__file__)
    # Return the parent directory (the package root)
    return os.path.dirname(test_dir)


@pytest.fixture
def data_resolver(local_path):
    """
    Fixture that provides a function to resolve test data file paths.

    This function resolves data file paths to work from both
    the project root and the package directory.
    """

    def resolve_data_path(data_path):
        """
        Resolve a test data file path to work from current location.

        Parameters
        ----------
        data_path : str
            Path to the data file relative to local_path
            (e.g., "tests/data/ref_TQU_signal.dat")

        Returns
        -------
        str
            Absolute path to the data file
        """
        # Find the project root by looking for src directory
        current_dir = os.getcwd()
        path_parts = current_dir.split(os.sep)

        # Determine if we need to add package prefix for relative paths
        package_prefix = ""
        try:
            # If we find 'src' in the path, we might be running from project root
            path_parts.index("src")
            # If current dir doesn't end with package name, add prefix
            if not current_dir.endswith("cosmoforge.cosmocore"):
                package_prefix = "src/cosmoforge.cosmocore/"
        except ValueError:
            # No 'src' in path - check if we're in the package directory
            if not current_dir.endswith("cosmoforge.cosmocore"):
                # Assume we need the full path from wherever we are
                package_prefix = "src/cosmoforge.cosmocore/"

        # Add package prefix if needed
        if data_path.startswith("tests/"):
            resolved_path = package_prefix + data_path
        else:
            resolved_path = data_path

        # If it's not absolute, make it relative to local_path
        if not os.path.isabs(resolved_path):
            if resolved_path.startswith(package_prefix):
                # Remove the package prefix since we're going from local_path
                relative_path = resolved_path[len(package_prefix) :]
                resolved_path = os.path.join(local_path, relative_path)
            else:
                resolved_path = os.path.join(local_path, resolved_path)

        return os.path.abspath(resolved_path)

    return resolve_data_path
