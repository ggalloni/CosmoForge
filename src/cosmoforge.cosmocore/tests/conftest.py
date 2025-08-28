import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest


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
