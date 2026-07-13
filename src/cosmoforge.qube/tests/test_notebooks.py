"""Execute the demo notebooks headless and assert they run disk-free.

Gated on the ``jupyter`` dependency group: ``importorskip`` makes a plain
``uv run pytest`` skip these, while a notebook CI job that installs the group
(``uv sync --group jupyter``) runs them. Each notebook is executed with its own
directory as cwd (their ``ROOT = getcwd()/..`` path logic) and must write no new
files anywhere under the package tree — the notebooks strip every ``out*`` path
so the in-memory pipeline leaves the working tree untouched (ADR-0015/0017).
"""

import os
import pathlib

import pytest

PKG_ROOT = pathlib.Path(__file__).resolve().parents[1]
NB_DIR = PKG_ROOT / "notebooks"
NOTEBOOKS = sorted(NB_DIR.glob("*.ipynb"))

_IGNORE = (".ipynb_checkpoints", "__pycache__")


def _snapshot(root: pathlib.Path) -> set[pathlib.Path]:
    return {
        p
        for p in root.rglob("*")
        if p.is_file() and not any(part in _IGNORE for part in p.parts)
    }


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_runs_disk_free(nb_path):
    pytest.importorskip("nbclient")
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(nb_path.parent)}},
    )

    before = _snapshot(PKG_ROOT)
    # The kernel is spawned as a subprocess and inherits this process's env.
    # Notebooks must run the way users run them (JIT on): if anything in the
    # test environment ever sets NUMBA_DISABLE_JIT, leaking it into the kernel
    # drops every Numba kernel to pure Python and inflates a ~15 s notebook to
    # ~5 min. Cheap insurance — this footgun has bitten once already.
    jit = os.environ.pop("NUMBA_DISABLE_JIT", None)
    try:
        client.execute()  # raises CellExecutionError on any failing cell
    finally:
        if jit is not None:
            os.environ["NUMBA_DISABLE_JIT"] = jit

    new_files = _snapshot(PKG_ROOT) - before
    assert not new_files, f"{nb_path.name} wrote files to disk: {sorted(new_files)}"
