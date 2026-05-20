Installation
============

CosmoForge is distributed as four PyPI packages that share a uv workspace in
this repository:

* **CosmoCore** (``cosmocore``) — shared algebra, fields, I/O, computation bases.
* **QUBE** (``qube``, PyPI name ``qube-qml``) — Fisher and QML estimation.
* **PICSLike** (``picslike``) — pixel-space likelihood.
* **CosmoForge** (``cosmoforge``) — umbrella metapackage that depends on the three above.

Requirements
------------

* Python 3.11–3.13

**cosmocore runtime dependencies** (installed automatically):

* ``numpy >= 2.2.6``
* ``scipy >= 1.16.1``
* ``healpy >= 1.18.1``
* ``numba >= 0.61.2``
* ``psutil >= 5.9``
* ``threadpoolctl >= 3.0``

**Optional extras:**

* ``cosmocore[mpi]`` — adds ``mpi4py >= 4.1.0`` for MPI parallel runs.
  ``qube-qml[mpi]``, ``picslike[mpi]``, ``cosmoforge[mpi]`` all pull this in transitively.
* ``qube-qml[pcl]`` — adds ``pymaster >= 2.6`` for the pseudo-Cl comparison utilities.

**Development tools** (uv dependency groups in the root ``pyproject.toml``):

* ``dev`` group: ``matplotlib``, ``pytest``, ``pytest-cov``, ``pyyaml``, ``ruff``.
* ``docs`` group: ``sphinx``, ``sphinx-rtd-theme``, ``myst-parser``,
  ``sphinx-autodoc-typehints``.
* ``pre-commit`` is currently only declared in
  ``src/cosmoforge.cosmocore/pyproject.toml``'s ``dev`` group.

Install from PyPI
-----------------

The umbrella distribution pulls in all three subpackages:

.. code-block:: bash

   pip install cosmoforge

Or pick subpackages individually:

.. code-block:: bash

   pip install cosmocore       # core utilities only
   pip install qube-qml        # adds QML / Fisher (imports as `qube`)
   pip install picslike        # adds pixel-space likelihood

Add the MPI extra if you intend to run in parallel:

.. code-block:: bash

   pip install "cosmoforge[mpi]"

Install from source (development)
---------------------------------

The repository is a `uv <https://docs.astral.sh/uv/>`_ workspace. Install
uv first, then:

.. code-block:: bash

   git clone https://github.com/ggalloni/CosmoForge.git
   cd CosmoForge
   uv sync --all-packages --all-extras --dev

This creates a ``.venv/`` and installs every workspace member in editable
mode together with all optional extras and the ``dev`` / ``docs`` groups.

Any subsequent Python command should be run through ``uv``:

.. code-block:: bash

   uv run python -c "import cosmocore"
   uv run pytest src/cosmoforge.cosmocore/tests/ -s
   uv run sphinx-build -b html docs/source docs/build/html

Verification
------------

After installation, confirm the imports succeed:

.. code-block:: python

   # cosmocore
   from cosmocore.settings import InputParams
   params = InputParams()
   print(f"cosmocore OK, nside={params.nside}")

   # qube
   try:
       from qube import Fisher, Spectra
       print("qube OK")
   except ImportError:
       print("qube not installed")

   # picslike
   try:
       from picslike import PICSLike
       print("picslike OK")
   except ImportError:
       print("picslike not installed")

The ``cosmoforge`` umbrella package itself exposes no module surface — it
only declares dependencies on the three packages above.
