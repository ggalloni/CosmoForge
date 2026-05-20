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
uv first, then clone and set up the minimal dev environment — every
workspace member installed editable, plus lint/test tooling:

.. code-block:: bash

   git clone https://github.com/ggalloni/CosmoForge.git
   cd CosmoForge
   uv sync --all-packages --dev

Any subsequent Python command should be run through ``uv``:

.. code-block:: bash

   uv run python -c "import cosmocore"
   uv run pytest src/cosmoforge.cosmocore/tests/ -s

The optional groups below are **opt-in** — add them only when you need
them. ``--dev`` always pulls in the root ``dev`` group; ``--group docs``
adds Sphinx + theme; ``--extra mpi`` / ``--extra pcl`` add runtime
extras on the workspace packages.

MPI parallel runs
^^^^^^^^^^^^^^^^^

Adds ``mpi4py``. Requires an MPI implementation (OpenMPI or MPICH) on
the system before running the sync.

.. code-block:: bash

   uv sync --all-packages --extra mpi --dev

Pseudo-Cl comparison (``pymaster``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``pcl`` extra pulls ``pymaster``, which is a C extension and needs
``healpix``, ``cfitsio``, and ``fftw`` installed at the system level
before any ``pip``/``uv`` install can succeed.

System packages first (pick the route that matches your OS):

.. code-block:: bash

   # Debian/Ubuntu
   sudo apt-get install libcfitsio-dev libhealpix-cxx-dev libfftw3-dev

   # macOS (Homebrew)
   brew install cfitsio healpix fftw

Then:

.. code-block:: bash

   uv sync --all-packages --extra pcl --dev

Or, if you'd rather sidestep system packages entirely, conda-forge ships
a pre-built ``pymaster`` with its C dependencies baked in:

.. code-block:: bash

   conda install -c conda-forge pymaster
   # then the regular uv sync inside the same env
   uv sync --all-packages --dev

Building docs locally
^^^^^^^^^^^^^^^^^^^^^

Docs are built and hosted on Read the Docs — both ``master`` and every
PR get an auto-build. Only install the ``docs`` group if you're
iterating on the documentation itself:

.. code-block:: bash

   uv sync --all-packages --group docs --dev
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
