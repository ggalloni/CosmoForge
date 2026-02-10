Installation
============

CosmoForge is organized as a namespace package with three main subpackages that can be
installed independently or together.

Requirements
------------

**Core Requirements (all packages):**

* Python 3.11+

**CosmoCore Requirements:**

* NumPy >= 2.2.6
* SciPy >= 1.16.1  
* HEALPy >= 1.18.1
* Numba >= 0.61.2
* mpi4py >= 4.1.0

**Development Requirements:**

* matplotlib >= 3.10.5
* pytest
* ruff (for linting)

Installation Options
--------------------

Complete Installation
^^^^^^^^^^^^^^^^^^^^^

Install the entire CosmoForge framework:

.. code-block:: bash

   git clone https://github.com/ggalloni/CosmoForge.git
   cd CosmoForge
   pip install -e .

This installs all subpackages: cosmocore, qube, and meta.

Individual Package Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Install specific subpackages as needed:

.. code-block:: bash

   # Install only CosmoCore
   cd CosmoForge/src/cosmoforge.cosmocore
   pip install -e .

   # Install only QUBE
   cd CosmoForge/src/cosmoforge.qube
   pip install -e .
   
   # Install only Meta
   cd CosmoForge/src/cosmoforge.meta
   pip install -e .

Development Installation
------------------------

For development work with all packages:

.. code-block:: bash

   git clone https://github.com/ggalloni/CosmoForge.git
   cd CosmoForge
   pip install -e ".[dev]"

This includes testing, documentation, and code quality tools.

Verification
------------

Verify your installation:

.. code-block:: python

   # Test CosmoCore
   from cosmoforge.cosmocore.cosmocore.settings import InputParams
   params = InputParams()
   print(f"CosmoCore installed! nside={params.nside}")

   # Test QUBE (if installed)
   try:
       from cosmoforge.qube import Fisher
       print("QUBE installed successfully!")
   except ImportError:
       print("QUBE not installed")
   
   # Test Meta (if installed)
   try:
       import cosmoforge.meta
       print("Meta installed successfully!")
   except ImportError:
       print("Meta not installed")
