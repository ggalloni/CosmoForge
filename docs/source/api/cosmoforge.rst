CosmoForge (umbrella metapackage)
=================================

**CosmoForge** (``cosmoforge``) is the umbrella PyPI distribution. It contains
no Python module surface of its own — installing it just pulls in the three
runtime packages:

* **CosmoCore** (:mod:`cosmocore`) — shared algebra, fields, computation bases, I/O.
* **QUBE** (:mod:`qube`, PyPI name ``qube-qml``) — Fisher and QML estimation.
* **PICSLike** (:mod:`picslike`) — pixel-space likelihood.

Install
-------

.. code-block:: bash

   pip install cosmoforge          # runtime
   pip install "cosmoforge[mpi]"   # adds mpi4py via cosmocore[mpi]

Source-install lives under the uv workspace — see :doc:`../installation`.

Note
----

The metapackage's source lives at ``src/cosmoforge.meta/`` inside the
repository (the directory name is historical); the distribution name on PyPI
is ``cosmoforge``. There is no importable ``cosmoforge`` Python module — the
public surface is exposed by the three subpackages above.
