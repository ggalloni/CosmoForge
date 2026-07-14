Command-Line Interface
======================

PICSLike installs a single console script, ``picslike-run``, which evaluates the
pixel-space Gaussian log-likelihood over the parameter grid declared in the
configuration. It is on ``PATH`` after ``pip install picslike`` — no repository
checkout required.

.. code-block:: bash

   picslike-run config.yaml
   picslike-run config.yaml --out results.npz

Grid points are partitioned across MPI ranks, and because a console script is an
ordinary executable, MPI works without a ``python`` prefix:

.. code-block:: bash

   mpirun -n 8 picslike-run config.yaml --out results.npz

Unlike QUBE there is no packaged default configuration, so the path is required.

Persistence is opt-in: without ``--out`` the grid is evaluated and the result
discarded. The command warns on completion when that is about to happen.

.. automodule:: picslike.cli
   :members:
   :undoc-members:
   :show-inheritance:
