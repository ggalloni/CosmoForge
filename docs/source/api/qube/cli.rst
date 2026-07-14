Command-Line Interface
======================

QUBE installs console scripts with the package, so they are on ``PATH`` after
``pip install qube-qml`` — no repository checkout required.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Command
     - Does
   * - ``qube-fisher-run``
     - Builds the Fisher matrix and stops.
   * - ``qube-spectra-run``
     - Estimates QML power spectra end to end.
   * - ``qube-run``
     - Alias for ``qube-spectra-run``.
   * - ``qube-memory-budget``
     - Predicts peak memory for a configuration without running it.

Both pipeline commands take an optional configuration path and fall back to the
packaged ``TEB_defaults.yaml``:

.. code-block:: bash

   qube-run                                  # packaged default config
   qube-run config.yaml                      # explicit config
   qube-spectra-run config.yaml --mode convolved --out spectra.txt
   qube-fisher-run config.yaml

Because a console script is an ordinary executable, MPI works without a
``python`` prefix:

.. code-block:: bash

   mpirun -n 8 qube-spectra-run config.yaml
   mpirun -n 16 qube-fisher-run config.yaml

Why two commands
----------------

``Spectra`` builds its own ``Fisher`` internally, so ``qube-run`` is the whole
pipeline in one step. ``qube-fisher-run`` exists for the cases where you want
**F** on its own:

* the two-job cluster workflow, where one job computes the Fisher matrix and a
  later job reuses it through the ``out*`` files (see :doc:`../../adr` on the
  Fisher → Spectra handoff);
* reusing a single Fisher across many map sets, since **F** does not depend on
  the data.

Persistence is opt-in
---------------------

Nothing is written unless you ask for it. ``qube-fisher-run`` persists the
matrix only when ``outfilefisher`` is set in the configuration, and
``qube-spectra-run`` writes the spectra only when ``--out`` is given. Both
commands warn on completion when the result they just computed is about to be
discarded.

.. automodule:: qube.cli
   :members:
   :undoc-members:
   :show-inheritance:
