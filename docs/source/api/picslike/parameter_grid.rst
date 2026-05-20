ParameterGrid Class
===================

.. currentmodule:: picslike.parameter_grid

The ParameterGrid class manages the parameter space for likelihood evaluation,
including grid generation, theoretical spectrum storage, and MPI work distribution.

Overview
--------

ParameterGrid provides:

* Cartesian product grid generation from parameter ranges
* Mapping between grid indices and parameter values
* Storage and retrieval of theoretical spectra for each grid point
* MPI-aware work distribution across processes

Class Documentation
-------------------

.. autoclass:: ParameterGrid
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Key Methods
-----------

Initialization
^^^^^^^^^^^^^^

.. automethod:: ParameterGrid.__init__
   :no-index:

Grid Navigation
^^^^^^^^^^^^^^^

.. automethod:: ParameterGrid.get_total_points
   :no-index:
.. automethod:: ParameterGrid.get_points_for_process
   :no-index:

Spectrum Access
^^^^^^^^^^^^^^^

.. automethod:: ParameterGrid.get_spectrum
   :no-index:

Usage Examples
--------------

Creating a Parameter Grid
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from picslike import ParameterGrid
   from cosmocore import InputParams

   # Define core parameters
   core_params = InputParams()
   core_params.update({
       "nside": 512,
       "lmax": 1000,
       "spins": [0, 2],
       "labels": ["T", "E", "B"],
   })

   # Define parameter ranges
   param_ranges = {
       "omega_b": np.linspace(0.020, 0.025, 11),
       "omega_c": np.linspace(0.10, 0.14, 11),
   }

   # Pre-compute theoretical spectra for each parameter combination
   theoretical_spectra = {}
   for omega_b in param_ranges["omega_b"]:
       for omega_c in param_ranges["omega_c"]:
           # Compute spectra using your Boltzmann code
           spectra = compute_spectra(omega_b, omega_c)
           theoretical_spectra[(omega_b, omega_c)] = spectra

   # Create parameter grid
   grid = ParameterGrid(
       core_params=core_params,
       parameter_ranges=param_ranges,
       theoretical_spectra=theoretical_spectra,
   )

   print(f"Total grid points: {grid.get_total_points()}")

MPI Work Distribution
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from mpi4py import MPI

   comm = MPI.COMM_WORLD
   rank = comm.Get_rank()
   size = comm.Get_size()

   # Get points assigned to this process
   my_points = grid.get_points_for_process(rank, size)

   print(f"Process {rank}: handling {len(my_points)} parameter points")

   for point in my_points:
       # Get theoretical spectra for this point
       spectra = grid.get_spectrum(point)
       # Compute likelihood at this point...

Iterating Over Grid Points
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Iterate over all grid points
   for point in grid.grid_points:
       omega_b, omega_c = point
       spectra = grid.get_spectrum(point)
       print(f"Point ({omega_b:.4f}, {omega_c:.4f}): {len(spectra)} spectra")

Grid Properties
---------------

The ParameterGrid stores:

* **parameter_ranges**: Dictionary mapping parameter names to value arrays
* **grid_points**: Set of all parameter combinations as tuples
* **theoretical_spectra**: Dictionary mapping parameter tuples to spectrum dictionaries

Spectrum Format
^^^^^^^^^^^^^^^

Theoretical spectra should be provided as dictionaries with keys for each
spectrum type:

.. code-block:: python

   spectra = {
       "TT": cl_tt,  # Temperature auto-spectrum
       "EE": cl_ee,  # E-mode auto-spectrum
       "BB": cl_bb,  # B-mode auto-spectrum
       "TE": cl_te,  # Temperature-E cross-spectrum
       "TB": cl_tb,  # Temperature-B cross-spectrum (usually zero)
       "EB": cl_eb,  # E-B cross-spectrum (usually zero)
   }

Each spectrum should be a 1D numpy array indexed by multipole :math:`\ell`.

See Also
--------

* :class:`picslike.picslike.PICSLike` : Main likelihood computation class
* :class:`picslike.likelihood_result.LikelihoodResult` : Results container
