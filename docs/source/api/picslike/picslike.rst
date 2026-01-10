PICSLike Class
==============

.. currentmodule:: picslike.picslike

The PICSLike class provides comprehensive pixel-based likelihood computation for
cosmological parameter estimation. The implementation features MPI parallelization,
exact covariance matrix treatment, and integration with the broader CosmoForge
analysis pipeline.

Mathematical Background
-----------------------

The pixel-based likelihood evaluates how well theoretical predictions match observed
data in pixel space:

.. math::

   \ln \mathcal{L}(\theta) = -\frac{1}{2} \chi^2(\theta) = -\frac{1}{2} (\mathbf{d} - \mathbf{s}(\theta))^T \mathbf{C}^{-1}(\theta) (\mathbf{d} - \mathbf{s}(\theta))

where:

- :math:`\mathbf{d}` is the observed data vector
- :math:`\mathbf{s}(\theta)` is the theoretical signal for parameters :math:`\theta`
- :math:`\mathbf{C}(\theta) = \mathbf{S}(\theta) + \mathbf{N}` is the total covariance matrix

Class Documentation
-------------------

.. autoclass:: PICSLike
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Key Methods
-----------

Initialization and Setup
^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: PICSLike.__init__
   :no-index:
.. automethod:: PICSLike.setup_parameter_grid
   :no-index:
.. automethod:: PICSLike.setup_maps
   :no-index:

Core Computation
^^^^^^^^^^^^^^^^

.. automethod:: PICSLike.compute_signal_matrix
   :no-index:
.. automethod:: PICSLike.prepare_covariance_matrix
   :no-index:
.. automethod:: PICSLike.compute
   :no-index:
.. automethod:: PICSLike.run
   :no-index:

Results Retrieval
^^^^^^^^^^^^^^^^^

.. automethod:: PICSLike.get_chi_squared
   :no-index:
.. automethod:: PICSLike.get_log_likelihood
   :no-index:
.. automethod:: PICSLike.get_best_fit
   :no-index:
.. automethod:: PICSLike.get_simulation_results
   :no-index:
.. automethod:: PICSLike.get_mean_likelihood_result
   :no-index:
.. automethod:: PICSLike.save_results
   :no-index:

Simulation Management
^^^^^^^^^^^^^^^^^^^^^

.. automethod:: PICSLike.set_simulation_index
   :no-index:

Usage Examples
--------------

Basic Likelihood Computation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from picslike import PICSLike

   # Initialize with configuration file
   picslike = PICSLike(params_file="config/analysis.yaml")

   # Run complete computation pipeline
   picslike.run()

   # Get results (master process only)
   if picslike.rank == 0:
       chi2 = picslike.get_chi_squared()
       log_like = picslike.get_log_likelihood()
       best_fit = picslike.get_best_fit()

       print(f"Chi-squared at best-fit: {chi2.min():.2f}")
       print(f"Best-fit parameters: {best_fit}")

Step-by-Step Setup
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from picslike import PICSLike

   # Initialize
   picslike = PICSLike(params_file="config/analysis.yaml")

   # Setup components in order
   picslike.setup_parameter_grid()
   picslike.setup_fields()
   picslike.setup_geometry()
   picslike.setup_covariance_matrices()
   picslike.setup_cls()
   picslike.setup_beams()
   picslike.setup_maps()

   # Run likelihood computation
   picslike.compute()

   # Save results
   if picslike.rank == 0:
       picslike.save_results("output/results.npz")

Multiple Simulations
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from picslike import PICSLike

   picslike = PICSLike(params_file="config/analysis.yaml")
   picslike.run()

   # Access individual simulation results
   if picslike.rank == 0:
       sim_results = picslike.get_simulation_results()
       for i, result in enumerate(sim_results):
           print(f"Sim {i}: chi2_min = {result.get_chi_squared_minimum():.2f}")

       # Get mean across simulations
       mean_result = picslike.get_mean_likelihood_result()

MPI Parallel Execution
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Run with 8 MPI processes
   mpirun -n 8 python -c "
   from picslike import PICSLike
   picslike = PICSLike(params_file='config.yaml')
   picslike.run()
   "

Computational Notes
-------------------

Performance Characteristics
^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Scaling**: :math:`O(N_{param} \times N_{pix}^3)` where :math:`N_{param}` is the number
  of parameter grid points and :math:`N_{pix}` is the number of active pixels
* **Memory**: Dominated by covariance matrix storage :math:`O(N_{pix}^2)`
* **Parallelization**: Parameter grid points distributed across MPI processes

Pipeline Order
^^^^^^^^^^^^^^

The setup methods must be called in the correct order:

1. ``setup_parameter_grid()`` - Define parameter space
2. ``setup_fields()`` - Configure field types (T, Q, U)
3. ``setup_geometry()`` - Set up pixel geometry and masks
4. ``setup_covariance_matrices()`` - Load noise covariance
5. ``setup_cls()`` - Load theoretical power spectra
6. ``setup_beams()`` - Load beam transfer functions
7. ``setup_maps()`` - Load observed maps
8. ``compute()`` - Run likelihood computation

Alternatively, ``run()`` executes all steps automatically.

See Also
--------

* :class:`picslike.parameter_grid.ParameterGrid` : Parameter space management
* :class:`picslike.likelihood_result.LikelihoodResult` : Results container
* :mod:`cosmocore` : Underlying computational infrastructure
