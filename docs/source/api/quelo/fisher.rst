Fisher Matrix Analysis
======================

.. currentmodule:: quelo.fisher

The Fisher class provides comprehensive Fisher information matrix computation for 
cosmological parameter forecasting and optimal experiment design. The implementation 
features MPI parallelization, exact covariance matrix treatment, and integration 
with the broader CosmoForge analysis pipeline.

Mathematical Background
-----------------------

The Fisher information matrix quantifies the information content of observations 
about model parameters. For CMB observations, it is computed as:

.. math::

   F_{ij} = \frac{1}{2} \text{Tr}\left[ \mathbf{C}^{-1} \frac{\partial \mathbf{C}}{\partial \theta_i} \mathbf{C}^{-1} \frac{\partial \mathbf{C}}{\partial \theta_j} \right]

where:
- :math:`\mathbf{C} = \mathbf{S} + \mathbf{N}` is the total covariance matrix
- :math:`\mathbf{S}` is the signal covariance from theoretical power spectra  
- :math:`\mathbf{N}` is the instrumental noise covariance
- :math:`\theta_i` are the cosmological parameters of interest

The inverse Fisher matrix provides the parameter covariance matrix under Gaussian 
posterior assumptions: :math:`\text{Cov}(\theta_i, \theta_j) = (F^{-1})_{ij}`.

Class Documentation
-------------------

.. autoclass:: Fisher
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:

Key Methods
-----------

Initialization and Setup
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: Fisher.__init__
   :no-index:
.. automethod:: Fisher.setup_signal_matrix
   :no-index:
.. automethod:: Fisher.prepare_covariance_matrices
   :no-index:
.. automethod:: Fisher.setup_fisher_matrices
   :no-index:

Core Computation
^^^^^^^^^^^^^^^^^

.. automethod:: Fisher.compute_fisher_element
   :no-index:
.. automethod:: Fisher.compute
   :no-index:
.. automethod:: Fisher.run
   :no-index:

Results Retrieval
^^^^^^^^^^^^^^^^^^

.. automethod:: Fisher.get_fisher_matrix
   :no-index:
.. automethod:: Fisher.get_parameter_errors
   :no-index:

Usage Examples
--------------

Basic Fisher Matrix Computation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from cosmoforge.quelo import Fisher
   
   # Initialize with configuration file
   fisher = Fisher("config/fisher_analysis.yaml")
   
   # Run complete computation pipeline
   fisher.run()
   
   # Get results (master process only)
   if fisher.rank == 0:
       F_matrix = fisher.get_fisher_matrix()
       param_errors = fisher.get_parameter_errors()
       
       print(f"Fisher matrix shape: {F_matrix.shape}")
       print(f"Parameter errors: {param_errors}")

MPI Parallel Execution
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Run with 8 MPI processes
   mpirun -n 8 python -c "
   from cosmoforge.quelo import Fisher
   fisher = Fisher('config.yaml')
   fisher.run()
   "

Parameter Forecasting
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Compute parameter constraints for different survey configurations
   surveys = ['current', 'near_future', 'ultimate']
   
   for survey in surveys:
       fisher = Fisher(f"config/{survey}_config.yaml")
       fisher.run()
       
       if fisher.rank == 0:
           errors = fisher.get_parameter_errors()
           # Analyze parameter constraints...

Computational Notes
-------------------

Performance Characteristics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Scaling**: :math:`O(N_{ell}^2 \times N_{pix}^3)` where :math:`N_{ell}` is the number 
  of multipole parameters and :math:`N_{pix}` is the number of active pixels
* **Memory**: Dominated by covariance matrix storage :math:`O(N_{pix}^2)`
* **Parallelization**: Fisher matrix elements distributed across MPI processes

Optimization Strategies
^^^^^^^^^^^^^^^^^^^^^^^

* Use appropriate multipole range (lmin, lmax) for target precision
* Leverage sky cuts to reduce active pixel count
* Distribute computation across multiple nodes for large analyses
* Monitor memory usage for high-resolution pixelizations

Typical Resource Requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For temperature + polarization analysis:

.. list-table::
   :header-rows: 1
   :widths: 15 20 20 25 20

   * - HEALPix nside
     - Active Pixels
     - Memory (GB)
     - Compute Time
     - Recommended Cores
   * - 256
     - ~200k
     - ~4
     - ~1 hour
     - 4-8
   * - 512  
     - ~800k
     - ~16
     - ~8 hours
     - 8-16
   * - 1024
     - ~3.2M
     - ~250
     - ~2 days
     - 16-32
   * - 2048
     - ~12.8M
     - ~4000
     - ~2 weeks
     - 32-64

Configuration Parameters
------------------------

Key YAML configuration parameters for Fisher analysis:

.. code-block:: yaml

   # HEALPix parameters
   nside: 512
   lmin: 2
   lmax: 3000
   
   # Field configuration
   nfields: 3  # T, Q, U
   physical_labels: ["T", "Q", "U"]
   
   # Analysis mode
   do_cross: false  # Auto-correlation analysis
   
   # Input files
   covmatfile1: "noise_cov_primary.bin"
   clfile: "fiducial_spectra.txt"
   beamfile: "beam_transfer.fits"
   
   # Output files  
   outfilefisher: "fisher_matrix.dat"
   outinvcovmatfile1: "inv_cov_primary.bin"

See Also
--------

* :class:`quelo.spectra.Spectra` : QML power spectrum estimation using Fisher results
* :mod:`cosmocore` : Underlying computational infrastructure
* :doc:`main_fisher` : Main execution script for Fisher analysis
