Power Spectrum Estimation
==========================

.. currentmodule:: qube.spectra

The Spectra class implements Quadratic Maximum Likelihood (QML) estimation for
power spectrum recovery from partial-sky observations of spin-0 and spin-2 fields.
This method provides unbiased, minimum-variance power spectrum estimates while properly
accounting for noise, masking, and correlations. It applies to any Gaussian field on
the sphere, such as CMB temperature and polarization, galaxy density, or 21 cm intensity.

Mathematical Background
-----------------------

QML Theory
^^^^^^^^^^

QML power spectrum estimation maximizes the likelihood function for Gaussian
random fields. For each multipole :math:`\ell`, the QML estimator is:

.. math::

   \hat{C}_\ell = \frac{\mathbf{x}^T \mathbf{E}_\ell \mathbf{x}}{\text{Tr}[\mathbf{E}_\ell \mathbf{S}]}

where:
- :math:`\mathbf{x}` is the observed data vector
- :math:`\mathbf{E}_\ell = \mathbf{C}^{-1} \frac{\partial \mathbf{C}}{\partial C_\ell} \mathbf{C}^{-1}` is the estimator matrix
- :math:`\mathbf{S}` is the total covariance matrix
- :math:`\mathbf{C} = \mathbf{S} + \mathbf{N}` includes signal and noise

This estimator is unbiased: :math:`\langle \hat{C}_\ell \rangle = C_\ell` and
achieves the minimum variance bound given by the Fisher information.

Bias Correction
^^^^^^^^^^^^^^^

The normalized QML estimator includes bias correction for finite-size effects:

.. math::

   q_\ell = \frac{\mathbf{x}^T \mathbf{E}_\ell \mathbf{x}}{\text{Tr}[\mathbf{E}_\ell \mathbf{M}]}

where :math:`\mathbf{M}` is the appropriate normalization matrix that accounts
for the specific covariance structure.

Class Documentation
-------------------

.. autoclass:: Spectra
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Key Methods
-----------

Initialization
^^^^^^^^^^^^^^

.. automethod:: Spectra.__init__
   :no-index:
.. automethod:: Spectra.setup_geometry
   :no-index:

Core Computation
^^^^^^^^^^^^^^^^^

.. automethod:: Spectra.setup_maps
   :no-index:
.. automethod:: Spectra.setup_fisher_inversion
   :no-index:
.. automethod:: Spectra.setup_qml_computation
   :no-index:
.. automethod:: Spectra.compute_e_operator
   :no-index:
.. automethod:: Spectra.compute_qml_spectra
   :no-index:
.. automethod:: Spectra.compute
   :no-index:
.. automethod:: Spectra.run
   :no-index:

Results Processing
^^^^^^^^^^^^^^^^^^

.. automethod:: Spectra.get_power_spectra
   :no-index:
.. automethod:: Spectra.get_noise_bias
   :no-index:
.. automethod:: Spectra.get_covariance
   :no-index:
.. automethod:: Spectra.get_error_bars
   :no-index:
.. automethod:: Spectra.convolve_theory
   :no-index:

Normalization Modes
-------------------

Power spectrum estimates can be retrieved in three normalization modes, each suited
to different analysis needs. The raw QML estimates :math:`\mathbf{y}` are related to
the true power spectrum :math:`C_\ell` through the Fisher matrix :math:`\mathbf{F}`:

.. math::

   \langle \mathbf{y} \rangle = \mathbf{F} \, \mathbf{C}_\ell

The three modes differ in how :math:`\mathbf{y}` is post-processed before being
returned to the user.

.. list-table::
   :header-rows: 1
   :widths: 18 15 22 45

   * - Mode
     - Formula
     - Covariance
     - Use case
   * - ``"deconvolved"``
     - :math:`\mathbf{F}^{-1} \mathbf{y}`
     - :math:`\mathbf{F}^{-1}` (correlated)
     - Standard analysis: estimates of the true :math:`C_\ell`
   * - ``"decorrelated"``
     - :math:`\mathbf{F}^{-1/2} \mathbf{y}`
     - :math:`\mathbf{I}` (identity)
     - Independent error bars for visualization
   * - ``"convolved"``
     - :math:`\mathbf{y}`
     - :math:`\mathbf{F}`
     - Theory comparison via :math:`\langle \mathbf{y} \rangle = \mathbf{W} C_\ell^{\rm th}`

Deconvolved (default)
^^^^^^^^^^^^^^^^^^^^^^

The standard QML output. The window function is inverted so the returned estimates
are direct estimates of the true :math:`C_\ell`. Errors are correlated between
multipoles; use ``get_covariance()`` for the full covariance matrix.

.. code-block:: python

   cl = spectra.get_power_spectra()  # or mode="deconvolved"
   cov = spectra.get_covariance()
   errors = spectra.get_error_bars()

Decorrelated
^^^^^^^^^^^^^

Applies :math:`\mathbf{F}^{-1/2}` so that the output covariance is the identity
matrix. Error bars are unity by construction, making this mode useful for quick
visualization. Information content is preserved but redistributed.

.. code-block:: python

   cl_decorr = spectra.get_power_spectra(mode="decorrelated")
   errors = spectra.get_error_bars(mode="decorrelated")  # all 1.0

Convolved
^^^^^^^^^^

Returns the raw estimates together with the window matrix :math:`\mathbf{W}` and a
convenience function to convolve a theoretical spectrum. This avoids inverting a
potentially ill-conditioned Fisher matrix.

.. code-block:: python

   y, W, convolve = spectra.get_power_spectra(mode="convolved")
   cl_theory_convolved = convolve(cl_theory)
   # Compare: y ≈ cl_theory_convolved

Chi-squared equivalence
^^^^^^^^^^^^^^^^^^^^^^^^

All three modes yield the same chi-squared value for any given data realization:

.. math::

   \chi^2_{\rm deconv} = (\hat{C} - C^{\rm th})^T \mathbf{F} (\hat{C} - C^{\rm th})
   = (\mathbf{y} - \mathbf{W} C^{\rm th})^T \mathbf{F}^{-1} (\mathbf{y} - \mathbf{W} C^{\rm th})
   = \| \mathbf{q} - \mathbf{F}^{1/2} C^{\rm th} \|^2

Usage Examples
--------------

Basic QML Estimation
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from cosmoforge.qube import Spectra

   # Initialize QML analysis
   spectra = Spectra("config/qml_config.yaml")

   # Run complete estimation pipeline
   spectra.run()

   # Get power spectrum estimates (master process)
   if spectra.rank == 0:
       cl = spectra.get_power_spectra()
       errors = spectra.get_error_bars()

Cross-Correlation Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Configure for cross-correlation between two datasets
   config = {
       'do_cross': True,
       'covmatfile1': 'dataset1_noise.bin',
       'covmatfile2': 'dataset2_noise.bin',
       'mapfiles1': ['map1_T.fits', 'map1_Q.fits', 'map1_U.fits'],
       'mapfiles2': ['map2_T.fits', 'map2_Q.fits', 'map2_U.fits'],
   }

   spectra = Spectra(config)
   spectra.run()

MPI Parallel Execution
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Run QML estimation with 16 processes
   mpirun -n 16 python -c "
   from cosmoforge.qube import Spectra
   spectra = Spectra('qml_config.yaml')
   spectra.run()
   "

Error Analysis
^^^^^^^^^^^^^^

.. code-block:: python

   # Compute error estimates from Fisher matrix
   from cosmoforge.qube import Fisher, Spectra

   # First run Fisher analysis for error forecasting
   fisher = Fisher("fisher_config.yaml")
   fisher.run()

   # Then run QML estimation
   spectra = Spectra("qml_config.yaml")
   spectra.run()

   if spectra.rank == 0:
       # Compare with Fisher forecasts
       fisher_errors = fisher.get_error_bars()
       qml_estimates = spectra.get_power_spectra()

Computational Performance
-------------------------

Algorithm Complexity
^^^^^^^^^^^^^^^^^^^^^

The QML algorithm scales as:

* **Matrix inversion**: :math:`O(N_{pix}^3)` one-time cost
* **Estimator matrices**: :math:`O(N_{ell} \times N_{pix}^2)`
* **Quadratic forms**: :math:`O(N_{ell} \times N_{pix}^2)` per data realization

where :math:`N_{pix}` is the number of active pixels and :math:`N_{ell}` is the
number of multipole bins.

Memory Requirements
^^^^^^^^^^^^^^^^^^^

Peak memory usage is dominated by:

* Covariance matrix: :math:`N_{pix}^2 \times 8` bytes
* Estimator matrices: :math:`N_{ell} \times N_{pix}^2 \times 8` bytes
* Working arrays: Additional :math:`\sim 2 \times N_{pix}^2 \times 8` bytes

Optimization Strategies
^^^^^^^^^^^^^^^^^^^^^^^

For large-scale analyses:

1. **Hierarchical approach**: Start with low resolution for initial estimates
2. **Block processing**: Process multipole blocks separately to reduce memory
3. **Iterative methods**: Use conjugate gradient for matrix operations
4. **Preconditioning**: Improve convergence with appropriate preconditioners

Resource Scaling
^^^^^^^^^^^^^^^^

Typical requirements for full-mission analyses:

.. list-table::
   :header-rows: 1
   :widths: 15 20 20 25 20

   * - Resolution
     - Active Pixels
     - Memory (GB)
     - Compute Time
     - Cores
   * - nside=128
     - ~50k
     - ~2
     - ~30 min
     - 4-8
   * - nside=256
     - ~200k
     - ~8
     - ~3 hours
     - 8-16
   * - nside=512
     - ~800k
     - ~128
     - ~1 day
     - 16-32
   * - nside=1024
     - ~3.2M
     - ~2000
     - ~1 week
     - 32-64

Configuration Options
---------------------

Essential QML configuration parameters:

.. code-block:: yaml

   # Analysis type
   do_cross: false  # Auto-correlation (true for cross-correlation)

   # Input data
   mapfiles1: ["map_T.fits", "map_Q.fits", "map_U.fits"]
   covmatfile1: "noise_covariance.bin"
   maskfile: "analysis_mask.fits"

   # Multipole range
   lmin: 2
   lmax: 2000
   nbands: 50  # Number of bandpowers

   # Output files
   outfile: "qml_power_spectra.dat"
   outfilelhood: "likelihood_data.dat"

   # Computation options
   compute_fisher: true  # Include Fisher matrix computation
   save_estimators: false  # Save estimator matrices (large files)

Advanced Features
-----------------

Likelihood Analysis
^^^^^^^^^^^^^^^^^^^

The QML estimates come with full likelihood information for parameter fitting:

.. code-block:: python

   # Access likelihood data for cosmological parameter fitting
   spectra = Spectra("config.yaml")
   spectra.run()

   if spectra.rank == 0:
       # Get bandpowers and covariance for likelihood
       bandpowers = spectra.get_power_spectra()
       covariance = spectra.get_covariance()

       # Use in cosmological parameter fitting pipeline

Systematic Error Studies
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Test robustness to different analysis choices
   systematics = {
       'lmax_values': [1500, 2000, 2500, 3000],
       'mask_apodization': [0.0, 1.0, 2.0, 5.0],
       'multipole_binning': ['linear', 'logarithmic', 'hybrid']
   }

   for test_case in systematic_combinations:
       config = base_config.copy()
       config.update(test_case)

       spectra = Spectra(config)
       spectra.run()
       # Analyze systematic differences...

See Also
--------

* :class:`qube.fisher.Fisher` : Fisher information matrix computation
* :mod:`cosmocore.harmonic` : Spherical harmonic transformations
* :doc:`main_qml` : Main execution script for QML analysis
* :doc:`produce_mock_inputs` : Mock data generation for testing
