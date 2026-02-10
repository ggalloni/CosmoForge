CosmoForge.PICSLike Package
============================

.. image:: /_static/logo_picslike.png
   :alt: PICSLike Logo
   :align: center
   :width: 40%

|

PICSLike is a pixel-based likelihood analysis package for cosmological parameter estimation.
It provides tools for computing the likelihood of CMB observations given theoretical predictions
directly in pixel space, offering an alternative to harmonic-space methods.

Overview
--------

PICSLike implements pixel-based likelihood analysis, which is particularly useful for:

* **Incomplete sky coverage**: Natural handling of masked regions without harmonic-space complications
* **Non-Gaussian features**: Direct treatment of non-Gaussian signals and systematics
* **Cross-correlation analysis**: Efficient computation of cross-correlations between different maps
* **Computational efficiency**: Potentially faster for certain analysis configurations

Mathematical Foundation
-----------------------

Pixel-Based Likelihood
^^^^^^^^^^^^^^^^^^^^^^

The pixel-based likelihood function is computed as:

.. math::

   \ln \mathcal{L}(\theta) = -\frac{1}{2} (\mathbf{d} - \mathbf{s}(\theta))^T \mathbf{C}^{-1} (\mathbf{d} - \mathbf{s}(\theta))

where:

- :math:`\mathbf{d}` is the observed data vector in pixel space
- :math:`\mathbf{s}(\theta)` is the theoretical signal prediction for parameters :math:`\theta`
- :math:`\mathbf{C} = \mathbf{S}(\theta) + \mathbf{N}` is the total covariance matrix

The chi-squared statistic is:

.. math::

   \chi^2(\theta) = (\mathbf{d} - \mathbf{s}(\theta))^T \mathbf{C}^{-1} (\mathbf{d} - \mathbf{s}(\theta))

Key Features
------------

Parameter Grid Evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^

* Systematic exploration of parameter space on a grid
* Automatic signal covariance computation for each parameter point
* MPI parallelization for efficient grid traversal
* Support for arbitrary number of cosmological parameters

Statistical Analysis
^^^^^^^^^^^^^^^^^^^^

* Chi-squared and log-likelihood computation at each grid point
* Best-fit parameter extraction from likelihood surface
* Confidence interval estimation via likelihood marginalization
* Support for multiple simulation realizations

High Performance Computing
^^^^^^^^^^^^^^^^^^^^^^^^^^

* MPI parallelization across parameter grid points
* Memory-optimized covariance matrix operations
* Integration with CosmoForge infrastructure

Package Contents
----------------

Core Analysis Classes
^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 2

   picslike/picslike
   picslike/parameter_grid
   picslike/likelihood_result

Quick Reference
^^^^^^^^^^^^^^^

The package provides comprehensive documentation for each component with mathematical
foundations, computational details, and practical usage examples.

Usage Examples
--------------

Basic Pixel-Based Likelihood Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from picslike import PICSLike

   # Initialize with configuration file
   picslike = PICSLike(params_file="config/pixel_analysis.yaml")

   # Run complete analysis pipeline
   picslike.run()

   # Get results (master process only)
   if picslike.rank == 0:
       chi2 = picslike.get_chi_squared()
       best_fit = picslike.get_best_fit()
       print(f"Best-fit parameters: {best_fit}")

Step-by-Step Pipeline
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from picslike import PICSLike

   # Initialize
   picslike = PICSLike(params_file="config/analysis.yaml")

   # Setup pipeline components
   picslike.setup_parameter_grid()
   picslike.setup_fields()
   picslike.setup_geometry()
   picslike.setup_covariance_matrices()
   picslike.setup_cls()
   picslike.setup_beams()
   picslike.setup_maps()

   # Compute likelihood across parameter grid
   picslike.compute()

   # Extract and save results
   if picslike.rank == 0:
       picslike.save_results("output/likelihood_results.npz")

MPI Parallel Execution
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Run with 8 MPI processes
   mpirun -n 8 python pixel_analysis.py config.yaml

Result Analysis
^^^^^^^^^^^^^^^

.. code-block:: python

   from picslike import LikelihoodResult

   # Load saved results
   result = LikelihoodResult.load("output/likelihood_results.npz")

   # Get confidence intervals
   intervals_68 = result.get_confidence_intervals(0.68)
   intervals_95 = result.get_confidence_intervals(0.95)

   # Marginalize over parameters
   marg_omega_b = result.get_marginalized_likelihood('omega_b')

   # Summary statistics
   summary = result.get_summary_statistics()

Performance Considerations
--------------------------

Memory Requirements
^^^^^^^^^^^^^^^^^^^

The primary memory bottleneck is covariance matrix storage, scaling as :math:`O(N_{pix}^2)`.
For typical analyses:

* **nside=256**: ~4 GB for temperature + polarization
* **nside=512**: ~16 GB for temperature + polarization
* **nside=1024**: ~250 GB for temperature + polarization

Computational Scaling
^^^^^^^^^^^^^^^^^^^^^

* **Likelihood evaluation**: :math:`O(N_{param} \times N_{pix}^3)` per parameter point
* **Matrix inversion**: Dominant computational cost
* **MPI parallelization**: Parameter grid points distributed across processes

Optimization Strategies
^^^^^^^^^^^^^^^^^^^^^^^

* Use appropriate HEALPix resolution (nside) for science goals
* Pre-compute theoretical spectra grids for efficiency
* Consider reduced covariance matrices for large-scale analyses
* Leverage MPI parallelization for compute-bound operations

Configuration
-------------

PICSLike uses YAML configuration files for analysis setup:

.. code-block:: yaml

   # HEALPix parameters
   nside: 512
   lmax: 1000

   # Field configuration
   spins: [0, 2]
   labels: ["T", "E", "B"]
   physical_labels: ["T", "Q", "U"]

   # Simulation settings
   nsims: 100

   # Parameter grid
   parameters:
     omega_b:
       min: 0.020
       max: 0.025
       n_points: 11
     omega_c:
       min: 0.10
       max: 0.14
       n_points: 11

   # Input files
   mapsfile1: "data/observed_maps.fits"
   covmatfile1: "data/noise_covariance.bin"
   clfile: "theory/theoretical_spectra.txt"
   beamfile: "data/beam_transfer.fits"

Integration with CosmoForge
---------------------------

PICSLike seamlessly integrates with other CosmoForge packages:

- **cosmocore**: Provides base functionality, field management, and mathematical utilities
- **qube**: Complementary QML power spectrum analysis for comparison
- **meta**: Workflow management and configuration utilities

References
----------

.. [Wandelt2004] Wandelt, B.D., Larson, D.L. & Lakshminarayanan, A. "Global, exact cosmic
   microwave background data analysis using Gibbs sampling"
   *Phys. Rev. D* **70**, 083511 (2004)

.. [Jewell2004] Jewell, J., Levin, S. & Anderson, C.H. "Application of Monte Carlo algorithms
   to the Bayesian analysis of the cosmic microwave background"
   *Astrophys. J.* **609**, 1-14 (2004)

.. [Eriksen2004] Eriksen, H.K. et al. "Power Spectrum Estimation from High-Resolution Maps by
   Gibbs Sampling" *Astrophys. J. Suppl.* **155**, 227-241 (2004)

.. [Planck2020] Planck Collaboration "Planck 2018 results. V. CMB power spectra and likelihoods"
   *Astron. Astrophys.* **641**, A5 (2020)
