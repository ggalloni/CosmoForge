QUBE Package
============

.. image:: /_static/logo_qube.png
   :alt: QUBE Logo
   :align: center
   :width: 40%

|

QUBE is the analysis engine of CosmoForge, implementing Fisher matrix analysis and
Quadratic Maximum Likelihood (QML) power spectrum estimation for any spin-0 or spin-2
field on the sphere. The package provides optimal statistical methods for power spectrum
recovery and parameter forecasting from noisy, partial-sky observations. While applicable
to a broad range of sky signals (e.g. CMB, galaxy surveys, 21 cm), its design is
particularly optimized for the case of complex noise covariance and incomplete sky coverage.

Overview
--------

QUBE provides two complementary analysis methods that together form a complete pipeline 
for CMB power spectrum analysis:

* **Fisher Matrix Analysis**: Parameter forecasting and optimal experiment design
* **QML Power Spectrum Estimation**: Unbiased, minimum-variance power spectrum recovery

Both methods are designed for large-scale analyses with full MPI parallelization support
and can handle realistic observational complexities including partial sky coverage,
inhomogeneous noise, and instrumental systematics.

Mathematical Foundation
-----------------------

Fisher Matrix Method
^^^^^^^^^^^^^^^^^^^^

The Fisher information matrix quantifies the information content of observations about 
model parameters:

.. math::

   F_{ij} = \left\langle \frac{\partial^2 \ln \mathcal{L}}{\partial \theta_i \partial \theta_j} \right\rangle = \frac{1}{2} \text{Tr}\left[ \mathbf{C}^{-1} \frac{\partial \mathbf{C}}{\partial \theta_i} \mathbf{C}^{-1} \frac{\partial \mathbf{C}}{\partial \theta_j} \right]

where :math:`\mathbf{C}` is the total covariance matrix and :math:`\theta_i` are the power
spectrum amplitudes (or other parameters of interest). The inverse Fisher matrix provides
the parameter covariance under Gaussian assumptions.

QML Power Spectrum Estimation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Quadratic Maximum Likelihood estimator provides optimal power spectrum estimates
[Tegmark1997]_ [Bond1998]_:

.. math::

   \hat{q}_\ell = \frac{1}{2} \mathbf{x}^T \mathbf{E}_\ell \mathbf{x}

where :math:`\mathbf{E}_\ell = \mathbf{C}^{-1} \frac{\partial \mathbf{C}}{\partial q_\ell} \mathbf{C}^{-1}`
is the quadratic estimator matrix [Oh1999]_. Final estimates are optimally combined using
the Fisher matrix. The method applies to any Gaussian field on the sphere, including but
not limited to CMB temperature and polarization.

Key Features
------------

Statistical Optimality
^^^^^^^^^^^^^^^^^^^^^^^

* **Minimum variance**: Achieves Cramér-Rao lower bound for parameter estimation
* **Unbiased estimation**: Proper treatment of noise bias in auto-correlation analyses
* **Optimal weighting**: Fisher matrix provides optimal combination of estimates
* **Exact likelihood**: No approximations in covariance matrix treatment
* **Normalization modes**: Three output normalizations (deconvolved, decorrelated, convolved)
  for different analysis workflows — see :doc:`qube/spectra` for details

Observational Realism
^^^^^^^^^^^^^^^^^^^^^

* **Partial sky coverage**: Exact treatment of sky cuts and masked regions
* **Inhomogeneous noise**: Pixel-dependent noise modeling
* **Beam convolution**: Instrumental beam effects in harmonic space
* **Cross-correlations**: Independent dataset combinations for systematics control
* **General applicability**: Any spin-0 or spin-2 Gaussian field (CMB, galaxy surveys, 21 cm, etc.)

Computational Efficiency
^^^^^^^^^^^^^^^^^^^^^^^^^

* **MPI parallelization**: Scalable computation across multiple processes
* **Memory optimization**: Efficient storage and manipulation of large matrices
* **HEALPix integration**: Native support for spherical pixelization schemes
* **Modular design**: Reusable components for different analysis workflows

Package Contents
----------------

Core Analysis Classes
^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 2

   qube/fisher
   qube/spectra
   qube/memory_budget

Main Execution Scripts
^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 2

   qube/main_fisher
   qube/main_qml

Utility Scripts
^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 2

   qube/produce_mock_inputs

Quick Reference
^^^^^^^^^^^^^^^

The package provides comprehensive documentation for each component with mathematical
foundations, computational details, and practical usage examples.

Usage Examples
--------------

Basic Fisher Matrix Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from qube import Fisher

   # Initialize Fisher matrix computation
   fisher = Fisher("config/analysis.yaml")

   # Run complete analysis pipeline
   fisher.run()

   # Get results (master process only)
   if fisher.rank == 0:
       F_matrix = fisher.get_fisher_matrix()
       errors = fisher.get_error_bars()
       print(f"Parameter constraints: {errors}")

QML Power Spectrum Estimation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from qube import Spectra

   # Initialize QML estimation
   spectra = Spectra("config/qml_analysis.yaml")

   # Run complete QML pipeline
   spectra.run()

   # Get power spectrum estimates (default: deconvolved)
   if spectra.rank == 0:
       power_spectra = spectra.get_power_spectra()
       noise_bias = spectra.get_noise_bias()

       # Other normalization modes
       cl_decorr = spectra.get_power_spectra(mode="decorrelated")
       y, W, convolve = spectra.get_power_spectra(mode="convolved")

Combined Fisher + QML Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from qube import Fisher, Spectra
   
   # Compute Fisher matrix first
   fisher = Fisher("config/fisher.yaml")
   fisher.run()
   
   # Reuse Fisher components for efficient QML
   spectra = Spectra("config/qml.yaml", fisher=fisher)
   spectra.run()
   
   # Analysis results available on master process
   if spectra.rank == 0:
       results = spectra.get_power_spectra()

MPI Parallel Execution
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Run Fisher matrix analysis with 8 processes
   mpirun -n 8 python main_fisher.py
   
   # Run QML estimation with 16 processes  
   mpirun -n 16 python main_qml.py

Mock Data Generation
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Generate synthetic CMB observations
   python produce_mock_inputs.py TQU /data/mocks config.yaml --mask

Performance Considerations
--------------------------

Memory Requirements
^^^^^^^^^^^^^^^^^^^

The primary memory bottleneck is covariance matrix storage, scaling as :math:`O(N_{pix}^2)`.
For typical analyses with three field components (e.g. T+Q+U):

* **nside=512**: ~16 GB
* **nside=1024**: ~250 GB
* **nside=2048**: ~4 TB

Computational Scaling
^^^^^^^^^^^^^^^^^^^^^^

* **Fisher matrix**: :math:`O(N_{ell}^2 \times N_{pix}^3)` 
* **QML estimation**: :math:`O(N_{ell} \times N_{pix}^3 + N_{ell} \times N_{pix}^2 \times N_{sims})`
* **Memory bandwidth**: Often the limiting factor for large matrices

Optimization Strategies
^^^^^^^^^^^^^^^^^^^^^^^

* Use appropriate HEALPix resolution for science goals
* Leverage MPI parallelization for compute-bound operations
* Consider iterative methods for very large covariance matrices
* Monitor memory usage and adjust process distribution accordingly

References
----------

.. [Tegmark1997] Tegmark, M. "How to measure CMB power spectra without losing information" 
   *Phys. Rev. D* **55**, 5895 (1997)

.. [Bond1998] Bond, J.R., Jaffe, A.H. & Knox, L. "Estimating the power spectrum of the 
   cosmic microwave background" *Phys. Rev. D* **57**, 2117 (1998)

.. [Oh1999] Oh, S.P., Spergel, D.N. & Hinshaw, G. "An efficient technique to determine
   the power spectrum from cosmic microwave background sky maps" 
   *Astrophys. J.* **510**, 551 (1999)

