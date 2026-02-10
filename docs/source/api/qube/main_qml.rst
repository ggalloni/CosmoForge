QML Power Spectrum Estimation Pipeline
======================================

.. currentmodule:: cosmoforge.qube

The QML (Quadratic Maximum Likelihood) power spectrum estimation pipeline provides 
unbiased, minimum-variance estimates of CMB power spectra from observed data. This 
script orchestrates the complete analysis workflow from data input through 
statistical inference.

Overview
--------

The ``main_qml.py`` script implements a comprehensive QML estimation pipeline 
optimized for CMB power spectrum analysis. Key features include:

* Unbiased power spectrum estimation with exact error propagation
* Support for auto-correlation and cross-correlation analyses
* MPI parallelization for computational efficiency
* Integration with Fisher information matrix results
* Comprehensive likelihood data for cosmological parameter fitting

The pipeline handles both simulated and observed CMB data, providing production-ready 
power spectrum estimates suitable for cosmological inference.

Mathematical Background
-----------------------

QML Estimator Theory
^^^^^^^^^^^^^^^^^^^^

For each multipole :math:`\ell`, the QML estimator provides the minimum-variance 
unbiased estimate:

.. math::

   \hat{C}_\ell = \frac{\mathbf{x}^T \mathbf{E}_\ell \mathbf{x}}{\text{Tr}[\mathbf{E}_\ell \mathbf{S}]}

where:
- :math:`\mathbf{x}` is the observed data vector (temperature and polarization maps)
- :math:`\mathbf{E}_\ell = \mathbf{C}^{-1} \frac{\partial \mathbf{C}}{\partial C_\ell} \mathbf{C}^{-1}` is the estimator matrix
- :math:`\mathbf{C}` is the total covariance matrix (signal + noise + systematics)
- :math:`\mathbf{S}` is the signal covariance matrix

Statistical Properties
^^^^^^^^^^^^^^^^^^^^^^

The QML estimator achieves optimal statistical properties:

* **Unbiasedness**: :math:`\langle \hat{C}_\ell \rangle = C_\ell` for all multipoles
* **Minimum Variance**: Saturates the Cramér-Rao bound given by Fisher information
* **Gaussianity**: Asymptotically Gaussian for large sky coverage
* **Correlation Structure**: Full covariance matrix available for likelihood analysis

Usage
-----

Command Line Interface
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Single-process execution (small analyses)
   python main_qml.py

   # MPI parallel execution (recommended)
   mpirun -n 16 python main_qml.py

   # With custom configuration file
   python main_qml.py --config qml_analysis.yaml

Script Workflow
^^^^^^^^^^^^^^^

The execution pipeline consists of the following stages:

1. **Environment Initialization**: Set up MPI communicator and process management
2. **Configuration Parsing**: Load and validate YAML configuration parameters
3. **Data Loading**: Read CMB maps, noise covariance, and auxiliary data files
4. **Geometry Setup**: Define pixel selection, multipole binning, and field combinations
5. **Matrix Preparation**: Compute or load estimator and normalization matrices
6. **QML Computation**: Execute parallel power spectrum estimation
7. **Result Gathering**: Collect and synchronize estimates across MPI processes
8. **Output Generation**: Save power spectra, covariance matrices, and likelihood data
9. **Validation**: Perform sanity checks and quality assessment

Configuration Format
--------------------

The script requires a YAML configuration file specifying analysis parameters:

.. code-block:: yaml

   # QML Analysis Configuration
   
   # Data specification
   mapfiles1: ["map_T.fits", "map_Q.fits", "map_U.fits"]
   covmatfile1: "noise_covariance.bin"
   maskfile: "analysis_mask.fits"
   
   # Cross-correlation (optional)
   do_cross: false
   mapfiles2: ["map2_T.fits", "map2_Q.fits", "map2_U.fits"]  # if do_cross=true
   covmatfile2: "noise_covariance2.bin"  # if do_cross=true
   
   # Analysis parameters
   nside: 512
   lmin: 2
   lmax: 2000
   nbands: 40  # Number of bandpowers
   nfields: 3  # T, Q, U
   physical_labels: ["T", "Q", "U"]
   
   # Fiducial model
   clfile: "fiducial_power_spectra.txt"
   beamfile: "beam_transfer_function.fits"
   
   # Output files
   outfile: "qml_power_spectra.dat"
   outfilelhood: "likelihood_data.dat"
   outfilecov: "bandpower_covariance.dat"
   
   # Computation options
   compute_fisher: true  # Include Fisher information
   save_intermediate: false  # Save estimator matrices (large files)

Implementation Features
-----------------------

Core Algorithms
^^^^^^^^^^^^^^^

The pipeline implements several sophisticated algorithms:

* **Iterative Matrix Methods**: Conjugate gradient for large matrix operations
* **Bandpower Binning**: Optimal multipole averaging for noise reduction  
* **Cross-Spectrum Handling**: Proper treatment of cross-correlation estimates
* **Numerical Stability**: Regularization and conditioning for robust computation

Performance Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^

Key optimization strategies include:

* **Distributed Memory**: MPI parallelization across estimator computation
* **Memory Management**: Efficient handling of large covariance matrices
* **I/O Optimization**: Parallel file operations and memory mapping
* **Computational Kernels**: Optimized linear algebra operations (BLAS/LAPACK)

Quality Control
^^^^^^^^^^^^^^^

The pipeline includes comprehensive validation:

* **Input Verification**: File format, dimensions, and consistency checking
* **Numerical Stability**: Matrix conditioning and eigenvalue analysis  
* **Statistical Tests**: Unbiasedness checks and chi-squared validation
* **Convergence Monitoring**: Iterative solver diagnostics

Output Products
---------------

Primary Outputs
^^^^^^^^^^^^^^^

**Power Spectrum File** (``qml_power_spectra.dat``):

.. code-block:: text

   # QML Power Spectrum Estimates
   # Column 1: Effective multipole
   # Column 2: TT power (μK²)
   # Column 3: EE power (μK²) 
   # Column 4: BB power (μK²)
   # Column 5: TE cross-power (μK²)
   # Column 6: TB cross-power (μK²)
   # Column 7: EB cross-power (μK²)
   2.5    5247.3    4.2    0.015    -142.1    0.002    0.001
   7.5    4923.8    12.8   0.023    -129.4    0.003    0.002
   ...

**Likelihood Data** (``likelihood_data.dat``):

Contains full statistical information for cosmological parameter fitting:
- Bandpower central values and uncertainties
- Complete covariance matrix between bandpowers
- Window functions and effective multipoles
- Calibration and beam uncertainties

**Covariance Matrix** (``bandpower_covariance.dat``):

Full covariance matrix enabling proper likelihood construction for parameter fitting.

Diagnostic Outputs
^^^^^^^^^^^^^^^^^^

**Analysis Log**: Detailed execution information including:
- Timing and memory usage statistics
- Convergence diagnostics for iterative methods
- Quality metrics and validation results
- Parameter summary and configuration record

**Intermediate Products** (optional):
- Estimator matrices for reanalysis
- Normalization factors and window functions
- Pixel weights and effective areas

Examples
--------

Standard Auto-Correlation Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   """
   Example: QML power spectrum estimation from CMB maps
   """
   import subprocess
   import yaml
   
   # Configure standard analysis
   config = {
       'mapfiles1': ['planck_T.fits', 'planck_Q.fits', 'planck_U.fits'],
       'covmatfile1': 'planck_noise_cov.bin',
       'maskfile': 'planck_analysis_mask.fits',
       'nside': 1024,
       'lmin': 2,
       'lmax': 2500,
       'nbands': 50,
       'clfile': 'planck2018_fiducial.txt',
       'beamfile': 'planck_beam.fits',
       'outfile': 'planck_qml_spectra.dat',
       'compute_fisher': True
   }
   
   # Save and run
   with open('qml_config.yaml', 'w') as f:
       yaml.dump(config, f)
   
   subprocess.run(['mpirun', '-n', '16', 'python', 'main_qml.py', 
                  '--config', 'qml_config.yaml'])

Cross-Correlation Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   """
   Example: Cross-correlation between two CMB datasets
   """
   
   # Configure cross-correlation
   cross_config = {
       'do_cross': True,
       'mapfiles1': ['dataset1_T.fits', 'dataset1_Q.fits', 'dataset1_U.fits'],
       'mapfiles2': ['dataset2_T.fits', 'dataset2_Q.fits', 'dataset2_U.fits'],
       'covmatfile1': 'dataset1_noise.bin',
       'covmatfile2': 'dataset2_noise.bin',
       'maskfile': 'common_mask.fits',
       'nside': 512,
       'lmin': 2,
       'lmax': 2000,
       'nbands': 30,
       'outfile': 'cross_correlation_spectra.dat'
   }

Multi-Scale Analysis Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Progressive resolution analysis
   
   # Low-resolution initial estimate
   python main_qml.py --config configs/qml_nside256.yaml
   
   # Medium resolution
   mpirun -n 8 python main_qml.py --config configs/qml_nside512.yaml
   
   # High resolution (production)
   mpirun -n 32 python main_qml.py --config configs/qml_nside1024.yaml

Systematic Error Studies
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   """
   Example: Impact of analysis choices on power spectrum estimates
   """
   
   # Test different multipole ranges
   lmax_values = [1500, 2000, 2500, 3000]
   
   for lmax in lmax_values:
       config = base_config.copy()
       config['lmax'] = lmax
       config['outfile'] = f'qml_spectra_lmax{lmax}.dat'
       
       with open(f'config_lmax{lmax}.yaml', 'w') as f:
           yaml.dump(config, f)
       
       subprocess.run(['mpirun', '-n', '16', 'python', 'main_qml.py',
                      '--config', f'config_lmax{lmax}.yaml'])

Performance Guidelines
----------------------

Computational Scaling
^^^^^^^^^^^^^^^^^^^^^^

QML estimation scales with system size as:

* **Memory**: :math:`O(N_{ell} \times N_{pix}^2)` for estimator matrices
* **Computation**: :math:`O(N_{ell} \times N_{pix}^2)` for quadratic forms
* **Communication**: :math:`O(N_{ell} \times N_{pix})` for MPI coordination

Resource Requirements
^^^^^^^^^^^^^^^^^^^^^

Typical resource needs for different analysis configurations:

.. list-table::
   :header-rows: 1
   :widths: 15 20 20 25 20

   * - Configuration
     - Memory (GB)
     - Compute Time  
     - Storage (GB)
     - Recommended Cores
   * - Quick test (nside=128)
     - ~4
     - ~1 hour
     - ~1
     - 4-8
   * - Standard (nside=512)
     - ~64
     - ~8 hours
     - ~10
     - 16-32
   * - High-res (nside=1024)
     - ~512
     - ~48 hours
     - ~100
     - 32-64
   * - Ultimate (nside=2048)
     - ~4000
     - ~2 weeks
     - ~1000
     - 64-128

Optimization Strategies
^^^^^^^^^^^^^^^^^^^^^^^

For large-scale analyses:

1. **Iterative Solvers**: Use preconditioned conjugate gradient for matrix operations
2. **Block Processing**: Process multipole ranges in blocks to reduce memory
3. **Checkpointing**: Save intermediate results for fault tolerance
4. **Load Balancing**: Distribute estimator computation optimally across processes

Validation and Quality Assessment
---------------------------------

Statistical Tests
^^^^^^^^^^^^^^^^^

The pipeline includes several validation checks:

**Unbiasedness Test**:
   Compare estimates from multiple simulations to input spectra

**Chi-squared Test**: 
   Verify statistical consistency of bandpower residuals

**Cross-Validation**:
   Compare estimates from different sky regions or data splits

**Null Tests**:
   Analyze difference maps and systematic-dominated modes

Error Analysis
^^^^^^^^^^^^^^

Comprehensive error characterization includes:

* **Statistical Errors**: From Fisher information matrix and Monte Carlo
* **Systematic Errors**: From pipeline variations and null tests  
* **Calibration Uncertainties**: Propagated through covariance matrices
* **Theoretical Modeling**: Impact of fiducial spectrum assumptions

Troubleshooting
---------------

Common Issues and Solutions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Convergence Problems**:
   * Check covariance matrix conditioning
   * Verify input map and mask consistency
   * Examine numerical precision requirements
   * Consider iterative solver tolerances

**Memory Limitations**:
   * Reduce nside or multipole range
   * Use more MPI processes for distribution
   * Enable out-of-core computation modes

**Unexpected Results**:
   * Validate input files and units
   * Run on known simulations for comparison
   * Check for systematic contamination
   * Verify calibration and beam corrections

**Performance Issues**:
   * Profile MPI communication patterns
   * Optimize I/O and memory access patterns
   * Consider alternative solver algorithms
   * Balance computation vs. communication

See Also
--------

* :doc:`spectra` : Spectra class API documentation
* :doc:`main_fisher` : Fisher matrix analysis pipeline
* :doc:`produce_mock_inputs` : Mock data generation utilities
* :mod:`cosmoforge.cosmocore.harmonic` : Spherical harmonic utilities
