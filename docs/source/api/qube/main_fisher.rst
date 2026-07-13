Fisher Matrix Analysis Pipeline
=================================

.. currentmodule:: cosmoforge.qube

The Fisher matrix analysis pipeline provides a complete framework for power spectrum
parameter forecasting from partial-sky observations of spin-0 and spin-2 fields.
This script orchestrates the full computation workflow from input validation through
results output.

Overview
--------

The ``main_fisher.py`` script implements a comprehensive Fisher information matrix 
computation pipeline optimized for cosmological parameter forecasting. It handles:

* Multi-field CMB analysis (temperature and polarization)
* MPI parallelization for large-scale computations  
* Flexible input/output file management
* Robust error handling and validation
* Integration with the broader CosmoForge analysis framework

Mathematical Foundation
-----------------------

The pipeline computes the Fisher information matrix:

.. math::

   F_{ij} = \frac{1}{2} \text{Tr}\left[ \mathbf{C}^{-1} \frac{\partial \mathbf{C}}{\partial \theta_i} \mathbf{C}^{-1} \frac{\partial \mathbf{C}}{\partial \theta_j} \right]

This quantifies the information content of observations about cosmological parameters 
:math:`\theta_i`, providing the foundation for parameter constraint forecasts and 
optimal survey design.

Usage
-----

Command Line Interface
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Single-process execution
   python main_fisher.py

   # MPI parallel execution  
   mpirun -n 8 python main_fisher.py

   # With a custom configuration (optional positional argument)
   python main_fisher.py custom_config.yaml

Script Workflow
^^^^^^^^^^^^^^^

The execution pipeline follows these stages:

1. **Environment Setup**: Initialize MPI communicator and process management
2. **Configuration Loading**: Parse YAML configuration with validation
3. **Input Verification**: Check file existence and format compatibility  
4. **Fisher Initialization**: Create Fisher analysis object with validated parameters
5. **Computation Execution**: Run distributed Fisher matrix calculation
6. **Results Collection**: Gather and synchronize results across processes
7. **Output Generation**: Save Fisher matrix and derived quantities
8. **Cleanup**: Release resources and finalize MPI environment

Configuration Requirements
--------------------------

The script expects a YAML configuration file with the following structure:

.. code-block:: yaml

   # Analysis configuration for Fisher matrix computation
   
   # HEALPix parameters
   nside: 512
   lmin: 2  
   lmax: 3000
   
   # Field specification
   nfields: 3
   physical_labels: ["T", "Q", "U"]
   
   # Analysis mode
   do_cross: false  # Auto-correlation analysis
   
   # Input files (relative to script directory)
   covmatfile1: "inputs/noise_covariance.bin"
   clfile: "inputs/fiducial_spectra.txt"
   beamfile: "inputs/beam_transfer.fits"  
   maskfile: "inputs/analysis_mask.fits"
   
   # Output files
   outfilefisher: "outputs/fisher_matrix.dat"
   outinvcovmatfile1: "outputs/inverse_covariance.bin"
   outgeometryfile: "outputs/analysis_geometry.dat"

Implementation Details
----------------------

Key Components
^^^^^^^^^^^^^^

The script integrates several critical components:

* **MPI Management**: Distributed computation with proper process synchronization
* **File I/O**: Robust handling of binary covariance matrices and FITS files
* **Memory Management**: Efficient allocation for large covariance matrices
* **Error Recovery**: Graceful handling of computational failures
* **Progress Monitoring**: Status reporting for long-running computations

Performance Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^

The pipeline includes several optimization strategies:

* **Lazy Loading**: Input files loaded only when needed
* **Memory Pooling**: Reuse of large array allocations  
* **Process Distribution**: Optimal work distribution across MPI ranks
* **I/O Optimization**: Parallel file operations where possible

Error Handling
^^^^^^^^^^^^^^

Comprehensive error handling covers:

* **Configuration Validation**: Parameter range and type checking
* **File System Errors**: Missing files, permission issues, corrupted data
* **Memory Errors**: Insufficient memory for large matrices
* **MPI Errors**: Process failures, communication timeouts
* **Numerical Errors**: Matrix singularities, convergence failures

Output Products
---------------

Standard Outputs
^^^^^^^^^^^^^^^^

When the corresponding ``out*`` paths are set (persistence is opt-in, ADR-0015),
the pipeline writes several output files:

1. **Fisher Matrix** (``fisher_matrix.dat``): 
   
   - Text format with parameter labels
   - Full Fisher information matrix F_ij
   - Parameter error estimates (diagonal of F^-1)
   
2. **Inverse Covariance** (``inverse_covariance.bin``):
   
   - Binary format for computational efficiency
   - Useful for subsequent QML analysis
   - Includes proper conditioning and regularization

3. **Geometry Information** (``analysis_geometry.dat``):
   
   - Pixel selection and indexing
   - Multipole binning specification  
   - Analysis mask and beam information

Diagnostic Outputs
^^^^^^^^^^^^^^^^^^

Additional diagnostic information includes:

* **Computation Log**: Timing, memory usage, convergence metrics
* **Parameter Summary**: Input configuration and derived quantities
* **Quality Metrics**: Matrix condition numbers, eigenvalue analysis

Examples
--------

Basic Fisher Analysis
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   """
   Example: Standard Fisher matrix computation for CMB forecasting
   """
   import subprocess
   import yaml
   
   # Create configuration
   config = {
       'nside': 256,
       'lmin': 2,
       'lmax': 2000,
       'nfields': 3,
       'physical_labels': ['T', 'Q', 'U'],
       'do_cross': False,
       'covmatfile1': 'inputs/planck_noise.bin',
       'clfile': 'inputs/planck2018_spectra.txt',
       'beamfile': 'inputs/planck_beam.fits',
       'outfilefisher': 'outputs/planck_fisher.dat'
   }
   
   # Save configuration
   with open('fisher_config.yaml', 'w') as f:
       yaml.dump(config, f)
   
   # Run Fisher analysis
   subprocess.run(['python', 'main_fisher.py', 'fisher_config.yaml'])

Cross-Survey Analysis
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Compare different survey configurations
   
   # Current generation (Planck-like)
   python main_fisher.py configs/current_survey.yaml
   
   # Next generation (LiteBIRD/CMB-S4)  
   mpirun -n 16 python main_fisher.py configs/future_survey.yaml
   
   # Ultimate precision (post-CMB-S4)
   mpirun -n 32 python main_fisher.py configs/ultimate_survey.yaml

Systematic Studies
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   """
   Example: Systematic error impact on Fisher forecasts
   """
   
   # Test different analysis choices
   systematics = {
       'lmax_test': [1500, 2000, 2500, 3000],
       'beam_uncertainty': [0.0, 0.5, 1.0, 2.0],  # arcmin FWHM
       'noise_scaling': [0.8, 1.0, 1.2, 1.5]
   }
   
   for lmax in systematics['lmax_test']:
       config['lmax'] = lmax
       config['outfilefisher'] = f'outputs/fisher_lmax{lmax}.dat'
       
       with open(f'config_lmax{lmax}.yaml', 'w') as f:
           yaml.dump(config, f)
       
       subprocess.run(['mpirun', '-n', '8', 'python', 'main_fisher.py',
                      f'config_lmax{lmax}.yaml'])

Performance Guidelines
----------------------

Resource Requirements
^^^^^^^^^^^^^^^^^^^^^

Typical resource needs for different analysis scales:

.. list-table::
   :header-rows: 1
   :widths: 15 20 20 25 20

   * - Analysis Scale
     - Active Pixels  
     - Memory (GB)
     - Compute Time
     - Recommended Cores
   * - Quick test
     - ~10k
     - ~1
     - ~10 min
     - 1-4
   * - Standard analysis
     - ~200k  
     - ~16
     - ~4 hours
     - 8-16
   * - High resolution
     - ~800k
     - ~128
     - ~24 hours  
     - 16-32
   * - Ultimate precision
     - ~3M
     - ~1000
     - ~1 week
     - 32-64

Optimization Tips
^^^^^^^^^^^^^^^^^

For optimal performance:

1. **Memory**: Ensure sufficient RAM for covariance matrices (8×N_pix² bytes)
2. **Storage**: Use fast I/O for large binary files (covariance matrices)  
3. **Network**: High-bandwidth interconnect for MPI communication
4. **CPU**: Prefer many cores over high clock speeds
5. **Scheduling**: Request appropriate walltime for job completion

Troubleshooting
---------------

Common Issues
^^^^^^^^^^^^^

**Memory Errors**:
   * Reduce nside or lmax parameters
   * Use more MPI processes to distribute memory load
   * Enable swap space (not recommended for production)

**Convergence Problems**:
   * Check covariance matrix conditioning  
   * Verify input file formats and units
   * Examine numerical precision requirements

**MPI Communication Failures**:
   * Verify MPI installation and configuration
   * Check network connectivity between nodes
   * Monitor for process failures or timeouts

**File I/O Errors**:
   * Confirm file paths and permissions
   * Verify sufficient disk space for outputs
   * Check file system compatibility (NFS issues)

See Also
--------

* :doc:`fisher` : Fisher class API documentation
* :doc:`main_qml` : QML analysis pipeline  
* :doc:`produce_mock_inputs` : Mock data generation
* :mod:`cosmoforge.cosmocore` : Core computational framework
