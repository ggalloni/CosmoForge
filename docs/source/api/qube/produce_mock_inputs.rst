Mock Data Generation
====================

.. currentmodule:: cosmoforge.qube

The mock data generation utility provides comprehensive tools for creating realistic 
CMB simulations for testing and validation of analysis pipelines. This script generates 
all input files required for Fisher matrix and QML analysis workflows.

Overview
--------

The ``produce_mock_inputs.py`` script creates a complete set of mock input data 
suitable for CosmoForge analysis pipelines. It generates:

* Realistic CMB sky maps with temperature and polarization
* Instrumental noise covariance matrices  
* Beam transfer functions and window functions
* Analysis masks with configurable sky cuts
* Fiducial power spectra for theoretical comparison

The generated mock data maintains full statistical consistency and includes realistic 
instrumental effects, making it ideal for algorithm validation, systematic studies, 
and pipeline development.

Features
--------

Comprehensive Mock Data Generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **CMB Realizations**: Gaussian random field generation from input power spectra
* **Instrumental Effects**: Beam convolution, pixelization, and noise addition
* **Systematic Effects**: Foreground contamination, calibration errors, systematic templates
* **Multiple Datasets**: Support for cross-correlation studies with independent realizations
* **Configurable Parameters**: Flexible control over all simulation aspects

Statistical Accuracy
^^^^^^^^^^^^^^^^^^^^^

* **Exact Covariance**: Proper treatment of pixel-pixel correlations
* **Beam Effects**: Full convolution with realistic beam profiles  
* **Noise Modeling**: Correlated noise from realistic instrumental specifications
* **Finite Map Effects**: Proper treatment of partial sky coverage and masking

Integration with Analysis Pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* **Format Compatibility**: Direct integration with Fisher and QML analysis scripts
* **Configuration Consistency**: Shared parameter definitions across analysis chain
* **Validation Framework**: Built-in comparison tools for method validation

Usage
-----

Command Line Interface
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Generate mock data with default parameters
   python produce_mock_inputs.py

   # Use custom configuration file
   python produce_mock_inputs.py --config mock_config.yaml

   # Generate multiple realizations
   python produce_mock_inputs.py --nrealizations 100

   # Specify output directory
   python produce_mock_inputs.py --output-dir simulation_outputs/

Script Workflow
^^^^^^^^^^^^^^^

The generation process follows these stages:

1. **Parameter Initialization**: Load configuration and set simulation parameters
2. **Sky Model Setup**: Define fiducial cosmological model and power spectra
3. **Instrumental Model**: Specify beam, noise, and systematic properties
4. **Map Generation**: Create CMB realizations with instrumental effects
5. **Covariance Construction**: Build pixel-pixel noise covariance matrices
6. **Mask Creation**: Generate analysis masks with appropriate sky cuts
7. **Output Formatting**: Save all products in analysis-ready formats
8. **Validation**: Perform basic consistency checks on generated data

Configuration Parameters
------------------------

The script accepts comprehensive configuration through YAML files:

.. code-block:: yaml

   # Mock Data Generation Configuration
   
   # HEALPix parameters
   nside: 512
   
   # Multipole range
   lmin: 2
   lmax: 3000
   
   # Field configuration  
   nfields: 3  # T, Q, U
   physical_labels: ["T", "Q", "U"]
   
   # Cosmological model
   fiducial_spectra: "inputs/planck2018_base_plikHM_TTTEEE_lowl_lowE.txt"
   
   # Instrumental specifications
   beam_fwhm: 7.0  # arcminutes
   noise_levels:
     T: 10.0  # μK-arcmin
     P: 14.1  # μK-arcmin (Q, U)
   
   # Sky cuts and masking
   galactic_mask: "GAL070"  # Standard galactic mask
   point_source_mask: true
   custom_mask_file: null  # Optional custom mask
   
   # Systematic effects
   include_foregrounds: false
   calibration_uncertainty: 0.002  # Fractional
   polarization_efficiency: 0.99
   
   # Output configuration
   output_directory: "mock_inputs/"
   file_prefix: "mock_"
   save_intermediate: false  # Save intermediate products
   
   # Generation options
   random_seed: 12345
   nrealizations: 1
   generate_cross_data: false  # For cross-correlation tests

Generated Products
------------------

CMB Sky Maps
^^^^^^^^^^^^

**Temperature Map** (``mock_map_T.fits``):
   HEALPix format temperature map in μK units with instrumental effects applied

**Polarization Maps** (``mock_map_Q.fits``, ``mock_map_U.fits``):
   Stokes Q and U polarization maps with proper beam convolution and noise

**Map Headers**: 
   Complete FITS headers with simulation parameters, beam information, and coordinate system

Noise Covariance Matrices
^^^^^^^^^^^^^^^^^^^^^^^^^^

**Primary Covariance** (``mock_noise_cov.bin``):
   Binary format pixel-pixel noise covariance matrix including:
   - Thermal noise from detector specifications
   - Correlated noise from atmospheric fluctuations  
   - Systematic noise templates
   - Proper conditioning for numerical stability

**Format Specification**:
   Single-precision binary with row-major ordering, compatible with CosmoForge analysis tools

Instrumental Data
^^^^^^^^^^^^^^^^^

**Beam Transfer Function** (``mock_beam.fits``):
   Multipole-space beam transfer function B(ℓ) including:
   - Main beam response
   - Far sidelobe contributions
   - Polarization cross-coupling terms

**Window Functions** (``mock_windows.fits``):
   Pixel window functions for HEALPix pixelization effects

Analysis Masks
^^^^^^^^^^^^^^

**Primary Mask** (``mock_mask.fits``):
   Binary analysis mask excluding:
   - Galactic plane (configurable latitude cut)
   - Known point sources above threshold
   - Bad or missing pixels
   - Custom user-defined regions

**Mask Statistics**: 
   Sky fraction, effective area, and multipole-dependent effective area

Fiducial Spectra
^^^^^^^^^^^^^^^^^

**Power Spectra** (``mock_fiducial_spectra.txt``):
   Text format theoretical power spectra used for generation:
   - TT, EE, BB auto-spectra  
   - TE, TB, EB cross-spectra
   - Primordial and lensed components

Implementation Details
----------------------

Random Field Generation
^^^^^^^^^^^^^^^^^^^^^^^

CMB maps are generated using the standard Gaussian random field approach:

.. math::

   a_{\ell m} = \sqrt{C_\ell} \times (X_{\ell m} + i Y_{\ell m})

where :math:`X_{\ell m}`, :math:`Y_{\ell m}` are independent Gaussian random variables 
and :math:`C_\ell` are the input power spectra.

Instrumental Effects
^^^^^^^^^^^^^^^^^^^^

**Beam Convolution**:
   Spherical harmonic coefficients are multiplied by beam transfer functions:
   :math:`a_{\ell m}^{obs} = B_\ell \times a_{\ell m}^{sky}`

**Noise Addition**:
   Pixel-space noise with specified power spectrum and correlation structure

**Systematic Effects**:
   Template-based systematic contamination with configurable amplitudes

Quality Control
^^^^^^^^^^^^^^^

Generated data includes validation against input specifications:

* **Power Spectrum Recovery**: Verify generated maps match input spectra
* **Noise Properties**: Confirm noise covariance matches specifications  
* **Statistical Tests**: Chi-squared tests for Gaussianity and isotropy
* **Cross-Correlation**: Verify independence of multiple realizations

Examples
--------

Basic Mock Generation
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   """
   Example: Generate basic mock data for testing
   """
   import subprocess
   import yaml
   
   # Configure mock generation
   config = {
       'nside': 256,
       'lmax': 2000,
       'beam_fwhm': 10.0,  # arcmin
       'noise_levels': {'T': 20.0, 'P': 28.3},  # μK-arcmin
       'galactic_mask': 'GAL060',
       'output_directory': 'test_mocks/',
       'nrealizations': 5
   }
   
   with open('mock_config.yaml', 'w') as f:
       yaml.dump(config, f)
   
   # Generate mock data
   subprocess.run(['python', 'produce_mock_inputs.py', 
                  '--config', 'mock_config.yaml'])

Systematic Studies
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   """
   Example: Generate mocks for systematic error studies
   """
   
   # Test different noise levels
   noise_levels = [
       {'T': 10.0, 'P': 14.1},  # Optimistic
       {'T': 20.0, 'P': 28.3},  # Realistic  
       {'T': 40.0, 'P': 56.6}   # Conservative
   ]
   
   for i, noise in enumerate(noise_levels):
       config = base_config.copy()
       config['noise_levels'] = noise
       config['output_directory'] = f'systematic_study/noise_case_{i}/'
       
       with open(f'mock_config_noise_{i}.yaml', 'w') as f:
           yaml.dump(config, f)
       
       subprocess.run(['python', 'produce_mock_inputs.py',
                      '--config', f'mock_config_noise_{i}.yaml'])

Cross-Correlation Testing
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Generate independent datasets for cross-correlation
   
   # Dataset 1
   python produce_mock_inputs.py --config cross_config.yaml \
     --output-dir cross_test/dataset1/ --random-seed 12345
   
   # Dataset 2 (independent realization)
   python produce_mock_inputs.py --config cross_config.yaml \
     --output-dir cross_test/dataset2/ --random-seed 67890

Pipeline Integration
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Complete analysis pipeline with mock data
   
   # 1. Generate mock inputs
   python produce_mock_inputs.py --config analysis_config.yaml
   
   # 2. Run Fisher analysis
   mpirun -n 8 python main_fisher.py --config analysis_config.yaml
   
   # 3. Run QML analysis  
   mpirun -n 16 python main_qml.py --config analysis_config.yaml
   
   # 4. Compare results with known input

Validation Framework
--------------------

Statistical Validation
^^^^^^^^^^^^^^^^^^^^^^^

The script includes comprehensive validation tools:

**Power Spectrum Check**:
   Compare recovered power spectra from generated maps with input

**Noise Validation**:
   Verify noise covariance properties through multiple realizations

**Gaussianity Tests**:
   Statistical tests for non-Gaussian signatures

**Isotropy Verification**:
   Check for spurious anisotropic signatures

Quality Metrics
^^^^^^^^^^^^^^^

Standard quality assessment includes:

* **Bias Estimation**: Mean deviation from input spectra
* **Variance Check**: Consistency with theoretical predictions  
* **Systematic Residuals**: Search for coherent deviations
* **Cross-Correlation**: Verify independence of separate realizations

Performance Optimization
-------------------------

Computational Efficiency
^^^^^^^^^^^^^^^^^^^^^^^^^

* **FFT Operations**: Optimized spherical harmonic transforms
* **Memory Management**: Efficient handling of large arrays
* **Parallel Generation**: Multi-threading for multiple realizations
* **I/O Optimization**: Efficient binary file operations

Resource Requirements
^^^^^^^^^^^^^^^^^^^^^

Typical resource needs for different configurations:

.. list-table::
   :header-rows: 1
   :widths: 15 20 20 25 20

   * - Resolution
     - Memory (GB)
     - Generation Time
     - Storage (GB)
     - Notes
   * - nside=128
     - ~1
     - ~5 min
     - ~0.1
     - Quick testing
   * - nside=256
     - ~4
     - ~15 min
     - ~0.5
     - Standard validation
   * - nside=512
     - ~16
     - ~1 hour
     - ~2
     - Production analysis
   * - nside=1024
     - ~64
     - ~4 hours
     - ~8
     - High-resolution studies

Advanced Features
-----------------

Foreground Modeling
^^^^^^^^^^^^^^^^^^^

Optional foreground contamination modeling:

.. code-block:: yaml

   # Foreground configuration
   include_foregrounds: true
   foreground_models:
     synchrotron: 
       amplitude: 20.0  # μK at 408 MHz
       spectral_index: -3.0
     thermal_dust:
       amplitude: 50.0  # μK at 545 GHz  
       spectral_index: 1.6
       temperature: 20.0  # K

Systematic Templates
^^^^^^^^^^^^^^^^^^^^

User-defined systematic contamination:

.. code-block:: yaml

   # Systematic effects
   systematic_templates:
     - name: "gain_variation"
       amplitude: 0.01  # Fractional
       template_file: "gain_template.fits"
     - name: "pointing_error"  
       amplitude: 2.0   # arcsec RMS
       correlation_length: 10.0  # degrees

Calibration Uncertainties
^^^^^^^^^^^^^^^^^^^^^^^^^^

Realistic calibration error modeling:

.. code-block:: yaml

   # Calibration parameters
   calibration:
     gain_uncertainty: 0.002      # Fractional
     polarization_angle: 0.5      # degrees  
     polarization_efficiency: 0.99
     beam_uncertainty: 0.01       # Fractional FWHM

See Also
--------

* :doc:`main_fisher` : Fisher matrix analysis using mock data
* :doc:`main_qml` : QML analysis pipeline
* :mod:`cosmoforge.cosmocore.pixel` : HEALPix utilities for map generation
* :mod:`cosmoforge.cosmocore.harmonic` : Spherical harmonic operations
