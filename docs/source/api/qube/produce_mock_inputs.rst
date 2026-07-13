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

The script takes three positional arguments and one optional flag:

.. code-block:: text

   produce_mock_inputs.py <compute_case> <output_path> <config_file> [--mask]

``compute_case`` is one of ``T``, ``QU``, ``TQU`` or ``TEB``; ``output_path`` is
the directory the mock products are written to; ``config_file`` is the YAML
parameter file. ``--mask`` applies a ±30° Galactic plane cut.

.. code-block:: bash

   # Full-sky temperature mocks
   python produce_mock_inputs.py T /data/mocks config.yaml

   # Cut-sky temperature + polarization mocks
   python produce_mock_inputs.py TQU /data/mocks config.yaml --mask

The number of realizations is not a command-line option: it is taken from
``nsims`` in the configuration file.

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

The config file is a standard :class:`~cosmocore.InputParams` YAML — the same one
the Fisher and QML runs consume. Of its keys, the script reads only:

.. code-block:: yaml

   nside: 8                              # HEALPix resolution of the mocks
   labels: ["T", "E", "B"]               # fixes the number of fields
   lmax: 32                              # band-limit of the simulated alms
   nsims: 100                            # number of Monte Carlo realizations
   covmatfile1: 'inputs/TQU_NCVM1.bin'   # noise covariance to draw noise from
   covmatfile2: 'inputs/TQU_NCVM2.bin'   # second covariance (cross runs)

See ``src/cosmoforge.qube/qube/TQU_defaults.yaml`` for a complete working config.

The random seed is fixed (``master_seed = 123456789``) so that runs are
reproducible; it is not configurable. The beam is read from the packaged
``beam_440TP_pixwin16.fits`` rather than from ``beam_file``.

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

.. code-block:: bash

   # Write nside, lmax, nsims, beam and noise settings into the YAML first,
   # then generate the mocks for the case you want to analyse.
   python produce_mock_inputs.py TQU /data/mocks config.yaml --mask

Pipeline Integration
^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Complete analysis pipeline with mock data

   # 1. Generate mock inputs
   python produce_mock_inputs.py TQU /data/mocks analysis_config.yaml

   # 2. Run Fisher analysis
   mpirun -n 8 python main_fisher.py analysis_config.yaml

   # 3. Run QML analysis
   mpirun -n 16 python main_qml.py analysis_config.yaml
   
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

Scope
-----

The script generates Gaussian CMB realizations, convolves them with the beam, and adds
noise drawn from the pixel noise covariance. It models **no** foregrounds, systematic
templates, or gain/pointing errors — there is no configuration for them.

See Also
--------

* :doc:`main_fisher` : Fisher matrix analysis using mock data
* :doc:`main_qml` : QML analysis pipeline
* :mod:`cosmoforge.cosmocore.pixel` : HEALPix utilities for map generation
* :mod:`cosmoforge.cosmocore.harmonic` : Spherical harmonic operations
